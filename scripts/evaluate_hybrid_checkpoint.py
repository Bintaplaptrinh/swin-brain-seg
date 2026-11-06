"""Evaluate a Swin Hybrid checkpoint or the complete CADx pipeline at native size."""
import argparse
from pathlib import Path
import gc
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from src.datasets.brisc import PLANES, Sample, discover, load_mask, restore_split
from src.utils.metrics import (binary_metrics, classifier_confusion_matrix,
                                plot_classifier_confusion_matrix,
                                print_classifier_confusion_matrix, summarize)
from src.trainers.baseline import evaluate, loader, write_json, write_rows
from src.trainers.swin_hybrid import PROTOCOL, make_model
from src.core.paths import relocate_config


def evaluate_cadx(test, pipeline, tolerance):
    rows = []
    for sample in tqdm(test, desc="CADx test"):
        result = pipeline.predict(sample.image)
        segmentation = result["stage2_segmentation"]
        target = load_mask(sample.mask)
        prediction = np.zeros_like(target) if segmentation is None else segmentation["mask"]
        rows.append(dict(filename=sample.image.name, tumor=sample.tumor, plane=sample.plane,
                         segmenter_ran=segmentation is not None,
                         **binary_metrics(prediction, target, tolerance)))
    return summarize(rows), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-no-tumor', action='store_true')
    parser.add_argument('--classifier-checkpoint', type=Path,
                        help='Evaluate the two-stage CADx pipeline with these classifier weights')
    options = parser.parse_args()
    if options.classifier_checkpoint and not options.include_no_tumor:
        parser.error('--classifier-checkpoint requires --include-no-tumor')
    checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=True)
    config = relocate_config(checkpoint['config'])
    if config['protocol'] != PROTOCOL:
        raise ValueError('Unsupported checkpoint protocol')
    args = argparse.Namespace(**config)
    args.tokenizer_checkpoint = Path(args.tokenizer_checkpoint)
    args.batch_size, args.workers = 1, 0
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    options.output.mkdir(parents=True, exist_ok=False)
    epoch, weights = checkpoint['epoch'], checkpoint['evaluation_weights']
    print(f'Evaluating epoch {epoch}, {weights}, size {args.size}', flush=True)
    _, _, test = restore_split(discover(Path(args.data_root), 'train'),
                              discover(Path(args.data_root), 'test'), checkpoint['splits'])
    if options.include_no_tumor:
        classification = Path(args.data_root)/'classification_task'/'test'
        all_images = sorted(classification.rglob('*.jpg'))
        positive_lookup = {s.image.name: s for s in test}
        expected_positive = {p.name for p in all_images if p.parent.name != 'no_tumor'}
        if expected_positive != set(positive_lookup):
            raise ValueError('Classification and segmentation positives differ')
        for image in all_images:
            if image.name in positive_lookup:
                if image.read_bytes() != positive_lookup[image.name].image.read_bytes():
                    raise ValueError(f'Classification image differs: {image.name}')
        mask_dir = options.output/'generated_no_tumor_masks'
        mask_dir.mkdir(exist_ok=True)
        negatives = []
        for image in all_images:
            if image.parent.name == 'no_tumor':
                mask = mask_dir/(image.stem+'.png')
                with Image.open(image) as source:
                    Image.new('L', source.size, 0).save(mask)
                negatives.append(Sample(image, mask, 'No Tumor', PLANES[image.stem.split('_')[-2]]))
        test = sorted(test+negatives, key=lambda s: s.image.name)
        if len(test) != 1000 or len(negatives) != 140:
            raise ValueError('Unexpected full-test membership')
        write_json(options.output/'test_manifest.json',
                   {'images': [str(s.image.resolve()) for s in test],
                    'negative_ground_truth': 'Empty masks generated from no_tumor classification labels'})
    metadata = dict(selected_epoch=epoch, selected_weights=weights,
                    checkpoint=str(options.checkpoint.resolve()),
                    protocol=PROTOCOL, validation=checkpoint['validation'], training_config=config)
    if options.classifier_checkpoint:
        from src.pipeline import CADxPipeline

        del checkpoint
        gc.collect()
        pipeline = CADxPipeline(classifier_checkpoint=options.classifier_checkpoint,
                                segmenter_checkpoint=options.checkpoint, device=args.device)
        metrics, rows = evaluate_cadx(test, pipeline, args.nsd_tolerance)
        # --- Image-level confusion matrix for Stage 1 classifier ---
        if options.include_no_tumor:
            cm = classifier_confusion_matrix(rows)
            print_classifier_confusion_matrix(cm, threshold=pipeline.threshold)
            metrics['classifier_confusion_matrix'] = cm
            write_json(options.output / 'classifier_confusion_matrix.json', cm)
            png_path = plot_classifier_confusion_matrix(
                cm,
                options.output / 'classifier_confusion_matrix.png',
                threshold=pipeline.threshold,
            )
            print(f'Confusion matrix image saved → {png_path}', flush=True)
        metadata.update(classifier_checkpoint=str(options.classifier_checkpoint.resolve()),
                        classifier_threshold=pipeline.threshold, segmenter_runs=sum(r['segmenter_ran'] for r in rows),
                        skipped_tumor_images=sum(not r['segmenter_ran'] and r['tumor'] != 'No Tumor' for r in rows),
                        evaluation_config=dict(device=args.device, batch_size=1, size=args.size, amp=False))
    else:
        model = make_model(args)
        if model.tokenizer.audit['sha256'] != checkpoint['tokenizer_sha256']:
            raise ValueError('Tokenizer source differs from training')
        model.load_state_dict(checkpoint['evaluation_model'], strict=True)
        metadata.update(interim_evaluation=True,
                        selection='latest completed checkpoint; not best-validation selection',
                        efficiency_note='Latency not benchmarked while training shares the GPU.',
                        parameter_counts=model.parameter_counts,
                        evaluation_config=dict(device=args.device, batch_size=1, workers=0,
                                               size=args.size, amp=args.amp))
        del checkpoint
        gc.collect()
        batches = loader(test, args, torch.Generator().manual_seed(args.seed+2))
        metrics, rows = evaluate(model, batches, args, f'Test epoch {epoch} {weights}')
    if options.include_no_tumor:
        negative_rows = [r for r in rows if r['tumor'] == 'No Tumor']
        negatives = summarize(negative_rows)
        metrics['per_class']['No Tumor'] = {k: negatives['overall'][k] for k in ('images', 'mIoU', 'Dice')}
        metrics['Weighted_mIoU'] = metrics['overall']['mIoU']
        metrics['tumor_only'] = summarize([r for r in rows if r['tumor'] != 'No Tumor'])
        metrics['no_tumor'] = dict(negatives['overall'],
                                 false_positive_images=sum(not r['empty_prediction'] for r in negative_rows),
                                 image_false_positive_rate=sum(not r['empty_prediction'] for r in negative_rows)/len(negative_rows),
                                 ground_truth='Empty masks from classification no_tumor labels',
                                 positive_rule='Any predicted foreground pixel; no postprocessing')
        metadata['metric_note'] = 'Per-image IoU/Dice: both empty = 1, false positive on empty target = 0. Overall includes all 1000 images.'
    write_json(options.output/'test_metrics.json', {**metadata, **metrics})
    write_rows(options.output/'test_per_image.csv', rows)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
