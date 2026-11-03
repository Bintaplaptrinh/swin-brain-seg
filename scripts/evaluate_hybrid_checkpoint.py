"""Read-only interim evaluation using the trainer's native-size test protocol."""
import argparse
from pathlib import Path
import gc
import json
import torch
from PIL import Image

from src.datasets.brisc import PLANES, Sample, discover, restore_split
from src.utils.metrics import summarize
from src.trainers.baseline import evaluate, loader, write_json, write_rows
from src.trainers.swin_hybrid import PROTOCOL, make_model
from src.core.paths import relocate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-no-tumor', action='store_true')
    options = parser.parse_args()
    checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=True)
    config = relocate_config(checkpoint['config'])
    if config['protocol'] != PROTOCOL:
        raise ValueError('Unsupported checkpoint protocol')
    args = argparse.Namespace(**config)
    args.tokenizer_checkpoint = Path(args.tokenizer_checkpoint)
    args.batch_size, args.workers = 1, 0
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    options.output.mkdir(parents=True, exist_ok=True)
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
    model = make_model(args)
    if model.tokenizer.audit['sha256'] != checkpoint['tokenizer_sha256']:
        raise ValueError('Tokenizer source differs from training')
    model.load_state_dict(checkpoint['evaluation_model'], strict=True)
    metadata = dict(selected_epoch=epoch, selected_weights=weights,
                    checkpoint=str(options.checkpoint.resolve()),
                    protocol=PROTOCOL, interim_evaluation=True,
                    selection='latest completed checkpoint; not best-validation selection',
                    validation=checkpoint['validation'], training_config=config,
                    parameter_counts=model.parameter_counts,
                    evaluation_config=dict(device=args.device, batch_size=1, workers=0,
                                           size=args.size, amp=args.amp),
                    efficiency_note='Latency not benchmarked while training shares the GPU.')
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
