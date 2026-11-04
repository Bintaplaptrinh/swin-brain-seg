"""Export the trained Swin V2 classifier for native CADx inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.core.paths import DATA_ROOT
from src.models.swin_classifier import (
    DEFAULT_BACKBONE,
    DEFAULT_CLASSIFIER_CHECKPOINT,
    DEFAULT_CLASSIFIER_THRESHOLD,
    SwinTumorClassifier,
    SwinTumorDataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT.with_suffix(".onnx"))
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Export already exists: {args.output}")

    torch.set_num_threads(4)
    model = SwinTumorClassifier(pretrained=False, dropout=0.0,
                                checkpoint_path=args.checkpoint, backbone_name=DEFAULT_BACKBONE).eval()
    dataset = SwinTumorDataset(DATA_ROOT, split="test")
    indices = [next(i for i, (_, label) in enumerate(dataset.samples) if label == target) for target in (0, 1)]
    samples = [dataset[i][0].unsqueeze(0) for i in indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Validate in memory before publishing the export.
    graph = io.BytesIO()
    with torch.inference_mode():
        torch.onnx.export(model, samples[0], graph, input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamo=False)
        contents = graph.getvalue()
        onnx.checker.check_model(onnx.load_from_string(contents))
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 4
        session = ort.InferenceSession(contents, sess_options=settings, providers=["CPUExecutionProvider"])
        errors = []
        for sample in samples:
            expected = model(sample).numpy()
            actual = session.run(["logits"], {"image": sample.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-4)
            threshold_logit = torch.logit(torch.tensor(DEFAULT_CLASSIFIER_THRESHOLD)).item()
            np.testing.assert_array_equal(actual >= threshold_logit, expected >= threshold_logit)
            errors.append(float(np.max(np.abs(actual - expected))))

    metadata = {"format": "brain-swinv2-classifier-onnx-v1",
                "model_sha256": hashlib.sha256(contents).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "validation_max_abs_logit_error": max(errors)}
    args.output.write_bytes(contents)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
