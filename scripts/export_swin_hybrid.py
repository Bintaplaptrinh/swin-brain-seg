"""Export saved Swin Hybrid evaluation weights to a fixed-size ONNX graph."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from src.core.paths import ROOT, relocate_config
from src.datasets.brisc import BRISCDataset, discover
from src.trainers.swin_hybrid import PROTOCOL, make_model


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images)
        return output[0] if isinstance(output, (list, tuple)) else output


class FixedContext(nn.Module):
    """Equivalent divisible average pooling for this fixed-resolution export."""
    def __init__(self, context, size):
        super().__init__()
        self.original = context
        self.size = size // 32

    def forward(self, skips):
        c = self.original
        maps = [proj(F.avg_pool2d(x, 2 ** (3 - index)))
                for index, (proj, x) in enumerate(zip(c.projections, skips[2:]))]
        global_map = c.global_projection(skips[-1].mean((2, 3), keepdim=True)).expand(-1, -1, self.size, self.size)
        return skips[-1] + c.scale * c.mix(torch.cat([*maps, global_map], 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/last.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/checkpoints/exports/swin_hybrid_epoch159.onnx")
    cli = parser.parse_args(argv)
    if cli.output.exists():
        raise FileExistsError(f"Export already exists: {cli.output}")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    config = relocate_config(checkpoint["config"])
    if config.get("protocol") != PROTOCOL or config.get("smoke_test"):
        raise ValueError("Export requires a full-data Swin Hybrid v2 checkpoint")
    args = argparse.Namespace(**config)
    args.device, args.tokenizer_checkpoint = "cpu", Path(config["tokenizer_checkpoint"])
    torch.set_num_threads(4)
    model = make_model(args).eval()
    model.load_state_dict(checkpoint["evaluation_model"], strict=True)
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
    wrapper = LogitsOnly(model).eval()
    # Real deterministic BRISC input checks the export against the training preprocessing.
    sample = BRISCDataset(discover(config["data_root"], "test")[:1], size=args.size)[0][0][None]
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), warnings.catch_warnings():
        expected = wrapper(sample).numpy()
        model.decoder.context = FixedContext(model.decoder.context, args.size)
        np.testing.assert_allclose(wrapper(sample).numpy(), expected, atol=1e-4, rtol=1e-4)
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(wrapper, sample, str(cli.output), input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load(str(cli.output))
    onnx.checker.check_model(graph)
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    session = ort.InferenceSession(str(cli.output), sess_options=settings, providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"image": sample.numpy()})[0]
    if not np.isfinite(actual).all():
        raise ValueError("ONNX output contains nonfinite logits")
    error = float(np.max(np.abs(expected - actual)))
    agreement = float((expected.argmax(1) == actual.argmax(1)).mean())
    if error > .01 or agreement < .9999:
        raise ValueError(f"ONNX parity failed: max_abs_error={error}, argmax_agreement={agreement}")
    metadata = {
        "format": "brain-swin-hybrid-onnx-v1", "protocol": PROTOCOL,
        "checkpoint": str(cli.checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(cli.output.read_bytes()).hexdigest(),
        "epoch": checkpoint["epoch"], "weights": checkpoint["evaluation_weights"],
        "size": args.size, "input": "image", "output": "logits", "input_shape": [1, 1, args.size, args.size],
        "output_shape": [1, 2, args.size, args.size], "precision": "float32",
        "preprocessing": "Pillow-compatible grayscale/antialiased bilinear letterbox; nonzero per-slice zscore",
        "postprocessing": "crop letterbox; bilinear restore logits (align_corners=false); argmax; PNG 0/255",
        "validation": {"real_test_input": discover(config["data_root"], "test")[0].image.name,
                       "max_abs_logit_error": error, "argmax_agreement": agreement,
                       "onnxruntime": ort.__version__},
    }
    cli.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
