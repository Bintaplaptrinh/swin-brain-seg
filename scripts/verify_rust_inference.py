"""Verify native Rust preprocessing and native-size masks against Python/ONNX."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torch.nn import functional as F

from src.core.paths import DATA_ROOT, ROOT
from src.datasets.brisc import BRISCDataset, Sample, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "inference/target/release/brain-inference.exe")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/checkpoints/exports/swin_hybrid_epoch159.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/runs/rust_parity")
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError("Choose a new parity output directory")
    source = DATA_ROOT / "segmentation_task/test/images"
    sources = [next(source.glob(f"*_pi_*.jpg")), next(source.glob("*_gl_*.jpg")), next(source.glob("*_me_*.jpg"))]
    cases = cli.output / "inputs"
    masks = cli.output / "predictions"
    cases.mkdir(parents=True)
    for index, path in enumerate(sources):
        with Image.open(path) as original:
            original.save(cases / f"case_{index}.png")
            if index == 0:
                original.resize((389, 277), Image.Resampling.BILINEAR).save(cases / "non_square.png")
                original.resize((701, 937), Image.Resampling.BILINEAR).save(cases / "downsample.png")
    # Also exercise the Rust JPEG decoder on the unmodified dataset image.
    (cases / "original.jpg").write_bytes(sources[1].read_bytes())
    runtime = Path(ort.__file__).parent / "capi/onnxruntime.dll"
    if not runtime.exists():
        runtime = next((Path(ort.__file__).parent / "capi").glob("libonnxruntime.so*"))
    subprocess.run([str(cli.binary), "--model", str(cli.model), "--input", str(cases),
                    "--output", str(masks), "--ort-dylib", str(runtime), "--segment-only", "--dump-input"], check=True)
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    session = ort.InferenceSession(str(cli.model), sess_options=settings, providers=["CPUExecutionProvider"])
    size = json.loads(cli.model.with_suffix(".json").read_text())["size"]
    reports = []
    for path in sorted(cases.iterdir()):
        with Image.open(path) as source_image:
            width, height = source_image.size
        # The image itself is a harmless dummy mask; only deterministic inputs are used.
        sample = Sample(path, path, "Glioma", "Axial")
        expected = BRISCDataset([sample], size=size)[0][0][None].numpy()
        native = np.fromfile(masks / f"{path.stem}.f32le", dtype="<f4").reshape(expected.shape)
        input_error = float(np.max(np.abs(expected - native)))
        mean_input_error = float(np.mean(np.abs(expected - native)))
        logits = torch.from_numpy(session.run(["logits"], {"image": expected})[0])
        rh, rw, top, left = geometry(height, width, size)
        restored = F.interpolate(logits[:, :, top:top + rh, left:left + rw], (height, width), mode="bilinear", align_corners=False)
        expected_mask = restored.argmax(1)[0].numpy() > 0
        with Image.open(masks / f"{path.stem}_mask.png") as native_image:
            native_mask = np.asarray(native_image) > 0
        agreement = float((expected_mask == native_mask).mean())
        report = {"image": path.name, "native_shape": [height, width],
                  "max_abs_preprocessing_error": input_error, "mean_abs_preprocessing_error": mean_input_error,
                  "native_mask_agreement": agreement}
        reports.append(report)
        # Independent JPEG decoders can round a black pixel to 1, which becomes
        # nonzero after z-score normalization. Check mean error plus final mask
        # agreement for JPEG; lossless PNG must match pointwise as well.
        input_failed = mean_input_error > .002 if path.suffix.lower() == ".jpg" else input_error > .002
        if input_failed or agreement < .9999:
            raise AssertionError(f"Rust parity failed: {report}")
    (cli.output / "parity.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
