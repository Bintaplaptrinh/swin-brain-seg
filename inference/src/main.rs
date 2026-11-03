mod preprocess;

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use image::GrayImage;
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(about = "Segment brain MRI slices with the exported Swin Hybrid checkpoint")]
struct Cli {
    #[arg(
        long,
        default_value = "outputs/checkpoints/exports/swin_hybrid_epoch159.onnx"
    )]
    model: PathBuf,
    /// One PNG/JPEG image, or a directory of PNG/JPEG images (nonrecursive).
    #[arg(long)]
    input: PathBuf,
    /// Destination directory for native-resolution binary masks and JSON reports.
    #[arg(long)]
    output: PathBuf,
    /// Path to onnxruntime.dll, libonnxruntime.so or libonnxruntime.dylib.
    #[arg(long)]
    ort_dylib: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cpu")]
    device: Device,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Write the normalized input as little-endian float32 for parity diagnostics.
    #[arg(long)]
    dump_input: bool,
}

#[derive(Deserialize)]
struct Metadata {
    format: String,
    size: usize,
    epoch: usize,
    weights: String,
    model_sha256: String,
}

fn grayscale(path: &Path) -> Result<GrayImage> {
    let decoded = image::open(path).with_context(|| format!("Cannot decode {}", path.display()))?;
    let rgb = decoded.to_rgb8();
    // Match Pillow's ITU-R 601 conversion rather than image crate's Rec.709 default.
    Ok(GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let p = rgb.get_pixel(x, y);
        image::Luma([((19595u32 * p[0] as u32
            + 38470u32 * p[1] as u32
            + 7471u32 * p[2] as u32
            + 32768)
            >> 16) as u8])
    }))
}

fn images(input: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = if input.is_file() {
        vec![input.to_path_buf()]
    } else {
        fs::read_dir(input)?
            .map(|v| v.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?
    };
    paths.retain(|p| {
        p.is_file()
            && p.extension().is_some_and(|e| {
                ["png", "jpg", "jpeg"]
                    .iter()
                    .any(|s| e.eq_ignore_ascii_case(s))
            })
    });
    paths.sort();
    ensure!(!paths.is_empty(), "No PNG/JPEG inputs found");
    let mut names = HashSet::new();
    for path in &paths {
        ensure!(
            names.insert(path.file_stem().unwrap().to_os_string()),
            "Duplicate input stem: {}",
            path.display()
        );
    }
    Ok(paths)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.threads > 0, "--threads must be positive");
    let metadata: Metadata = serde_json::from_slice(
        &fs::read(cli.model.with_extension("json"))
            .context("Missing model metadata; export with python -m scripts.export_swin_hybrid")?,
    )?;
    ensure!(
        metadata.format == "brain-swin-hybrid-onnx-v1",
        "Unsupported export format"
    );
    ensure!(
        metadata.size >= 64 && metadata.size % 32 == 0,
        "Invalid model input size"
    );
    let digest = format!("{:x}", Sha256::digest(fs::read(&cli.model)?));
    ensure!(
        digest == metadata.model_sha256,
        "ONNX checksum differs from export metadata"
    );
    if let Some(path) = &cli.ort_dylib {
        ort::init_from(path.to_string_lossy()).commit()?;
    }
    let builder = Session::builder()?.with_intra_threads(cli.threads)?;
    let builder = match cli.device {
        Device::Cpu => builder,
        Device::Cuda => {
            #[cfg(feature = "cuda")]
            {
                builder.with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default()
                        .build()
                        .error_on_failure(),
                ])?
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA support requires cargo build --release --features cuda");
        }
    };
    let mut session = builder
        .commit_from_file(&cli.model)
        .context("Cannot load ONNX graph")?;
    let paths = images(&cli.input)?;
    // Validate all destinations before producing the first output.
    for path in &paths {
        let stem = path.file_stem().unwrap().to_string_lossy();
        for suffix in ["_mask.png", ".json", ".f32le"] {
            ensure!(
                !cli.output.join(format!("{stem}{suffix}")).exists(),
                "Output already exists for {stem}"
            );
        }
    }
    fs::create_dir_all(&cli.output)?;
    for path in paths {
        let start = Instant::now();
        let image = grayscale(&path)?;
        let (input, geometry) = preprocess::prepare(&image, metadata.size);
        let stem = path.file_stem().unwrap().to_string_lossy();
        if cli.dump_input {
            let bytes: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
            fs::write(cli.output.join(format!("{stem}.f32le")), bytes)?;
        }
        let tensor = Tensor::from_array(([1usize, 1, metadata.size, metadata.size], input))?;
        let forward_start = Instant::now();
        let outputs = session.run(ort::inputs!["image" => tensor])?;
        let forward_seconds = forward_start.elapsed().as_secs_f64();
        let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, 2, metadata.size as i64, metadata.size as i64],
            "Unexpected output shape: {shape:?}"
        );
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "Nonfinite model logits"
        );
        let mask = preprocess::restore(logits, geometry);
        let target = cli.output.join(format!("{stem}_mask.png"));
        mask.save(&target)?;
        let report = json!({"input": path, "mask": target, "model": cli.model,
            "epoch": metadata.epoch, "weights": metadata.weights, "device": format!("{:?}", cli.device),
            "width": image.width(), "height": image.height(),
            "foreground_pixels": mask.as_raw().iter().filter(|p| **p > 0).count(),
            "forward_seconds": forward_seconds, "total_seconds": start.elapsed().as_secs_f64()});
        fs::write(
            cli.output.join(format!("{stem}.json")),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{}", report);
    }
    Ok(())
}
