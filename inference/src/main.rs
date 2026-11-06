mod clahe;
#[cfg(feature = "gui")]
mod gui;
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
#[command(about = "Screen brain MRI slices and segment classifier-positive slices")]
struct Cli {
    /// Open the native desktop interface (requires the gui feature).
    #[arg(long, conflicts_with_all = ["input", "output", "segment_only", "dump_input"])]
    gui: bool,
    /// Enable CLAHE before classification and segmentation.
    #[arg(long)]
    clahe: bool,
    #[arg(
        long,
        default_value = "outputs/checkpoints/exports/swin_hybrid_epoch159.onnx"
    )]
    model: PathBuf,
    /// Exported Swin V2-Tiny classifier graph.
    #[arg(
        long,
        default_value = "outputs/checkpoints/classifer_swinv2_tiny/classifer_swinv2_tiny_binary.onnx"
    )]
    classifier: PathBuf,
    /// Minimum tumor probability required for segmentation.
    #[arg(long, default_value_t = 0.40)]
    threshold: f32,
    /// Bypass screening for segmenter benchmarks and parity checks.
    #[arg(long)]
    segment_only: bool,
    /// One PNG/JPEG image, or a directory of PNG/JPEG images (nonrecursive).
    #[arg(long, required_unless_present = "gui")]
    input: Option<PathBuf>,
    /// Destination directory for native-resolution binary masks and JSON reports.
    #[arg(long, required_unless_present = "gui")]
    output: Option<PathBuf>,
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

#[derive(Deserialize)]
struct ClassifierMetadata {
    format: String,
    model_sha256: String,
}

fn checked_graph(path: &Path, expected_digest: &str) -> Result<Vec<u8>> {
    let graph = fs::read(path).with_context(|| format!("Cannot read {}", path.display()))?;
    ensure!(
        format!("{:x}", Sha256::digest(&graph)) == expected_digest,
        "ONNX checksum differs from export metadata for {}",
        path.display()
    );
    Ok(graph)
}

fn screen(session: &mut Session, image: &GrayImage) -> Result<f32> {
    let size = preprocess::CLASSIFIER_SIZE;
    let input = Tensor::from_array((
        [1usize, 3, size, size],
        preprocess::prepare_classifier(image),
    ))?;
    let outputs = session.run(ort::inputs!["image" => input])?;
    let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
    ensure!(
        shape.as_ref() == [1] && logits[0].is_finite(),
        "Invalid classifier output"
    );
    Ok(1. / (1. + (-logits[0]).exp()))
}

fn grayscale(path: &Path) -> Result<GrayImage> {
    let decoded = image::open(path).with_context(|| format!("Cannot decode {}", path.display()))?;
    Ok(to_grayscale(&decoded.to_rgb8()))
}

fn to_grayscale(rgb: &image::RgbImage) -> GrayImage {
    // Match Pillow's ITU-R 601 conversion rather than image crate's Rec.709 default.
    GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let p = rgb.get_pixel(x, y);
        image::Luma([((19595u32 * p[0] as u32
            + 38470u32 * p[1] as u32
            + 7471u32 * p[2] as u32
            + 32768)
            >> 16) as u8])
    })
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

struct Engine {
    metadata: Metadata,
    classifier: Option<Session>,
    segmenter: Session,
    threshold: f32,
}

struct Prediction {
    screening: Option<f32>,
    segmenter_ran: bool,
    mask: GrayImage,
    forward_seconds: f64,
}

impl Engine {
    fn load(cli: &Cli) -> Result<Self> {
        ensure!(cli.threads > 0, "--threads must be positive");
        let metadata: Metadata =
            serde_json::from_slice(&fs::read(cli.model.with_extension("json")).context(
                "Missing model metadata; export with python -m scripts.export_swin_hybrid",
            )?)?;
        ensure!(
            metadata.format == "brain-swin-hybrid-onnx-v1",
            "Unsupported export format"
        );
        ensure!(
            metadata.size >= 64 && metadata.size % 32 == 0,
            "Invalid model input size"
        );
        ensure!(
            (0.0..=1.0).contains(&cli.threshold),
            "--threshold must be between 0 and 1"
        );
        let segmenter_graph = checked_graph(&cli.model, &metadata.model_sha256)?;
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
        let classifier = if cli.segment_only {
            None
        } else {
            let metadata: ClassifierMetadata =
            serde_json::from_slice(&fs::read(cli.classifier.with_extension("json")).context(
                "Missing classifier metadata; export with python -m scripts.export_swin_classifier",
            )?)?;
            ensure!(
                metadata.format == "brain-swinv2-classifier-onnx-v1",
                "Unsupported classifier format"
            );
            let graph = checked_graph(&cli.classifier, &metadata.model_sha256)?;
            Some(
                builder
                    .clone()
                    .commit_from_memory(&graph)
                    .context("Cannot load classifier graph")?,
            )
        };
        let segmenter = builder
            .commit_from_memory(&segmenter_graph)
            .context("Cannot load segmenter graph")?;
        Ok(Self {
            metadata,
            classifier,
            segmenter,
            threshold: cli.threshold,
        })
    }

    fn predict(&mut self, image: &GrayImage, dump_input: Option<&Path>) -> Result<Prediction> {
        let screening = self
            .classifier
            .as_mut()
            .map(|session| screen(session, image))
            .transpose()?;
        let segmenter_ran = screening.is_none_or(|probability| probability >= self.threshold);
        let (mask, forward_seconds) = if segmenter_ran {
            let (input, geometry) = preprocess::prepare(image, self.metadata.size);
            if let Some(path) = dump_input {
                let bytes: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
                fs::write(path, bytes)?;
            }
            let tensor =
                Tensor::from_array(([1usize, 1, self.metadata.size, self.metadata.size], input))?;
            let forward_start = Instant::now();
            let outputs = self.segmenter.run(ort::inputs!["image" => tensor])?;
            let forward_seconds = forward_start.elapsed().as_secs_f64();
            let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            ensure!(
                shape.as_ref() == [1, 2, self.metadata.size as i64, self.metadata.size as i64],
                "Unexpected output shape: {shape:?}"
            );
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "Nonfinite model logits"
            );
            (preprocess::restore(logits, geometry), forward_seconds)
        } else {
            (GrayImage::new(image.width(), image.height()), 0.)
        };
        Ok(Prediction {
            screening,
            segmenter_ran,
            mask,
            forward_seconds,
        })
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.gui {
        #[cfg(feature = "gui")]
        return gui::launch(cli);
        #[cfg(not(feature = "gui"))]
        anyhow::bail!("GUI support requires cargo build --release --features gui");
    }
    let mut engine = Engine::load(&cli)?;
    let output = cli.output.as_ref().context("--output is required")?;
    let paths = images(cli.input.as_deref().context("--input is required")?)?;
    // Validate all destinations before producing the first output.
    for path in &paths {
        let stem = path.file_stem().unwrap().to_string_lossy();
        for suffix in ["_mask.png", ".json", ".f32le"] {
            ensure!(
                !output.join(format!("{stem}{suffix}")).exists(),
                "Output already exists for {stem}"
            );
        }
    }
    fs::create_dir_all(output)?;
    for path in paths {
        let start = Instant::now();
        let mut image = grayscale(&path)?;
        if cli.clahe {
            image = clahe::apply(&image);
        }
        let stem = path.file_stem().unwrap().to_string_lossy();
        let dump = cli.dump_input.then(|| output.join(format!("{stem}.f32le")));
        let Prediction {
            screening,
            segmenter_ran,
            mask,
            forward_seconds,
        } = engine.predict(&image, dump.as_deref())?;
        let target = output.join(format!("{stem}_mask.png"));
        mask.save(&target)?;
        let report = json!({"input": path, "mask": target, "model": cli.model,
            "clahe": cli.clahe, "epoch": engine.metadata.epoch, "weights": engine.metadata.weights, "device": format!("{:?}", cli.device),
            "screening": screening.map(|probability| json!({"model": cli.classifier,
                "tumor_probability": probability, "threshold": cli.threshold, "is_positive": segmenter_ran})),
            "segmenter_ran": segmenter_ran,
            "width": image.width(), "height": image.height(),
            "foreground_pixels": mask.as_raw().iter().filter(|p| **p > 0).count(),
            "forward_seconds": forward_seconds, "total_seconds": start.elapsed().as_secs_f64()});
        fs::write(
            output.join(format!("{stem}.json")),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{}", report);
    }
    Ok(())
}
