use crate::{Cli, Engine, Prediction, clahe, to_grayscale};
use anyhow::{Context as _, Result, ensure};
use eframe::egui::{self, ColorImage, TextureHandle};
use image::{GrayImage, Rgb, RgbImage};
use std::{
    path::Path,
    sync::mpsc::{self, Receiver, Sender},
};

struct Scenario {
    label: &'static str,
    filename: &'static str,
    tumor: bool,
}

const SCENARIOS: [Scenario; 4] = [
    Scenario {
        label: "Glioma",
        filename: "brisc2025_test_00162_gl_co_t1.jpg",
        tumor: true,
    },
    Scenario {
        label: "Meningioma",
        filename: "brisc2025_test_00496_me_sa_t1.jpg",
        tumor: true,
    },
    Scenario {
        label: "U tuyến yên",
        filename: "brisc2025_test_00835_pi_co_t1.jpg",
        tumor: true,
    },
    Scenario {
        label: "Không có khối u",
        filename: "brisc2025_test_00616_no_co_t1.jpg",
        tumor: false,
    },
];

struct Reference {
    label: &'static str,
    tumor: bool,
    mask: Option<GrayImage>,
}

struct Case {
    filename: String,
    original: RgbImage,
    reference: Option<Reference>,
}

impl Case {
    fn open(path: &Path, reference: Option<Reference>) -> Result<Self> {
        let original = image::open(path)
            .with_context(|| format!("Không thể mở {}", path.display()))?
            .into_rgb8();
        if let Some(mask) = reference.as_ref().and_then(|r| r.mask.as_ref()) {
            ensure!(
                original.dimensions() == mask.dimensions(),
                "Kích thước ảnh và mask khác nhau"
            );
        }
        Ok(Self {
            filename: path
                .file_name()
                .unwrap_or(path.as_os_str())
                .to_string_lossy()
                .into_owned(),
            original,
            reference,
        })
    }
}

impl Scenario {
    fn load(&self) -> Result<Case> {
        let root = Path::new("data/data/brisc2025");
        let path = if self.tumor {
            root.join("segmentation_task/test/images")
        } else {
            root.join("classification_task/test/no_tumor")
        }
        .join(self.filename);
        let mask = if self.tumor {
            let mask_path = root
                .join("segmentation_task/test/masks")
                .join(self.filename)
                .with_extension("png");
            Some(
                image::open(&mask_path)
                    .with_context(|| format!("Không thể mở nhãn {}", mask_path.display()))?
                    .into_luma8(),
            )
        } else {
            None
        };
        Case::open(
            &path,
            Some(Reference {
                label: self.label,
                tumor: self.tumor,
                mask,
            }),
        )
    }
}

fn overlay(image: &RgbImage, mask: &GrayImage, color: [u8; 3]) -> RgbImage {
    RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let pixel = image.get_pixel(x, y);
        if mask.get_pixel(x, y)[0] >= 128 {
            Rgb(std::array::from_fn(|c| {
                ((u16::from(pixel[c]) + u16::from(color[c])) / 2) as u8
            }))
        } else {
            *pixel
        }
    })
}

fn dice(prediction: &GrayImage, target: &GrayImage) -> f64 {
    let (mut intersection, mut area) = (0, 0);
    for (&p, &t) in prediction.as_raw().iter().zip(target.as_raw()) {
        let (p, t) = (p >= 128, t >= 128);
        intersection += usize::from(p && t);
        area += usize::from(p) + usize::from(t);
    }
    if area == 0 {
        1.
    } else {
        2. * intersection as f64 / area as f64
    }
}

fn texture(ctx: &egui::Context, name: &str, image: &RgbImage) -> TextureHandle {
    ctx.load_texture(
        name,
        ColorImage::from_rgb(
            [image.width() as usize, image.height() as usize],
            image.as_raw(),
        ),
        egui::TextureOptions::LINEAR,
    )
}

fn panel(
    ui: &mut egui::Ui,
    title: &str,
    image: Option<&TextureHandle>,
    placeholder: &str,
    height: f32,
) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.heading(title);
        let size = egui::vec2(ui.available_width(), height);
        ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Center), |ui| {
            ui.set_min_size(size);
            if let Some(image) = image {
                let dimensions = image.size_vec2();
                let scale = (size.x / dimensions.x).min(size.y / dimensions.y);
                ui.add(egui::Image::new(image).fit_to_exact_size(dimensions * scale));
            } else {
                ui.add_space(height * 0.35);
                ui.label(placeholder);
            }
        });
    });
}

struct App {
    selected: Option<usize>,
    case: Option<Case>,
    clahe: bool,
    input: Option<GrayImage>,
    original: Option<TextureHandle>,
    prepared: Option<TextureHandle>,
    reference: Option<TextureHandle>,
    output: Option<TextureHandle>,
    status: String,
    verdict: String,
    running: bool,
    requests: Sender<GrayImage>,
    results: Receiver<Result<Prediction>>,
}

impl App {
    fn new(ctx: &egui::Context, cli: Cli) -> Self {
        let (requests, receive) = mpsc::channel();
        let (send, results) = mpsc::channel();
        let initial_clahe = cli.clahe;
        let notify = ctx.clone();
        std::thread::spawn(move || {
            let mut engine = None;
            for image in receive {
                let result = (|| {
                    if engine.is_none() {
                        engine = Some(Engine::load(&cli)?);
                    }
                    engine.as_mut().unwrap().predict(&image, None)
                })();
                if send.send(result).is_err() {
                    break;
                }
                notify.request_repaint();
            }
        });

        Self {
            selected: None,
            case: None,
            clahe: initial_clahe,
            input: None,
            original: None,
            prepared: None,
            reference: None,
            output: None,
            status: String::new(),
            verdict: String::new(),
            running: false,
            requests,
            results,
        }
    }

    fn load_case(&mut self, ctx: &egui::Context, case: Result<Case>) {
        self.case = None;
        self.original = None;
        self.reference = None;
        match case {
            Ok(case) => {
                self.original = Some(texture(ctx, "original", &case.original));
                self.reference =
                    case.reference
                        .as_ref()
                        .and_then(|r| r.mask.as_ref())
                        .map(|mask| {
                            texture(
                                ctx,
                                "reference",
                                &overlay(&case.original, mask, [26, 225, 128]),
                            )
                        });
                self.case = Some(case);
                self.prepare(ctx);
            }
            Err(error) => {
                self.prepare(ctx);
                self.status = format!("{error:#}");
            }
        }
    }

    fn prepare(&mut self, ctx: &egui::Context) {
        self.input = self.case.as_ref().map(|case| {
            let gray = to_grayscale(&case.original);
            if self.clahe {
                clahe::apply(&gray)
            } else {
                gray
            }
        });
        self.prepared = self.input.as_ref().map(|input| {
            texture(
                ctx,
                "prepared",
                &image::DynamicImage::ImageLuma8(input.clone()).into_rgb8(),
            )
        });
        self.verdict = "Chưa phân tích".into();
        self.status.clear();
        self.output = None;
    }

    fn show_result(&mut self, ctx: &egui::Context, result: Result<Prediction>) {
        self.running = false;
        let prediction = match result {
            Ok(prediction) => prediction,
            Err(error) => {
                self.verdict = "Phân tích thất bại".into();
                self.status = format!("{error:#}");
                return;
            }
        };
        let positive = prediction.segmenter_ran;
        let label = if positive {
            "Có khối u"
        } else {
            "Không có khối u"
        };
        let probability = prediction.screening.unwrap_or_default();
        self.verdict = format!("{label} · Xác suất u {:.2}%", f64::from(probability) * 100.);
        let Some(case) = &self.case else { return };
        if positive && prediction.mask.as_raw().iter().any(|p| *p >= 128) {
            let marked = overlay(&case.original, &prediction.mask, [255, 60, 60]);
            self.output = Some(texture(ctx, "prediction", &marked));
        }
        self.status = if positive {
            if self.output.is_some() {
                "Mask dự đoán màu đỏ".into()
            } else {
                "Mask dự đoán rỗng".into()
            }
        } else {
            "Đã bỏ qua phân đoạn".into()
        };
        if let Some(reference) = &case.reference {
            let matches = if positive == reference.tumor {
                "khớp"
            } else {
                "lệch"
            };
            self.status.push_str(&format!(" · Sàng lọc {matches} nhãn"));
            if let Some(mask) = &reference.mask {
                self.status.push_str(&format!(
                    " · Dice {:.2}%",
                    100. * dice(&prediction.mask, mask)
                ));
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.running {
            match self.results.try_recv() {
                Ok(result) => self.show_result(ctx, result),
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.show_result(ctx, Err(anyhow::anyhow!("Luồng inference đã dừng")))
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Brain CADx");
            ui.label("Sàng lọc Swin V2-Tiny và phân đoạn Swin Hybrid");
            ui.add_space(8.);
            ui.add_enabled_ui(!self.running, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let previous = self.selected;
                    egui::ComboBox::from_label("Kịch bản")
                        .selected_text(
                            self.selected
                                .map_or("Chọn kịch bản", |i| SCENARIOS[i].label),
                        )
                        .show_ui(ui, |ui| {
                            for (i, scenario) in SCENARIOS.iter().enumerate() {
                                ui.selectable_value(&mut self.selected, Some(i), scenario.label);
                            }
                        });
                    if self.selected != previous {
                        if let Some(i) = self.selected {
                            self.load_case(ctx, SCENARIOS[i].load());
                        }
                    }
                    if ui.button("Mở ảnh JPG/PNG…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("MRI", &["jpg", "jpeg", "png"])
                            .pick_file()
                        {
                            self.selected = None;
                            self.load_case(ctx, Case::open(&path, None));
                        }
                    }
                    if ui.checkbox(&mut self.clahe, "Bật CLAHE").changed() {
                        self.prepare(ctx);
                    }
                    if ui
                        .add_enabled(self.input.is_some(), egui::Button::new("Chạy phân tích"))
                        .clicked()
                    {
                        if let Some(image) = &self.input {
                            self.output = None;
                            self.running = self.requests.send(image.clone()).is_ok();
                            self.verdict = if self.running {
                                "Đang phân tích…"
                            } else {
                                "Luồng inference đã dừng"
                            }
                            .into();
                            self.status.clear();
                        }
                    }
                });
            });
            ui.separator();
            ui.strong(if self.verdict.is_empty() {
                "Chưa phân tích"
            } else {
                &self.verdict
            });
            if !self.status.is_empty() {
                ui.label(&self.status);
            }
            let height = ((ui.available_height() - 120.) / 2.).max(120.);
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.columns(2, |columns| {
                    panel(
                        &mut columns[0],
                        if self.clahe {
                            "Đầu vào · CLAHE"
                        } else {
                            "Đầu vào · Ảnh xám"
                        },
                        self.prepared.as_ref(),
                        "Chọn kịch bản hoặc mở ảnh.",
                        height,
                    );
                    panel(
                        &mut columns[1],
                        "Kết quả mô hình",
                        self.output.as_ref(),
                        if self.status.is_empty() {
                            "Mask dự đoán sẽ hiện ở đây."
                        } else {
                            &self.status
                        },
                        height,
                    );
                });
                ui.add_space(8.);
                ui.separator();
                ui.strong("Tệp gốc và nhãn tham chiếu");
                ui.columns(2, |columns| {
                    let filename = self
                        .case
                        .as_ref()
                        .map_or("Tệp gốc", |case| case.filename.as_str());
                    panel(
                        &mut columns[0],
                        filename,
                        self.original.as_ref(),
                        "Chưa có ảnh.",
                        height,
                    );
                    let reference = self.case.as_ref().and_then(|case| case.reference.as_ref());
                    let title = reference.map_or_else(
                        || "Nhãn tham chiếu".to_owned(),
                        |r| format!("Nhãn · {}", r.label),
                    );
                    let placeholder = if reference.is_some() {
                        "Ảnh âm tính có mask tham chiếu rỗng."
                    } else {
                        "Chọn kịch bản BRISC để xem nhãn."
                    };
                    panel(
                        &mut columns[1],
                        &title,
                        self.reference.as_ref(),
                        placeholder,
                        height,
                    );
                });
            });
        });
    }
}

pub fn launch(cli: Cli) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1160., 850.])
            .with_min_inner_size([780., 580.]),
        ..Default::default()
    };
    eframe::run_native(
        "Brain CADx",
        options,
        Box::new(|cc| {
            if let Some(data) = std::env::var_os("WINDIR")
                .and_then(|path| std::fs::read(Path::new(&path).join("Fonts/segoeui.ttf")).ok())
            {
                let mut fonts = egui::FontDefinitions::default();
                fonts
                    .font_data
                    .insert("Segoe UI".into(), egui::FontData::from_owned(data).into());
                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .insert(0, "Segoe UI".into());
                cc.egui_ctx.set_fonts(fonts);
            }
            Ok(Box::new(App::new(&cc.egui_ctx, cli)))
        }),
    )
    .map_err(|error| anyhow::anyhow!("Không thể mở giao diện: {error}"))
}
