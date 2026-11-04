use image::GrayImage;

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    pub resized_width: usize,
    pub resized_height: usize,
    pub left: usize,
    pub top: usize,
    pub size: usize,
}

impl Geometry {
    pub fn new(width: usize, height: usize, size: usize) -> Self {
        let scale = (size as f64 / width as f64).min(size as f64 / height as f64);
        // Match Python round(), including ties to even.
        let rw = (width as f64 * scale).round_ties_even().max(1.) as usize;
        let rh = (height as f64 * scale).round_ties_even().max(1.) as usize;
        Self {
            width,
            height,
            resized_width: rw,
            resized_height: rh,
            left: (size - rw) / 2,
            top: (size - rh) / 2,
            size,
        }
    }
}

// Pillow's separable BILINEAR antialiasing uses a widened triangle when shrinking,
// 22-bit fixed-point coefficients and an 8-bit intermediate after each axis.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let support = scale.max(1.);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = (((center + support + 0.5) as usize).min(input)).max(start);
            let raw: Vec<f64> = (start..end)
                .map(|x| (1. - ((x as f64 - center + 0.5) / support).abs()).max(0.))
                .collect();
            let sum: f64 = raw.iter().sum();
            (
                start,
                raw.iter()
                    .map(|x| (x / sum * (1 << 22) as f64 + 0.5) as i32)
                    .collect(),
            )
        })
        .collect()
}

pub fn resize_bilinear(image: &GrayImage, width: usize, height: usize) -> Vec<u8> {
    let (iw, ih) = (image.width() as usize, image.height() as usize);
    let mut horizontal = vec![0u8; width * ih];
    let xc = coefficients(iw, width);
    for y in 0..ih {
        for (x, (start, weights)) in xc.iter().enumerate() {
            let sum = weights
                .iter()
                .enumerate()
                .fold(1i64 << 21, |v, (j, weight)| {
                    v + image.as_raw()[y * iw + start + j] as i64 * *weight as i64
                });
            horizontal[y * width + x] = (sum >> 22).clamp(0, 255) as u8;
        }
    }
    let mut result = vec![0u8; width * height];
    for (y, (start, weights)) in coefficients(ih, height).iter().enumerate() {
        for x in 0..width {
            let sum = weights
                .iter()
                .enumerate()
                .fold(1i64 << 21, |v, (j, weight)| {
                    v + horizontal[(start + j) * width + x] as i64 * *weight as i64
                });
            result[y * width + x] = (sum >> 22).clamp(0, 255) as u8;
        }
    }
    result
}

fn normalize_foreground(resized: Vec<u8>) -> Vec<f32> {
    let values: Vec<f32> = resized.into_iter().map(|p| p as f32 / 255.).collect();
    let nonzero: Vec<f64> = values
        .iter()
        .filter(|v| **v > 0.)
        .map(|v| *v as f64)
        .collect();
    let count = nonzero.len().max(1) as f64;
    let mean = nonzero.iter().sum::<f64>() / count;
    let std = (nonzero.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count)
        .sqrt()
        .max(1e-6);
    values
        .into_iter()
        .map(|v| {
            if v > 0. {
                ((v as f64 - mean) / std) as f32
            } else {
                0.
            }
        })
        .collect()
}

pub const CLASSIFIER_SIZE: usize = 224;

pub fn prepare_classifier(image: &GrayImage) -> Vec<f32> {
    let values = normalize_foreground(resize_bilinear(image, CLASSIFIER_SIZE, CLASSIFIER_SIZE));
    [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)]
        .into_iter()
        .flat_map(|(mean, std)| values.iter().map(move |v| (v - mean) / std))
        .collect()
}

pub fn prepare(image: &GrayImage, size: usize) -> (Vec<f32>, Geometry) {
    let g = Geometry::new(image.width() as usize, image.height() as usize, size);
    let values = normalize_foreground(resize_bilinear(image, g.resized_width, g.resized_height));
    let mut input = vec![0f32; size * size];
    for y in 0..g.resized_height {
        let source = y * g.resized_width;
        let dest = (y + g.top) * size + g.left;
        input[dest..dest + g.resized_width]
            .copy_from_slice(&values[source..source + g.resized_width]);
    }
    (input, g)
}

// PyTorch interpolate(mode="bilinear", align_corners=false), applied to
// continuous foreground-minus-background logits before the binary decision.
pub fn restore(logits: &[f32], g: Geometry) -> GrayImage {
    let plane = g.size * g.size;
    let at = |y: usize, x: usize| {
        let i = (y + g.top) * g.size + x + g.left;
        logits[plane + i] - logits[i]
    };
    GrayImage::from_fn(g.width as u32, g.height as u32, |x, y| {
        let fy = ((y as f32 + 0.5) * (g.resized_height as f32 / g.height as f32) - 0.5).max(0.);
        let fx = ((x as f32 + 0.5) * (g.resized_width as f32 / g.width as f32) - 0.5).max(0.);
        let y0 = (fy.floor() as usize).min(g.resized_height - 1);
        let x0 = (fx.floor() as usize).min(g.resized_width - 1);
        let y1 = (y0 + 1).min(g.resized_height - 1);
        let x1 = (x0 + 1).min(g.resized_width - 1);
        let (wy, wx) = (fy - y0 as f32, fx - x0 as f32);
        let a = at(y0, x0) * (1. - wx) + at(y0, x1) * wx;
        let b = at(y1, x0) * (1. - wx) + at(y1, x1) * wx;
        image::Luma([if a * (1. - wy) + b * wy > 0. { 255 } else { 0 }])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_keeps_zero_padding_and_zero_background() {
        let image = GrayImage::from_fn(8, 4, |x, _| image::Luma([if x < 4 { 0 } else { 100 }]));
        let (input, g) = prepare(&image, 64);
        assert_eq!(
            (g.resized_width, g.resized_height, g.top, g.left),
            (64, 32, 16, 0)
        );
        assert!(input[..16 * 64].iter().all(|x| *x == 0.));
        assert!(input.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn resizing_preserves_identity_and_constant() {
        let image = GrayImage::from_fn(13, 17, |x, y| image::Luma([((x * 3 + y * 7) % 256) as u8]));
        assert_eq!(resize_bilinear(&image, 13, 17), *image.as_raw());
        assert!(
            resize_bilinear(&GrayImage::from_pixel(19, 5, image::Luma([73])), 8, 12)
                .iter()
                .all(|x| *x == 73)
        );
    }

    #[test]
    fn restoration_uses_logits_and_background_on_ties() {
        let g = Geometry::new(7, 3, 64);
        let mut logits = vec![0.; 2 * 64 * 64];
        assert!(restore(&logits, g).as_raw().iter().all(|v| *v == 0));
        logits[64 * 64..].fill(2.);
        assert_eq!(restore(&logits, g).dimensions(), (7, 3));
        assert!(restore(&logits, g).as_raw().iter().all(|v| *v == 255));
    }
}
