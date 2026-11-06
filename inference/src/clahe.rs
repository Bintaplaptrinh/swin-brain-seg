use image::{GrayImage, Luma};

const GRID: usize = 8;
const CLIP_LIMIT: usize = 2;

/// Equalize clipped tile histograms, then interpolate the four neighboring maps.
pub fn apply(image: &GrayImage) -> GrayImage {
    let (width, height) = (image.width() as usize, image.height() as usize);
    if width == 0 || height == 0 {
        return image.clone();
    }
    let (cols, rows) = (width.min(GRID), height.min(GRID));
    let (tw, th) = (width.div_ceil(cols), height.div_ceil(rows));
    let area = tw * th;
    let limit = (CLIP_LIMIT * area / 256).max(1);
    let mut maps = vec![[0.; 256]; cols * rows];

    for ty in 0..rows {
        for tx in 0..cols {
            let mut hist = [0usize; 256];
            for y in ty * th..(ty + 1) * th {
                for x in tx * tw..(tx + 1) * tw {
                    let offset = y.min(height - 1) * width + x.min(width - 1);
                    hist[image.as_raw()[offset] as usize] += 1;
                }
            }
            let excess: usize = hist.iter().map(|n| n.saturating_sub(limit)).sum();
            let remainder = excess % 256;
            let stride = 256 / remainder.max(1);
            let mut cumulative = 0;
            for (i, count) in hist.into_iter().enumerate() {
                let extra = usize::from(i.is_multiple_of(stride) && i / stride < remainder);
                cumulative += count.min(limit) + excess / 256 + extra;
                maps[ty * cols + tx][i] = 255. * cumulative as f32 / area as f32;
            }
        }
    }

    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let interpolate = |pixel: u32, tile: usize, count: usize| {
            let f = (pixel as f32 / tile as f32 - 0.5).clamp(0., (count - 1) as f32);
            let low = f.floor() as usize;
            (low, (low + 1).min(count - 1), f.fract())
        };
        let (x0, x1, wx) = interpolate(x, tw, cols);
        let (y0, y1, wy) = interpolate(y, th, rows);
        let value = image.get_pixel(x, y)[0] as usize;
        let a = maps[y0 * cols + x0][value] * (1. - wx) + maps[y0 * cols + x1][value] * wx;
        let b = maps[y1 * cols + x0][value] * (1. - wx) + maps[y1 * cols + x1][value] * wx;
        Luma([(a * (1. - wy) + b * wy).round() as u8])
    })
}
