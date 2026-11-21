//! Finding a single bright point of light (a phone's flashlight, an LED, a lit ball) in a camera frame, to
//! sub-pixel accuracy: what the wand calibration measures.

use crate::frame::VideoFrame;

/// The light must be at least this bright (0 to 255, the brightest channel) to count.
const MIN_BRIGHTNESS: u8 = 190;
/// A pixel belongs to the light when it is at least this fraction of the brightest.
const RELATIVE_THRESHOLD: f32 = 0.78;
/// Smallest and largest area, pixels, of a light: a speck is noise, a patch is a window or a lamp.
const MIN_AREA: usize = 3;
const MAX_AREA: usize = 3000;
/// More lit pixels than this and the frame is washed out.
const MAX_CANDIDATES: usize = 60_000;
/// The brightest patch must hold this many times the light of the next, or the frame is ambiguous.
const DOMINANCE: f32 = 2.0;

/// Where a light is in a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightSpot {
    /// Pixel coordinates of its centre (a pixel's centre is at +0.5).
    pub x: f64,
    pub y: f64,
    /// Lit pixels.
    pub area: usize,
    /// The brightest channel value at its peak.
    pub peak: u8,
}

/// The one light in `frame`, or `None` when there is none, several of similar strength, or it is too
/// small, too large or too faint to trust.
pub fn find_light(frame: &VideoFrame) -> Option<LightSpot> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    if w < 8 || h < 8 || !frame.is_well_formed() {
        return None;
    }
    let data = &frame.data;
    let value = |x: usize, y: usize| -> u8 {
        let i = (y * w + x) * 4;
        data[i].max(data[i + 1]).max(data[i + 2])
    };
    // The brightest value, looking at every second pixel.
    let mut peak = 0u8;
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            peak = peak.max(value(x, y));
        }
    }
    if peak < MIN_BRIGHTNESS {
        return None;
    }
    let threshold = ((f32::from(peak) * RELATIVE_THRESHOLD) as u8).max(MIN_BRIGHTNESS - 20);
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        let row = y * w * 4;
        for x in 0..w {
            let i = row + x * 4;
            if data[i].max(data[i + 1]).max(data[i + 2]) >= threshold {
                candidates.push((x, y));
                if candidates.len() > MAX_CANDIDATES {
                    return None;
                }
            }
        }
    }
    // Connected patches of lit pixels, weighted by how far above the threshold they are.
    let mut visited = vec![false; w * h];
    let mut patches: Vec<(f64, f64, f64, usize, u8)> = Vec::new(); // (sum w, sum w x, sum w y, area, peak)
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &(sx, sy) in &candidates {
        if visited[sy * w + sx] {
            continue;
        }
        visited[sy * w + sx] = true;
        stack.push((sx, sy));
        let (mut sw, mut swx, mut swy, mut area, mut top) = (0.0f64, 0.0f64, 0.0f64, 0usize, 0u8);
        while let Some((x, y)) = stack.pop() {
            let v = value(x, y);
            let weight = f64::from(v - threshold) + 1.0;
            sw += weight;
            swx += weight * (x as f64 + 0.5);
            swy += weight * (y as f64 + 0.5);
            area += 1;
            top = top.max(v);
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if !visited[ny * w + nx] && value(nx, ny) >= threshold {
                    visited[ny * w + nx] = true;
                    stack.push((nx, ny));
                }
            }
        }
        patches.push((sw, swx, swy, area, top));
    }
    patches.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (sw, swx, swy, area, top) = *patches.first()?;
    if !(MIN_AREA..=MAX_AREA).contains(&area) {
        return None;
    }
    if patches.get(1).is_some_and(|second| sw < second.0 * f64::from(DOMINANCE)) {
        return None;
    }
    Some(LightSpot { x: swx / sw, y: swy / sw, area, peak: top })
}
