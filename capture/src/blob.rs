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

/// Finds a moving light in a lit room, where a plain "brightest point" would find a monitor or a window: it learns
/// what each part of the picture normally looks like and looks for something that is suddenly brighter, bright
/// and small. One tracker per camera, fed every frame.
#[derive(Clone, Debug, Default)]
pub struct LightTracker {
    size: (usize, usize),
    background: Vec<f32>,
    frames: u32,
    status: &'static str,
}

/// How much the picture is shrunk before looking for change (each cell is the brightest of its pixels).
const CELL: usize = 4;
/// A cell is lit by a light when it is this much brighter than usual, and at least this bright.
const MIN_RISE: f32 = 45.0;
const MIN_LEVEL: f32 = 140.0;
/// The size of a light, in cells.
const MAX_CELLS: usize = 700;
/// Frames spent learning before it reports anything.
const WARM_UP: u32 = 8;

impl LightTracker {
    /// A tracker that has seen nothing yet.
    pub fn new() -> Self {
        Self { status: "learning the picture", ..Self::default() }
    }

    /// Why the last frame gave what it gave: for showing to the person waving the light.
    pub fn status(&self) -> &'static str {
        if self.status.is_empty() {
            "learning the picture"
        } else {
            self.status
        }
    }

    /// Looks for the light in `frame` and learns from it.
    pub fn update(&mut self, frame: &VideoFrame) -> Option<LightSpot> {
        let (w, h) = (frame.width as usize, frame.height as usize);
        if w < 2 * CELL || h < 2 * CELL || !frame.is_well_formed() {
            return None;
        }
        let (cw, ch) = (w / CELL, h / CELL);
        let data = &frame.data;
        let mut small = vec![0.0f32; cw * ch];
        for cy in 0..ch {
            for cx in 0..cw {
                let mut best = 0u8;
                for dy in 0..CELL {
                    let row = ((cy * CELL + dy) * w + cx * CELL) * 4;
                    for dx in 0..CELL {
                        let i = row + dx * 4;
                        best = best.max(data[i].max(data[i + 1]).max(data[i + 2]));
                    }
                }
                small[cy * cw + cx] = f32::from(best);
            }
        }
        if self.size != (cw, ch) || self.background.len() != small.len() {
            self.size = (cw, ch);
            self.background = small.clone();
            self.frames = 0;
            self.status = "learning the picture";
            return None;
        }
        self.frames += 1;
        let rise: Vec<f32> = small.iter().zip(&self.background).map(|(s, b)| s - b).collect();
        let lit = |i: usize| rise[i] >= MIN_RISE && small[i] >= MIN_LEVEL;
        // Learn: quickly at first, then slowly, and hardly at all where something just lit up.
        let rate = if self.frames < WARM_UP { 0.4 } else { 0.03 };
        for i in 0..small.len() {
            let r = if lit(i) { rate * 0.05 } else { rate };
            self.background[i] += (small[i] - self.background[i]) * r;
        }
        if self.frames < WARM_UP {
            self.status = "learning the picture";
            return None;
        }
        // Connected patches of lit cells, by how much light they added.
        let mut seen = vec![false; small.len()];
        let mut patches: Vec<(f32, Vec<usize>)> = Vec::new();
        let mut stack = Vec::new();
        for start in 0..small.len() {
            if seen[start] || !lit(start) {
                continue;
            }
            seen[start] = true;
            stack.push(start);
            let (mut weight, mut cells) = (0.0f32, Vec::new());
            while let Some(i) = stack.pop() {
                weight += rise[i];
                cells.push(i);
                let (x, y) = (i % cw, i / cw);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= cw as i32 || ny >= ch as i32 {
                        continue;
                    }
                    let j = ny as usize * cw + nx as usize;
                    if !seen[j] && lit(j) {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            patches.push((weight, cells));
        }
        patches.sort_by(|a, b| b.0.total_cmp(&a.0));
        let Some((weight, cells)) = patches.first() else {
            self.status = "no new bright point: wave the light in view";
            return None;
        };
        let count = cells.len();
        if count > MAX_CELLS {
            self.status = "too much light changed: keep the room steady, only the light should move";
            return None;
        }
        if patches.get(1).is_some_and(|second| *weight < second.0 * 1.4) {
            self.status = "more than one bright point is moving: cover the others";
            return None;
        }
        // The exact spot, from the full-resolution pixels around the patch.
        let (sx, sy) = cells.iter().fold((0.0f64, 0.0f64), |(ax, ay), &i| (ax + (i % cw) as f64, ay + (i / cw) as f64));
        let (cx, cy) = ((sx / count as f64 + 0.5) * CELL as f64, (sy / count as f64 + 0.5) * CELL as f64);
        let reach = (CELL as f64 * (2.0 + (count as f64).sqrt())).ceil() as i64;
        let (x0, x1) = ((cx as i64 - reach).max(0), (cx as i64 + reach).min(w as i64 - 1));
        let (y0, y1) = ((cy as i64 - reach).max(0), (cy as i64 + reach).min(h as i64 - 1));
        let value = |x: i64, y: i64| {
            let i = (y as usize * w + x as usize) * 4;
            data[i].max(data[i + 1]).max(data[i + 2])
        };
        let peak = (y0..=y1).flat_map(|y| (x0..=x1).map(move |x| (x, y))).map(|(x, y)| value(x, y)).max().unwrap_or(0);
        if peak < MIN_LEVEL as u8 {
            self.status = "the bright point is too dim";
            return None;
        }
        let threshold = f32::from(peak) * 0.7;
        let (mut sw, mut swx, mut swy, mut area) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let v = f32::from(value(x, y));
                if v >= threshold {
                    let wgt = f64::from(v - threshold) + 1.0;
                    sw += wgt;
                    swx += wgt * (x as f64 + 0.5);
                    swy += wgt * (y as f64 + 0.5);
                    area += 1;
                }
            }
        }
        if area == 0 {
            self.status = "the bright point is too dim";
            return None;
        }
        self.status = "seeing the light";
        Some(LightSpot { x: swx / sw, y: swy / sw, area, peak })
    }
}
