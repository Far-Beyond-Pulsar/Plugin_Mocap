//! One clock for sources that each have their own: a short history of every source's frames,
//! sampling at any moment by interpolation, and the estimate of how late one source's clock is
//! against another's from how they moved.
//!
//! Cameras, VMC performers and sensors do not deliver in step: each stamps its frames when it
//! captured them, at its own rate, with its own latency. Merging "the latest of each" mixes moments
//! that never happened together. Instead every source is sampled at the same instants, between its
//! own frames, after its clock offset is removed.

use crate::frame::PoseFrame;
use mocap_format::signal::nlerp;
use mocap_format::{ChannelKind, RigProfile};
use std::collections::VecDeque;
use std::sync::Arc;

/// How the columns of a frame are interpolated: quaternions by normalised lerp, everything else
/// linearly.
#[derive(Clone, Debug, Default)]
pub struct ColumnPlan {
    groups: Vec<(usize, ChannelKind)>,
    columns: usize,
}

impl ColumnPlan {
    /// The plan for frames of `rig`.
    pub fn new(rig: &RigProfile) -> Self {
        let mut at = 0;
        let groups = rig
            .channels
            .iter()
            .map(|c| {
                let start = at;
                at += c.kind.width();
                (start, c.kind)
            })
            .collect();
        Self { groups, columns: at }
    }

    /// Columns per frame.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// `a` blended toward `b` by `f` (0 is `a`, 1 is `b`). Where only one side has a value, the nearer
    /// frame's value is used, so a dropout is a gap and never a made-up number.
    pub fn blend(&self, a: &[f32], b: &[f32], f: f32) -> Vec<f32> {
        let mut out = vec![f32::NAN; self.columns];
        let nearer = if f < 0.5 { a } else { b };
        for &(start, kind) in &self.groups {
            let width = kind.width();
            let (Some(ga), Some(gb)) = (a.get(start..start + width), b.get(start..start + width)) else { continue };
            let finite = |g: &[f32]| g.iter().all(|v| v.is_finite());
            if finite(ga) && finite(gb) {
                if kind == ChannelKind::Rotation4 {
                    out[start..start + 4].copy_from_slice(&nlerp([ga[0], ga[1], ga[2], ga[3]], [gb[0], gb[1], gb[2], gb[3]], f));
                } else {
                    for k in 0..width {
                        out[start + k] = ga[k] + (gb[k] - ga[k]) * f;
                    }
                }
            } else if let Some(g) = nearer.get(start..start + width) {
                out[start..start + width].copy_from_slice(g);
            }
        }
        out
    }
}

/// A source's recent frames, oldest first.
#[derive(Debug, Default)]
pub struct History {
    frames: VecDeque<Arc<PoseFrame>>,
    keep_s: f64,
}

impl History {
    /// A history that keeps the last `keep_s` seconds.
    pub fn new(keep_s: f64) -> Self {
        Self { frames: VecDeque::new(), keep_s }
    }

    /// Adds a frame; one that is not newer than the last is dropped (a clock must not go back).
    pub fn push(&mut self, frame: Arc<PoseFrame>) {
        if self.frames.back().is_some_and(|last| frame.time_s <= last.time_s) {
            return;
        }
        let newest = frame.time_s;
        self.frames.push_back(frame);
        while self.frames.front().is_some_and(|f| newest - f.time_s > self.keep_s) {
            self.frames.pop_front();
        }
    }

    /// The newest frame's time.
    pub fn newest(&self) -> Option<f64> {
        self.frames.back().map(|f| f.time_s)
    }

    /// The oldest frame's time.
    pub fn oldest(&self) -> Option<f64> {
        self.frames.front().map(|f| f.time_s)
    }

    /// Number of frames held.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// True when no frame is held.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The source's values at `t`: interpolated between the frames around it. After the newest frame
    /// the last values are held, for at most `max_gap` seconds; across a hole wider than `max_gap`, or
    /// before the first frame, there is no value.
    pub fn sample(&self, t: f64, plan: &ColumnPlan, max_gap: f64) -> Option<Vec<f32>> {
        let next = self.frames.partition_point(|f| f.time_s <= t);
        if next == self.frames.len() {
            let last = self.frames.back()?;
            return (t - last.time_s <= max_gap).then(|| last.values.clone());
        }
        if next == 0 {
            return None;
        }
        let (a, b) = (&self.frames[next - 1], &self.frames[next]);
        if b.time_s - a.time_s > max_gap {
            return None;
        }
        let f = ((t - a.time_s) / (b.time_s - a.time_s)) as f32;
        Some(plan.blend(&a.values, &b.values, f))
    }

    /// How much the source moved at each of `steps` instants `1 / rate_hz` apart from `from`: the
    /// summed change of its columns since the instant before, `NaN` where it has no value.
    pub fn motion(&self, from: f64, steps: usize, rate_hz: f64, plan: &ColumnPlan, max_gap: f64) -> Vec<f32> {
        let mut out = Vec::with_capacity(steps);
        let mut previous: Option<Vec<f32>> = None;
        for i in 0..steps {
            let now = self.sample(from + i as f64 / rate_hz, plan, max_gap);
            let energy = match (&previous, &now) {
                (Some(p), Some(n)) => {
                    let (mut sum, mut count) = (0.0f32, 0usize);
                    for (a, b) in p.iter().zip(n) {
                        if a.is_finite() && b.is_finite() {
                            sum += (a - b).abs();
                            count += 1;
                        }
                    }
                    if count == 0 {
                        f32::NAN
                    } else {
                        sum / count as f32
                    }
                }
                _ => f32::NAN,
            };
            out.push(energy);
            previous = now;
        }
        out
    }
}

/// How late a source's clock is, and how sure the estimate is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyncEstimate {
    /// Seconds the other source's timestamps are late against the reference's (subtract it).
    pub offset_s: f64,
    /// How much higher the best match is than the best one that is not next to it (correlation units): a
    /// low margin means more than one lag fits, so the answer is ambiguous.
    pub margin: f64,
    /// The correlation at the peak, 0 to 1.
    pub correlation: f64,
}

/// Zero mean, unit variance, a little smoothing; gaps become the mean (zero).
fn normalised(signal: &[f32]) -> Vec<f64> {
    let finite: Vec<f64> = signal.iter().filter(|v| v.is_finite()).map(|v| f64::from(*v)).collect();
    if finite.is_empty() {
        return vec![0.0; signal.len()];
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let filled: Vec<f64> = signal.iter().map(|v| if v.is_finite() { f64::from(*v) - mean } else { 0.0 }).collect();
    let smooth: Vec<f64> = (0..filled.len())
        .map(|i| {
            let (lo, hi) = (i.saturating_sub(1), (i + 1).min(filled.len() - 1));
            filled[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64
        })
        .collect();
    let var = smooth.iter().map(|v| v * v).sum::<f64>() / smooth.len() as f64;
    let sd = var.sqrt().max(1e-9);
    smooth.iter().map(|v| v / sd).collect()
}

/// How late `other` is against `reference`, from two motion signals sampled at `rate_hz` over the same
/// window: the lag (within `max_lag_s` either way) at which they correlate best. `None` when the
/// signals are too short or too flat to say.
pub fn estimate_offset(reference: &[f32], other: &[f32], rate_hz: f64, max_lag_s: f64) -> Option<SyncEstimate> {
    let n = reference.len().min(other.len());
    let max_lag = (max_lag_s * rate_hz).round() as i64;
    if n < (4 * max_lag.max(1)) as usize || n < 20 {
        return None;
    }
    let (r, o) = (normalised(&reference[..n]), normalised(&other[..n]));
    let lags: Vec<i64> = (-max_lag..=max_lag).collect();
    let corr: Vec<f64> = lags
        .iter()
        .map(|&lag| {
            let (mut sum, mut count) = (0.0, 0usize);
            for i in 0..n as i64 {
                let j = i + lag;
                if j >= 0 && (j as usize) < n {
                    sum += r[i as usize] * o[j as usize];
                    count += 1;
                }
            }
            if count < n / 2 {
                0.0
            } else {
                sum / count as f64
            }
        })
        .collect();
    let (best, &peak) = corr.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    // The best match outside the peak's own lobe (a smooth movement correlates over a neighbourhood).
    let lobe = (0.15 * rate_hz).ceil() as usize;
    let second = corr.iter().enumerate().filter(|(i, _)| i.abs_diff(best) > lobe).map(|(_, c)| *c).fold(0.0f64, f64::max);
    // Parabolic refinement between samples.
    let mut lag = lags[best] as f64;
    if best > 0 && best + 1 < corr.len() {
        let (a, b, c) = (corr[best - 1], corr[best], corr[best + 1]);
        let denom = a - 2.0 * b + c;
        if denom.abs() > 1e-12 {
            lag += (0.5 * (a - c) / denom).clamp(-0.5, 0.5);
        }
    }
    Some(SyncEstimate { offset_s: lag / rate_hz, margin: peak - second, correlation: peak })
}
