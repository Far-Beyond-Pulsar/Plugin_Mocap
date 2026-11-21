//! Combining any number of pose and sensor sources into one stream and one rig.
//!
//! Every source keeps its own rig and bus. For a take they are merged: with one
//! source the rig is that source's (names unchanged); with several, every joint
//! and channel is namespaced `"<source id>/<name>"` so nothing collides (two
//! VMC performers, a suit plus an IMU, a camera estimator plus a glove).

use crate::bus::PoseBus;
use crate::frame::PoseFrame;
use crate::sync::{estimate_offset, ColumnPlan, History, SyncEstimate};
use crate::recorder::MAX_HOLD_S;
use mocap_format::{ChannelDesc, Joint, RigProfile};
use std::ops::Range;
use std::sync::Arc;

/// One source's contribution to a mix.
pub struct MixInput {
    pub id: String,
    pub rig: RigProfile,
    pub bus: Arc<PoseBus>,
}

/// Merges rigs; returns the merged rig and each input's column range in it.
pub fn merge_rigs(inputs: &[(String, &RigProfile)]) -> (RigProfile, Vec<Range<usize>>) {
    if let [(_, only)] = inputs {
        return ((*only).clone(), vec![0..only.columns()]);
    }
    let mut merged = RigProfile { name: "mix".to_owned(), ..RigProfile::default() };
    let mut ranges = Vec::with_capacity(inputs.len());
    for (id, rig) in inputs {
        let joint_base = merged.joints.len() as u16;
        let start = merged.columns();
        merged.joints.extend(rig.joints.iter().map(|j| Joint {
            name: format!("{id}/{}", j.name),
            parent: j.parent.map(|p| p + joint_base),
        }));
        merged.channels.extend(rig.channels.iter().map(|c| ChannelDesc {
            name: format!("{id}/{}", c.name),
            joint: c.joint.map(|j| j + joint_base),
            ..c.clone()
        }));
        ranges.push(start..merged.columns());
    }
    (merged, ranges)
}

/// The rate the merged stream is delivered at.
pub const DEFAULT_RATE_HZ: f64 = 60.0;
/// How far behind the newest data the merged stream runs, so the slower sources have a frame
/// after each instant to interpolate toward.
pub const DEFAULT_DELAY_S: f64 = 0.1;
/// How much of each source's past is kept (clock alignment looks at this much).
pub const HISTORY_S: f64 = 12.0;
/// What the clock estimate looks at, in Hz.
const SYNC_RATE_HZ: f64 = 100.0;
/// The furthest apart two sources' clocks are searched for, seconds.
pub const MAX_SYNC_LAG_S: f64 = 0.6;
/// An estimate must beat every other lag by this margin and be this correlated.
const MIN_SYNC_MARGIN: f64 = 0.05;
const MIN_SYNC_CORRELATION: f64 = 0.3;

struct Track {
    id: String,
    bus: Arc<PoseBus>,
    range: Range<usize>,
    plan: ColumnPlan,
    history: History,
    /// Seconds this source's timestamps are late (its true time is its stamp minus this).
    offset_s: f64,
}

/// Turns the buses of several sources into one merged pose stream on one clock.
///
/// Frames are emitted on a fixed grid (`rate_hz`). Each source is sampled at the grid instant after
/// its clock offset is removed, by interpolating between its own frames (quaternions by normalised
/// lerp), so a fast source and a slow one describe the same moment. A source silent for longer than
/// [`MAX_HOLD_S`] around the instant contributes gaps, never stale data.
pub struct Mixer {
    tracks: Vec<Track>,
    width: usize,
    sequence: u64,
    rate_hz: f64,
    delay_s: f64,
    next: Option<f64>,
}

impl Mixer {
    /// A mixer over `inputs`, and the merged rig its frames are in.
    pub fn new(inputs: Vec<MixInput>) -> (Self, RigProfile) {
        let rigs: Vec<(String, &RigProfile)> = inputs.iter().map(|i| (i.id.clone(), &i.rig)).collect();
        let (rig, ranges) = merge_rigs(&rigs);
        let tracks = inputs
            .into_iter()
            .zip(ranges)
            .map(|(input, range)| Track { plan: ColumnPlan::new(&input.rig), id: input.id, bus: input.bus, range, history: History::new(HISTORY_S), offset_s: 0.0 })
            .collect();
        (Self { tracks, width: rig.columns(), sequence: 0, rate_hz: DEFAULT_RATE_HZ, delay_s: DEFAULT_DELAY_S, next: None }, rig)
    }

    /// Sets the output rate and the delay behind the newest data.
    pub fn configure(&mut self, rate_hz: f64, delay_s: f64) {
        self.rate_hz = rate_hz.clamp(10.0, 240.0);
        self.delay_s = delay_s.clamp(0.0, 1.0);
    }

    /// Sets how late source `id`'s timestamps are, seconds (negative if early). Unknown ids are ignored.
    pub fn set_offset(&mut self, id: &str, offset_s: f64) {
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            t.offset_s = offset_s;
        }
    }

    /// Each source's clock offset now, in input order.
    pub fn offsets(&self) -> Vec<(String, f64)> {
        self.tracks.iter().map(|t| (t.id.clone(), t.offset_s)).collect()
    }

    /// Drains every source into its history, then appends the merged frames, oldest first.
    pub fn pump(&mut self, out: &mut Vec<PoseFrame>) {
        let mut taken = Vec::new();
        for track in &mut self.tracks {
            taken.clear();
            track.bus.drain(&mut taken);
            for frame in taken.drain(..) {
                track.history.push(frame);
            }
        }
        let newest = self.tracks.iter().filter_map(|t| t.history.newest().map(|n| n - t.offset_s)).fold(f64::NEG_INFINITY, f64::max);
        if !newest.is_finite() {
            return;
        }
        let horizon = newest - self.delay_s;
        let step = 1.0 / self.rate_hz;
        let mut at = self.next.unwrap_or(horizon);
        // After a stall, restart from the present instead of replaying it.
        if horizon - at > 2.0 {
            at = horizon;
        }
        while at <= horizon {
            let mut values = vec![f32::NAN; self.width];
            let mut any = false;
            for track in &self.tracks {
                if let Some(v) = track.history.sample(at + track.offset_s, &track.plan, MAX_HOLD_S) {
                    values[track.range.clone()].copy_from_slice(&v);
                    any = true;
                }
            }
            if any {
                self.sequence += 1;
                out.push(PoseFrame { sequence: self.sequence, time_s: at, values });
            }
            at += step;
        }
        self.next = Some(at);
    }

    /// Estimates how late each source's clock is against `reference`'s from how they moved over the last
    /// `window_s` seconds (the subject must be moving, in view of or carrying every source), and applies
    /// the estimates that are clear. Returns every source with its estimate (`None`: no movement to go
    /// by or no clear match, its offset is left alone).
    pub fn align(&mut self, reference: &str, window_s: f64) -> Vec<(String, Option<SyncEstimate>)> {
        let Some(ref_index) = self.tracks.iter().position(|t| t.id == reference) else { return Vec::new() };
        let end = self.tracks.iter().filter_map(|t| t.history.newest()).fold(f64::INFINITY, f64::min);
        if !end.is_finite() {
            return Vec::new();
        }
        let steps = (window_s * SYNC_RATE_HZ) as usize;
        let from = end - window_s;
        let signals: Vec<Vec<f32>> = self.tracks.iter().map(|t| t.history.motion(from, steps, SYNC_RATE_HZ, &t.plan, 0.5)).collect();
        let ref_offset = self.tracks[ref_index].offset_s;
        let mut report = Vec::new();
        for (i, track) in self.tracks.iter_mut().enumerate() {
            if i == ref_index {
                report.push((track.id.clone(), Some(SyncEstimate { offset_s: 0.0, margin: f64::INFINITY, correlation: 1.0 })));
                continue;
            }
            let estimate = estimate_offset(&signals[ref_index], &signals[i], SYNC_RATE_HZ, MAX_SYNC_LAG_S);
            if let Some(e) = estimate.filter(|e| e.margin >= MIN_SYNC_MARGIN && e.correlation >= MIN_SYNC_CORRELATION) {
                track.offset_s = ref_offset + e.offset_s;
            }
            report.push((track.id.clone(), estimate));
        }
        report
    }
}
