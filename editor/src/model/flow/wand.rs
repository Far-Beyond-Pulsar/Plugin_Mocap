//! Calibrating the cameras from a moving light: every camera's frames are searched for the one bright
//! point, the moments several cameras saw it together are collected, and the solve runs as an editor task.

use super::{CalibrationFlow, Method, Phase, SessionModel, SYNC_WINDOW_S};
use editor_task_queue::{TaskDescription, TaskDuration};
use mocap_capture::{find_light, CameraSpec, Check, WandCalibrator, WAND_SAMPLES_TARGET};
use std::sync::Arc;

/// What a camera last showed.
#[derive(Clone, Debug, Default)]
pub(super) struct WandView {
    seq: u64,
    time_s: f64,
    spot: Option<[f64; 2]>,
    size: (u32, u32),
}

impl CalibrationFlow {
    /// The light's last place in each camera, as keypoint-like overlays (`x`, `y` in 0..1, score 1), keyed by camera.
    pub fn wand_overlays(&self) -> Vec<(String, [f32; 3])> {
        if self.phase != Phase::Guiding || self.method != Method::Wand {
            return Vec::new();
        }
        self.wand_cams
            .iter()
            .zip(&self.wand_views)
            .filter_map(|(id, v)| {
                let spot = v.spot?;
                Some((id.clone(), [(spot[0] / f64::from(v.size.0.max(1))) as f32, (spot[1] / f64::from(v.size.1.max(1))) as f32, 1.0]))
            })
            .collect()
    }

    /// Moments collected, and where the light has been, per camera: `(camera id, moments seen, coverage 0..1)`.
    pub fn wand_progress(&self) -> Vec<(String, usize, f64)> {
        self.wand_cams
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), self.wand_seen.get(i).copied().unwrap_or(0), self.wand_coverage.get(i).copied().unwrap_or(0.0)))
            .collect()
    }
}

impl SessionModel {
    /// The running cameras (sources with video that are not a pose tracker).
    pub fn video_camera_count(&self) -> usize {
        self.sources.iter().filter(|s| s.descriptor.has_video && !s.descriptor.id.starts_with("pose:") && s.running()).count()
    }

    /// Starts collecting the light from every running camera.
    pub fn start_wand_calibration(&mut self) {
        let mut ids = Vec::new();
        let mut specs = Vec::new();
        let mut views = Vec::new();
        for s in self.sources.iter().filter(|s| s.descriptor.has_video && !s.descriptor.id.starts_with("pose:") && s.running()) {
            let Some(frame) = s.video.latest() else {
                self.status = format!("Camera {} has not delivered a frame yet", s.descriptor.id);
                return;
            };
            ids.push(s.descriptor.id.clone());
            specs.push(CameraSpec { id: s.descriptor.id.clone(), width: frame.width, height: frame.height });
            views.push(WandView { size: (frame.width, frame.height), ..WandView::default() });
        }
        if ids.len() < 2 {
            self.status = "Calibrating with a light needs at least two running cameras".to_owned();
            return;
        }
        let flow = &mut self.calib;
        flow.wand_seen = vec![0; ids.len()];
        flow.wand_coverage = vec![0.0; ids.len()];
        flow.wand = Some(WandCalibrator::new(specs));
        flow.wand_cams = ids;
        flow.wand_views = views;
        flow.checks.clear();
        flow.rms_px = None;
        flow.collected = 0;
        flow.phase = Phase::Guiding;
        self.status = "Wave the light through the space the cameras share".to_owned();
    }

    /// One tick of collecting: searches each new frame for the light, and offers the moment to the calibrator.
    pub(super) fn collect_wand(&mut self) -> bool {
        let Self { sources, calib, .. } = self;
        let mut fresh = false;
        for (i, id) in calib.wand_cams.clone().iter().enumerate() {
            let Some(entry) = sources.iter().find(|s| s.descriptor.id == *id && s.running()) else {
                calib.phase = Phase::Failed(format!("{id} stopped while calibrating"));
                return true;
            };
            let Some(frame) = entry.video.latest() else { continue };
            let view = &mut calib.wand_views[i];
            if frame.sequence == view.seq {
                continue;
            }
            *view = WandView { seq: frame.sequence, time_s: frame.time_s, spot: find_light(&frame).map(|s| [s.x, s.y]), size: (frame.width, frame.height) };
            fresh = true;
        }
        if !fresh {
            return false;
        }
        let newest = calib.wand_views.iter().map(|v| v.time_s).fold(f64::NEG_INFINITY, f64::max);
        let pixels: Vec<Option<[f64; 2]>> = calib.wand_views.iter().map(|v| if newest - v.time_s <= SYNC_WINDOW_S { v.spot } else { None }).collect();
        let mut solve = false;
        if let Some(wand) = calib.wand.as_mut() {
            wand.offer(&pixels);
            calib.collected = wand.samples();
            calib.wand_seen = wand.seen_by();
            calib.wand_coverage = wand.coverage();
            solve = wand.samples() >= WAND_SAMPLES_TARGET || wand.full();
        }
        if solve {
            self.solve_wand_calibration();
        }
        true
    }

    /// Solves with what was collected (as an editor task).
    pub fn solve_wand_calibration(&mut self) {
        let Some(wand) = self.calib.wand.clone() else { return };
        if self.calib.phase != Phase::Guiding {
            return;
        }
        self.calib.phase = Phase::Solving;
        let (slot, distance) = (Arc::clone(&self.calib.slot), self.calib.wand_distance);
        let description = TaskDescription::new("Solve camera calibration from a light", "mocap.calibration", TaskDuration::Long);
        editor_task_queue::global().submit(description, move |ctx| {
            ctx.set_phase("Adjusting the cameras");
            let result = wand.solve((0, 1, distance)).map(|r| {
                let ids: Vec<String> = r.rig.cameras.iter().map(|c| c.id.clone()).collect();
                let mut checks = vec![Check { name: "Reprojection", passed: r.rig.rms_px < 1.0, detail: format!("{:.2} px over {} points", r.rig.rms_px, r.points) }];
                for (i, id) in ids.iter().enumerate() {
                    let (focal, k1, k2) = r.lens[i];
                    checks.push(Check { name: "Lens", passed: true, detail: format!("{id}: focal {focal:.0} px, distortion {k1:+.3} {k2:+.3}") });
                    checks.push(Check { name: "Coverage", passed: r.coverage[i] >= 0.5, detail: format!("{id}: the light covered {:.0}% of the image", r.coverage[i] * 100.0) });
                }
                (r.rig, checks)
            });
            let failed = result.as_ref().err().map(ToString::to_string);
            *slot.lock() = Some(result.map_err(|e| e.to_string()));
            failed.map_or(Ok(()), Err)
        });
    }

    /// Adjusts the distance between the first two cameras (metres) that sets the size of the space.
    pub fn adjust_wand_distance(&mut self, delta: f64) {
        self.calib.wand_distance = (self.calib.wand_distance + delta).clamp(0.2, 20.0);
    }
}
