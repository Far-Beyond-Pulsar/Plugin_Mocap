//! Calibrating cameras from a moving point of light: wave a phone's flashlight, an LED or a bright
//! ball through the space the cameras share, and every moment gives each camera that sees it one exact
//! point. Thousands of such points seen by several cameras are what a bundle adjustment needs to find
//! each camera's pose, its focal length and its lens distortion to well under a pixel, with no printed
//! target and without the subject's body (whose keypoints are blurry, flat and different in every camera).
//!
//! The method is the multi-camera self-calibration of Svoboda et al.: pairs of views give a first guess
//! (essential matrix, then the other cameras by resection), the adjustment ([`crate::bundle`]) refines
//! it, and the size of the scene is set by one distance you give (two cameras a known distance apart).

use crate::bundle::{bundle_adjust, Observation, Options};
use crate::calibration::{CameraModel, CameraRig, CameraSpec};
use crate::epipolar::{essential_ransac, recover_pose, triangulate, Rng};
use nalgebra::{DMatrix, Matrix3, Vector3, SVD};
use std::fmt;

/// The fewest points (seen by two or more cameras) to solve from.
pub const MIN_SAMPLES: usize = 60;
/// Enough points: more add nothing.
pub const SAMPLES_TARGET: usize = 400;
/// The most points kept.
pub const MAX_SAMPLES: usize = 800;
/// A point must have moved this far (pixels, in some camera) since the last one kept, so the samples
/// spread over the volume instead of piling up where the light rests.
const MIN_MOVE_PX: f64 = 14.0;
/// What a camera's first focal guess assumes: a horizontal field of view of about 65 degrees.
const FOCAL_GUESS_PER_WIDTH: f64 = 0.785;
/// A calibration that reprojects worse than this (pixels, RMS) is refused.
pub const MAX_RMS_PX: f64 = 2.0;
/// The grid the image is divided into to see how much of it the light covered.
const COVERAGE_COLUMNS: usize = 4;
const COVERAGE_ROWS: usize = 3;

/// Why a calibration could not be made.
#[derive(Clone, Debug, PartialEq)]
pub enum WandError {
    TooFewCameras,
    TooFewSamples { have: usize, need: usize },
    /// No two cameras saw the light together often enough.
    NoPair,
    /// These cameras could not be tied to the rest (too few shared points).
    Unplaced(Vec<String>),
    /// The two cameras the scale is taken from are not both calibrated or too close together.
    BadScale,
    PoorFit { rms: f64 },
}

impl fmt::Display for WandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewCameras => write!(f, "calibration needs at least two cameras"),
            Self::TooFewSamples { have, need } => write!(f, "only {have} points seen by two or more cameras, {need} needed: wave the light where the cameras overlap"),
            Self::NoPair => write!(f, "no two cameras saw the light at the same moments: wave it where two of them overlap"),
            Self::Unplaced(ids) => write!(f, "cameras {ids:?} saw the light too rarely together with the others: wave it in front of them too"),
            Self::BadScale => write!(f, "the two cameras the size is taken from are too close together or not calibrated"),
            Self::PoorFit { rms } => write!(f, "the cameras do not agree on where the light was ({rms:.1} px): check that it is a single bright point, the room is dim and the cameras did not move"),
        }
    }
}

impl std::error::Error for WandError {}

/// A calibration result with what to check.
#[derive(Clone, Debug)]
pub struct WandResult {
    pub rig: CameraRig,
    /// Points the cameras were adjusted against.
    pub points: usize,
    /// Per camera: the fraction of the image grid the light passed through (0 to 1). Lens distortion is
    /// only known where the light was, so a low value means the edges are a guess.
    pub coverage: Vec<f64>,
    /// Per camera: the focal length found, pixels, and the radial distortion `(k1, k2)`.
    pub lens: Vec<(f64, f64, f64)>,
}

/// Collects the light's position in every camera, moment by moment.
#[derive(Clone)]
pub struct WandCalibrator {
    cameras: Vec<CameraSpec>,
    /// One entry per kept moment: the light's pixel in each camera, `None` where it was not seen.
    samples: Vec<Vec<Option<[f64; 2]>>>,
    last: Vec<Option<[f64; 2]>>,
}

impl WandCalibrator {
    /// A calibrator for `cameras` (index = position in every sample).
    pub fn new(cameras: Vec<CameraSpec>) -> Self {
        let last = vec![None; cameras.len()];
        Self { cameras, samples: Vec::new(), last }
    }

    /// Moments kept so far.
    pub fn samples(&self) -> usize {
        self.samples.len()
    }

    /// True when enough are in to solve.
    pub fn ready(&self) -> bool {
        self.samples.len() >= MIN_SAMPLES
    }

    /// True when no more are wanted.
    pub fn full(&self) -> bool {
        self.samples.len() >= MAX_SAMPLES
    }

    /// Offers one moment: `pixels[i]` is where camera `i` saw the light. Kept when at least two cameras saw
    /// it and it has moved since the last kept one. True when kept.
    pub fn offer(&mut self, pixels: &[Option<[f64; 2]>]) -> bool {
        let views: Vec<Option<[f64; 2]>> = (0..self.cameras.len()).map(|i| pixels.get(i).copied().flatten()).collect();
        if views.iter().flatten().count() < 2 || self.full() {
            return false;
        }
        let moved = views.iter().zip(&self.last).any(|(now, before)| match (now, before) {
            (Some(a), Some(b)) => ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() >= MIN_MOVE_PX,
            (Some(_), None) => true,
            _ => false,
        });
        if !moved {
            return false;
        }
        for (slot, now) in self.last.iter_mut().zip(&views) {
            if now.is_some() {
                *slot = *now;
            }
        }
        self.samples.push(views);
        true
    }

    /// How many kept moments each camera saw the light in.
    pub fn seen_by(&self) -> Vec<usize> {
        (0..self.cameras.len()).map(|c| self.samples.iter().filter(|s| s[c].is_some()).count()).collect()
    }

    /// Solves. `scale` is `(camera a, camera b, metres between them)`: how big the space is.
    pub fn solve(&self, scale: (usize, usize, f64)) -> Result<WandResult, WandError> {
        let n = self.cameras.len();
        if n < 2 {
            return Err(WandError::TooFewCameras);
        }
        if self.samples.len() < MIN_SAMPLES {
            return Err(WandError::TooFewSamples { have: self.samples.len(), need: MIN_SAMPLES });
        }
        let mut cameras: Vec<CameraModel> = self.cameras.iter().map(first_guess).collect();
        // The first guesses, kept apart from the cameras that are about to be placed.
        let guess: Vec<(f64, f64, f64)> = cameras.iter().map(|m| (f64::from(m.width) * 0.5, f64::from(m.height) * 0.5, m.focal_px)).collect();
        let normalised = |c: usize, p: [f64; 2]| -> [f64; 2] { [(p[0] - guess[c].0) / guess[c].2, (p[1] - guess[c].1) / guess[c].2] };
        // The pair that saw the light together most.
        let shared = |a: usize, b: usize| self.samples.iter().filter(|s| s[a].is_some() && s[b].is_some()).count();
        let (mut best, mut pair) = (0, (0, 1));
        for a in 0..n {
            for b in a + 1..n {
                let k = shared(a, b);
                if k > best {
                    best = k;
                    pair = (a, b);
                }
            }
        }
        if best < 20 {
            return Err(WandError::NoPair);
        }
        let (ia, ib) = pair;
        let both: Vec<usize> = (0..self.samples.len()).filter(|&i| self.samples[i][ia].is_some() && self.samples[i][ib].is_some()).collect();
        let pa: Vec<[f64; 2]> = both.iter().map(|&i| normalised(ia, self.samples[i][ia].unwrap_or_default())).collect();
        let pb: Vec<[f64; 2]> = both.iter().map(|&i| normalised(ib, self.samples[i][ib].unwrap_or_default())).collect();
        let threshold = 4.0 / cameras[ia].focal_px;
        let mut rng = Rng::new(7);
        let (e, inliers) = essential_ransac(&pa, &pb, threshold, 300, &mut rng).ok_or(WandError::NoPair)?;
        let (pose, _) = recover_pose(&e, &pa, &pb, &inliers).ok_or(WandError::NoPair)?;
        cameras[ia].rotation = Matrix3::identity();
        cameras[ia].translation = Vector3::zeros();
        cameras[ib].rotation = pose.r;
        cameras[ib].translation = pose.t;
        let mut placed = vec![false; n];
        placed[ia] = true;
        placed[ib] = true;
        adjust(&self.samples, &mut cameras, &placed, false);
        // Place the others by resection against the points the placed cameras triangulate.
        loop {
            let points = triangulate_samples(&self.samples, &cameras, &placed);
            let mut candidate: Option<(usize, usize)> = None;
            for c in (0..n).filter(|&c| !placed[c]) {
                let k = points.iter().enumerate().filter(|(i, p)| p.is_some() && self.samples[*i][c].is_some()).count();
                if k >= 12 && candidate.is_none_or(|(_, best)| k > best) {
                    candidate = Some((c, k));
                }
            }
            let Some((c, _)) = candidate else { break };
            let correspondences: Vec<(Vector3<f64>, [f64; 2])> = points
                .iter()
                .enumerate()
                .filter_map(|(i, p)| Some(((*p)?, normalised(c, self.samples[i][c]?))))
                .collect();
            let Some((r, t)) = resection(&correspondences, 40.0 / cameras[c].focal_px, &mut rng) else { break };
            cameras[c].rotation = r;
            cameras[c].translation = t;
            placed[c] = true;
            adjust(&self.samples, &mut cameras, &placed, false);
        }
        if placed.iter().any(|p| !p) {
            let missing = (0..n).filter(|&c| !placed[c]).map(|c| self.cameras[c].id.clone()).collect();
            return Err(WandError::Unplaced(missing));
        }
        // Adjust: every camera's pose, focal and distortion, with every point.
        let mut points: Vec<Option<Vector3<f64>>> = triangulate_samples(&self.samples, &cameras, &placed);
        let mut index_of = vec![usize::MAX; self.samples.len()];
        let mut xs: Vec<Vector3<f64>> = Vec::new();
        for (i, p) in points.iter().enumerate() {
            if let Some(p) = p {
                index_of[i] = xs.len();
                xs.push(*p);
            }
        }
        let mut obs: Vec<Observation> = Vec::new();
        for (i, s) in self.samples.iter().enumerate() {
            if index_of[i] == usize::MAX {
                continue;
            }
            for (c, p) in s.iter().enumerate() {
                if let Some(p) = p {
                    obs.push(Observation { camera: c, point: index_of[i], pixel: *p, weight: 1.0 });
                }
            }
        }
        let options = Options { fix_first_pose: true, focal: true, distortion: false, sigma_px: 1.5, max_iterations: 30 };
        bundle_adjust(&mut cameras, &mut xs, &obs, &options);
        // With the focal lengths near right, the distortion can be found; then again with the outliers dropped.
        let options = Options { distortion: true, ..options };
        let first = bundle_adjust(&mut cameras, &mut xs, &obs, &options);
        let kept: Vec<Observation> = obs.iter().zip(&first.inliers).filter(|(_, ok)| **ok).map(|(o, _)| *o).collect();
        let options = Options { sigma_px: 0.8, max_iterations: 40, ..options };
        let report = bundle_adjust(&mut cameras, &mut xs, &kept, &options);
        // The points again, from the refined cameras.
        for (i, slot) in points.iter_mut().enumerate() {
            if index_of[i] != usize::MAX {
                *slot = Some(xs[index_of[i]]);
            }
        }
        if report.rms_px > MAX_RMS_PX {
            return Err(WandError::PoorFit { rms: report.rms_px });
        }
        // Metric scale from the given distance.
        let (sa, sb, metres) = scale;
        if sa >= n || sb >= n || sa == sb || metres <= 0.0 {
            return Err(WandError::BadScale);
        }
        let apart = (Vector3::from_row_slice(&cameras[sa].position()) - Vector3::from_row_slice(&cameras[sb].position())).norm();
        if apart < 1e-6 {
            return Err(WandError::BadScale);
        }
        let factor = metres / apart;
        for c in cameras.iter_mut() {
            c.translation *= factor;
        }
        for x in xs.iter_mut() {
            *x *= factor;
        }
        let notes = align_to_floor(&mut cameras, &mut xs);
        let coverage = self.coverage();
        let mut notes: Vec<&'static str> = notes;
        if coverage.iter().any(|c| *c < 0.5) {
            notes.push("The light covered less than half of some camera's image: its lens distortion is only known where the light passed");
        }
        let lens = cameras.iter().map(|c| (c.focal_px, c.k1, c.k2)).collect();
        let parallax = median_parallax(&cameras, &xs);
        let rig = CameraRig { per_camera_rms: report.per_camera_rms, cameras, rms_px: report.rms_px, samples: self.samples.len(), parallax_deg: parallax, notes };
        Ok(WandResult { rig, points: xs.len(), coverage, lens })
    }
}

/// Refines the placed cameras' poses (the focal lengths stay at their guesses: from two or three views they cannot be told from depth) against the points
/// they triangulate, so the next camera is placed from a structure that is right and not just plausible.
fn adjust(samples: &[Vec<Option<[f64; 2]>>], cameras: &mut [CameraModel], placed: &[bool], distortion: bool) {
    let points = triangulate_samples(samples, cameras, placed);
    let mut xs: Vec<Vector3<f64>> = Vec::new();
    let mut obs: Vec<Observation> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let Some(p) = p else { continue };
        for (c, pixel) in samples[i].iter().enumerate() {
            if let (true, Some(pixel)) = (placed[c], pixel) {
                obs.push(Observation { camera: c, point: xs.len(), pixel: *pixel, weight: 1.0 });
            }
        }
        xs.push(*p);
    }
    let first = (0..cameras.len()).find(|&c| placed[c]).unwrap_or(0);
    // Only the placed cameras go in, the first of them held; the rest keep their guesses.
    let index: Vec<usize> = (0..cameras.len()).filter(|&c| placed[c]).collect();
    let mut subset: Vec<CameraModel> = index.iter().map(|&c| cameras[c].clone()).collect();
    for o in obs.iter_mut() {
        o.camera = index.iter().position(|&c| c == o.camera).unwrap_or(0);
    }
    let _ = first;
    let options = Options { fix_first_pose: true, focal: false, distortion, sigma_px: 3.0, max_iterations: 25 };
    bundle_adjust(&mut subset, &mut xs, &obs, &options);
    for (k, &c) in index.iter().enumerate() {
        cameras[c] = subset[k].clone();
    }
}

/// A camera with its focal length guessed from the frame width, distortion zero, at the origin.
fn first_guess(spec: &CameraSpec) -> CameraModel {
    CameraModel {
        id: spec.id.clone(),
        width: spec.width,
        height: spec.height,
        focal_px: FOCAL_GUESS_PER_WIDTH * f64::from(spec.width),
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
        k1: 0.0,
        k2: 0.0,
    }
}

/// Each moment seen by two or more placed cameras, triangulated; `None` where it cannot be or fits badly.
fn triangulate_samples(samples: &[Vec<Option<[f64; 2]>>], cameras: &[CameraModel], placed: &[bool]) -> Vec<Option<Vector3<f64>>> {
    samples
        .iter()
        .map(|s| {
            let views: Vec<(usize, [f64; 2])> = s
                .iter()
                .enumerate()
                .filter_map(|(c, p)| {
                    let p = (*p)?;
                    placed[c].then(|| {
                        let m = &cameras[c];
                        (c, [(p[0] - f64::from(m.width) * 0.5) / m.focal_px, (p[1] - f64::from(m.height) * 0.5) / m.focal_px])
                    })
                })
                .collect();
            if views.len() < 2 {
                return None;
            }
            let input: Vec<_> = views.iter().map(|(c, p)| (&cameras[*c].rotation, &cameras[*c].translation, *p, 1.0)).collect();
            let x = triangulate(&input)?;
            // In front of every camera and roughly consistent with every view (the focal lengths and the lens are only guessed so far).
            let ok = views.iter().all(|(c, p)| {
                let m = &cameras[*c];
                let q = m.rotation * x + m.translation;
                q[2] > 1e-6 && ((q[0] / q[2] - p[0]).powi(2) + (q[1] / q[2] - p[1]).powi(2)).sqrt() * m.focal_px < 15.0
            });
            ok.then_some(x)
        })
        .collect()
}

/// The camera pose `x_cam = R x + t` that sees `points` at their normalised pixels: a direct linear
/// transform in RANSAC, then least squares over its inliers.
fn resection(correspondences: &[(Vector3<f64>, [f64; 2])], threshold: f64, rng: &mut Rng) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let n = correspondences.len();
    if n < 8 {
        return None;
    }
    let mut best: Option<(Matrix3<f64>, Vector3<f64>, usize)> = None;
    for _ in 0..300 {
        let picks: Vec<usize> = (0..8).map(|_| rng.below(n)).collect();
        let Some((r, t)) = dlt(&picks.iter().map(|&i| correspondences[i]).collect::<Vec<_>>()) else { continue };
        let count = correspondences.iter().filter(|(x, p)| reprojection(&r, &t, x, *p) < threshold).count();
        if best.as_ref().is_none_or(|(_, _, c)| count > *c) {
            best = Some((r, t, count));
        }
    }
    let (r, t, count) = best?;
    if count < 8 {
        return None;
    }
    let inliers: Vec<(Vector3<f64>, [f64; 2])> = correspondences.iter().filter(|(x, p)| reprojection(&r, &t, x, *p) < threshold).copied().collect();
    dlt(&inliers).or(Some((r, t)))
}

fn reprojection(r: &Matrix3<f64>, t: &Vector3<f64>, x: &Vector3<f64>, p: [f64; 2]) -> f64 {
    let c = r * x + t;
    if c[2] <= 1e-6 {
        return f64::INFINITY;
    }
    ((c[0] / c[2] - p[0]).powi(2) + (c[1] / c[2] - p[1]).powi(2)).sqrt()
}

/// The direct linear transform: the projection matrix of normalised points, split into a rotation and a translation.
fn dlt(points: &[(Vector3<f64>, [f64; 2])]) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    if points.len() < 6 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros((points.len() * 2).max(12), 12);
    for (i, (x, p)) in points.iter().enumerate() {
        let xh = [x[0], x[1], x[2], 1.0];
        for k in 0..4 {
            a[(2 * i, k)] = xh[k];
            a[(2 * i, 8 + k)] = -p[0] * xh[k];
            a[(2 * i + 1, 4 + k)] = xh[k];
            a[(2 * i + 1, 8 + k)] = -p[1] * xh[k];
        }
    }
    let svd = SVD::new(a, false, true);
    let v_t = svd.v_t?;
    let (index, _) = svd.singular_values.iter().enumerate().min_by(|x, y| x.1.total_cmp(y.1))?;
    let row = v_t.row(index.min(v_t.nrows() - 1));
    let m = Matrix3::new(row[0], row[1], row[2], row[4], row[5], row[6], row[8], row[9], row[10]);
    let t = Vector3::new(row[3], row[7], row[11]);
    let svd = SVD::new(m, true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut r = u * v_t;
    let scale = svd.singular_values.iter().sum::<f64>() / 3.0;
    if scale.abs() < 1e-12 {
        return None;
    }
    let mut t = t / scale;
    if r.determinant() < 0.0 {
        r = -r;
        t = -t;
    }
    // The sign of the whole projection is fixed by the determinant (a rotation has +1), so the points
    // are in front exactly when the data are consistent; `reprojection` refuses those behind.
    Some((r, t))
}

/// Turns the cameras' frame into a world: +Y up (the average of the cameras' own up), the origin below
/// the middle of the points on the floor (the lowest the light went), +Z along the first camera's view.
fn align_to_floor(cameras: &mut [CameraModel], points: &mut [Vector3<f64>]) -> Vec<&'static str> {
    // Cameras stand level (no roll): each one's right vector is horizontal, so up is perpendicular to them all
    // (the direction they span the least). Their own up vectors are only roughly up: they are pitched.
    let mut rows = nalgebra::DMatrix::<f64>::zeros(cameras.len().max(3), 3);
    for (i, c) in cameras.iter().enumerate() {
        let right = c.rotation.transpose() * Vector3::new(1.0, 0.0, 0.0);
        for k in 0..3 {
            rows[(i, k)] = right[k];
        }
    }
    let svd = SVD::new(rows, false, true);
    let rough: Vector3<f64> = cameras.iter().map(|c| -(c.rotation.transpose() * Vector3::new(0.0, 1.0, 0.0))).sum();
    let level = svd.v_t.as_ref().and_then(|v| {
        let (i, _) = svd.singular_values.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?;
        let v = Vector3::new(v[(i, 0)], v[(i, 1)], v[(i, 2)]);
        Some(if v.dot(&rough) < 0.0 { -v } else { v })
    });
    let up = match level {
        Some(u) if cameras.len() >= 2 && u.norm() > 1e-9 => u.normalize(),
        _ if rough.norm() > 1e-9 => rough.normalize(),
        _ => Vector3::new(0.0, 1.0, 0.0),
    };
    let forward = {
        let f = cameras[0].rotation.transpose() * Vector3::new(0.0, 0.0, 1.0);
        let flat = f - up * f.dot(&up);
        if flat.norm() > 1e-6 {
            flat.normalize()
        } else {
            up.cross(&Vector3::new(1.0, 0.0, 0.0)).normalize()
        }
    };
    let right = up.cross(&forward).normalize();
    // Rows: world x (right), y (up), z (forward) expressed in the old frame.
    let to_world = Matrix3::from_rows(&[right.transpose(), up.transpose(), forward.transpose()]);
    let mut heights: Vec<f64> = points.iter().map(|p| up.dot(p)).collect();
    heights.sort_by(f64::total_cmp);
    let floor = heights.get(heights.len() / 20).copied().unwrap_or(0.0);
    let centre = if points.is_empty() { Vector3::zeros() } else { points.iter().sum::<Vector3<f64>>() / points.len() as f64 };
    let origin_old = centre - up * (up.dot(&centre) - floor);
    for c in cameras.iter_mut() {
        // x_cam = R x_old + t, x_old = to_world^T x_new + origin_old.
        c.translation += c.rotation * origin_old;
        c.rotation *= to_world.transpose();
    }
    for p in points.iter_mut() {
        *p = to_world * (*p - origin_old);
    }
    vec!["The floor is taken to be the lowest point the light reached: touch it to the floor first", "Up is taken from the cameras standing level (no roll)"]
}

/// The median angle between two cameras' rays to the same point.
fn median_parallax(cameras: &[CameraModel], points: &[Vector3<f64>]) -> f64 {
    let mut angles = Vec::new();
    for x in points.iter().step_by((points.len() / 50).max(1)) {
        let rays: Vec<Vector3<f64>> = cameras.iter().map(|c| (x - Vector3::from_row_slice(&c.position())).normalize()).collect();
        for i in 0..rays.len() {
            for j in i + 1..rays.len() {
                angles.push(rays[i].dot(&rays[j]).clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
    }
    angles.sort_by(f64::total_cmp);
    angles.get(angles.len() / 2).copied().unwrap_or(0.0)
}

impl WandCalibrator {
    /// Per camera, the fraction of its image grid (4 by 3) the light has passed through so far.
    pub fn coverage(&self) -> Vec<f64> {
        (0..self.cameras.len())
            .map(|c| {
                let mut cells = [false; COVERAGE_COLUMNS * COVERAGE_ROWS];
                for s in &self.samples {
                    if let Some(p) = s[c] {
                        let col = ((p[0] / f64::from(self.cameras[c].width)) * COVERAGE_COLUMNS as f64).clamp(0.0, COVERAGE_COLUMNS as f64 - 1.0) as usize;
                        let row = ((p[1] / f64::from(self.cameras[c].height)) * COVERAGE_ROWS as f64).clamp(0.0, COVERAGE_ROWS as f64 - 1.0) as usize;
                        cells[row * COVERAGE_COLUMNS + col] = true;
                    }
                }
                cells.iter().filter(|c| **c).count() as f64 / cells.len() as f64
            })
            .collect()
    }
}
