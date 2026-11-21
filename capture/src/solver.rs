//! One body for every camera: the subject's joints are found together, from every camera's
//! detections at once, under the constraint that bones keep their length.
//!
//! Triangulating each keypoint on its own throws away what is known about a body: a wrist cannot
//! be further from its elbow than the forearm is long, and a joint one camera sees is not free
//! along that camera's ray. The solver finds, per frame, the 17 joint positions that minimise
//!
//! - the reprojection error of every detection (robust to a detector's gross misses, weighted by its
//!   score and by how trustworthy that keypoint is),
//! - the difference of each bone's length from the subject's, learned from the first frames and held
//!   (left and right limbs share a length), and
//! - a weak pull toward where the joint was the frame before, which carries a joint through a frame
//!   in which no camera sees it and picks a point on the ray for one only one camera sees.
//!
//! The result is smoother and more accurate than independent triangulation, and it has an answer for
//! joints that fewer than two cameras see.

use crate::anatomy::Anatomy;
use crate::calibration::{good, CameraModel, CameraRig, KEYPOINTS, RELIABILITY};
use nalgebra::{DMatrix, DVector, Vector3};

/// Reprojection noise assumed for a detection, pixels.
const SIGMA_PX: f64 = 4.0;
/// A residual this many sigmas out is down-weighted to a third of its weight (Cauchy scale).
const ROBUST_SCALE: f64 = 2.5;
/// The tolerance of a bone's length, as a fraction of it, and at least this many metres.
const BONE_SIGMA_FRACTION: f64 = 0.025;
const BONE_SIGMA_MIN_M: f64 = 0.008;
/// How far a joint may move between frames at 60 Hz before the pull toward its old place matters, metres.
const TEMPORAL_SIGMA_M: f64 = 0.12;
/// The links the solver holds to a length: the 16 bones, then the face's loose ties to the shoulders (the
/// face is not otherwise joined to the body).
const LINKS: [(usize, usize); LINK_COUNT] = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (3, 5), (4, 6), (0, 5), (0, 6),
];
/// The number of links.
pub const LINK_COUNT: usize = 20;
/// What a link measures before it is learned (metres on a subject with a 0.38 m shoulder width), for the
/// links the cameras rarely both see: the face. The limbs have none: they are only ever learned.
const PRIORS: [Option<f64>; LINK_COUNT] = [
    Some(0.04), Some(0.04), Some(0.085), Some(0.085), None, None, None, None, None, None, None, None, None, None, None, None,
    Some(0.2), Some(0.2), Some(0.26), Some(0.26),
];
/// The shoulder width the priors are for, metres.
const PRIOR_SHOULDER_M: f64 = 0.38;

/// Samples of a bone's length before it is trusted.
const LENGTH_SAMPLES: usize = 30;
/// A joint is predicted from its last place for at most this many frames without a detection.
const MAX_HOLD_FRAMES: u32 = 6;
/// A joint whose detections disagree with the result by more than this (mean, pixels) is dropped.
const MAX_JOINT_ERROR_PX: f64 = 25.0;
/// Only frames that fit this well (mean pixels) teach the bone lengths.
const LEARN_MAX_ERROR_PX: f64 = 6.0;
const ITERATIONS: usize = 6;

/// One frame's joints.
#[derive(Clone, Debug, PartialEq)]
pub struct Solved {
    /// Joint positions, metres (`NaN` where the joint is not known).
    pub points: [[f32; 3]; KEYPOINTS],
    /// How many cameras detected each joint this frame (0: carried from before or estimated).
    pub views: [u8; KEYPOINTS],
    /// Joints no camera saw that the body's anatomy filled in (see [`crate::Anatomy`]).
    pub estimated: [bool; KEYPOINTS],
}

/// The left or right counterpart of a COCO joint.
fn mirror(joint: usize) -> usize {
    match joint {
        0 => 0,
        j if j % 2 == 1 => j + 1,
        j => j - 1,
    }
}

/// Fits the subject's body to every camera's keypoints, frame after frame.
pub struct BodySolver {
    cameras: Vec<CameraModel>,
    /// Each bone's length, metres, once learned.
    lengths: [Option<f64>; LINK_COUNT],
    samples: Vec<Vec<f64>>,
    previous: [Option<Vector3<f64>>; KEYPOINTS],
    missing: [u32; KEYPOINTS],
    anatomy: Anatomy,
    /// Fill in the joints nobody sees from the body's anatomy.
    pub use_anatomy: bool,
}

impl BodySolver {
    /// A solver for the cameras of `rig`.
    pub fn new(rig: &CameraRig) -> Self {
        Self {
            cameras: rig.cameras.clone(),
            lengths: [None; LINK_COUNT],
            samples: vec![Vec::new(); LINK_COUNT],
            previous: [None; KEYPOINTS],
            missing: [0; KEYPOINTS],
            anatomy: Anatomy::new(),
            use_anatomy: true,
        }
    }

    /// The learned lengths, metres: the 16 bones in `COCO17_EDGES` order, then the face-to-shoulder links.
    pub fn lengths(&self) -> [Option<f64>; LINK_COUNT] {
        self.lengths
    }

    /// The length of link `e` and how far it may stray: what was learned, or the prior for the face.
    fn link(&self, e: usize) -> Option<(f64, f64)> {
        if let Some(l) = self.lengths[e] {
            let soft = e >= 16;
            return Some((l, if soft { (0.2 * l).max(0.04) } else { (BONE_SIGMA_FRACTION * l).max(BONE_SIGMA_MIN_M) }));
        }
        let scale = self.lengths[4].map_or(1.0, |w| w / PRIOR_SHOULDER_M);
        PRIORS[e].map(|p| (p * scale, 0.4 * p * scale))
    }

    /// Sets the bone lengths (for example from a take's stored subject), skipping the learning.
    pub fn set_lengths(&mut self, lengths: [Option<f64>; LINK_COUNT]) {
        self.lengths = lengths;
    }

    /// Forgets the body and what was learned about it (a different subject steps in).
    pub fn reset(&mut self) {
        self.lengths = [None; LINK_COUNT];
        self.samples = vec![Vec::new(); LINK_COUNT];
        self.previous = [None; KEYPOINTS];
        self.missing = [0; KEYPOINTS];
        self.anatomy.reset();
    }

    /// Solves one frame: `views[i]` is camera `i`'s keypoints (`x`, `y` in 0..1 of the frame, score).
    pub fn solve(&mut self, views: &[Option<&[[f32; 3]]>]) -> Solved {
        // Detections per joint: (camera, pixel, weight).
        let mut detections: Vec<Vec<(usize, [f64; 2], f64)>> = vec![Vec::new(); KEYPOINTS];
        for (c, view) in views.iter().enumerate().take(self.cameras.len()) {
            let Some(view) = view else { continue };
            for (k, p) in view.iter().enumerate().take(KEYPOINTS) {
                if good(p) {
                    detections[k].push((c, self.cameras[c].pixel(*p), f64::from(p[2]).min(1.0).sqrt() * RELIABILITY[k].sqrt()));
                }
            }
        }
        let mut x = self.initial(&detections);
        let active: Vec<usize> = (0..KEYPOINTS).filter(|&k| x[k].is_some()).collect();
        if active.is_empty() {
            self.age_all();
            return Solved { points: [[f32::NAN; 3]; KEYPOINTS], views: [0; KEYPOINTS], estimated: [false; KEYPOINTS] };
        }
        self.refine(&mut x, &active, &detections, true);
        // Frames that fit well teach the bone lengths; then solve again with them.
        if self.learn(&x, &detections) {
            self.refine(&mut x, &active, &detections, true);
        }
        let mut points = [[f32::NAN; 3]; KEYPOINTS];
        let mut counts = [0u8; KEYPOINTS];
        for &k in &active {
            let Some(p) = x[k] else { continue };
            let seen = detections[k].len();
            let error = self.joint_error(&p, &detections[k]);
            if seen > 0 && error > MAX_JOINT_ERROR_PX {
                // Detections that cannot be made to agree with the body: no answer rather than a wrong one.
                self.previous[k] = None;
                continue;
            }
            points[k] = [p[0] as f32, p[1] as f32, p[2] as f32];
            counts[k] = seen.min(255) as u8;
            self.previous[k] = Some(p);
            self.missing[k] = if seen > 0 { 0 } else { self.missing[k] + 1 };
        }
        for k in 0..KEYPOINTS {
            if x[k].is_none() {
                self.age(k);
            }
        }
        let estimated = if self.use_anatomy { self.anatomy.complete(&mut points, &self.lengths, Vector3::new(0.0, 1.0, 0.0)) } else { [false; KEYPOINTS] };
        Solved { points, views: counts, estimated }
    }

    fn age(&mut self, k: usize) {
        self.missing[k] += 1;
        if self.missing[k] > MAX_HOLD_FRAMES {
            self.previous[k] = None;
        }
    }

    fn age_all(&mut self) {
        for k in 0..KEYPOINTS {
            self.age(k);
        }
    }

    /// Mean reprojection error of `p` against `detections`, pixels.
    fn joint_error(&self, p: &Vector3<f64>, detections: &[(usize, [f64; 2], f64)]) -> f64 {
        if detections.is_empty() {
            return 0.0;
        }
        let total: f64 = detections
            .iter()
            .map(|(c, o, _)| self.cameras[*c].project(*p).map_or(1e3, |q| ((q[0] - o[0]).powi(2) + (q[1] - o[1]).powi(2)).sqrt()))
            .sum();
        total / detections.len() as f64
    }

    /// A starting place for every joint that can have one: triangulated from two or more cameras, on the
    /// ray of the one camera that sees it at the depth it had (or its neighbours have), or where it was.
    fn initial(&self, detections: &[Vec<(usize, [f64; 2], f64)>]) -> [Option<Vector3<f64>>; KEYPOINTS] {
        let mut out: [Option<Vector3<f64>>; KEYPOINTS] = [None; KEYPOINTS];
        for k in 0..KEYPOINTS {
            if detections[k].len() >= 2 {
                out[k] = self.triangulate(&detections[k]);
            }
        }
        // Depth of the body along a ray, from what is already placed: the mean distance of the placed joints.
        for _pass in 0..4 {
        for k in 0..KEYPOINTS {
            if out[k].is_some() {
                continue;
            }
            if let Some(&(c, pixel, _)) = detections[k].first() {
                let cam = &self.cameras[c];
                let centre = Vector3::from_row_slice(&cam.position());
                let depth_of = |p: &Vector3<f64>| (cam.rotation * p + cam.translation)[2];
                let depth = self.previous[k]
                    .as_ref()
                    .map(depth_of)
                    .or_else(|| {
                        let placed: Vec<f64> = out.iter().flatten().map(depth_of).collect();
                        (!placed.is_empty()).then(|| placed.iter().sum::<f64>() / placed.len() as f64)
                    })
                    .or_else(|| {
                        let old: Vec<f64> = self.previous.iter().flatten().map(depth_of).collect();
                        (!old.is_empty()).then(|| old.iter().sum::<f64>() / old.len() as f64)
                    });
                let ray = ray_of(cam, pixel);
                // A placed neighbour and a known bone length fix the depth: the ray meets the sphere around the
                // neighbour at the bone's length (the root nearer the expected depth, when there are two).
                if let Some(p) = self.on_sphere(k, &out, &centre, &ray, depth) {
                    out[k] = Some(p);
                    continue;
                }
                if let Some(z) = depth.filter(|z| *z > 0.2) {
                    let ray = ray_of(cam, pixel);
                    // Along the ray to depth `z` (the ray is unit length in camera z).
                    out[k] = Some(centre + ray * z);
                    continue;
                }
            }
            if self.missing[k] <= MAX_HOLD_FRAMES {
                out[k] = self.previous[k];
            }
        }
        }
        out
    }

    /// Where the ray `centre + ray * s` meets the sphere around a placed bone neighbour of joint `k`.
    fn on_sphere(&self, k: usize, placed: &[Option<Vector3<f64>>; KEYPOINTS], centre: &Vector3<f64>, ray: &Vector3<f64>, expected_depth: Option<f64>) -> Option<Vector3<f64>> {
        for (e, &(a, b)) in LINKS.iter().enumerate() {
            let other = if a == k { b } else if b == k { a } else { continue };
            let (Some((length, _)), Some(c)) = (self.link(e), placed[other]) else { continue };
            // |centre + ray*s - c|^2 = length^2
            let d = centre - c;
            let (qa, qb, qc) = (ray.dot(ray), 2.0 * ray.dot(&d), d.dot(&d) - length * length);
            let disc = qb * qb - 4.0 * qa * qc;
            let roots: Vec<f64> = if disc >= 0.0 {
                let r = disc.sqrt();
                vec![(-qb - r) / (2.0 * qa), (-qb + r) / (2.0 * qa)]
            } else {
                // The ray misses the sphere: the point of the ray nearest the neighbour.
                vec![-qb / (2.0 * qa)]
            };
            let want = expected_depth.unwrap_or(f64::NAN);
            let best = roots.into_iter().filter(|s| *s > 0.0).min_by(|x, y| {
                let (dx, dy) = ((x - want).abs(), (y - want).abs());
                if want.is_finite() { dx.total_cmp(&dy) } else { x.total_cmp(y) }
            })?;
            return Some(centre + ray * best);
        }
        None
    }

    /// The least-squares point for the detections of one joint (two or more cameras).
    fn triangulate(&self, detections: &[(usize, [f64; 2], f64)]) -> Option<Vector3<f64>> {
        let mut x: Vector3<f64> = Vector3::zeros();
        // Linear start: the midpoint of the rays of the first two cameras is good enough for Gauss-Newton.
        let (a, b) = (&detections[0], &detections[1]);
        let (ca, cb) = (Vector3::from_row_slice(&self.cameras[a.0].position()), Vector3::from_row_slice(&self.cameras[b.0].position()));
        let (da, db) = (ray_of(&self.cameras[a.0], a.1), ray_of(&self.cameras[b.0], b.1));
        let w = ca - cb;
        let (p, q, r, s, t) = (da.dot(&da), da.dot(&db), db.dot(&db), da.dot(&w), db.dot(&w));
        let denom = p * r - q * q;
        if denom.abs() < 1e-9 {
            return None;
        }
        let (sa, sb) = ((q * t - r * s) / denom, (p * t - q * s) / denom);
        if sa <= 0.0 || sb <= 0.0 {
            return None;
        }
        x += ((ca + da * sa) + (cb + db * sb)) * 0.5;
        // Refine against every view.
        for _ in 0..4 {
            let mut jtj = nalgebra::Matrix3::<f64>::zeros();
            let mut jtr = Vector3::<f64>::zeros();
            for (c, o, weight) in detections {
                let Some((q, j)) = project_with_jacobian(&self.cameras[*c], &x) else { return None };
                let r = nalgebra::Vector2::new(q[0] - o[0], q[1] - o[1]);
                let w = weight * weight / (1.0 + (r.norm() / (ROBUST_SCALE * SIGMA_PX)).powi(2));
                jtj += j.transpose() * j * w;
                jtr += j.transpose() * r * w;
            }
            let step = (jtj + nalgebra::Matrix3::identity() * 1e-6).lu().solve(&(-jtr))?;
            x += step;
            if step.norm() < 1e-5 {
                break;
            }
        }
        Some(x)
    }

    /// Gauss-Newton with Levenberg damping and robust (Cauchy) weights over the active joints.
    fn refine(&self, x: &mut [Option<Vector3<f64>>; KEYPOINTS], active: &[usize], detections: &[Vec<(usize, [f64; 2], f64)>], temporal: bool) {
        let index_of = |k: usize| active.iter().position(|&a| a == k);
        let n = active.len() * 3;
        let mut lambda = 1e-3;
        for _ in 0..ITERATIONS {
            let mut jtj = DMatrix::<f64>::zeros(n, n);
            let mut jtr = DVector::<f64>::zeros(n);
            let mut cost = 0.0;
            // Reprojection.
            for (i, &k) in active.iter().enumerate() {
                let Some(p) = x[k] else { continue };
                for (c, o, weight) in &detections[k] {
                    let Some((q, j)) = project_with_jacobian(&self.cameras[*c], &p) else { continue };
                    let r = nalgebra::Vector2::new((q[0] - o[0]) / SIGMA_PX, (q[1] - o[1]) / SIGMA_PX);
                    let w = weight * weight / (1.0 + (r.norm() / ROBUST_SCALE).powi(2));
                    let jr = j / SIGMA_PX;
                    let block = jr.transpose() * jr * w;
                    let grad = jr.transpose() * r * w;
                    for a in 0..3 {
                        jtr[3 * i + a] += grad[a];
                        for b in 0..3 {
                            jtj[(3 * i + a, 3 * i + b)] += block[(a, b)];
                        }
                    }
                    cost += w * r.norm_squared();
                }
            }
            // Bone lengths.
            for (e, &(a, b)) in LINKS.iter().enumerate() {
                let (Some((length, sigma)), Some(ia), Some(ib), Some(pa), Some(pb)) = (self.link(e), index_of(a), index_of(b), x[a], x[b]) else { continue };
                let d = pa - pb;
                let dist = d.norm().max(1e-6);
                let r = (dist - length) / sigma;
                let g = d / dist / sigma;
                for (u, su) in [(ia, 1.0), (ib, -1.0)] {
                    for a_ in 0..3 {
                        jtr[3 * u + a_] += su * g[a_] * r;
                        for (v, sv) in [(ia, 1.0), (ib, -1.0)] {
                            for b_ in 0..3 {
                                jtj[(3 * u + a_, 3 * v + b_)] += su * sv * g[a_] * g[b_];
                            }
                        }
                    }
                }
                cost += r * r;
            }
            // Pull toward where each joint was.
            if temporal {
                for (i, &k) in active.iter().enumerate() {
                    let (Some(p), Some(old)) = (x[k], self.previous[k]) else { continue };
                    // Stronger for a joint with fewer detections: it has less else to go by.
                    let w = if detections[k].is_empty() { 1.0 } else { 0.15 / detections[k].len() as f64 };
                    for a in 0..3 {
                        let r = (p[a] - old[a]) / TEMPORAL_SIGMA_M;
                        jtr[3 * i + a] += w * r / TEMPORAL_SIGMA_M;
                        jtj[(3 * i + a, 3 * i + a)] += w / (TEMPORAL_SIGMA_M * TEMPORAL_SIGMA_M);
                        cost += w * r * r;
                    }
                }
            }
            for d in 0..n {
                jtj[(d, d)] += lambda * jtj[(d, d)].max(1e-3) + 1e-6;
            }
            let Some(step) = jtj.clone().lu().solve(&(-&jtr)) else { return };
            // Accept the step if it does not make things much worse; damp harder otherwise.
            let mut trial = *x;
            for (i, &k) in active.iter().enumerate() {
                if let Some(p) = trial[k].as_mut() {
                    *p += Vector3::new(step[3 * i], step[3 * i + 1], step[3 * i + 2]);
                }
            }
            let new_cost = self.cost(&trial, active, detections, temporal);
            if new_cost <= cost * 1.0001 + 1e-9 {
                *x = trial;
                lambda = (lambda * 0.5).max(1e-6);
                if step.norm() < 1e-5 {
                    break;
                }
            } else {
                lambda = (lambda * 4.0).min(1e3);
            }
        }
    }

    /// The robust cost of a configuration (the same terms as [`Self::refine`]).
    fn cost(&self, x: &[Option<Vector3<f64>>; KEYPOINTS], active: &[usize], detections: &[Vec<(usize, [f64; 2], f64)>], temporal: bool) -> f64 {
        let mut cost = 0.0;
        for &k in active {
            let Some(p) = x[k] else { continue };
            for (c, o, weight) in &detections[k] {
                let Some(q) = self.cameras[*c].project(p) else { continue };
                let r = nalgebra::Vector2::new((q[0] - o[0]) / SIGMA_PX, (q[1] - o[1]) / SIGMA_PX).norm();
                cost += weight * weight * (ROBUST_SCALE * ROBUST_SCALE) * (1.0 + (r / ROBUST_SCALE).powi(2)).ln();
            }
            if temporal {
                if let Some(old) = self.previous[k] {
                    let w = if detections[k].is_empty() { 1.0 } else { 0.15 / detections[k].len() as f64 };
                    cost += w * ((p - old) / TEMPORAL_SIGMA_M).norm_squared() * 0.5;
                }
            }
        }
        for (e, &(a, b)) in LINKS.iter().enumerate() {
            if let (Some((length, sigma)), Some(pa), Some(pb)) = (self.link(e), x[a], x[b]) {
                cost += (((pa - pb).norm() - length) / sigma).powi(2) * 0.5;
            }
        }
        cost
    }

    /// Takes the bone lengths of a well-fitting frame as samples; sets a length once there are enough.
    /// True when a length was just set.
    fn learn(&mut self, x: &[Option<Vector3<f64>>; KEYPOINTS], detections: &[Vec<(usize, [f64; 2], f64)>]) -> bool {
        let mut newly = false;
        for (e, &(a, b)) in LINKS.iter().enumerate() {
            if self.lengths[e].is_some() {
                continue;
            }
            let (Some(pa), Some(pb)) = (x[a], x[b]) else { continue };
            if detections[a].len() < 2 || detections[b].len() < 2 {
                continue;
            }
            if self.joint_error(&pa, &detections[a]) > LEARN_MAX_ERROR_PX || self.joint_error(&pb, &detections[b]) > LEARN_MAX_ERROR_PX {
                continue;
            }
            self.samples[e].push((pa - pb).norm());
            if self.samples[e].len() >= LENGTH_SAMPLES {
                let mut sorted = self.samples[e].clone();
                sorted.sort_by(f64::total_cmp);
                self.lengths[e] = Some(sorted[sorted.len() / 2]);
                newly = true;
            }
        }
        if newly {
            // Left and right limbs are the same length.
            for (e, &(a, b)) in LINKS.iter().enumerate() {
                let (ma, mb) = (mirror(a), mirror(b));
                let Some(m) = LINKS.iter().position(|&(c, d)| (c, d) == (ma, mb) || (c, d) == (mb, ma)) else { continue };
                if m == e {
                    continue;
                }
                match (self.lengths[e], self.lengths[m]) {
                    (Some(l), Some(r)) => {
                        let mean = (l + r) * 0.5;
                        self.lengths[e] = Some(mean);
                        self.lengths[m] = Some(mean);
                    }
                    (Some(l), None) => self.lengths[m] = Some(l),
                    _ => {}
                }
            }
        }
        newly
    }
}

/// The unit-z ray of a pixel in world space: the camera centre plus `ray * depth` is the point at that depth.
fn ray_of(cam: &CameraModel, pixel: [f64; 2]) -> Vector3<f64> {
    let (cx, cy) = (f64::from(cam.width) * 0.5, f64::from(cam.height) * 0.5);
    let dir = Vector3::new((pixel[0] - cx) / cam.focal_px, (pixel[1] - cy) / cam.focal_px, 1.0);
    cam.rotation.transpose() * dir
}

/// A point's pixel and the derivative of that pixel with respect to the point.
fn project_with_jacobian(cam: &CameraModel, p: &Vector3<f64>) -> Option<([f64; 2], nalgebra::Matrix2x3<f64>)> {
    let c = cam.rotation * p + cam.translation;
    if c[2] < 1e-3 {
        return None;
    }
    let (cx, cy) = (f64::from(cam.width) * 0.5, f64::from(cam.height) * 0.5);
    let f = cam.focal_px;
    let (inv, inv2) = (1.0 / c[2], 1.0 / (c[2] * c[2]));
    let mut j = nalgebra::Matrix2x3::<f64>::zeros();
    for k in 0..3 {
        j[(0, k)] = f * (cam.rotation[(0, k)] * inv - c[0] * cam.rotation[(2, k)] * inv2);
        j[(1, k)] = f * (cam.rotation[(1, k)] * inv - c[1] * cam.rotation[(2, k)] * inv2);
    }
    Some(([cx + f * c[0] * inv, cy + f * c[1] * inv], j))
}
