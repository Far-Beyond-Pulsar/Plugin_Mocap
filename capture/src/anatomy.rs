//! What the body allows: joints no camera sees are estimated from the ones that are seen, the
//! subject's bone lengths and how a human skeleton can bend, instead of being left as gaps.
//!
//! - The torso is completed first: missing hips are placed below the shoulders along the torso (and the
//!   other way round), a missing shoulder or hip of a pair from its partner and the pair's width.
//! - Each limb is a chain (shoulder, elbow, wrist; hip, knee, ankle). With the middle joint known and the end
//!   not, the end continues the limb at the bend it had a moment ago. With the end known and the middle
//!   not, two-bone inverse kinematics places the middle on the circle the two bone lengths allow, on the side
//!   the joint bends toward (elbows back, knees forward). With neither, the limb hangs relaxed.
//! - Elbows and knees are hinges: they bend one way, up to 150 degrees, and cannot hyperextend; a limb that
//!   would is bent back to straight.
//! - Estimates are smoothed against their last value, and reported as estimated so a viewer can show them
//!   as less certain than what a camera saw.

use nalgebra::Vector3;

/// COCO joint indices used here.
const L_SHOULDER: usize = 5;
const R_SHOULDER: usize = 6;
const L_HIP: usize = 11;
const R_HIP: usize = 12;

/// The limb chains: (root, middle, end, link index of the upper bone, link index of the lower bone,
/// is an arm). Link indices are the solver's: 5, 6 left arm; 7, 8 right arm; 12, 13 left leg; 14, 15 right leg.
const CHAINS: [(usize, usize, usize, usize, usize, bool); 4] = [
    (5, 7, 9, 5, 6, true),
    (6, 8, 10, 7, 8, true),
    (11, 13, 15, 12, 13, false),
    (12, 14, 16, 14, 15, false),
];

/// Lengths for a 1.75 m person (metres), used where none has been learned yet.
const PRIOR_UPPER_ARM: f64 = 0.29;
const PRIOR_FOREARM: f64 = 0.26;
const PRIOR_THIGH: f64 = 0.42;
const PRIOR_SHIN: f64 = 0.40;
const PRIOR_TORSO: f64 = 0.50;
const PRIOR_SHOULDERS: f64 = 0.38;
const PRIOR_HIPS: f64 = 0.20;
/// The most an elbow or knee flexes, radians.
const MAX_FLEXION: f64 = 150.0 * std::f64::consts::PI / 180.0;
/// How far past straight a hinge may go, radians.
const MAX_HYPEREXTENSION: f64 = 4.0 * std::f64::consts::PI / 180.0;
/// How much of its previous value an estimated joint keeps (smoothing).
const KEEP: f64 = 0.5;
/// The flexion a limb with a known middle joint is assumed to hold, radians, until it has been seen.
const REST_FLEXION_ARM: f64 = 0.25;
const REST_FLEXION_LEG: f64 = 0.10;

/// Fills in the joints that are not known.
#[derive(Default)]
pub struct Anatomy {
    previous: [Option<Vector3<f64>>; 17],
    /// The flexion of each chain when it was last seen.
    flexion: [Option<f64>; 4],
    estimated_before: [bool; 17],
    /// The torso's direction (hips to shoulders) while both were seen; it relaxes toward up when they are not.
    axis: Option<Vector3<f64>>,
}

fn finite(p: &[f32; 3]) -> Option<Vector3<f64>> {
    (p[0].is_finite() && p[1].is_finite() && p[2].is_finite()).then(|| Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])))
}

fn store(p: &mut [f32; 3], v: Vector3<f64>) {
    *p = [v[0] as f32, v[1] as f32, v[2] as f32];
}

/// `v` rotated by `angle` about the unit `axis`.
fn rotate(v: Vector3<f64>, axis: Vector3<f64>, angle: f64) -> Vector3<f64> {
    nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle) * v
}

impl Anatomy {
    /// A new estimator with no history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forgets everything seen so far.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Completes `points` (metres, +Y up, `NaN` where unknown) in place and says which joints were estimated.
    /// `lengths` are the solver's learned link lengths (`None` where not yet learned).
    pub fn complete(&mut self, points: &mut [[f32; 3]; 17], lengths: &[Option<f64>], up: Vector3<f64>) -> [bool; 17] {
        let seen: Vec<Option<Vector3<f64>>> = points.iter().map(finite).collect();
        let mut estimated = [false; 17];
        let len = |i: usize, prior: f64| lengths.get(i).copied().flatten().unwrap_or(prior);
        let scale = len(4, PRIOR_SHOULDERS) / PRIOR_SHOULDERS;
        let shoulder_width = len(4, PRIOR_SHOULDERS);
        let hip_width = len(11, PRIOR_HIPS * scale);
        let torso = (len(9, PRIOR_TORSO * scale) + len(10, PRIOR_TORSO * scale)) * 0.5;

        // The torso.
        let mut at: Vec<Option<Vector3<f64>>> = seen.clone();
        let mid = |a: Option<Vector3<f64>>, b: Option<Vector3<f64>>| match (a, b) {
            (Some(a), Some(b)) => Some((a + b) * 0.5),
            (Some(a), None) | (None, Some(a)) => Some(a),
            _ => None,
        };
        let shoulders = mid(at[L_SHOULDER], at[R_SHOULDER]);
        let hips = mid(at[L_HIP], at[R_HIP]);
        // The torso's direction: measured when both ends are seen (smoothed), else the last one relaxing toward up,
        // so a guess never feeds on its own guesses.
        let both_seen = seen[L_SHOULDER].is_some() && seen[R_SHOULDER].is_some() && seen[L_HIP].is_some() && seen[R_HIP].is_some();
        let axis = match (both_seen, shoulders.zip(hips), self.axis) {
            (true, Some((s, h)), old) if (s - h).norm() > 1e-6 => {
                let measured = (s - h).normalize();
                old.map_or(measured, |o| (o * 0.6 + measured * 0.4).normalize())
            }
            (_, _, Some(old)) => (old * 0.97 + up * 0.03).normalize(),
            _ => up,
        };
        self.axis = Some(axis);
        // Left, as a direction: from the shoulders if both are known, else from the hips, else as it was.
        let lateral = {
            let pair = |l: usize, r: usize| match (at[l], at[r]) {
                (Some(a), Some(b)) if (a - b).norm() > 1e-6 => Some((a - b).normalize()),
                _ => None,
            };
            pair(L_SHOULDER, R_SHOULDER).or_else(|| pair(L_HIP, R_HIP)).or_else(|| match (self.previous[L_SHOULDER], self.previous[R_SHOULDER]) {
                (Some(a), Some(b)) if (a - b).norm() > 1e-6 => Some((a - b).normalize()),
                _ => None,
            })
        };
        if let Some(lateral) = lateral {
            let lateral = (lateral - axis * lateral.dot(&axis)).try_normalize(1e-9).unwrap_or(lateral);
            // The far end of the torso from the near one.
            if hips.is_none() {
                if let Some(s) = shoulders {
                    let h = s - axis * torso;
                    at[L_HIP] = Some(h + lateral * hip_width * 0.5);
                    at[R_HIP] = Some(h - lateral * hip_width * 0.5);
                    estimated[L_HIP] = true;
                    estimated[R_HIP] = true;
                }
            } else if shoulders.is_none() {
                if let Some(h) = hips {
                    let s = h + axis * torso;
                    at[L_SHOULDER] = Some(s + lateral * shoulder_width * 0.5);
                    at[R_SHOULDER] = Some(s - lateral * shoulder_width * 0.5);
                    estimated[L_SHOULDER] = true;
                    estimated[R_SHOULDER] = true;
                }
            }
            // One of a pair missing: from its partner and the width.
            for (l, r, width) in [(L_SHOULDER, R_SHOULDER, shoulder_width), (L_HIP, R_HIP, hip_width)] {
                match (at[l], at[r]) {
                    (Some(a), None) => {
                        at[r] = Some(a - lateral * width);
                        estimated[r] = true;
                    }
                    (None, Some(b)) => {
                        at[l] = Some(b + lateral * width);
                        estimated[l] = true;
                    }
                    _ => {}
                }
            }
        }
        // The way the subject faces: from their left to up.
        let forward = match (lateral, at[L_SHOULDER].zip(at[R_SHOULDER]).or(at[L_HIP].zip(at[R_HIP]))) {
            (Some(l), Some(_)) => l.cross(&axis).try_normalize(1e-9),
            _ => None,
        };
        let down = -axis;

        // The limbs.
        for (c, &(a, b, e, upper, lower, arm)) in CHAINS.iter().enumerate() {
            let Some(root) = at[a] else { continue };
            let side = if arm { 1.0 } else { 1.0 } * if a % 2 == 1 { 1.0 } else { -1.0 };
            let l1 = len(upper, if arm { PRIOR_UPPER_ARM } else { PRIOR_THIGH } * scale);
            let l2 = len(lower, if arm { PRIOR_FOREARM } else { PRIOR_SHIN } * scale);
            // The way the middle joint points: elbows back, knees forward.
            let pole = forward.map(|f| if arm { -f } else { f });
            let outward = lateral.map_or(Vector3::zeros(), |l| l * side);
            match (at[b], at[e]) {
                (Some(mid_joint), None) => {
                    let dir = (mid_joint - root).try_normalize(1e-9).unwrap_or(down);
                    let flex = self.flexion[c].unwrap_or(if arm { REST_FLEXION_ARM } else { REST_FLEXION_LEG });
                    // The lower bone bends from the upper one toward the front (arms) or the back (legs).
                    let bend_axis = match forward {
                        Some(f) => f.cross(&dir).try_normalize(1e-9).unwrap_or(outward),
                        None => outward,
                    };
                    let sign = if arm { 1.0 } else { -1.0 };
                    let lower_dir = rotate(dir, bend_axis, sign * flex);
                    at[e] = Some(mid_joint + lower_dir * l2);
                    estimated[e] = true;
                }
                (None, Some(end)) => {
                    at[b] = Some(two_bone(root, end, l1, l2, pole.unwrap_or(outward)).0);
                    estimated[b] = true;
                    if (end - root).norm() > l1 + l2 {
                        at[e] = Some(two_bone(root, end, l1, l2, pole.unwrap_or(outward)).1);
                    }
                }
                (None, None) => {
                    // Hanging: down, a little away from the body; the forearm a little forward.
                    let upper_dir = (down + outward * 0.12).normalize();
                    let lower_dir = (down + forward.map_or(Vector3::zeros(), |f| f * if arm { 0.18 } else { -0.04 })).normalize();
                    let m = root + upper_dir * l1;
                    at[b] = Some(m);
                    at[e] = Some(m + lower_dir * l2);
                    estimated[b] = true;
                    estimated[e] = true;
                }
                (Some(_), Some(_)) => {}
            }
            // Remember how far the limb was bent while it was seen, and keep it a hinge.
            if let (Some(m), Some(end)) = (at[b], at[e]) {
                let (u, v) = ((m - root).normalize(), (end - m).normalize());
                let flex = u.dot(&v).clamp(-1.0, 1.0).acos();
                if !estimated[b] && !estimated[e] {
                    self.flexion[c] = Some(flex.min(MAX_FLEXION));
                }
                if let Some(f) = forward {
                    let fixed = limit_hinge(root, m, end, l2, f, arm);
                    if let Some(fixed) = fixed {
                        at[e] = Some(fixed);
                        estimated[e] = true;
                    }
                }
            }
        }

        // Smooth what was estimated against what it was a moment ago, keep the face where it was seen.
        for k in 0..17 {
            let Some(mut v) = at[k] else { continue };
            if estimated[k] {
                if let (Some(old), true) = (self.previous[k], self.estimated_before[k] || seen[k].is_none()) {
                    v = old * KEEP + v * (1.0 - KEEP);
                }
                store(&mut points[k], v);
            }
        }
        for k in 0..17 {
            self.previous[k] = finite(&points[k]);
            self.estimated_before[k] = estimated[k];
        }
        estimated
    }
}

/// The middle joint of a two-bone chain from `root` to `end`, and the end (pulled in if out of reach):
/// the middle joint is on the circle the two lengths allow, toward `pole`.
fn two_bone(root: Vector3<f64>, end: Vector3<f64>, l1: f64, l2: f64, pole: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let d = end - root;
    let dist = d.norm().max(1e-9);
    let dir = d / dist;
    let reach = (l1 + l2) * 0.9999;
    if dist >= reach {
        let straight_end = root + dir * reach;
        return (root + dir * l1, straight_end);
    }
    let dist = dist.max((l1 - l2).abs() + 1e-6);
    let x = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist);
    let h = (l1 * l1 - x * x).max(0.0).sqrt();
    let off = (pole - dir * pole.dot(&dir)).try_normalize(1e-9).unwrap_or_else(|| {
        let any = if dir[1].abs() < 0.9 { Vector3::new(0.0, 1.0, 0.0) } else { Vector3::new(1.0, 0.0, 0.0) };
        (any - dir * any.dot(&dir)).normalize()
    });
    (root + dir * x + off * h, end)
}

/// Keeps a hinge a hinge: if the lower bone bent the wrong way past the allowed hyperextension, or flexed
/// beyond the limit, the end is moved back; returns the corrected end, or `None` when it was fine.
fn limit_hinge(root: Vector3<f64>, mid: Vector3<f64>, end: Vector3<f64>, l2: f64, forward: Vector3<f64>, arm: bool) -> Option<Vector3<f64>> {
    let upper = (mid - root).try_normalize(1e-9)?;
    let lower = (end - mid).try_normalize(1e-9)?;
    // The bend plane's normal, from the upper bone and the way the joint bends: the lower bone should turn
    // toward `toward` (forward for elbows, backward for knees).
    let toward = if arm { forward } else { -forward };
    let side = upper.cross(&toward).try_normalize(1e-9)?;
    let in_plane = (toward - upper * toward.dot(&upper)).try_normalize(1e-9)?;
    // The lower bone in the plane's own axes: along the upper bone, and toward the bend side.
    let (along, bend, out_of_plane) = (lower.dot(&upper), lower.dot(&in_plane), lower.dot(&side));
    let angle = bend.atan2(along);
    let flexion_ok = angle >= -MAX_HYPEREXTENSION && angle <= MAX_FLEXION;
    // Out of the plane is allowed a little (the joints are not perfect hinges, and detections are noisy).
    if flexion_ok {
        return None;
    }
    let clamped = angle.clamp(-MAX_HYPEREXTENSION, MAX_FLEXION);
    let planar = (upper * clamped.cos() + in_plane * clamped.sin()) * (1.0 - out_of_plane * out_of_plane).max(0.0).sqrt() + side * out_of_plane;
    Some(mid + planar.try_normalize(1e-9)? * l2)
}
