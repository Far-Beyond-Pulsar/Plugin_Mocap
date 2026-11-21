//! Bundle adjustment: every camera's pose, focal length and lens distortion, and every observed
//! 3D point, refined together so that the points reproject onto what the cameras saw.
//!
//! This is what makes a calibration accurate: an initial solution from pairs of views is only as good
//! as its worst pair, while the adjustment uses every observation at once and also finds the focal
//! lengths and the radial distortion a webcam's lens has. It is a Levenberg-Marquardt solve with a
//! robust (Cauchy) loss, the points eliminated with the Schur complement so the system that remains
//! is only as large as the cameras (nine numbers each).

use crate::calibration::CameraModel;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

type Cam = SVector<f64, 9>;
type CamBlock = SMatrix<f64, 9, 9>;
type Cp = SMatrix<f64, 9, 3>;
type J2c = SMatrix<f64, 2, 9>;
type J2p = SMatrix<f64, 2, 3>;

/// One camera's measurement of one point.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub camera: usize,
    pub point: usize,
    /// Where the camera saw it, pixels in the raw image (distortion included).
    pub pixel: [f64; 2],
    /// 0 to 1: how much to trust it.
    pub weight: f64,
}

/// What may change.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// The first camera's pose is held (the frame the rest is measured in).
    pub fix_first_pose: bool,
    /// Refine the focal lengths.
    pub focal: bool,
    /// Refine the radial distortion (k1, k2).
    pub distortion: bool,
    /// Detection noise, pixels: the robust loss starts to ignore residuals beyond a few of these.
    pub sigma_px: f64,
    pub max_iterations: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { fix_first_pose: true, focal: true, distortion: true, sigma_px: 1.0, max_iterations: 40 }
    }
}

/// How the adjustment went.
#[derive(Clone, Debug)]
pub struct Report {
    /// Root mean square reprojection error of the observations kept as inliers, pixels.
    pub rms_px: f64,
    /// The same per camera.
    pub per_camera_rms: Vec<f64>,
    /// Observations judged inliers at the end (`residual < 4 sigma`).
    pub inliers: Vec<bool>,
    pub iterations: usize,
}

/// The pixel a point appears at in `cam` after the parameter step `d` (rotation vector, translation,
/// log focal, k1, k2): `None` behind the camera.
fn project(cam: &CameraModel, d: &Cam, x: &Vector3<f64>) -> Option<[f64; 2]> {
    let rotation = nalgebra::Rotation3::from_scaled_axis(Vector3::new(d[0], d[1], d[2])).into_inner() * cam.rotation;
    let c = rotation * x + cam.translation + Vector3::new(d[3], d[4], d[5]);
    if c[2] < 1e-6 {
        return None;
    }
    let (k1, k2) = (cam.k1 + d[7], cam.k2 + d[8]);
    let focal = cam.focal_px * d[6].exp();
    let (u, v) = (c[0] / c[2], c[1] / c[2]);
    let r2 = u * u + v * v;
    let s = 1.0 + k1 * r2 + k2 * r2 * r2;
    Some([f64::from(cam.width) * 0.5 + focal * u * s, f64::from(cam.height) * 0.5 + focal * v * s])
}

/// Applies a parameter step to a camera.
fn apply(cam: &mut CameraModel, d: &Cam) {
    cam.rotation = nalgebra::Rotation3::from_scaled_axis(Vector3::new(d[0], d[1], d[2])).into_inner() * cam.rotation;
    cam.translation += Vector3::new(d[3], d[4], d[5]);
    cam.focal_px *= d[6].exp();
    cam.k1 += d[7];
    cam.k2 += d[8];
}

fn residual(cam: &CameraModel, x: &Vector3<f64>, o: &Observation) -> Option<[f64; 2]> {
    let q = project(cam, &Cam::zeros(), x)?;
    Some([q[0] - o.pixel[0], q[1] - o.pixel[1]])
}

/// The Jacobians of one observation's residual: with respect to its camera's nine parameters and its point.
fn jacobians(cam: &CameraModel, x: &Vector3<f64>, scale: f64) -> Option<(J2c, J2p)> {
    let mut jc = J2c::zeros();
    let steps = [1e-6, 1e-6, 1e-6, 1e-6 * scale, 1e-6 * scale, 1e-6 * scale, 1e-6, 1e-5, 1e-5];
    for (k, h) in steps.iter().enumerate() {
        let (mut plus, mut minus) = (Cam::zeros(), Cam::zeros());
        plus[k] = *h;
        minus[k] = -*h;
        let (a, b) = (project(cam, &plus, x)?, project(cam, &minus, x)?);
        jc[(0, k)] = (a[0] - b[0]) / (2.0 * h);
        jc[(1, k)] = (a[1] - b[1]) / (2.0 * h);
    }
    let mut jp = J2p::zeros();
    let h = 1e-6 * scale;
    for k in 0..3 {
        let mut dx = Vector3::zeros();
        dx[k] = h;
        let (a, b) = (project(cam, &Cam::zeros(), &(x + dx))?, project(cam, &Cam::zeros(), &(x - dx))?);
        jp[(0, k)] = (a[0] - b[0]) / (2.0 * h);
        jp[(1, k)] = (a[1] - b[1]) / (2.0 * h);
    }
    Some((jc, jp))
}

/// The robust cost of the observations (Cauchy loss with scale `scale_px`).
fn cost(cameras: &[CameraModel], points: &[Vector3<f64>], obs: &[Observation], scale_px: f64) -> f64 {
    obs.iter()
        .map(|o| match residual(&cameras[o.camera], &points[o.point], o) {
            Some(r) => {
                let q = (r[0] * r[0] + r[1] * r[1]) / (scale_px * scale_px);
                o.weight * scale_px * scale_px * (1.0 + q).ln()
            }
            None => o.weight * 1e6,
        })
        .sum()
}

/// Refines `cameras` and `points` against `obs`. Observations behind their camera count against the
/// solution heavily; points no camera sees are left alone.
pub fn bundle_adjust(cameras: &mut [CameraModel], points: &mut [Vector3<f64>], obs: &[Observation], options: &Options) -> Report {
    let n_cam = cameras.len();
    let scale = (points.iter().map(|p| p.norm()).sum::<f64>() / points.len().max(1) as f64).max(1e-3);
    let robust = options.sigma_px * 2.5;
    let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (i, o) in obs.iter().enumerate() {
        by_point[o.point].push(i);
    }
    let frozen = |c: usize, k: usize| -> bool {
        (options.fix_first_pose && c == 0 && k < 6) || (!options.focal && k == 6) || (!options.distortion && k >= 7)
    };
    let mut lambda = 1e-3;
    let mut current = cost(cameras, points, obs, robust);
    let mut iterations = 0;
    for _ in 0..options.max_iterations {
        iterations += 1;
        let mut u = vec![CamBlock::zeros(); n_cam];
        let mut e_c = vec![Cam::zeros(); n_cam];
        let mut v = vec![Matrix3::<f64>::zeros(); points.len()];
        let mut e_p = vec![Vector3::<f64>::zeros(); points.len()];
        let mut w_entry: Vec<Option<(J2c, J2p, f64)>> = vec![None; obs.len()];
        for (i, o) in obs.iter().enumerate() {
            let Some(r) = residual(&cameras[o.camera], &points[o.point], o) else { continue };
            let Some((mut jc, jp)) = jacobians(&cameras[o.camera], &points[o.point], scale) else { continue };
            for k in 0..9 {
                if frozen(o.camera, k) {
                    jc.set_column(k, &SVector::<f64, 2>::zeros());
                }
            }
            let rn = (r[0] * r[0] + r[1] * r[1]) / (robust * robust);
            let w = o.weight / (1.0 + rn);
            let rv = SVector::<f64, 2>::new(r[0], r[1]);
            u[o.camera] += jc.transpose() * jc * w;
            e_c[o.camera] += jc.transpose() * rv * w;
            v[o.point] += jp.transpose() * jp * w;
            e_p[o.point] += jp.transpose() * rv * w;
            w_entry[i] = Some((jc, jp, w));
        }
        // Damping, and the frozen parameters held at zero.
        for c in 0..n_cam {
            for k in 0..9 {
                let d = u[c][(k, k)];
                u[c][(k, k)] = if frozen(c, k) { 1.0 } else { d + lambda * d.max(1e-6) + 1e-9 };
            }
        }
        let mut v_inv = vec![Matrix3::<f64>::zeros(); points.len()];
        for p in 0..points.len() {
            let mut m = v[p];
            for k in 0..3 {
                m[(k, k)] += lambda * m[(k, k)].max(1e-6) + 1e-9;
            }
            v_inv[p] = m.try_inverse().unwrap_or_else(Matrix3::zeros);
        }
        // The reduced camera system.
        let mut s = nalgebra::DMatrix::<f64>::zeros(9 * n_cam, 9 * n_cam);
        let mut rhs = nalgebra::DVector::<f64>::zeros(9 * n_cam);
        for c in 0..n_cam {
            for a in 0..9 {
                for b in 0..9 {
                    s[(9 * c + a, 9 * c + b)] += u[c][(a, b)];
                }
                rhs[9 * c + a] -= e_c[c][a];
            }
        }
        for (p, list) in by_point.iter().enumerate() {
            let entries: Vec<(usize, Cp)> = list
                .iter()
                .filter_map(|&i| w_entry[i].map(|(jc, jp, w)| (obs[i].camera, jc.transpose() * jp * w)))
                .collect();
            for &(ci, wi) in &entries {
                let wv = wi * v_inv[p];
                let vec_part = wv * e_p[p];
                for a in 0..9 {
                    rhs[9 * ci + a] += vec_part[a];
                }
                for &(cj, wj) in &entries {
                    let block = wv * wj.transpose();
                    for a in 0..9 {
                        for b in 0..9 {
                            s[(9 * ci + a, 9 * cj + b)] -= block[(a, b)];
                        }
                    }
                }
            }
        }
        let Some(delta) = s.lu().solve(&rhs) else { break };
        // Back-substitute the points.
        let mut trial_cameras: Vec<CameraModel> = cameras.to_vec();
        let mut trial_points: Vec<Vector3<f64>> = points.to_vec();
        for c in 0..n_cam {
            let d = Cam::from_fn(|k, _| delta[9 * c + k]);
            apply(&mut trial_cameras[c], &d);
        }
        for (p, list) in by_point.iter().enumerate() {
            let mut wt_dc = Vector3::<f64>::zeros();
            for &i in list {
                if let Some((jc, jp, w)) = w_entry[i] {
                    let d = Cam::from_fn(|k, _| delta[9 * obs[i].camera + k]);
                    wt_dc += (jc.transpose() * jp * w).transpose() * d;
                }
            }
            trial_points[p] += -(v_inv[p] * (e_p[p] + wt_dc));
        }
        let new_cost = cost(&trial_cameras, &trial_points, obs, robust);
        if new_cost < current {
            let relative = (current - new_cost) / current.max(1e-12);
            cameras.clone_from_slice(&trial_cameras);
            points.clone_from_slice(&trial_points);
            current = new_cost;
            lambda = (lambda * 0.4).max(1e-9);
            if relative < 1e-7 {
                break;
            }
        } else {
            lambda *= 5.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    // The report.
    let mut inliers = vec![false; obs.len()];
    let (mut sum, mut count) = (0.0, 0usize);
    let (mut cam_sum, mut cam_count) = (vec![0.0; n_cam], vec![0usize; n_cam]);
    for (i, o) in obs.iter().enumerate() {
        if let Some(r) = residual(&cameras[o.camera], &points[o.point], o) {
            let e2 = r[0] * r[0] + r[1] * r[1];
            if e2.sqrt() < 4.0 * options.sigma_px.max(0.5) * 2.0 {
                inliers[i] = true;
                sum += e2;
                count += 1;
                cam_sum[o.camera] += e2;
                cam_count[o.camera] += 1;
            }
        }
    }
    Report {
        rms_px: (sum / count.max(1) as f64).sqrt(),
        per_camera_rms: cam_sum.iter().zip(&cam_count).map(|(s, n)| (s / (*n).max(1) as f64).sqrt()).collect(),
        inliers,
        iterations,
    }
}
