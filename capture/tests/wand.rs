//! Calibrating from a moving point of light: focal lengths and lens distortion found, not assumed.

mod common;

use common::{truth_cameras, Noise};
use mocap_capture::{CameraModel, CameraSpec, WandCalibrator, WandError};
use nalgebra::Vector3;

/// The light's position at step `i`: a long smooth path through the volume, floor to head height.
fn light(i: usize) -> Vector3<f64> {
    let t = i as f64 * 0.011;
    Vector3::new(0.9 * (1.3 * t).sin() + 0.2 * (4.1 * t).cos(), 1.05 + 0.95 * (0.9 * t + 0.5).sin() * (0.37 * t).cos(), 0.9 * (1.7 * t + 1.0).cos() + 0.2 * (3.3 * t).sin())
}

fn lensed(count: usize) -> Vec<CameraModel> {
    truth_cameras(count)
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            c.k1 = -0.10 + 0.03 * i as f64;
            c.k2 = 0.02;
            c
        })
        .collect()
}

fn observe(cams: &[CameraModel], x: Vector3<f64>, noise: &mut Noise, sigma: f64, dropout: f64, outliers: f64) -> Vec<Option<[f64; 2]>> {
    cams.iter()
        .map(|c| {
            let p = c.project_raw(x)?;
            if !(0.0..f64::from(c.width)).contains(&p[0]) || !(0.0..f64::from(c.height)).contains(&p[1]) || noise.uniform() < dropout {
                return None;
            }
            if noise.uniform() < outliers {
                let miss = 30.0 + 50.0 * noise.uniform();
                let angle = std::f64::consts::TAU * noise.uniform();
                return Some([p[0] + miss * angle.cos(), p[1] + miss * angle.sin()]);
            }
            Some([p[0] + sigma * noise.gaussian(), p[1] + sigma * noise.gaussian()])
        })
        .collect()
}

fn calibrator_for(cams: &[CameraModel]) -> WandCalibrator {
    WandCalibrator::new(cams.iter().map(|c| CameraSpec { id: c.id.clone(), width: c.width, height: c.height }).collect())
}

fn distance(a: &CameraModel, b: &CameraModel) -> f64 {
    (Vector3::from_row_slice(&a.position()) - Vector3::from_row_slice(&b.position())).norm()
}

#[test]
fn the_cameras_focal_lengths_and_distortion_are_found_from_a_waved_light() {
    let cams = lensed(4);
    let mut calibrator = calibrator_for(&cams);
    let mut noise = Noise(3);
    for i in 0..6000 {
        calibrator.offer(&observe(&cams, light(i), &mut noise, 0.6, 0.05, 0.01));
    }
    assert!(calibrator.ready());
    let result = calibrator.solve((0, 1, distance(&cams[0], &cams[1]))).expect("a calibration");
    eprintln!("rms {:.2} px over {} points, coverage {:?}", result.rig.rms_px, result.points, result.coverage);
    assert!(result.rig.rms_px < 1.0, "{:.2}", result.rig.rms_px);
    for (found, real) in result.rig.cameras.iter().zip(&cams) {
        eprintln!("{}: focal {:.1} vs {:.1}, k1 {:.3} vs {:.3}, k2 {:.3} vs {:.3}", found.id, found.focal_px, real.focal_px, found.k1, real.k1, found.k2, real.k2);
        assert!((found.focal_px / real.focal_px - 1.0).abs() < 0.015, "{}: focal {:.1} vs {:.1}", found.id, found.focal_px, real.focal_px);
        assert!((found.k1 - real.k1).abs() < 0.04, "{}: k1 {:.3} vs {:.3}", found.id, found.k1, real.k1);
    }
    // The cameras stand where they did, up to where the world's origin and heading were put.
    for i in 0..cams.len() {
        for j in i + 1..cams.len() {
            let (found, real) = (distance(&result.rig.cameras[i], &result.rig.cameras[j]), distance(&cams[i], &cams[j]));
            assert!((found - real).abs() < 0.03, "cameras {i} and {j}: {found:.3} vs {real:.3} m");
        }
    }
    // Up is up: every camera's height over the floor agrees with the real ones to the floor's offset.
    let truth_height: Vec<f64> = cams.iter().map(|c| c.position()[1]).collect();
    let found_height: Vec<f64> = result.rig.cameras.iter().map(|c| c.position()[1]).collect();
    let offset = found_height[0] - truth_height[0];
    for (f, t) in found_height.iter().zip(&truth_height) {
        assert!((f - t - offset).abs() < 0.04, "height {f:.3} vs {t:.3} (offset {offset:.3})");
    }
    assert!(offset.abs() < 0.4, "the floor is taken to be near the lowest the light went: {offset:.3}");
}

#[test]
fn it_solves_with_only_a_pinhole_camera_pair_too() {
    let cams: Vec<CameraModel> = truth_cameras(2);
    let mut calibrator = calibrator_for(&cams);
    let mut noise = Noise(9);
    for i in 0..4000 {
        calibrator.offer(&observe(&cams, light(i), &mut noise, 0.5, 0.0, 0.0));
    }
    let result = calibrator.solve((0, 1, distance(&cams[0], &cams[1]))).expect("a calibration");
    assert!(result.rig.rms_px < 0.9, "{:.2}", result.rig.rms_px);
    assert!((distance(&result.rig.cameras[0], &result.rig.cameras[1]) - distance(&cams[0], &cams[1])).abs() < 0.03);
}

#[test]
fn too_little_light_or_cameras_that_never_overlap_are_refused_with_the_reason() {
    let cams = lensed(3);
    let mut few = calibrator_for(&cams);
    let mut noise = Noise(1);
    for i in 0..40 {
        few.offer(&observe(&cams, light(i * 40), &mut noise, 0.5, 0.0, 0.0));
    }
    assert!(matches!(few.solve((0, 1, 2.0)), Err(WandError::TooFewSamples { .. })));
    // The third camera never sees the light with the others.
    let mut split = calibrator_for(&cams);
    for i in 0..6000 {
        let mut views = observe(&cams, light(i), &mut noise, 0.5, 0.0, 0.0);
        views[2] = None;
        split.offer(&views);
    }
    let two_only = split.solve((0, 1, 2.0));
    assert!(matches!(two_only, Err(WandError::Unplaced(ref ids)) if ids == &vec!["camera:2".to_owned()]), "{two_only:?}");
    assert!(WandError::NoPair.to_string().contains("overlap"));
}

#[test]
fn a_light_that_rests_is_not_collected_again() {
    let cams = lensed(2);
    let mut calibrator = calibrator_for(&cams);
    let mut noise = Noise(4);
    let still = observe(&cams, light(100), &mut noise, 0.0, 0.0, 0.0);
    let kept = (0..200).filter(|_| calibrator.offer(&still)).count();
    assert_eq!(kept, 1);
}
