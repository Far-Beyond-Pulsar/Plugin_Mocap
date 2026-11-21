//! Joints no camera sees are estimated from the body's anatomy.

mod common;

use common::realism::{observe_real, Quality};
use common::{human as subject, truth_cameras, Noise};
use mocap_capture::{Anatomy, BodySolver, CameraRig};
use nalgebra::Vector3;

/// Runs the solver on three cameras with `hidden` keypoints invisible to every camera from frame 120 on;
/// returns, for each hidden joint, the mean error (metres) and the share of frames it had an answer, and
/// whether it was flagged as estimated.
fn hide(hidden: &[usize], frames: usize) -> Vec<(usize, f64, f64, bool)> {
    let cams = truth_cameras(3);
    let rig = CameraRig::from_cameras(cams.clone());
    let mut solver = BodySolver::new(&rig);
    let mut noise = Noise(17);
    let q = Quality::webcam().human();
    let mut sums = vec![(0.0, 0usize, 0usize, true); hidden.len()];
    let mut total = 0usize;
    for f in 0..frames {
        let t = f as f64 / 60.0 * 2.0;
        let mut views: Vec<Option<Vec<[f32; 3]>>> = cams.iter().map(|c| observe_real(c, t, &mut noise, &q)).collect();
        if f >= 120 {
            for v in views.iter_mut().flatten() {
                for &k in hidden {
                    v[k] = [f32::NAN, f32::NAN, 0.0];
                }
            }
        }
        let refs: Vec<Option<&[[f32; 3]]>> = views.iter().map(|v| v.as_deref()).collect();
        let solved = solver.solve(&refs);
        if f < 160 {
            continue; // the carried frames, before the estimate takes over
        }
        total += 1;
        let truth = subject(t);
        for (i, &k) in hidden.iter().enumerate() {
            let p = solved.points[k];
            if p[0].is_finite() {
                sums[i].0 += (Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])) - truth[k]).norm();
                sums[i].1 += 1;
                sums[i].3 &= solved.estimated[k];
            }
        }
    }
    hidden.iter().zip(&sums).map(|(&k, s)| (k, s.0 / s.1.max(1) as f64, s.1 as f64 / total as f64, s.3)).collect()
}

#[test]
fn legs_no_camera_sees_are_estimated_standing_not_left_as_gaps() {
    for (k, error, answered, flagged) in hide(&[13, 14, 15, 16], 360) {
        eprintln!("joint {k}: {error:.3} m, answered {answered:.2}");
        assert!(answered > 0.95, "joint {k} answered {answered:.2}");
        assert!(flagged, "joint {k} is marked as estimated");
        // Nobody sees the legs swing, so they hang: the knees are close, the swinging ankles are as far off as the swing.
        let limit = if k <= 14 { 0.2 } else { 0.4 };
        assert!(error < limit, "joint {k}: {error:.3} m");
    }
}

#[test]
fn hips_nobody_sees_come_from_the_shoulders_and_the_torso() {
    for (k, error, answered, _) in hide(&[11, 12], 360) {
        eprintln!("hip {k}: {error:.3} m, answered {answered:.2}");
        assert!(answered > 0.95);
        assert!(error < 0.12, "hip {k}: {error:.3} m");
    }
}

#[test]
fn an_arm_nobody_sees_hangs_where_an_arm_can_be() {
    for (k, error, answered, _) in hide(&[7, 9], 360) {
        eprintln!("arm joint {k}: {error:.3} m, answered {answered:.2}");
        assert!(answered > 0.95);
        assert!(error < 0.5, "joint {k}: {error:.3} m");
    }
}

fn standing() -> [[f32; 3]; 17] {
    let mut p = [[f32::NAN; 3]; 17];
    for (k, v) in [
        (5, [0.19, 1.45, 0.0]),
        (6, [-0.19, 1.45, 0.0]),
        (11, [0.1, 0.95, 0.0]),
        (12, [-0.1, 0.95, 0.0]),
        (13, [0.1, 0.53, 0.0]),
        (14, [-0.1, 0.53, 0.0]),
        (15, [0.1, 0.12, 0.0]),
        (16, [-0.1, 0.12, 0.0]),
        (7, [0.19, 1.16, 0.0]),
        (8, [-0.19, 1.16, 0.0]),
    ] {
        p[k] = v;
    }
    p
}

#[test]
fn an_elbow_cannot_hyperextend_and_a_knee_cannot_either() {
    let mut anatomy = Anatomy::new();
    let mut p = standing();
    // The left forearm tilted back (behind the upper arm, the subject faces +Z): an elbow cannot do that.
    p[9] = [0.19, 0.9, -0.12];
    // The right shin tilted forward of the thigh: a knee cannot do that.
    p[16] = [-0.1, 0.13, 0.12];
    let estimated = anatomy.complete(&mut p, &[None; 20], Vector3::new(0.0, 1.0, 0.0));
    assert!(estimated[9] && estimated[16], "both were corrected");
    assert!(p[9][2] > -0.03, "the wrist is no longer behind the elbow: {:?}", p[9]);
    assert!(p[16][2] < 0.03, "the ankle is no longer ahead of the knee: {:?}", p[16]);
    let forearm = (Vector3::from_row_slice(&p[9]) - Vector3::from_row_slice(&p[7])).norm();
    assert!((forearm - 0.26).abs() < 0.04, "the bone keeps its length: {forearm}");
}

#[test]
fn a_bent_elbow_and_knee_are_left_alone() {
    let mut anatomy = Anatomy::new();
    let mut p = standing();
    p[9] = [0.19, 1.0, 0.2]; // forearm forward: a flexed elbow
    p[15] = [-0.1, 0.2, -0.2]; // not used: right leg only
    p[15] = [0.1, 0.2, -0.25]; // shin back: a flexed knee
    let before = p;
    let estimated = anatomy.complete(&mut p, &[None; 20], Vector3::new(0.0, 1.0, 0.0));
    assert!(!estimated[9] && !estimated[15], "seen and plausible: not touched");
    assert_eq!(p[9], before[9]);
    assert_eq!(p[15], before[15]);
}

#[test]
fn two_bone_ik_puts_a_missing_elbow_behind_and_a_missing_knee_in_front() {
    let mut anatomy = Anatomy::new();
    let mut p = standing();
    p[7] = [f32::NAN; 3];
    p[9] = [0.25, 1.0, 0.15]; // the hand in front of the body
    p[13] = [f32::NAN; 3];
    p[15] = [0.1, 0.3, 0.15]; // the foot raised forward
    anatomy.complete(&mut p, &[None; 20], Vector3::new(0.0, 1.0, 0.0));
    let (shoulder, wrist) = (Vector3::from_row_slice(&p[5]), Vector3::from_row_slice(&p[9]));
    let elbow = Vector3::from_row_slice(&p[7]);
    let (u, l) = ((elbow - shoulder).norm(), (wrist - elbow).norm());
    assert!((u - 0.29).abs() < 0.02 && (l - 0.26).abs() < 0.02, "bone lengths {u} {l}");
    let line_mid = (shoulder + wrist) * 0.5;
    assert!(elbow[2] < line_mid[2], "the elbow is behind the line from shoulder to hand: {elbow:?}");
    let (hip, ankle, knee) = (Vector3::from_row_slice(&p[11]), Vector3::from_row_slice(&p[15]), Vector3::from_row_slice(&p[13]));
    assert!(knee[2] > ((hip + ankle) * 0.5)[2], "the knee is in front of the line from hip to foot: {knee:?}");
}
