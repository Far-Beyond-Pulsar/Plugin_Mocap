//! The body solver against independent triangulation, on a detector that is noisy, misses and sees
//! only part of the body from some cameras.

mod common;
use common::realism::{forearm_and_legs, observe_real, torso_down, up_to_hips, Quality};
use common::{subject, truth_cameras, Noise};
use mocap_capture::{BodySolver, CameraRig, COCO17_EDGES};
use nalgebra::Vector3;

struct Outcome {
    /// RMS position error over the joints each method had an answer for, metres.
    solver_rms: f64,
    independent_rms: f64,
    /// Joints answered per frame, of 17.
    solver_answered: f64,
    independent_answered: f64,
    /// The standard deviation of the forearm length over the frames (a rigid body keeps it).
    solver_forearm_sd: f64,
    independent_forearm_sd: f64,
}

fn run(qualities: &[Quality], seconds: f64, seed: u64) -> Outcome {
    let cams = truth_cameras(qualities.len());
    let rig = CameraRig::from_cameras(cams.clone());
    let mut solver = BodySolver::new(&rig);
    let mut noise = Noise(seed);
    let (mut s_err, mut i_err, mut s_n, mut i_n, mut frames) = (0.0, 0.0, 0usize, 0usize, 0usize);
    let (mut s_len, mut i_len): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut per: Vec<(f64, usize, usize)> = vec![(0.0, 0, 0); 17];
    for f in 0..(seconds * 60.0) as usize {
        let t = f as f64 / 60.0 * 2.0;
        let views: Vec<Option<Vec<[f32; 3]>>> = cams.iter().zip(qualities).map(|(c, q)| observe_real(c, t, &mut noise, q)).collect();
        let refs: Vec<Option<&[[f32; 3]]>> = views.iter().map(|v| v.as_deref()).collect();
        let solved = solver.solve(&refs);
        let independent = rig.triangulate_frame(&refs);
        if f < 120 {
            continue; // the bone lengths are being learned
        }
        frames += 1;
        let truth = subject(t);
        for k in 0..17 {
            let p = solved.points[k];
            if p[0].is_finite() {
                s_err += (Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])) - truth[k]).norm_squared();
                per[k].0 += (Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])) - truth[k]).norm_squared();
                per[k].1 += 1;
                per[k].2 += usize::from(solved.views[k]);
                s_n += 1;
            }
            let q = independent[k];
            // The same acceptance the editor applies to a triangulated point.
            if q[0].is_finite() && q[3] <= 15.0 {
                i_err += (Vector3::new(f64::from(q[0]), f64::from(q[1]), f64::from(q[2])) - truth[k]).norm_squared();
                i_n += 1;
            }
        }
        let length = |p: &[[f32; 3]], a: usize, b: usize| {
            let (x, y) = (Vector3::from_row_slice(&p[a]), Vector3::from_row_slice(&p[b]));
            (x - y).norm()
        };
        let (a, b) = (7, 9);
        if solved.points[a][0].is_finite() && solved.points[b][0].is_finite() {
            s_len.push(f64::from(length(&solved.points, a, b)));
        }
        let ip: Vec<[f32; 3]> = independent.iter().map(|q| [q[0], q[1], q[2]]).collect();
        if ip[a][0].is_finite() && ip[b][0].is_finite() && independent[a][3] <= 15.0 && independent[b][3] <= 15.0 {
            i_len.push(f64::from(length(&ip, a, b)));
        }
    }
    if std::env::var("SOLVER_DEBUG").is_ok() {
        for (k, (e, n, v)) in per.iter().enumerate() {
            eprintln!("joint {k}: rms {:.3} over {n} frames, {:.1} views", (e / (*n).max(1) as f64).sqrt(), *v as f64 / (*n).max(1) as f64);
        }
    }
    let sd = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
        (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    };
    Outcome {
        solver_rms: (s_err / s_n.max(1) as f64).sqrt(),
        independent_rms: (i_err / i_n.max(1) as f64).sqrt(),
        solver_answered: s_n as f64 / frames as f64,
        independent_answered: i_n as f64 / frames as f64,
        solver_forearm_sd: sd(&s_len),
        independent_forearm_sd: sd(&i_len),
    }
}

#[test]
fn with_three_full_cameras_the_solver_is_more_accurate_and_steadier() {
    let o = run(&[Quality::webcam(), Quality::webcam(), Quality::webcam()], 14.0, 11);
    eprintln!("rms solver {:.4} independent {:.4}; forearm sd {:.4} vs {:.4}", o.solver_rms, o.independent_rms, o.solver_forearm_sd, o.independent_forearm_sd);
    assert!(o.solver_rms < o.independent_rms, "{:.4} vs {:.4}", o.solver_rms, o.independent_rms);
    assert!(o.solver_rms < 0.04, "{:.4}", o.solver_rms);
    assert!(o.solver_forearm_sd < 0.5 * o.independent_forearm_sd, "a rigid forearm: {:.4} vs {:.4}", o.solver_forearm_sd, o.independent_forearm_sd);
}

#[test]
fn it_answers_for_joints_that_only_one_camera_sees() {
    // One camera sees the top half, one the bottom half, one everything: the legs and arms are each seen by two,
    // but with a camera of the three missing there are frames in which only one sees a joint.
    let qualities = [Quality::webcam().seeing(up_to_hips), Quality::webcam().seeing(torso_down), Quality::webcam().seeing(forearm_and_legs)];
    let o = run(&qualities, 14.0, 21);
    eprintln!("answered solver {:.1} independent {:.1}; rms {:.4} vs {:.4}", o.solver_answered, o.independent_answered, o.solver_rms, o.independent_rms);
    assert!(o.solver_answered >= o.independent_answered, "{:.1} vs {:.1}", o.solver_answered, o.independent_answered);
    assert!(o.solver_rms < 0.08, "{:.4}", o.solver_rms);
}

#[test]
fn gross_misses_do_not_drag_the_body() {
    let bad = Quality { outliers: 0.15, ..Quality::webcam() };
    let o = run(&[bad.clone(), bad.clone(), bad], 14.0, 31);
    eprintln!("outliers: rms {:.4} vs {:.4}", o.solver_rms, o.independent_rms);
    assert!(o.solver_rms < 0.06, "{:.4}", o.solver_rms);
    assert!(o.solver_rms <= o.independent_rms * 1.1, "{:.4} vs {:.4}", o.solver_rms, o.independent_rms);
}

#[test]
fn the_learned_bone_lengths_match_the_subject_and_left_matches_right() {
    let cams = truth_cameras(3);
    let rig = CameraRig::from_cameras(cams.clone());
    let mut solver = BodySolver::new(&rig);
    let mut noise = Noise(5);
    let q = Quality::webcam();
    for f in 0..240 {
        let t = f as f64 / 30.0;
        let views: Vec<Option<Vec<[f32; 3]>>> = cams.iter().map(|c| observe_real(c, t, &mut noise, &q)).collect();
        let refs: Vec<Option<&[[f32; 3]]>> = views.iter().map(|v| v.as_deref()).collect();
        solver.solve(&refs);
    }
    let truth = subject(0.0);
    let lengths = solver.lengths();
    for (e, &(a, b)) in COCO17_EDGES.iter().enumerate() {
        let real = (truth[a] - truth[b]).norm();
        let found = lengths[e].unwrap_or_else(|| panic!("bone {e} was not learned"));
        assert!((found - real).abs() < 0.03 + 0.1 * real, "bone {a}-{b}: {found:.3} vs {real:.3}");
    }
    // Upper arms and forearms: left equals right.
    assert_eq!(lengths[5], lengths[7], "left and right upper arm");
    assert_eq!(lengths[6], lengths[8], "left and right forearm");
}

#[test]
fn a_joint_nobody_sees_is_carried_briefly_and_then_dropped() {
    let cams = truth_cameras(3);
    let rig = CameraRig::from_cameras(cams.clone());
    let mut solver = BodySolver::new(&rig);
    let mut noise = Noise(7);
    let q = Quality { dropout: 0.0, outliers: 0.0, ..Quality::webcam() };
    let seen = |t: f64, noise: &mut Noise| -> Vec<Option<Vec<[f32; 3]>>> { cams.iter().map(|c| observe_real(c, t, noise, &q)).collect() };
    for f in 0..90 {
        let views = seen(f as f64 / 30.0, &mut noise);
        let refs: Vec<Option<&[[f32; 3]]>> = views.iter().map(|v| v.as_deref()).collect();
        solver.solve(&refs);
    }
    let mut hidden = |f: usize| {
        let mut views = seen(f as f64 / 30.0, &mut noise);
        for v in views.iter_mut().flatten() {
            v[9] = [f32::NAN, f32::NAN, 0.0];
        }
        let refs: Vec<Option<&[[f32; 3]]>> = views.iter().map(|v| v.as_deref()).collect();
        solver.solve(&refs)
    };
    assert!(hidden(90).points[9][0].is_finite(), "carried for a frame");
    assert_eq!(hidden(91).views[9], 0);
    let mut last = hidden(92);
    for f in 93..110 {
        last = hidden(f);
    }
    assert!(last.points[9][0].is_nan(), "dropped after a short while");
    assert!(last.points[7][0].is_finite(), "the rest of the body is unaffected");
}
