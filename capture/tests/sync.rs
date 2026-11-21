//! Several sources on one clock: sampling between frames, clock offsets, estimating them from motion.

use mocap_capture::{estimate_offset, MixInput, Mixer, PoseBus, PoseFrame};
use mocap_format::{ChannelDesc, ChannelKind, Quantisation, RigProfile};

fn rig(kind: ChannelKind) -> RigProfile {
    RigProfile {
        name: "r".into(),
        joints: Vec::new(),
        channels: vec![ChannelDesc { name: "v".into(), kind, joint: None, quantisation: Quantisation::Lossless }],
    }
}

/// A trajectory with bursts of movement and rests, so that a lag shows.
fn moving(t: f64) -> f64 {
    let envelope = (0.5 + 0.5 * (std::f64::consts::TAU * 0.13 * t).sin()).powi(2);
    envelope * ((std::f64::consts::TAU * 1.1 * t).sin() + 0.6 * (std::f64::consts::TAU * 2.3 * t + 1.0).sin())
}

struct Feed {
    bus: std::sync::Arc<PoseBus>,
    rate: f64,
    /// Seconds the stamps are late.
    late: f64,
    seq: u64,
    next: f64,
    value: fn(f64) -> Vec<f32>,
}

impl Feed {
    fn new(rate: f64, late: f64, value: fn(f64) -> Vec<f32>) -> Self {
        Self { bus: PoseBus::new(4096), rate, late, seq: 0, next: 0.0, value }
    }

    /// Delivers the frames up to true time `now`: the values of a frame are those of its true time,
    /// its stamp is `late` later.
    fn run(&mut self, now: f64) {
        while self.next <= now {
            self.seq += 1;
            self.bus.push(PoseFrame { sequence: self.seq, time_s: self.next + self.late, values: (self.value)(self.next) });
            self.next += 1.0 / self.rate;
        }
    }
}

fn scalar(t: f64) -> Vec<f32> {
    vec![moving(t) as f32]
}

fn scaled(t: f64) -> Vec<f32> {
    vec![(3.0 * moving(t) + 0.2) as f32]
}

fn input(id: &str, feed: &Feed) -> MixInput {
    MixInput { id: id.into(), rig: rig(ChannelKind::Scalar), bus: feed.bus.clone() }
}

fn run_mix(feeds: &mut [&mut Feed], seconds: f64, mixer: &mut Mixer) -> Vec<PoseFrame> {
    let mut out = Vec::new();
    let mut now = 0.0;
    while now < seconds {
        for f in feeds.iter_mut() {
            f.run(now + 0.2);
        }
        mixer.pump(&mut out);
        now += 0.01;
    }
    out
}

fn two_sources(late_b: f64) -> (Mixer, Feed, Feed) {
    let (a, b) = (Feed::new(30.0, 0.0, scalar), Feed::new(100.0, late_b, scalar));
    let (mixer, _) = Mixer::new(vec![input("a", &a), input("b", &b)]);
    (mixer, a, b)
}

fn worst_gap(out: &[PoseFrame], skip: usize) -> f32 {
    out.iter().skip(skip).map(|f| (f.values[0] - f.values[1]).abs()).fold(0.0f32, f32::max)
}

#[test]
fn sources_of_different_rates_are_sampled_at_the_same_instants() {
    let (mut mixer, mut a, mut b) = two_sources(0.0);
    let out = run_mix(&mut [&mut a, &mut b], 6.0, &mut mixer);
    assert!(out.len() > 300, "{}", out.len());
    let step = out[1].time_s - out[0].time_s;
    assert!((step - 1.0 / 60.0).abs() < 1e-6, "a fixed grid, got {step}");
    assert!(out.windows(2).all(|w| (w[1].time_s - w[0].time_s - step).abs() < 1e-6), "no jitter, no doubled frames");
    let mut worst: f32 = 0.0;
    for f in out.iter().skip(30) {
        assert!(f.values.iter().all(|v| v.is_finite()));
        worst = worst.max((f.values[0] - f.values[1]).abs()).max((f.values[1] - moving(f.time_s) as f32).abs());
    }
    assert!(worst < 0.05, "worst disagreement {worst}");
}

#[test]
fn a_known_clock_offset_is_removed() {
    let (mut mixer, mut a, mut b) = two_sources(0.13);
    mixer.set_offset("b", 0.13);
    let out = run_mix(&mut [&mut a, &mut b], 6.0, &mut mixer);
    let aligned = worst_gap(&out, 60);
    assert!(aligned < 0.05, "aligned: {aligned}");
    let (mut unaligned, mut a, mut b) = two_sources(0.13);
    let out = run_mix(&mut [&mut a, &mut b], 6.0, &mut unaligned);
    let off = worst_gap(&out, 60);
    assert!(off > 0.2, "without the offset the sources disagree ({off})");
}

#[test]
fn the_offset_is_found_from_how_the_sources_moved() {
    for late in [0.137, -0.21, 0.4] {
        let mut a = Feed::new(30.0, 0.0, scalar);
        let mut b = Feed::new(50.0, late, scaled);
        let (mut mixer, _) = Mixer::new(vec![input("a", &a), input("b", &b)]);
        run_mix(&mut [&mut a, &mut b], 11.0, &mut mixer);
        let report = mixer.align("a", 8.0);
        let (_, estimate) = report.iter().find(|(id, _)| id == "b").expect("b");
        let estimate = estimate.expect("an estimate");
        assert!((estimate.offset_s - late).abs() < 0.012, "late {late}: found {estimate:?}");
        assert!(estimate.margin > 0.05, "{estimate:?}");
        let applied = mixer.offsets().into_iter().find(|(id, _)| id == "b").expect("b").1;
        assert!((applied - late).abs() < 0.012, "applied {applied}");
    }
}

#[test]
fn a_source_that_did_not_move_keeps_its_offset() {
    let mut a = Feed::new(30.0, 0.0, scalar);
    let mut b = Feed::new(30.0, 0.3, |_| vec![0.5]);
    let (mut mixer, _) = Mixer::new(vec![input("a", &a), input("b", &b)]);
    run_mix(&mut [&mut a, &mut b], 11.0, &mut mixer);
    mixer.align("a", 8.0);
    assert_eq!(mixer.offsets().into_iter().find(|(id, _)| id == "b").expect("b").1, 0.0);
}

#[test]
fn a_silent_source_is_a_gap() {
    let mut a = Feed::new(30.0, 0.0, scalar);
    let mut b = Feed::new(30.0, 0.0, scalar);
    let (mut mixer, _) = Mixer::new(vec![input("a", &a), input("b", &b)]);
    let mut out = Vec::new();
    let mut now = 0.0;
    while now < 4.0 {
        a.run(now + 0.2);
        if now < 1.5 {
            b.run(now + 0.2);
        }
        mixer.pump(&mut out);
        now += 0.01;
    }
    let late: Vec<&PoseFrame> = out.iter().filter(|f| f.time_s > 2.5).collect();
    assert!(!late.is_empty());
    assert!(late.iter().all(|f| f.values[1].is_nan() && f.values[0].is_finite()), "the silent source is a gap");
    let early: Vec<&PoseFrame> = out.iter().filter(|f| f.time_s > 0.5 && f.time_s < 1.2).collect();
    assert!(early.len() > 20 && early.iter().all(|f| f.values[1].is_finite()));
}

#[test]
fn rotations_are_interpolated_as_rotations() {
    let q = |angle: f64| vec![0.0, 0.0, (angle / 2.0).sin() as f32, (angle / 2.0).cos() as f32];
    let bus = PoseBus::new(64);
    let (mut mixer, _) = Mixer::new(vec![MixInput { id: "imu".into(), rig: rig(ChannelKind::Rotation4), bus: bus.clone() }]);
    mixer.configure(100.0, 0.05);
    bus.push(PoseFrame { sequence: 1, time_s: 0.0, values: q(0.0) });
    bus.push(PoseFrame { sequence: 2, time_s: 0.1, values: q(1.0) });
    bus.push(PoseFrame { sequence: 3, time_s: 0.2, values: q(2.0) });
    let mut out = Vec::new();
    mixer.pump(&mut out);
    assert!(!out.is_empty());
    for f in &out {
        let norm: f32 = f.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "still a unit quaternion: {norm}");
    }
}

#[test]
fn estimating_needs_enough_signal() {
    assert!(estimate_offset(&[0.0; 10], &[0.0; 10], 100.0, 0.5).is_none(), "too short");
    let flat = vec![1.0f32; 800];
    let e = estimate_offset(&flat, &flat, 100.0, 0.5);
    assert!(e.is_none_or(|e| e.margin < 0.05 || e.correlation < 0.3), "nothing to line up: {e:?}");
}
