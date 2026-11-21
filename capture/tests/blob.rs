use mocap_capture::{find_light, PixelFormat, VideoFrame};
use std::sync::Arc;

fn frame(w: u32, h: u32, lights: &[(f64, f64, f64, f64)], background: u8) -> VideoFrame {
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut v = f64::from(background);
            for &(cx, cy, sigma, peak) in lights {
                let d2 = (f64::from(x) + 0.5 - cx).powi(2) + (f64::from(y) + 0.5 - cy).powi(2);
                v += peak * (-d2 / (2.0 * sigma * sigma)).exp();
            }
            let v = v.clamp(0.0, 255.0) as u8;
            let i = ((y * w + x) * 4) as usize;
            data[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    VideoFrame { width: w, height: h, format: PixelFormat::Rgba8, data: Arc::from(data), sequence: 1, time_s: 0.0 }
}

#[test]
fn a_light_is_found_to_a_fraction_of_a_pixel() {
    for (x, y) in [(300.3, 200.7), (17.5, 400.2), (900.9, 100.1), (640.0, 360.0)] {
        let spot = find_light(&frame(960, 540, &[(x, y, 2.2, 255.0)], 12)).expect("a light");
        assert!((spot.x - x).abs() < 0.3 && (spot.y - y).abs() < 0.3, "({x}, {y}) found at ({:.2}, {:.2})", spot.x, spot.y);
    }
}

#[test]
fn no_light_a_faint_one_or_a_washed_out_frame_is_nothing() {
    assert!(find_light(&frame(640, 360, &[], 30)).is_none());
    assert!(find_light(&frame(640, 360, &[(300.0, 200.0, 2.0, 120.0)], 20)).is_none(), "too faint");
    assert!(find_light(&frame(640, 360, &[], 230)).is_none(), "everything is lit");
    assert!(find_light(&frame(640, 360, &[(300.0, 200.0, 60.0, 255.0)], 0)).is_none(), "a lit wall is not a point");
}

#[test]
fn two_lights_of_the_same_strength_are_ambiguous_but_a_much_brighter_one_wins() {
    assert!(find_light(&frame(640, 360, &[(100.0, 100.0, 2.0, 255.0), (500.0, 250.0, 2.0, 255.0)], 10)).is_none());
    let spot = find_light(&frame(640, 360, &[(100.0, 100.0, 5.0, 255.0), (500.0, 250.0, 1.2, 255.0)], 10)).expect("the larger, brighter one");
    assert!((spot.x - 100.0).abs() < 0.5);
}
