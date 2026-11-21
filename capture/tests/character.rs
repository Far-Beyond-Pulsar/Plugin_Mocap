//! The skinned character: loading a humanoid glTF, mapping its bones, posing it from a body.

mod common;

use common::human;
use mocap_capture::{BodyPoints, Bone, Character, CharacterError, Side};
use nalgebra::Vector3;

const GLB: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../editor/assets/character.glb"));

fn character() -> Character {
    Character::from_glb(GLB).expect("the bundled character loads")
}

fn coco(t: f64) -> [[f32; 3]; 17] {
    let j = human(t);
    std::array::from_fn(|i| [j[i][0] as f32, j[i][1] as f32, j[i][2] as f32])
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (Vector3::from_row_slice(&a) - Vector3::from_row_slice(&b)).norm()
}

#[test]
fn the_character_loads_with_its_skeleton_and_mesh() {
    let c = character();
    assert!(c.vertices.len() > 1000 && c.indices.len() % 3 == 0, "{} vertices", c.vertices.len());
    assert!(c.joint_count() >= 20, "{} joints", c.joint_count());
    assert!(c.vertices.iter().all(|v| v.joints.iter().all(|j| usize::from(*j) < c.joint_count())));
    assert!(c.vertices.iter().all(|v| (v.weights.iter().sum::<f32>() - 1.0).abs() < 0.05), "weights are normalised");
    let h = c.rest_height();
    assert!((1.6..1.95).contains(&h), "{h} m tall");
}

#[test]
fn bones_are_found_by_name_in_any_common_scheme_and_missing_ones_are_reported() {
    // The bundled skeleton uses VRM-style names; the loader maps them all.
    let c = character();
    let body = BodyPoints::from_coco17(&coco(0.0));
    for bone in [Bone::Hips, Bone::Head, Bone::UpperArm(Side::Left), Bone::Foot(Side::Right), Bone::Toes(Side::Left)] {
        assert!(c.bone_position(&body, bone).is_some(), "{bone:?}");
    }
    assert!(matches!(Character::from_glb(b"not a glb"), Err(CharacterError::Invalid(_))));
}

#[test]
fn the_rest_matrices_leave_the_character_as_modelled() {
    let c = character();
    let rest = c.rest_matrices();
    // A vertex bound to one joint stays where it was.
    let v = c.vertices.iter().find(|v| v.weights[0] > 0.999).expect("a rigid vertex");
    let m = rest[usize::from(v.joints[0])];
    let p = [
        m[0][0] * v.position[0] + m[1][0] * v.position[1] + m[2][0] * v.position[2] + m[3][0],
        m[0][1] * v.position[0] + m[1][1] * v.position[1] + m[2][1] * v.position[2] + m[3][1],
        m[0][2] * v.position[0] + m[1][2] * v.position[1] + m[2][2] * v.position[2] + m[3][2],
    ];
    assert!(dist(p, v.position) < 1e-3, "{p:?} vs {:?}", v.position);
}

#[test]
fn the_pose_puts_each_bone_where_the_subjects_joint_is() {
    let c = character();
    for t in [0.0, 1.3, 2.9, 4.4] {
        let points = coco(t);
        let body = BodyPoints::from_coco17(&points);
        // The hips are the pelvis, exactly; the limbs follow the subject's within the difference in proportions.
        let pelvis = body.pelvis.expect("pelvis");
        assert!(dist(c.bone_position(&body, Bone::Hips).expect("hips"), pelvis) < 1e-3);
        for (bone, joint, limit) in [
            (Bone::LowerArm(Side::Left), 7, 0.06),
            (Bone::Hand(Side::Left), 9, 0.10),
            (Bone::LowerArm(Side::Right), 8, 0.06),
            (Bone::Hand(Side::Right), 10, 0.10),
            (Bone::LowerLeg(Side::Left), 13, 0.06),
            (Bone::Foot(Side::Left), 15, 0.10),
            (Bone::LowerLeg(Side::Right), 14, 0.06),
            (Bone::Foot(Side::Right), 16, 0.10),
        ] {
            let found = c.bone_position(&body, bone).expect("bone");
            let error = dist(found, points[joint]);
            assert!(error < limit, "t {t}: {bone:?} is {error:.3} m from joint {joint}");
        }
    }
}

#[test]
fn the_character_is_scaled_to_the_subject_and_faces_the_way_they_do() {
    let c = character();
    let points = coco(0.0);
    let body = BodyPoints::from_coco17(&points);
    let mut skinned: Vec<[f32; 3]> = Vec::new();
    let matrices = c.pose(&body).expect("a pose");
    for v in &c.vertices {
        let mut p = [0.0f32; 3];
        for k in 0..4 {
            let (m, w) = (matrices[usize::from(v.joints[k])], v.weights[k]);
            for (r, out) in p.iter_mut().enumerate() {
                *out += w * (m[0][r] * v.position[0] + m[1][r] * v.position[1] + m[2][r] * v.position[2] + m[3][r]);
            }
        }
        skinned.push(p);
    }
    assert!(skinned.iter().all(|p| p.iter().all(|x| x.is_finite())));
    let top = skinned.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let bottom = skinned.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let head_y = points[0][1];
    assert!(top > head_y && top - head_y < 0.3, "the head top is {} above the nose", top - head_y);
    assert!(bottom.abs() < 0.2, "the feet are near the floor: {bottom}");
    // The nose side of the head is forward (+Z at yaw 0): the character's front is where the subject's nose is.
    let nose = points[0];
    let nearest = skinned.iter().map(|p| dist(*p, nose)).fold(f32::INFINITY, f32::min);
    assert!(nearest < 0.08, "the mesh has a surface near the nose: {nearest}");
}

#[test]
fn a_body_without_a_pelvis_or_neck_cannot_be_posed() {
    let c = character();
    assert!(c.pose(&BodyPoints::default()).is_none());
}
