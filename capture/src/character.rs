//! A skinned humanoid character (a glTF/GLB with one skin) and the posing of it from the body the
//! cameras and sensors found.
//!
//! Any humanoid glTF works: its bones are recognised by name (VRM, Mixamo, Unreal-style, Blender and
//! Quaternius naming), its rest pose is read from the file, and the pose is computed by aiming each
//! bone along the matching bone of the subject (swing only for limbs, full frames for the torso and head).
//! The result is one skinning matrix per joint, ready for a vertex shader.

use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion, Vector3};
use std::collections::HashMap;

/// One vertex of the skin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

/// Why a character could not be loaded.
#[derive(Debug, Clone, PartialEq)]
pub enum CharacterError {
    /// The file is not a glTF or has no usable skin.
    Invalid(String),
    /// Bones of a humanoid the skeleton lacks.
    MissingBones(Vec<&'static str>),
}

impl std::fmt::Display for CharacterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "not a usable skinned character: {why}"),
            Self::MissingBones(names) => write!(f, "the skeleton lacks humanoid bones: {}", names.join(", ")),
        }
    }
}

impl std::error::Error for CharacterError {}

/// The bones of a humanoid that the posing drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bone {
    Hips,
    Neck,
    Head,
    Shoulder(Side),
    UpperArm(Side),
    LowerArm(Side),
    Hand(Side),
    UpperLeg(Side),
    LowerLeg(Side),
    Foot(Side),
    Toes(Side),
}

/// Left or right of the character.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Left,
    Right,
}

/// A subject's body, as the posing needs it (metres, +Y up, facing +Z at yaw 0, left at +X). Any may be missing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyPoints {
    pub pelvis: Option<[f32; 3]>,
    /// The base of the neck.
    pub neck: Option<[f32; 3]>,
    /// The middle of the head.
    pub head: Option<[f32; 3]>,
    pub nose: Option<[f32; 3]>,
    pub ears: [Option<[f32; 3]>; 2],
    pub shoulders: [Option<[f32; 3]>; 2],
    pub elbows: [Option<[f32; 3]>; 2],
    pub wrists: [Option<[f32; 3]>; 2],
    pub hips: [Option<[f32; 3]>; 2],
    pub knees: [Option<[f32; 3]>; 2],
    pub ankles: [Option<[f32; 3]>; 2],
}

fn good(p: &[f32; 3]) -> Option<[f32; 3]> {
    (p[0].is_finite() && p[1].is_finite() && p[2].is_finite()).then_some(*p)
}

fn middle(a: Option<[f32; 3]>, b: Option<[f32; 3]>) -> Option<[f32; 3]> {
    match (a, b) {
        (Some(a), Some(b)) => Some([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]),
        (Some(a), None) | (None, Some(a)) => Some(a),
        _ => None,
    }
}

impl BodyPoints {
    /// From the 17 COCO keypoints (indices as in `COCO17`; `NaN` where unknown), left at the odd indices.
    pub fn from_coco17(p: &[[f32; 3]; 17]) -> Self {
        let at = |i: usize| good(&p[i]);
        let pair = |l: usize, r: usize| [at(l), at(r)];
        let (shoulders, hips) = (pair(5, 6), pair(11, 12));
        let ears = pair(3, 4);
        Self {
            pelvis: middle(hips[0], hips[1]),
            neck: middle(shoulders[0], shoulders[1]),
            head: middle(ears[0], ears[1]).or(at(0)),
            nose: at(0),
            ears,
            shoulders,
            elbows: pair(7, 8),
            wrists: pair(9, 10),
            hips,
            knees: pair(13, 14),
            ankles: pair(15, 16),
        }
    }

    /// From named joints (VRM humanoid names, as `Hips`, `LeftUpperArm`, ...; any prefix before a `/` is ignored).
    pub fn from_named(names: &[String], joints: &[[f32; 3]]) -> Self {
        let find = |wanted: &str| -> Option<[f32; 3]> {
            names.iter().zip(joints).find(|(n, _)| n.rsplit('/').next().unwrap_or(n) == wanted).and_then(|(_, p)| good(p))
        };
        let pair = |l: &str, r: &str| [find(&format!("Left{l}")), find(&format!("Right{r}"))];
        let head_base = find("Head");
        Self {
            pelvis: find("Hips"),
            neck: find("Neck").or_else(|| middle(find("LeftShoulder"), find("RightShoulder"))),
            head: head_base,
            nose: None,
            ears: [None, None],
            shoulders: pair("UpperArm", "UpperArm"),
            elbows: pair("LowerArm", "LowerArm"),
            wrists: pair("Hand", "Hand"),
            hips: pair("UpperLeg", "UpperLeg"),
            knees: pair("LowerLeg", "LowerLeg"),
            ankles: pair("Foot", "Foot"),
        }
    }
}

/// A skinned humanoid.
pub struct Character {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// The material's base colour (linear RGBA).
    pub base_color: [f32; 4],
    names: Vec<String>,
    parents: Vec<Option<usize>>,
    /// Joints in an order where a parent precedes its children.
    order: Vec<usize>,
    /// Each joint's rest world position and rotation (in the file's space).
    rest_position: Vec<Vector3<f64>>,
    rest_rotation: Vec<UnitQuaternion<f64>>,
    inverse_bind: Vec<Matrix4<f64>>,
    map: HashMap<Bone, usize>,
    spine: Vec<usize>,
}

/// A bone name as plain lowercase letters and digits, with the rig prefixes dropped.
fn plain(name: &str) -> String {
    let lower: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
    lower.trim_start_matches("mixamorig").trim_start_matches("armature").to_owned()
}

/// Which side a bone name says (`Left`, `_l`, `.R`, ...).
fn side_of(name: &str) -> Option<Side> {
    let lower = name.to_lowercase();
    if lower.contains("left") {
        return Some(Side::Left);
    }
    if lower.contains("right") {
        return Some(Side::Right);
    }
    let parts: Vec<&str> = lower.split(|c: char| !c.is_ascii_alphanumeric()).filter(|p| !p.is_empty()).collect();
    if parts.contains(&"l") || lower.ends_with(".l") || lower.ends_with("_l") {
        return Some(Side::Left);
    }
    if parts.contains(&"r") || lower.ends_with(".r") || lower.ends_with("_r") {
        return Some(Side::Right);
    }
    // Names like "LeftHand" with a leading L/R and a capital next: "LUpperArm".
    let mut chars = name.chars();
    match (chars.next(), chars.next()) {
        (Some('L'), Some(c)) if c.is_ascii_uppercase() => Some(Side::Left),
        (Some('R'), Some(c)) if c.is_ascii_uppercase() => Some(Side::Right),
        _ => None,
    }
}

fn is_finger_or_twist(plain_name: &str) -> bool {
    ["twist", "roll", "index", "middle", "ring", "pinky", "thumb", "little", "finger", "end", "top", "eye", "jaw", "hair"].iter().any(|w| plain_name.contains(w))
}

/// Picks the joint whose plain name has the keyword and the given side, preferring the shortest name.
fn pick(names: &[String], side: Option<Side>, any_of: &[&str], none_of: &[&str]) -> Option<usize> {
    names
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            let p = plain(n);
            side_of(n) == side && any_of.iter().any(|k| p.contains(k)) && !none_of.iter().any(|k| p.contains(k)) && !is_finger_or_twist(&p)
        })
        .min_by_key(|(_, n)| plain(n).len())
        .map(|(i, _)| i)
}

fn mat(m: [[f32; 4]; 4]) -> Matrix4<f64> {
    Matrix4::from_fn(|r, c| f64::from(m[c][r]))
}

fn rotation_of(m: &Matrix4<f64>) -> UnitQuaternion<f64> {
    let r: Matrix3<f64> = m.fixed_view::<3, 3>(0, 0).into();
    // Remove any scale before taking the rotation.
    let cols: Vec<Vector3<f64>> = (0..3).map(|c| r.column(c).normalize()).collect();
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[cols[0], cols[1], cols[2]])))
}

impl Character {
    /// Loads a GLB (or self-contained glTF) with at least one skin.
    pub fn from_glb(bytes: &[u8]) -> Result<Self, CharacterError> {
        let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| CharacterError::Invalid(e.to_string()))?;
        let doc = &gltf.document;
        let blob = gltf.blob.as_deref();
        let source = |b: gltf::Buffer| match b.source() {
            gltf::buffer::Source::Bin => blob,
            gltf::buffer::Source::Uri(_) => None,
        };
        let skin = doc.skins().next().ok_or_else(|| CharacterError::Invalid("no skin".into()))?;
        let joints: Vec<gltf::Node> = skin.joints().collect();
        if joints.is_empty() {
            return Err(CharacterError::Invalid("the skin has no joints".into()));
        }
        // World matrices of every node.
        let mut world: HashMap<usize, Matrix4<f64>> = HashMap::new();
        fn walk(node: gltf::Node, parent: Matrix4<f64>, world: &mut HashMap<usize, Matrix4<f64>>, parent_of: &mut HashMap<usize, usize>) {
            let m = parent * mat(node.transform().matrix());
            world.insert(node.index(), m);
            for child in node.children() {
                parent_of.insert(child.index(), node.index());
                walk(child, m, world, parent_of);
            }
        }
        let mut parent_of = HashMap::new();
        for scene in doc.scenes() {
            for node in scene.nodes() {
                walk(node, Matrix4::identity(), &mut world, &mut parent_of);
            }
        }
        let joint_index: HashMap<usize, usize> = joints.iter().enumerate().map(|(i, n)| (n.index(), i)).collect();
        let names: Vec<String> = joints.iter().map(|n| n.name().unwrap_or("").to_owned()).collect();
        let parents: Vec<Option<usize>> = joints
            .iter()
            .map(|n| {
                let mut up = parent_of.get(&n.index()).copied();
                while let Some(p) = up {
                    if let Some(&j) = joint_index.get(&p) {
                        return Some(j);
                    }
                    up = parent_of.get(&p).copied();
                }
                None
            })
            .collect();
        let rest: Vec<Matrix4<f64>> = joints.iter().map(|n| world.get(&n.index()).copied().unwrap_or_else(Matrix4::identity)).collect();
        let rest_position: Vec<Vector3<f64>> = rest.iter().map(|m| Vector3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])).collect();
        let rest_rotation: Vec<UnitQuaternion<f64>> = rest.iter().map(rotation_of).collect();
        let inverse_bind: Vec<Matrix4<f64>> = match skin.reader(source).read_inverse_bind_matrices() {
            Some(it) => it.map(mat).collect(),
            None => rest.iter().map(|m| m.try_inverse().unwrap_or_else(Matrix4::identity)).collect(),
        };
        // Parents before children.
        let mut order = Vec::with_capacity(joints.len());
        let mut done = vec![false; joints.len()];
        while order.len() < joints.len() {
            let before = order.len();
            for j in 0..joints.len() {
                if !done[j] && parents[j].is_none_or(|p| done[p]) {
                    done[j] = true;
                    order.push(j);
                }
            }
            if order.len() == before {
                return Err(CharacterError::Invalid("the skeleton has a cycle".into()));
            }
        }
        // The mesh: every primitive of every mesh that uses this skin.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut base_color = [0.85, 0.82, 0.78, 1.0];
        for node in doc.nodes().filter(|n| n.skin().is_some_and(|s| s.index() == skin.index())) {
            let Some(mesh) = node.mesh() else { continue };
            for primitive in mesh.primitives() {
                let reader = primitive.reader(source);
                let Some(positions) = reader.read_positions() else { continue };
                let positions: Vec<[f32; 3]> = positions.collect();
                let normals: Vec<[f32; 3]> = reader.read_normals().map_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()], |n| n.collect());
                let joint_ids: Vec<[u16; 4]> = reader.read_joints(0).map_or_else(|| vec![[0; 4]; positions.len()], |j| j.into_u16().collect());
                let weights: Vec<[f32; 4]> = reader.read_weights(0).map_or_else(|| vec![[1.0, 0.0, 0.0, 0.0]; positions.len()], |w| w.into_f32().collect());
                let base = vertices.len() as u32;
                for i in 0..positions.len() {
                    vertices.push(Vertex { position: positions[i], normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]), joints: joint_ids.get(i).copied().unwrap_or([0; 4]), weights: weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]) });
                }
                match reader.read_indices() {
                    Some(idx) => indices.extend(idx.into_u32().map(|i| i + base)),
                    None => indices.extend((0..positions.len() as u32).map(|i| i + base)),
                }
                base_color = primitive.material().pbr_metallic_roughness().base_color_factor();
            }
        }
        if vertices.is_empty() {
            return Err(CharacterError::Invalid("the skin has no mesh".into()));
        }
        // The humanoid.
        let mut map = HashMap::new();
        let mut missing: Vec<&'static str> = Vec::new();
        let mut need = |bone: Bone, label: &'static str, found: Option<usize>, required: bool| match found {
            Some(i) => {
                map.insert(bone, i);
            }
            None if required => missing.push(label),
            None => {}
        };
        let hips = names.iter().position(|n| {
            let p = plain(n);
            p.contains("hips") || p.contains("pelvis")
        });
        need(Bone::Hips, "hips", hips, true);
        need(Bone::Neck, "neck", pick(&names, None, &["neck"], &[]), true);
        need(Bone::Head, "head", pick(&names, None, &["head"], &[]), true);
        for (side, label) in [(Side::Left, "left"), (Side::Right, "right")] {
            let s = Some(side);
            let leg_words: [&str; 3] = ["upperleg", "thigh", "upleg"];
            need(Bone::Shoulder(side), if label == "left" { "left shoulder" } else { "right shoulder" }, pick(&names, s, &["shoulder", "clavicle"], &[]), false);
            need(Bone::UpperArm(side), if label == "left" { "left upper arm" } else { "right upper arm" }, pick(&names, s, &["upperarm", "uparm", "arm"], &["forearm", "lowerarm", "lowarm"]), true);
            need(Bone::LowerArm(side), if label == "left" { "left lower arm" } else { "right lower arm" }, pick(&names, s, &["lowerarm", "forearm", "lowarm"], &[]), true);
            need(Bone::Hand(side), if label == "left" { "left hand" } else { "right hand" }, pick(&names, s, &["hand"], &[]), true);
            need(Bone::UpperLeg(side), if label == "left" { "left upper leg" } else { "right upper leg" }, pick(&names, s, &leg_words, &[]), true);
            need(Bone::LowerLeg(side), if label == "left" { "left lower leg" } else { "right lower leg" }, pick(&names, s, &["lowerleg", "calf", "shin"], &[]), true);
            need(Bone::Foot(side), if label == "left" { "left foot" } else { "right foot" }, pick(&names, s, &["foot"], &["ball"]), true);
            need(Bone::Toes(side), if label == "left" { "left toes" } else { "right toes" }, pick(&names, s, &["toe", "ball"], &[]), false);
        }
        if !missing.is_empty() {
            return Err(CharacterError::MissingBones(missing));
        }
        // The spine: every spine or chest bone between the hips and the neck, lowest first.
        let (hips_i, neck_i) = (map[&Bone::Hips], map[&Bone::Neck]);
        let mut spine: Vec<usize> = (0..names.len())
            .filter(|&i| {
                let p = plain(&names[i]);
                side_of(&names[i]).is_none() && (p.contains("spine") || p.contains("chest")) && i != hips_i && i != neck_i && !is_finger_or_twist(&p)
            })
            .collect();
        spine.sort_by(|&a, &b| rest_position[a][1].total_cmp(&rest_position[b][1]));
        Ok(Self { vertices, indices, base_color, names, parents, order, rest_position, rest_rotation, inverse_bind, map, spine })
    }

    /// Joints in the skin.
    pub fn joint_count(&self) -> usize {
        self.names.len()
    }

    /// The rest pose's skinning matrices (identity-equivalent): the character as modelled.
    pub fn rest_matrices(&self) -> Vec<[[f32; 4]; 4]> {
        (0..self.names.len())
            .map(|j| {
                let w = Matrix4::new_translation(&self.rest_position[j]) * self.rest_rotation[j].to_homogeneous();
                to_gpu(&(w * self.inverse_bind[j]))
            })
            .collect()
    }

    /// The height of the character at rest (metres in the file's space).
    pub fn rest_height(&self) -> f32 {
        let (lo, hi) = self.vertices.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v.position[1]), hi.max(v.position[1])));
        hi - lo
    }

    /// The skinning matrices that pose the character as `body`; `None` when the body does not have the
    /// pelvis and the base of the neck, which fix its place and size.
    pub fn pose(&self, body: &BodyPoints) -> Option<Vec<[[f32; 4]; 4]>> {
        let world = self.pose_world(body)?;
        Some((0..self.names.len()).map(|j| to_gpu(&(world[j] * self.inverse_bind[j]))).collect())
    }

    /// The posed world matrices (rotation, position and the body's scale) of every joint.
    pub fn pose_world(&self, body: &BodyPoints) -> Option<Vec<Matrix4<f64>>> {
        let v = |p: [f32; 3]| Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        let (pelvis, neck) = (v(body.pelvis?), v(body.neck?));
        let hips = self.map[&Bone::Hips];
        let mesh_neck = self.rest_position[self.map[&Bone::Neck]];
        let mesh_torso = (mesh_neck - self.rest_position[hips]).norm().max(1e-6);
        let scale = ((neck - pelvis).norm() / mesh_torso).clamp(0.25, 4.0);
        let up = (neck - pelvis).try_normalize(1e-9)?;
        let pt = |p: Option<[f32; 3]>| p.map(v);
        let side_idx = |s: Side| match s {
            Side::Left => 0,
            Side::Right => 1,
        };
        // Left across the body: from the hips or shoulders; else from the pelvis frame.
        let across = |pair: &[Option<[f32; 3]>; 2]| match (pair[0], pair[1]) {
            (Some(l), Some(r)) => (v(l) - v(r)).try_normalize(1e-9),
            _ => None,
        };
        let lateral_hips = across(&body.hips).or_else(|| across(&body.shoulders)).unwrap_or(Vector3::new(1.0, 0.0, 0.0));
        let lateral_chest = across(&body.shoulders).or(Some(lateral_hips))?;
        let frame = |lateral: Vector3<f64>, up: Vector3<f64>| -> UnitQuaternion<f64> {
            let y = up.normalize();
            let x = (lateral - y * lateral.dot(&y)).try_normalize(1e-9).unwrap_or_else(|| y.cross(&Vector3::z()).normalize());
            let z = x.cross(&y);
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[x, y, z])))
        };
        let d_hips = frame(lateral_hips, up);
        let d_chest = frame(lateral_chest, up);
        let head_target = pt(body.head);
        let d_head = match head_target {
            Some(h) => {
                let head_up = (h - neck).try_normalize(1e-9).unwrap_or(up);
                let ears = across(&body.ears).unwrap_or(lateral_chest);
                let forward_hint = match (pt(body.nose), pt(body.ears[0]), pt(body.ears[1])) {
                    (Some(n), Some(l), Some(r)) => (n - (l + r) * 0.5).try_normalize(1e-9),
                    _ => None,
                };
                match forward_hint {
                    Some(f) => {
                        let f = (f - head_up * f.dot(&head_up)).try_normalize(1e-9).unwrap_or(Vector3::z());
                        let x = head_up.cross(&f).normalize();
                        let z = x.cross(&head_up);
                        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[x, head_up, z])))
                    }
                    None => frame(ears, head_up),
                }
            }
            None => d_chest,
        };
        // Target world rotations of the bones that have one.
        let mut target: HashMap<usize, UnitQuaternion<f64>> = HashMap::new();
        target.insert(hips, d_hips * self.rest_rotation[hips]);
        let n = self.spine.len();
        for (k, &j) in self.spine.iter().enumerate() {
            let t = (k + 1) as f64 / (n + 1) as f64;
            target.insert(j, d_hips.slerp(&d_chest, t) * self.rest_rotation[j]);
        }
        target.insert(self.map[&Bone::Neck], d_chest.slerp(&d_head, 0.5) * self.rest_rotation[self.map[&Bone::Neck]]);
        target.insert(self.map[&Bone::Head], d_head * self.rest_rotation[self.map[&Bone::Head]]);
        // A limb bone: rotate it to point from `from` to `to` (swing only, from its rest direction to `child`).
        let mut aim = |bone: Bone, child: Bone, from: Option<Vector3<f64>>, to: Option<Vector3<f64>>| {
            let (Some(&j), Some(&c)) = (self.map.get(&bone), self.map.get(&child)) else { return };
            let (Some(a), Some(b)) = (from, to) else { return };
            let (Some(d1), Some(d0)) = ((b - a).try_normalize(1e-9), (self.rest_position[c] - self.rest_position[j]).try_normalize(1e-9)) else { return };
            let swing = UnitQuaternion::rotation_between(&d0, &d1).unwrap_or_else(UnitQuaternion::identity);
            target.insert(j, swing * self.rest_rotation[j]);
        };
        for side in [Side::Left, Side::Right] {
            let i = side_idx(side);
            aim(Bone::UpperArm(side), Bone::LowerArm(side), pt(body.shoulders[i]), pt(body.elbows[i]));
            aim(Bone::LowerArm(side), Bone::Hand(side), pt(body.elbows[i]), pt(body.wrists[i]));
            aim(Bone::UpperLeg(side), Bone::LowerLeg(side), pt(body.hips[i]), pt(body.knees[i]));
            aim(Bone::LowerLeg(side), Bone::Foot(side), pt(body.knees[i]), pt(body.ankles[i]));
        }
        // World rotation of every joint: the target, or the parent's with this joint's rest offset from it.
        let count = self.names.len();
        let mut rotation: Vec<UnitQuaternion<f64>> = self.rest_rotation.clone();
        let mut position: Vec<Vector3<f64>> = self.rest_position.clone();
        let shoulder_frame = d_chest;
        for &j in &self.order {
            let parent = self.parents[j];
            rotation[j] = if let Some(t) = target.get(&j) {
                *t
            } else if self.map.get(&Bone::Shoulder(Side::Left)) == Some(&j) || self.map.get(&Bone::Shoulder(Side::Right)) == Some(&j) {
                shoulder_frame * self.rest_rotation[j]
            } else if let Some(p) = parent {
                // Keep this joint's rest rotation relative to its parent.
                rotation[p] * (self.rest_rotation[p].inverse() * self.rest_rotation[j])
            } else {
                self.rest_rotation[j]
            };
            position[j] = match parent {
                _ if j == hips => pelvis,
                Some(p) => position[p] + rotation[p] * (self.rest_rotation[p].inverse() * (self.rest_position[j] - self.rest_position[p])) * scale,
                None => self.rest_position[j] * scale,
            };
        }
        Some((0..count).map(|j| Matrix4::new_translation(&position[j]) * rotation[j].to_homogeneous() * Matrix4::new_scaling(scale)).collect())
    }

    /// The world position of a bone in the pose `body` makes (for tests and for debugging).
    pub fn bone_position(&self, body: &BodyPoints, bone: Bone) -> Option<[f32; 3]> {
        let world = self.pose_world(body)?;
        let j = *self.map.get(&bone)?;
        Some([world[j][(0, 3)] as f32, world[j][(1, 3)] as f32, world[j][(2, 3)] as f32])
    }
}

/// A matrix as column-major `f32` rows for a GPU buffer.
fn to_gpu(m: &Matrix4<f64>) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = m[(r, c)] as f32;
        }
    }
    out
}
