"""Builds the stand-in character (editor/assets/character.glb) with Blender: a smooth mannequin skinned to a
humanoid skeleton (VRM-style bone names, A-pose, facing glTF +Z). It stands in until a professional asset
(for example a Quaternius Universal Base Character, CC0) is dropped in its place: the loader maps bones by
name, so any humanoid glTF works.

    blender -b -P make_character.py -- <out.glb>
"""

import math
import sys

import bmesh
import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "character.glb"

bpy.ops.wm.read_factory_settings(use_empty=True)

# Joint positions in Blender space (Z up, the character faces -Y, which becomes glTF +Z). Left is +X.
A = math.radians(33)


def arm(side, shoulder_x=0.19):
    s = side
    sh = Vector((s * shoulder_x, 0.0, 1.45))
    el = sh + Vector((s * 0.29 * math.sin(A), 0.0, -0.29 * math.cos(A)))
    wr = el + Vector((s * 0.26 * math.sin(A * 0.8), 0.0, -0.26 * math.cos(A * 0.8)))
    hand = wr + Vector((s * 0.085 * math.sin(A * 0.8), 0.0, -0.085 * math.cos(A * 0.8)))
    return sh, el, wr, hand


def leg(side):
    s = side
    hip = Vector((s * 0.10, 0.0, 0.93))
    kn = Vector((s * 0.10, -0.01, 0.52))
    an = Vector((s * 0.10, 0.0, 0.09))
    toe = Vector((s * 0.10, -0.17, 0.03))
    return hip, kn, an, toe


J = {
    "Hips": Vector((0, 0, 0.95)),
    "Spine": Vector((0, 0, 1.10)),
    "Chest": Vector((0, 0, 1.28)),
    "Neck": Vector((0, 0, 1.49)),
    "Head": Vector((0, 0, 1.60)),
    "HeadTop": Vector((0, 0, 1.80)),
}
for name, side in (("Left", 1.0), ("Right", -1.0)):
    sh, el, wr, hand = arm(side)
    J[name + "Shoulder"] = Vector((side * 0.045, 0.0, 1.46))
    J[name + "UpperArm"], J[name + "LowerArm"], J[name + "Hand"], J[name + "HandEnd"] = sh, el, wr, hand
    hip, kn, an, toe = leg(side)
    J[name + "UpperLeg"], J[name + "LowerLeg"], J[name + "Foot"], J[name + "Toes"] = hip, kn, an, toe
    J[name + "ToesEnd"] = toe + Vector((0, -0.07, 0))

BONES = [  # (name, parent, tail joint)
    ("Hips", None, "Spine"),
    ("Spine", "Hips", "Chest"),
    ("Chest", "Spine", "Neck"),
    ("Neck", "Chest", "Head"),
    ("Head", "Neck", "HeadTop"),
]
for n in ("Left", "Right"):
    BONES += [
        (n + "Shoulder", "Chest", n + "UpperArm"),
        (n + "UpperArm", n + "Shoulder", n + "LowerArm"),
        (n + "LowerArm", n + "UpperArm", n + "Hand"),
        (n + "Hand", n + "LowerArm", n + "HandEnd"),
        (n + "UpperLeg", "Hips", n + "LowerLeg"),
        (n + "LowerLeg", n + "UpperLeg", n + "Foot"),
        (n + "Foot", n + "LowerLeg", n + "Toes"),
        (n + "Toes", n + "Foot", n + "ToesEnd"),
    ]

# The armature.
arm_data = bpy.data.armatures.new("Armature")
armature = bpy.data.objects.new("Armature", arm_data)
bpy.context.collection.objects.link(armature)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="EDIT")
for name, parent, tail in BONES:
    b = arm_data.edit_bones.new(name)
    b.head = J[name]
    b.tail = J[tail]
    if parent:
        b.parent = arm_data.edit_bones[parent]
        if (b.head - arm_data.edit_bones[parent].tail).length < 1e-4:
            b.use_connect = True
bpy.ops.object.mode_set(mode="OBJECT")

# The skin: smooth tapered limbs and rounded body parts, joined into one mesh.
skin = bpy.data.materials.new("Skin")
skin.use_nodes = True
bsdf = skin.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.86, 0.80, 0.74, 1.0)
bsdf.inputs["Roughness"].default_value = 0.55

parts = []


def capsule(a, b, ra, rb, name):
    bm = bmesh.new()
    d = b - a
    steps = 14
    rings = 10
    # A lofted tube with rounded ends.
    verts = []
    for r in range(rings + 1):
        t = r / rings
        centre = a + d * t
        radius = ra + (rb - ra) * t
        ring = []
        for k in range(steps):
            ang = math.tau * k / steps
            ring.append((centre, radius, ang))
        verts.append(ring)
    axis = d.normalized()
    ref = Vector((1, 0, 0)) if abs(axis.x) < 0.9 else Vector((0, 1, 0))
    u = axis.cross(ref).normalized()
    v = axis.cross(u).normalized()
    grid = []
    for ring in verts:
        grid.append([bm.verts.new(c + (u * math.cos(ang) + v * math.sin(ang)) * radius) for c, radius, ang in ring])
    for r in range(rings):
        for k in range(steps):
            bm.faces.new((grid[r][k], grid[r][(k + 1) % steps], grid[r + 1][(k + 1) % steps], grid[r + 1][k]))
    for end, direction in ((grid[0], -1), (grid[-1], 1)):
        centre = bm.verts.new((a if direction < 0 else b) + axis * direction * (ra if direction < 0 else rb) * 0.7)
        for k in range(steps):
            f = (end[k], end[(k + 1) % steps], centre) if direction > 0 else (end[(k + 1) % steps], end[k], centre)
            bm.faces.new(f)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    parts.append(ob)
    return ob


def blob(centre, scale, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=centre, segments=24, ring_count=14)
    ob = bpy.context.active_object
    ob.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    ob.name = name
    parts.append(ob)
    return ob


blob(J["Hips"] + Vector((0, 0, 0.0)), (0.17, 0.11, 0.13), "pelvis")
capsule(J["Spine"], J["Chest"] + Vector((0, 0, 0.05)), 0.145, 0.17, "abdomen")
blob(J["Chest"] + Vector((0, 0, 0.07)), (0.19, 0.12, 0.17), "chest")
capsule(J["Neck"] - Vector((0, 0, 0.03)), J["Head"], 0.05, 0.045, "neck")
blob(J["Head"] + Vector((0, 0, 0.08)), (0.095, 0.105, 0.115), "head")
blob(J["Head"] + Vector((0, -0.095, 0.075)), (0.018, 0.022, 0.022), "nose")
for name, side in (("Left", 1.0), ("Right", -1.0)):
    capsule(J[name + "Shoulder"], J[name + "UpperArm"] + Vector((side * 0.02, 0, 0)), 0.045, 0.05, name + "shoulder")
    capsule(J[name + "UpperArm"], J[name + "LowerArm"], 0.052, 0.042, name + "upperarm")
    capsule(J[name + "LowerArm"], J[name + "Hand"], 0.042, 0.032, name + "forearm")
    blob((J[name + "Hand"] + J[name + "HandEnd"]) * 0.5, (0.04, 0.025, 0.055), name + "hand")
    capsule(J[name + "UpperLeg"], J[name + "LowerLeg"], 0.082, 0.056, name + "thigh")
    capsule(J[name + "LowerLeg"], J[name + "Foot"], 0.056, 0.04, name + "shin")
    blob((J[name + "Foot"] + J[name + "Toes"]) * 0.5 + Vector((0, 0.0, 0.0)), (0.05, 0.12, 0.045), name + "foot")

bpy.ops.object.select_all(action="DESELECT")
for ob in parts:
    ob.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
body = bpy.context.active_object
body.name = "Body"
body.data.materials.append(skin)
for p in body.data.polygons:
    p.use_smooth = True

# Skin it to the armature with automatic weights.
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_skins=True, export_animations=False, export_apply=False, export_yup=True)
print("exported", OUT, len(body.data.vertices), "vertices")
