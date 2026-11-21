//! Draws the posed skinned character on a WGPU surface: skinning in the vertex shader (one matrix per
//! joint), depth testing, 4x multisampling and a three-light shading (key, fill and a rim), through the same
//! orthographic orbit the scene panel's overlay lines use so both line up.

use bytemuck::{Pod, Zeroable};
use mocap_capture::{Character, CharacterVertex};
use nalgebra::{Matrix4, Rotation3, Vector3};

/// The most joints the shader skins with.
pub const MAX_JOINTS: usize = 128;
const SAMPLES: u32 = 4;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const SHADER: &str = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    key_dir: vec4<f32>,
    base: vec4<f32>,
    eye_dir: vec4<f32>,
};
struct Skin {
    joints: array<mat4x4<f32>, 128>,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<uniform> skin: Skin;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) joints: vec4<u32>,
    @location(3) weights: vec4<f32>,
};
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var p = vec4<f32>(0.0);
    var n = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let m = skin.joints[v.joints[k]];
        p = p + (m * vec4<f32>(v.position, 1.0)) * v.weights[k];
        n = n + (mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * v.normal) * v.weights[k];
    }
    var out: VertexOut;
    out.clip = g.view_proj * vec4<f32>(p.xyz, 1.0);
    out.normal = normalize(n);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let key = max(dot(n, normalize(g.key_dir.xyz)), 0.0);
    let fill = max(dot(n, normalize(vec3<f32>(-g.key_dir.x, 0.2, -g.key_dir.z))), 0.0) * 0.35;
    let sky = mix(0.18, 0.42, n.y * 0.5 + 0.5);
    let rim = pow(1.0 - max(dot(n, normalize(g.eye_dir.xyz)), 0.0), 3.0) * 0.35;
    let lit = g.base.rgb * (sky + key * 0.75 + fill) + vec3<f32>(rim);
    let srgb = pow(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    normal: [f32; 3],
    joints: [u16; 4],
    weights: [f32; 4],
}

impl From<&CharacterVertex> for GpuVertex {
    fn from(v: &CharacterVertex) -> Self {
        Self { position: v.position, normal: v.normal, joints: v.joints, weights: v.weights }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    key_dir: [f32; 4],
    base: [f32; 4],
    eye_dir: [f32; 4],
}

/// How the scene is looked at: the same orbit the overlay's projection uses.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub yaw: f32,
    pub pitch: f32,
    /// Metres of stage the window's height shows (the overlay's `VIEW_HEIGHT_M` over the zoom).
    pub height_m: f32,
    pub pivot: [f32; 3],
}

impl View {
    /// The orthographic view-projection for a target of `width` by `height` pixels.
    fn matrix(&self, width: f32, height: f32) -> Matrix4<f32> {
        let to_view = Rotation3::from_axis_angle(&Vector3::x_axis(), self.pitch) * Rotation3::from_axis_angle(&Vector3::y_axis(), self.yaw);
        let ortho = {
            let sy = 2.0 / self.height_m;
            let sx = sy * height / width.max(1.0);
            // Nearer is larger z in the view; the depth range is the stage with room around it.
            Matrix4::new(sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, -0.05, 0.5, 0.0, 0.0, 0.0, 1.0)
        };
        ortho * to_view.to_homogeneous() * Matrix4::new_translation(&-Vector3::from(self.pivot))
    }
}

struct Targets {
    size: (u32, u32),
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
}

/// The character's GPU resources.
pub struct CharacterRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    globals: wgpu::Buffer,
    skin: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    base_color: [f32; 4],
    format: wgpu::TextureFormat,
    targets: Option<Targets>,
}

impl CharacterRenderer {
    /// Uploads `character` for drawing into targets of `format`.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, character: &Character) -> Option<Self> {
        if character.joint_count() > MAX_JOINTS {
            return None;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mocap character"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("mocap character layout"), entries: &[uniform(0), uniform(1)] });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mocap character pipeline layout"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let attributes = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint16x4, offset: 24, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 3 },
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mocap character"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<GpuVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &attributes })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..wgpu::PrimitiveState::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });
        let vertices: Vec<GpuVertex> = character.vertices.iter().map(GpuVertex::from).collect();
        let buffer = |label: &str, usage: wgpu::BufferUsages, size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let globals = buffer("mocap character globals", wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, std::mem::size_of::<Globals>() as u64);
        let skin = buffer("mocap character skin", wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, (MAX_JOINTS * 64) as u64);
        let vertex_buffer = buffer("mocap character vertices", wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, (vertices.len() * std::mem::size_of::<GpuVertex>()) as u64);
        let index_buffer = buffer("mocap character indices", wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, (character.indices.len() * 4) as u64);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mocap character bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: skin.as_entire_binding() }],
        });
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&character.indices));
        Some(Self { pipeline, bind_group, globals, skin, vertices: vertex_buffer, indices: index_buffer, index_count: character.indices.len() as u32, base_color: character.base_color, format, targets: None })
    }

    /// The multisampled colour and depth targets for `size`, made again when it changes.
    fn targets(&mut self, device: &wgpu::Device, size: (u32, u32)) -> &Targets {
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            let make = |format: wgpu::TextureFormat, label: &str| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: SAMPLES,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            self.targets = Some(Targets { size, color: make(self.format, "mocap character colour"), depth: make(DEPTH_FORMAT, "mocap character depth") });
        }
        self.targets.as_ref().expect("just made")
    }

    /// Clears `target` to `background` and, when `pose` is given (one matrix per joint), draws the character into it.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: (&wgpu::TextureView, (u32, u32)),
        view: &View,
        background: [f64; 3],
        pose: Option<&[[[f32; 4]; 4]]>,
    ) -> wgpu::SubmissionIndex {
        let (target, size) = target;
        if let Some(matrices) = pose {
            let m = view.matrix(size.0 as f32, size.1 as f32);
            let mut view_proj = [[0.0f32; 4]; 4];
            for c in 0..4 {
                for r in 0..4 {
                    view_proj[c][r] = m[(r, c)];
                }
            }
            // The key light from the front-left-above of the viewer; the eye direction in the same (world) space.
            let to_view = Rotation3::from_axis_angle(&Vector3::x_axis(), view.pitch) * Rotation3::from_axis_angle(&Vector3::y_axis(), view.yaw);
            let toward_eye = to_view.inverse() * Vector3::z();
            let key = to_view.inverse() * Vector3::new(-0.45, 0.8, 0.6).normalize();
            let globals = Globals { view_proj, key_dir: [key[0], key[1], key[2], 0.0], base: self.base_color, eye_dir: [toward_eye[0], toward_eye[1], toward_eye[2], 0.0] };
            queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
            let mut skin = vec![[[0.0f32; 4]; 4]; MAX_JOINTS];
            for (slot, m) in skin.iter_mut().zip(matrices) {
                *slot = *m;
            }
            queue.write_buffer(&self.skin, 0, bytemuck::cast_slice(&skin));
        }
        self.targets(device, size);
        let targets = self.targets.as_ref().expect("targets");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mocap character") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mocap character"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: background[0], g: background[1], b: background[2], a: 1.0 }), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if pose.is_some() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }
        queue.submit(Some(encoder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mocap_capture::BodyPoints;

    fn standing() -> [[f32; 3]; 17] {
        let mut p = [[0.0f32; 3]; 17];
        let side = |s: f32| s;
        for (k, v) in [
            (0, [0.0, 1.62, 0.1]),
            (1, [0.03, 1.66, 0.08]),
            (2, [-0.03, 1.66, 0.08]),
            (3, [0.08, 1.64, 0.0]),
            (4, [-0.08, 1.64, 0.0]),
            (5, [side(0.19), 1.45, 0.0]),
            (6, [side(-0.19), 1.45, 0.0]),
            (7, [0.27, 1.2, 0.04]),
            (8, [-0.27, 1.2, 0.04]),
            (9, [0.31, 0.97, 0.12]),
            (10, [-0.31, 0.97, 0.12]),
            (11, [0.1, 0.95, 0.0]),
            (12, [-0.1, 0.95, 0.0]),
            (13, [0.1, 0.52, 0.05]),
            (14, [-0.1, 0.52, 0.0]),
            (15, [0.1, 0.1, 0.0]),
            (16, [-0.1, 0.1, 0.0]),
        ] {
            p[k] = v;
        }
        p
    }

    /// Draws the posed character into an offscreen texture on whatever GPU the machine has and checks that something
    /// character-shaped came out (skipped when there is no adapter).
    #[test]
    fn the_posed_character_renders_to_a_texture() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = futures::executor::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let Ok((device, queue)) = futures::executor::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) else { return };
        let character = Character::from_glb(include_bytes!("../assets/character.glb")).expect("the character loads");
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let mut renderer = CharacterRenderer::new(&device, &queue, format, &character).expect("a renderer");
        let (w, h) = (640u32, 800u32);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let pose = character.pose(&BodyPoints::from_coco17(&standing())).expect("a pose");
        let orbit = View { yaw: 0.45, pitch: 0.2, height_m: 2.6, pivot: [0.0, 0.9, 0.5] };
        let background = [0.1, 0.1, 0.12];
        let submission = renderer.draw(&device, &queue, (&view, (w, h)), &orbit, background, Some(&pose));
        let padded = (w * 4).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: u64::from(padded * h), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let _ = submission;
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("the GPU finishes");
        let data = slice.get_mapped_range().expect("mapped");
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        let mut lit = 0usize;
        for row in data.chunks(padded as usize).take(h as usize) {
            for px in row[..(w * 4) as usize].chunks_exact(4) {
                // Bgra to rgba.
                rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
                let (b, g, r) = (f32::from(px[0]) / 255.0, f32::from(px[1]) / 255.0, f32::from(px[2]) / 255.0);
                // The background is (0.1, 0.1, 0.12) in linear, written as it is (no sRGB target): compare loosely.
                if (r - 0.1).abs() + (g - 0.1).abs() + (b - 0.12).abs() > 0.08 {
                    lit += 1;
                }
            }
        }
        if let Ok(path) = std::env::var("MOCAP_RENDER_OUT") {
            image::save_buffer(path, &rgba, w, h, image::ColorType::Rgba8).expect("saved");
        }
        let share = lit as f32 / (w * h) as f32;
        assert!(share > 0.03 && share < 0.6, "a character covers part of the frame: {share:.3}");
    }
}
