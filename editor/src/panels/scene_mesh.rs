//! The scene panel's character: the bundled skinned humanoid, posed from the fused subject and drawn on a
//! WGPU surface under the panel's overlay (cameras, sensors, floor).

use crate::character_view::{CharacterRenderer, View};
use gpui::*;
use mocap_capture::{BodyPoints, Character, FigureSource, SceneFigure};
use std::sync::{Arc, OnceLock};

/// The surface's pixel format (the window compositor's, not sRGB).
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// The character that ships with the plugin (a humanoid glTF: any other can replace it, bones are found by name).
static CHARACTER_GLB: &[u8] = include_bytes!("../../assets/character.glb");

fn bundled() -> Option<Arc<Character>> {
    static CHARACTER: OnceLock<Option<Arc<Character>>> = OnceLock::new();
    CHARACTER
        .get_or_init(|| match Character::from_glb(CHARACTER_GLB) {
            Ok(c) => Some(Arc::new(c)),
            Err(error) => {
                tracing::error!("the bundled character did not load: {error}");
                None
            }
        })
        .clone()
}

/// The body a figure describes, for posing.
pub fn body_of(figure: &SceneFigure) -> Option<BodyPoints> {
    match figure.source {
        // A single camera's keypoints are flat (no depth), but the character is posed in that plane, facing the camera.
        FigureSource::Keypoints | FigureSource::Triangulated => {
            let mut p = [[f32::NAN; 3]; 17];
            for (slot, name) in p.iter_mut().zip(mocap_capture::COCO17) {
                if let Some(i) = figure.names.iter().position(|n| n == name) {
                    *slot = figure.joints[i];
                }
            }
            Some(BodyPoints::from_coco17(&p))
        }
        FigureSource::Skeleton => Some(BodyPoints::from_named(&figure.names, &figure.joints)),
    }
}

/// The surface, the character and its GPU resources.
#[derive(Default)]
pub struct SceneMesh {
    surface: Option<WgpuSurfaceHandle>,
    renderer: Option<CharacterRenderer>,
    character: Option<Arc<Character>>,
}

impl SceneMesh {
    fn ensure_surface(&mut self, window: &mut Window) {
        if self.surface.as_ref().is_some_and(WgpuSurfaceHandle::is_device_lost) {
            self.surface = None;
            self.renderer = None;
        }
        if self.surface.is_some() {
            return;
        }
        let size = window.bounds().size;
        let (w, h) = ((size.width.to_f64() as u32).max(64), (size.height.to_f64() as u32).max(64));
        self.surface = window.create_wgpu_surface(w, h, SURFACE_FORMAT).filter(|s| !s.is_device_lost());
    }

    /// Shows `character` instead of the bundled one.
    pub fn set_character(&mut self, character: Arc<Character>) {
        self.character = Some(character);
        self.renderer = None;
    }

    /// Draws the figure (if it can be posed) over `background` and returns the surface's element; the bool says whether
    /// a character was drawn (so the overlay can leave out its stick figure).
    pub fn element(&mut self, window: &mut Window, view: &View, background: [f64; 3], figure: Option<&SceneFigure>) -> (AnyElement, bool) {
        self.ensure_surface(window);
        if self.character.is_none() {
            self.character = bundled();
        }
        let mut drawn = false;
        if let (Some(surface), Some(character)) = (&self.surface, &self.character) {
            let busy = window.is_window_resizing() || surface.is_resize_pending() || surface.has_unconsumed_frame();
            if !busy {
                if let Some(_submit) = surface.try_submit_guard() {
                    let (device, queue) = (surface.device(), surface.queue());
                    if self.renderer.is_none() {
                        self.renderer = CharacterRenderer::new(device, queue, surface.format(), character);
                    }
                    let pose = figure.and_then(body_of).and_then(|body| character.pose(&body));
                    if let (Some(renderer), Some((target, size))) = (self.renderer.as_mut(), surface.back_view_with_size()) {
                        let submission = renderer.draw(device, queue, (&target, size), view, background, pose.as_deref());
                        drawn = pose.is_some();
                        drop(target);
                        surface.present_synced_silent(submission);
                    }
                }
            } else {
                // A frame is in flight: the overlay keeps its figure out of the way only if the mesh is what is showing.
                drawn = figure.and_then(body_of).is_some_and(|b| character.pose(&b).is_some());
            }
        }
        let element = match &self.surface {
            Some(handle) => wgpu_surface(handle.clone()).defer_resize_until_mouse_up(true).size_full().into_any_element(),
            None => div().size_full().into_any_element(),
        };
        (element, drawn)
    }
}
