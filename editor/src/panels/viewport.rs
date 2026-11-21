//! The camera viewport: the live video on a WGPU surface, with the skeleton and keypoints drawn
//! over it and a thin header of view toggles. No floating cards: the controls live in the editor's
//! toolbar and status bar and in the Setup panel.

use crate::blit::FrameBlit;
use crate::model::SessionModel;
use gpui::prelude::*;
use gpui::*;
use std::collections::HashMap;
use ui::PixelsExt as _;
use ui::button::{Button, ButtonVariants as _};
use ui::dock::{Panel, PanelEvent};
use ui::{h_flex, v_flex, ActiveTheme, IconName, Selectable as _, Sizable as _};

/// The surface's pixel format (the window compositor's, not sRGB).
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// The camera view.
pub struct ViewportPanel {
    model: Entity<SessionModel>,
    focus_handle: FocusHandle,
    surface: Option<WgpuSurfaceHandle>,
    blit: Option<FrameBlit>,
    seen_video: HashMap<String, u64>,
}

impl ViewportPanel {
    pub fn new(model: Entity<SessionModel>, cx: &mut Context<Self>) -> Self {
        cx.observe(&model, |_, _, cx| cx.notify()).detach();
        Self { model, focus_handle: cx.focus_handle(), surface: None, blit: None, seen_video: HashMap::new() }
    }

    fn ensure_surface(&mut self, window: &mut Window) {
        if self.surface.as_ref().is_some_and(WgpuSurfaceHandle::is_device_lost) {
            self.surface = None;
            self.blit = None;
            self.seen_video.clear();
        }
        if self.surface.is_some() {
            return;
        }
        let size = window.bounds().size;
        let width = (size.width.to_f64() as u32).max(64);
        let height = (size.height.to_f64() as u32).max(64);
        self.surface = window
            .create_wgpu_surface(width, height, SURFACE_FORMAT)
            .filter(|surface| !surface.is_device_lost());
    }

    /// Uploads a newer camera frame and publishes the surface.
    fn paint_surface(&mut self, window: &Window, cx: &App) {
        let Some(surface) = &self.surface else { return };
        if window.is_window_resizing() || surface.is_resize_pending() || surface.has_unconsumed_frame() {
            return;
        }
        let Some(_submit) = surface.try_submit_guard() else { return };
        let (device, queue) = (surface.device(), surface.queue());
        let blit = self.blit.get_or_insert_with(|| FrameBlit::new(device, surface.format()));
        let model = self.model.read(cx);
        let feeds = model.video_feeds();
        let keys: Vec<String> = feeds.iter().map(|(key, _)| key.clone()).collect();
        blit.retain(&keys);
        self.seen_video.retain(|key, _| keys.contains(key));
        for (key, mailbox) in &feeds {
            let seen = self.seen_video.get(key).copied().unwrap_or(0);
            if let Some((count, frame)) = mailbox.latest_if_newer(seen) {
                blit.upload(device, queue, key, &frame);
                self.seen_video.insert(key.clone(), count);
            }
        }
        let Some((view, size)) = surface.back_view_with_size() else { return };
        let submission = blit.draw(device, queue, &view, size, &keys, model.mirror);
        drop(view);
        surface.present_synced_silent(submission);
    }

    /// The thin strip above the video: what is shown, and the view toggles.
    fn header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let (overlay, mirror) = {
            let m = self.model.read(cx);
            (m.show_overlay, m.mirror)
        };
        let feeds: Vec<String> = self
            .model
            .read(cx)
            .video_feeds()
            .into_iter()
            .filter_map(|(key, _)| self.blit.as_ref()?.video_size(&key).map(|(w, h)| format!("{key} {w}x{h}")))
            .collect();
        let toggle = |id: &'static str, icon: IconName, label: &'static str, on: bool| Button::new(id).icon(icon).label(label).ghost().xsmall().selected(on);
        h_flex()
            .w_full()
            .h(px(28.))
            .px_2()
            .gap_1()
            .items_center()
            .flex_shrink_0()
            .border_b_1()
            .border_color(theme.border)
            .child(toggle("view-skeleton", IconName::Eye, "Skeleton", overlay).on_click(cx.listener(|this, _, _, cx| {
                this.model.update(cx, |m, cx| {
                    m.show_overlay = !m.show_overlay;
                    cx.notify();
                });
            })))
            .child(toggle("view-mirror", IconName::Camera, "Mirror", mirror).on_click(cx.listener(|this, _, _, cx| {
                this.model.update(cx, |m, cx| {
                    m.mirror = !m.mirror;
                    cx.notify();
                });
            })))
            .child(div().flex_1().min_w_0().truncate().text_xs().text_color(theme.muted_foreground).text_right().child(feeds.join("   ")))
    }
}

impl EventEmitter<PanelEvent> for ViewportPanel {}

impl Focusable for ViewportPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Panel for ViewportPanel {
    fn panel_name(&self) -> &'static str {
        "mocap-viewport"
    }

    fn title(&self, _window: &Window, _cx: &App) -> AnyElement {
        div().text_sm().child("Camera").into_any_element()
    }

    fn dump(&self, _cx: &App) -> ui::dock::PanelState {
        ui::dock::PanelState { panel_name: self.panel_name().to_string(), ..Default::default() }
    }

    fn inner_padding(&self, _cx: &App) -> bool {
        false
    }
}

impl Render for ViewportPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.ensure_surface(window);
        self.paint_surface(window, cx);
        let theme = cx.theme().clone();
        let video_live = self.model.read(cx).live_video();
        let (skeleton, mirror) = {
            let model = self.model.read(cx);
            let drawn = if model.show_overlay { model.current_figure().map(|f| (f.joints, f.bones)) } else { None };
            (drawn, model.mirror)
        };
        let accent = theme.accent;
        let tiles: Vec<(usize, usize, (u32, u32), Vec<[f32; 3]>)> = {
            let model = self.model.read(cx);
            let keys: Vec<String> = model.video_feeds().into_iter().map(|(key, _)| key).collect();
            if model.show_overlay {
                model
                    .keypoint_overlays()
                    .into_iter()
                    .filter_map(|overlay| {
                        let index = keys.iter().position(|key| *key == overlay.camera_id)?;
                        let size = self.blit.as_ref()?.video_size(&overlay.camera_id)?;
                        Some((index, keys.len(), size, overlay.points))
                    })
                    .collect()
            } else {
                Vec::new()
            }
        };
        let surface = match &self.surface {
            Some(handle) => wgpu_surface(handle.clone()).defer_resize_until_mouse_up(true).size_full().into_any_element(),
            None => div().size_full().into_any_element(),
        };
        let hint = div().absolute().inset_0().flex().items_center().justify_center().text_color(theme.muted_foreground).when(
            !video_live,
            |d| d.child("No camera running. Add one in Sources, or start a pose source."),
        );
        // The picture and what is drawn on it (the keypoints belong to the video, so they are drawn in place).
        let view = div()
            .relative()
            .flex_1()
            .min_h_0()
            .w_full()
            .child(div().absolute().inset_0().child(surface))
            .child(div().absolute().inset_0().when_some(skeleton, |d, (joints, bones)| {
                d.child(canvas(|bounds, _, _| bounds, move |_, bounds, window, _| crate::paint::draw_front_skeleton(window, bounds, &joints, &bones, mirror, accent)).size_full())
            }))
            .child(div().absolute().inset_0().child(canvas(|bounds, _, _| bounds, move |_, bounds, window, _| {
                let target = (bounds.size.width.as_f32() as u32, bounds.size.height.as_f32() as u32);
                for (index, count, size, points) in &tiles {
                    let rect = crate::tile_rect(*size, target, *index, *count);
                    crate::paint::draw_keypoints(window, bounds, rect, points, mirror, accent);
                }
            }).size_full()))
            .child(hint);
        v_flex().size_full().bg(theme.background).child(self.header(cx)).child(view)
    }
}
