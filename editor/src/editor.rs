//! The mocap editor: one dock workspace over one session model.

use crate::model::SessionModel;
use crate::panels::{InspectorPanel, ScenePanel, SetupPanel, SourcesPanel, TimelinePanel, ViewportPanel};
use gpui::*;
use std::path::PathBuf;
use std::sync::Arc;
use ui::dock::{DockChannel, DockItem, Panel, PanelEvent, PanelView};
use ui::workspace::Workspace;
use ui::button::{Button, ButtonVariants as _};
use ui::{h_flex, v_flex, ActiveTheme, IconName, Sizable as _};

/// The dock channel the mocap editor's panels share (others are 1 and 3 to 7).
const DOCK_CHANNEL: u32 = 8;

/// The editor panel opened on a `.mocap` file.
pub struct MocapEditor {
    path: PathBuf,
    model: Entity<SessionModel>,
    workspace: Option<Entity<Workspace>>,
    focus_handle: FocusHandle,
}

impl MocapEditor {
    /// A setup illustration shown large in the middle of the editor, over a dimmed backdrop; a click
    /// on the backdrop or the Close button dismisses it.
    fn preview_modal(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let theme = cx.theme().clone();
        let diagram = self.model.read(cx).preview?;
        let image = diagram.image()?;
        let close = {
            let model = self.model.clone();
            move |cx: &mut App| {
                model.update(cx, |m, cx| {
                    m.preview = None;
                    cx.notify();
                });
            }
        };
        let (close_backdrop, close_button) = (close.clone(), close);
        Some(
            div()
                .id("mocap-preview-backdrop")
                .absolute()
                .inset_0()
                .occlude()
                .flex()
                .items_center()
                .justify_center()
                .bg(gpui::black().opacity(0.62))
                .on_click(move |_, _, cx| close_backdrop(cx))
                .child(
                    v_flex()
                        .id("mocap-preview-card")
                        .w(relative(0.86))
                        .h(relative(0.86))
                        .rounded(theme.radius)
                        .border_1()
                        .border_color(theme.border)
                        .bg(theme.background)
                        .shadow_lg()
                        .overflow_hidden()
                        .on_click(|_, _, cx| cx.stop_propagation())
                        .child(
                            h_flex()
                                .w_full()
                                .h(px(36.))
                                .px_3()
                                .flex_shrink_0()
                                .items_center()
                                .justify_between()
                                .border_b_1()
                                .border_color(theme.border)
                                .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(diagram.title()))
                                .child(Button::new("mocap-preview-close").icon(IconName::Close).label("Close").ghost().small().on_click(move |_, _, cx| close_button(cx))),
                        )
                        .child(div().flex_1().min_h_0().p_2().child(img(image).size_full().object_fit(ObjectFit::Contain))),
                )
                .into_any_element(),
        )
    }

    /// An editor on `path`: a non-empty file opens as a take, an empty or new one starts a session.
    pub fn new(path: PathBuf, cx: &mut Context<Self>) -> Self {
        let model = cx.new(|_| SessionModel::new(path.clone()));
        // Re-render (and so pump) when a task or the panels change the model.
        cx.observe(&model, |_, _, cx| cx.notify()).detach();
        Self { path, model, workspace: None, focus_handle: cx.focus_handle() }
    }

    fn initialize_workspace(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let model = self.model.clone();
        let workspace = cx.new(|cx| Workspace::new_with_channel("mocap-workspace", DockChannel(DOCK_CHANNEL), window, cx));
        workspace.update(cx, |workspace, cx| {
            let dock = workspace.dock_area().downgrade();
            let viewport: Arc<dyn PanelView> = Arc::new(cx.new(|cx| ViewportPanel::new(model.clone(), cx)));
            let scene: Arc<dyn PanelView> = Arc::new(cx.new(|cx| ScenePanel::new(model.clone(), cx)));
            let sources: Arc<dyn PanelView> = Arc::new(cx.new(|cx| SourcesPanel::new(model.clone(), cx)));
            let inspector: Arc<dyn PanelView> = Arc::new(cx.new(|cx| InspectorPanel::new(model.clone(), cx)));
            let setup: Arc<dyn PanelView> = Arc::new(cx.new(|cx| SetupPanel::new(model.clone(), cx)));
            let timeline: Arc<dyn PanelView> = Arc::new(cx.new(|cx| TimelinePanel::new(model.clone(), cx)));
            let center = DockItem::split(
                Axis::Horizontal,
                vec![
                    DockItem::tabs(vec![viewport], Some(0), &dock, window, cx),
                    DockItem::tabs(vec![scene], Some(0), &dock, window, cx),
                ],
                &dock,
                window,
                cx,
            );
            let left = DockItem::tabs(vec![sources], Some(0), &dock, window, cx);
            let right = DockItem::tabs(vec![setup, inspector], Some(0), &dock, window, cx);
            let bottom = DockItem::tabs(vec![timeline], Some(0), &dock, window, cx);
            workspace.initialize(center, Some(left), Some(right), Some(bottom), window, cx);
        });
        self.workspace = Some(workspace);
    }
}

impl MocapEditor {
    /// The strip above the panels: recording, and the state of the session at a glance.
    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let (recording, blocker, summary) = {
            let m = self.model.read(cx);
            (m.is_recording(), m.record_blocker(), m.calibration_summary())
        };
        let record = Button::new("mocap-record")
            .icon(if recording { IconName::Pause } else { IconName::Circle })
            .label(if recording { "Stop" } else { "Record" })
            .tooltip(blocker.unwrap_or(if recording { "Stop recording" } else { "Record every running pose and sensor source" }))
            .small()
            .on_click(cx.listener(|this, _, _, cx| {
                this.model.update(cx, |m, cx| {
                    if m.is_recording() {
                        m.stop_recording();
                    } else {
                        m.start_recording();
                    }
                    cx.notify();
                });
            }));
        let record = if recording { record.danger() } else { record.ghost() };
        h_flex()
            .w_full()
            .h(px(34.))
            .px_2()
            .gap_1()
            .items_center()
            .flex_shrink_0()
            .border_b_1()
            .border_color(theme.border)
            .child(record)
            .child(Button::new("mocap-marker").icon(IconName::Bookmark).label("Marker").ghost().small().on_click(cx.listener(|this, _, _, cx| {
                this.model.update(cx, |m, cx| {
                    m.mark();
                    cx.notify();
                });
            })))
            .child(div().flex_1())
            .child(div().text_xs().text_color(theme.muted_foreground).child(summary.unwrap_or_else(|| "Cameras not set up: see the Setup tab".to_owned())))
    }

    /// The strip below the panels: what the session is doing.
    fn status_bar(&self, cx: &App) -> impl IntoElement {
        let theme = cx.theme();
        let m = self.model.read(cx);
        let live = m.sources.iter().filter(|s| s.running()).count();
        let recording = if m.is_recording() {
            let stats = m.record_stats;
            format!("REC {} frames, {} gap, {} dropped", stats.frames, stats.gap_frames, m.bus_dropped())
        } else {
            m.record_blocker().map(|r| format!("Record: {r}")).unwrap_or_default()
        };
        h_flex()
            .w_full()
            .h(px(24.))
            .px_2()
            .gap_3()
            .items_center()
            .flex_shrink_0()
            .border_t_1()
            .border_color(theme.border)
            .text_xs()
            .text_color(theme.muted_foreground)
            .child(div().flex_1().min_w_0().truncate().child(m.status.clone()))
            .child(div().child(recording))
            .child(div().child(format!("{live} of {} sources live", m.sources.len())))
    }
}

impl EventEmitter<PanelEvent> for MocapEditor {}

impl Focusable for MocapEditor {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Panel for MocapEditor {
    fn panel_name(&self) -> &'static str {
        "Mocap Editor"
    }

    fn panel_file_path(&self, _cx: &App) -> Option<PathBuf> {
        Some(self.path.clone())
    }

    fn title(&self, _window: &Window, _cx: &App) -> AnyElement {
        let name = self.path.file_name().and_then(|n| n.to_str()).unwrap_or("Mocap").to_owned();
        h_flex().gap_2().items_center().child(div().text_sm().child(name)).into_any_element()
    }

    fn dump(&self, _cx: &App) -> ui::dock::PanelState {
        ui::dock::PanelState { panel_name: self.panel_name().to_string(), ..Default::default() }
    }

    fn inner_padding(&self, _cx: &App) -> bool {
        false
    }
}

impl Render for MocapEditor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.workspace.is_none() {
            self.initialize_workspace(window, cx);
        }
        // One tick per frame while anything is live: results of tasks, the pose
        // bus, playback. The model notifies the panels that changed.
        let keep_going = self.model.update(cx, |model, cx| {
            if model.pump() {
                cx.notify();
            }
            model.needs_frames()
        });
        if keep_going {
            window.request_animation_frame();
        }
        let content = match &self.workspace {
            Some(workspace) => workspace.clone().into_any_element(),
            None => div().size_full().into_any_element(),
        };
        let modal = self.preview_modal(cx);
        div()
            .relative()
            .size_full()
            .child(
        v_flex()
            .size_full()
            .bg(cx.theme().background)
            .child(self.toolbar(cx))
            .child(div().flex_1().min_h_0().w_full().child(content))
            .child(self.status_bar(cx))
            )
            .children(modal)
    }
}
