//! The Setup panel: how the cameras are placed and calibrated, step by step, in one scrolling
//! column: the cameras that are ready, the method (with its illustration), its numbers, then the
//! action, the guided movements and the result. A dockable tab, not an overlay.

use crate::diagrams::Diagram;
use crate::model::{CalibrationMethod, CalibrationPhase, SessionModel};
use gpui::prelude::*;
use gpui::*;
use mocap_capture::{Arrangement, GuideState, Step, MIN_SAMPLES, SAMPLES_TARGET};
use ui::button::{Button, ButtonVariants as _};
use ui::dock::{Panel, PanelEvent};
use ui::{h_flex, v_flex, ActiveTheme, Disableable as _, Selectable as _, Sizable as _};

/// The calibration and camera setup.
pub struct SetupPanel {
    model: Entity<SessionModel>,
    focus_handle: FocusHandle,
    /// The "space to keep clear" picture is shown.
    show_space: bool,
}

impl SetupPanel {
    pub fn new(model: Entity<SessionModel>, cx: &mut Context<Self>) -> Self {
        cx.observe(&model, |_, _, cx| cx.notify()).detach();
        Self { model, focus_handle: cx.focus_handle(), show_space: false }
    }
}

/// A titled block of the column.
fn section(title: &'static str, cx: &App) -> Div {
    v_flex()
        .px_2()
        .py_1p5()
        .gap_1()
        .border_b_1()
        .border_color(cx.theme().border)
        .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(cx.theme().muted_foreground).child(title))
}

/// An illustration that opens enlarged when clicked.
fn picture(model: &Entity<SessionModel>, diagram: Diagram, image: std::sync::Arc<RenderImage>) -> impl IntoElement {
    let model = model.clone();
    div()
        .id(SharedString::from(format!("picture-{diagram:?}")))
        .w_full()
        .aspect_ratio(1.6)
        .cursor_pointer()
        .tooltip(|window, cx| ui::tooltip::Tooltip::new("Click to enlarge").build(window, cx))
        .on_click(move |_, _, cx| {
            model.update(cx, |m, cx| {
                m.preview = Some(diagram);
                cx.notify();
            });
        })
        .child(img(image).size_full().object_fit(ObjectFit::Contain))
}

fn note(text: impl Into<SharedString>, cx: &App) -> Div {
    div().text_xs().text_color(cx.theme().muted_foreground).child(text.into())
}

fn action(model: &Entity<SessionModel>, id: &'static str, label: &'static str, run: impl Fn(&mut SessionModel) + 'static) -> Button {
    let model = model.clone();
    Button::new(id).label(label).small().on_click(move |_, _, cx| {
        model.update(cx, |m, cx| {
            run(m);
            cx.notify();
        });
    })
}

fn describe(method: CalibrationMethod) -> (&'static str, &'static str, Diagram) {
    match method {
        CalibrationMethod::Layout(Arrangement::Ring) => ("Ring", "Cameras evenly around you, all aimed at you: the best coverage of the whole body. Placed exactly from the numbers below; no movement needed.", Diagram::Ring),
        CalibrationMethod::Layout(Arrangement::Arc) => ("Arc", "Cameras over a half circle in front of you, like a desk or a stage. Placed exactly from the numbers below; no movement needed.", Diagram::Arc),
        CalibrationMethod::Layout(Arrangement::Line) => ("Line", "Cameras side by side in a row facing you, a wall of cameras. Placed exactly from the numbers below; no movement needed.", Diagram::Line),
        CalibrationMethod::Body => ("Movement", "Cameras anywhere: their positions are worked out from you doing a few guided movements. Needs the cameras to see overlapping parts of you; the least exact method.", Diagram::Body),
    }
}

fn method_picker(model: &Entity<SessionModel>, current: CalibrationMethod, locked: bool) -> impl IntoElement {
    let choose = |id: &'static str, method: CalibrationMethod| {
        let handle = model.clone();
        Button::new(id)
            .label(describe(method).0)
            .small()
            .ghost()
            .selected(method == current)
            .disabled(locked)
            .on_click(move |_, _, cx| {
                handle.update(cx, |m, cx| {
                    m.set_calibration_method(method);
                    cx.notify();
                });
            })
    };
    h_flex()
        .gap_1()
        .flex_wrap()
        .child(choose("m-ring", CalibrationMethod::Layout(Arrangement::Ring)))
        .child(choose("m-arc", CalibrationMethod::Layout(Arrangement::Arc)))
        .child(choose("m-line", CalibrationMethod::Layout(Arrangement::Line)))
        .child(choose("m-body", CalibrationMethod::Body))
}

fn number(model: &Entity<SessionModel>, name: &'static str, value: String, id: &'static str, step: (f64, f64, f64), cx: &App) -> impl IntoElement {
    let nudge = |suffix: &'static str, label: &'static str, sign: f64| {
        let handle = model.clone();
        Button::new(SharedString::from(format!("{id}{suffix}"))).label(label).ghost().xsmall().on_click(move |_, _, cx| {
            handle.update(cx, |m, cx| {
                m.adjust_layout(step.0 * sign, step.1 * sign, step.2 * sign);
                cx.notify();
            });
        })
    };
    h_flex()
        .gap_1()
        .items_center()
        .text_xs()
        .child(div().w(px(96.)).flex_shrink_0().truncate().text_color(cx.theme().muted_foreground).child(name))
        .child(nudge("-", "-", -1.0))
        .child(div().w(px(56.)).flex_shrink_0().whitespace_nowrap().text_center().child(value))
        .child(nudge("+", "+", 1.0))
}

fn checklist(model: &SessionModel, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    let guide = &model.calib.guide;
    let done = guide.completed();
    let current = guide.current();
    v_flex().gap_0p5().children(Step::ALL.iter().map(|step| {
        let (mark, color) = if done.contains(step) {
            ("done", theme.success)
        } else if Some(*step) == current {
            ("now ", theme.accent)
        } else {
            ("    ", theme.muted_foreground)
        };
        h_flex().gap_2().text_xs().text_color(color).child(div().w(px(30.)).flex_shrink_0().child(mark)).child(div().flex_1().min_w_0().truncate().child(step.name()))
    }))
}

fn guiding(model: &Entity<SessionModel>, cx: &App) -> Div {
    let theme = cx.theme();
    let m = model.read(cx);
    let (text, progress, detected) = match m.calib.guide.state() {
        GuideState::Step { step, progress, detected } => (step.instruction(), progress, detected),
        GuideState::Finished => ("Steps done. Keep moving: walk around, turn, reach with both arms", (m.calib.collected as f32 / SAMPLES_TARGET as f32).min(1.0), true),
    };
    let ready = m.calib.collected >= MIN_SAMPLES;
    let hint = match m.calib.guide.hint() {
        "" if detected => "Got it: keep going".to_owned(),
        "" => "Waiting for the movement. Every camera should see you.".to_owned(),
        hint => hint.to_owned(),
    };
    section("Guided movements", cx)
        .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(text))
        .child(div().h(px(6.)).w_full().bg(theme.border).child(div().h_full().w(relative(progress)).bg(if detected { theme.success } else { theme.accent })))
        .child(div().text_xs().text_color(if detected { theme.success } else { theme.muted_foreground }).child(hint))
        .child(checklist(m, cx))
        .child(note(format!("{} samples (it solves at {SAMPLES_TARGET}, or press Solve now from {MIN_SAMPLES})", m.calib.collected), cx))
        .child(
            h_flex()
                .gap_1()
                .child(action(model, "calib-skip", "Skip step", SessionModel::skip_calibration_step).ghost())
                .when(ready, |row| row.child(action(model, "calib-solve", "Solve now", SessionModel::solve_camera_calibration)))
                .child(action(model, "calib-cancel", "Cancel", SessionModel::cancel_camera_calibration).ghost()),
        )
}

fn result(model: &Entity<SessionModel>, cx: &App) -> Div {
    let theme = cx.theme();
    let m = model.read(cx);
    let checks = m.calib.checks.iter().map(|check| {
        let (mark, color) = if check.passed { ("pass", theme.success) } else { ("FAIL", theme.danger) };
        h_flex().gap_2().items_start().text_xs().child(div().w(px(32.)).flex_shrink_0().text_color(color).child(mark)).child(div().flex_1().min_w_0().child(format!("{}: {}", check.name, check.detail)))
    });
    section("Result", cx)
        .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(theme.success).child(m.calibration_summary().unwrap_or_default()))
        .when_some(m.camera_rig.as_ref(), |col, rig| {
            let per_camera: Vec<String> = rig.cameras.iter().zip(&rig.per_camera_rms).map(|(c, r)| format!("{} {:.1}px", c.id, r)).collect();
            let detail = if rig.samples == 0 {
                format!("{} cameras, spread {:.0} deg between them.", rig.cameras.len(), rig.parallax_deg)
            } else {
                format!("{} samples, spread {:.0} deg between cameras. {}", rig.samples, rig.parallax_deg, per_camera.join(", "))
            };
            col.child(note(detail, cx)).children(rig.notes.iter().map(|n| note(*n, cx)))
        })
        .children(checks)
        .child(h_flex().gap_1().child(action(model, "calib-redo", "Change setup", SessionModel::forget_to_setup).ghost()))
}

impl SetupPanel {
    fn cameras_section(&self, cx: &App) -> Div {
        let theme = cx.theme();
        let m = self.model.read(cx);
        let ready = m.tracked_camera_count();
        let (text, color) = match ready {
            0 => ("No camera is tracking the body. Add cameras in Sources and press Track body on each.".to_owned(), theme.warning),
            1 => ("1 camera is tracking the body. Calibration needs at least two.".to_owned(), theme.warning),
            n => (format!("{n} cameras are tracking the body."), theme.success),
        };
        section("1  Cameras", cx).child(div().text_xs().text_color(color).child(text))
    }

    fn method_section(&self, cx: &mut Context<Self>) -> Div {
        let (current, locked, failure) = {
            let m = self.model.read(cx);
            let locked = matches!(m.calib.phase, CalibrationPhase::Guiding | CalibrationPhase::Solving);
            let failure = match &m.calib.phase {
                CalibrationPhase::Failed(reason) => Some(reason.clone()),
                _ => None,
            };
            (m.calib.method, locked, failure)
        };
        let (_, text, diagram) = describe(current);
        let theme = cx.theme().clone();
        section("2  Method", cx)
            .child(method_picker(&self.model, current, locked))
            .when_some(diagram.image(), |col, image| col.child(picture(&self.model, diagram, image)))
            .child(note(text, cx))
            .when_some(failure, |col, reason| col.child(div().text_xs().text_color(theme.danger).child(reason)))
    }

    fn layout_section(&self, cx: &mut Context<Self>) -> Div {
        let (distance, height, fov) = {
            let m = self.model.read(cx);
            (m.calib.distance, m.calib.height, m.calib.hfov_deg)
        };
        let weak = cx.entity().downgrade();
        let space = Diagram::Space.image().filter(|_| self.show_space);
        section("3  Distances", cx)
            .child(number(&self.model, "Distance", format!("{distance:.1} m"), "d", (0.25, 0.0, 0.0), cx))
            .child(number(&self.model, "Lens height", format!("{height:.2} m"), "h", (0.0, 0.1, 0.0), cx))
            .child(number(&self.model, "Field of view", format!("{fov:.0} deg"), "f", (0.0, 0.0, 5.0), cx))
            .child(
                Button::new("space-toggle")
                    .label(if self.show_space { "Hide the space you need" } else { "Show the space you need" })
                    .ghost()
                    .xsmall()
                    .on_click(move |_, _, cx| {
                        weak.update(cx, |this, cx| {
                            this.show_space = !this.show_space;
                            cx.notify();
                        })
                        .ok();
                    }),
            )
            .when_some(space, |col, image| col.child(picture(&self.model, Diagram::Space, image)))
    }

    fn start_section(&self, cx: &App) -> Div {
        let layout = matches!(self.model.read(cx).calib.method, CalibrationMethod::Layout(_));
        let step = if layout { "4  Place the cameras" } else { "3  Calibrate" };
        section(step, cx)
            .child(note("Cameras are assigned in the order of their sources.", cx))
            .child(h_flex().gap_1().child(if layout {
                action(&self.model, "calib-apply", "Place cameras", SessionModel::apply_camera_layout).primary()
            } else {
                action(&self.model, "calib-start", "Start guided calibration", SessionModel::start_camera_calibration).primary()
            }))
    }

    fn pose_section(&self, cx: &App) -> Div {
        let m = self.model.read(cx);
        let state = match (&m.calibration, m.calibration.as_ref().map(|c| c.scale)) {
            (Some(_), Some(scale)) => format!("Neutral pose taken (height scale {scale:.3})"),
            _ => "No neutral pose taken".to_owned(),
        };
        section("Neutral pose", cx)
            .child(note(state, cx))
            .child(note("Stand in a T-pose and press Take: the rig is zeroed to it.", cx))
            .child(
                h_flex()
                    .gap_1()
                    .child(action(&self.model, "pose-take", "Take T-pose", SessionModel::calibrate).ghost())
                    .child(action(&self.model, "pose-clear", "Clear", SessionModel::clear_calibration).ghost()),
            )
    }
}

impl EventEmitter<PanelEvent> for SetupPanel {}

impl Focusable for SetupPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Panel for SetupPanel {
    fn panel_name(&self) -> &'static str {
        "mocap-setup"
    }

    fn title(&self, _window: &Window, _cx: &App) -> AnyElement {
        div().text_sm().child("Setup").into_any_element()
    }

    fn dump(&self, _cx: &App) -> ui::dock::PanelState {
        ui::dock::PanelState { panel_name: self.panel_name().to_string(), ..Default::default() }
    }

    fn inner_padding(&self, _cx: &App) -> bool {
        false
    }
}

impl Render for SetupPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (phase, calibrated, layout) = {
            let m = self.model.read(cx);
            (m.calib.phase.clone(), m.camera_rig.is_some(), matches!(m.calib.method, CalibrationMethod::Layout(_)))
        };
        let column = v_flex().id("mocap-setup-scroll").size_full().overflow_x_hidden().overflow_y_scroll().bg(cx.theme().background).child(self.cameras_section(cx));
        let column = match phase {
            CalibrationPhase::Guiding => column.child(guiding(&self.model, cx)),
            CalibrationPhase::Solving => column.child(section("Calibrating", cx).child(note("Solving the camera positions...", cx))),
            CalibrationPhase::Done => column.child(result(&self.model, cx)),
            CalibrationPhase::Idle if calibrated => column.child(result(&self.model, cx)),
            _ => {
                let column = column.child(self.method_section(cx));
                let column = if layout { column.child(self.layout_section(cx)) } else { column };
                column.child(self.start_section(cx))
            }
        };
        column.child(self.pose_section(cx))
    }
}
