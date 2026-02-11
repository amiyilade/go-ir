from manim import *
import numpy as np

# ─── Data from actual results ────────────────────────────────────────────────

# RQ2: Attention-AST alignment mean (layers 0-11)
RQ2_UNIXCODER = [0.658, 0.595, 0.582, 0.560, 0.574, 0.404, 0.528, 0.678, 0.436, 0.515, 0.565, 0.492]
RQ2_CODEBERT  = [0.406, 0.547, 0.559, 0.603, 0.581, 0.408, 0.345, 0.589, 0.417, 0.455, 0.441, 0.436]

# RQ3a: Structural probing Spearman ρ (layers 0-12)
RQ3_PROBE_UNIXCODER = [0.350, 0.600, 0.590, 0.593, 0.569, 0.610, 0.609, 0.468, 0.577, 0.560, 0.516, 0.476, 0.300]
RQ3_PROBE_CODEBERT  = [0.627, 0.591, 0.623, 0.593, 0.599, 0.574, 0.602, 0.410, 0.533, 0.528, 0.536, 0.505, 0.491]

# RQ3b: Tree induction F1 (layers 0-12)
RQ3_TREE_UNIXCODER = [0.250, 0.344, 0.328, 0.301, 0.307, 0.315, 0.285, 0.253, 0.237, 0.223, 0.252, 0.254, 0.234]
RQ3_TREE_CODEBERT  = [0.205, 0.187, 0.193, 0.193, 0.177, 0.181, 0.152, 0.146, 0.146, 0.167, 0.178, 0.202, 0.108]

# Colors
C_UNI   = "#4FC3F7"  # light blue  — UniXcoder
C_CB    = "#FF8A65"  # orange      — CodeBERT
C_HIGH  = "#FFD54F"  # yellow      — highlight
C_BG    = "#0d1117"  # dark        — background
C_GRID  = "#2d333b"  # subtle grid
C_TEXT  = "#e6edf3"  # near-white


def make_axes(x_range, y_range, x_label, y_label, width=9, height=4.5):
    ax = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=width,
        y_length=height,
        axis_config={"color": C_TEXT, "stroke_width": 1.5,
                     "include_tip": False, "include_ticks": True},
        x_axis_config={"numbers_to_include": list(range(0, x_range[1]+1, 2)),
                       "font_size": 18},
        y_axis_config={"numbers_to_include": np.round(
            np.arange(y_range[0], y_range[1]+0.01, y_range[2]), 2),
                       "font_size": 18},
    )
    xl = ax.get_x_axis_label(Text(x_label, font_size=22, color=C_TEXT),
                             edge=DOWN, direction=DOWN, buff=0.3)
    yl = ax.get_y_axis_label(Text(y_label, font_size=22, color=C_TEXT),
                             edge=LEFT, direction=LEFT, buff=0.5)
    return ax, xl, yl


def draw_curve(ax, data, color, label_text, dot_scale=0.06):
    layers = list(range(len(data)))
    points = [ax.c2p(l, v) for l, v in zip(layers, data)]
    curve = VMobject(color=color, stroke_width=3)
    curve.set_points_smoothly(points)
    dots = VGroup(*[
        Dot(p, radius=dot_scale, color=color).set_z_index(2)
        for p in points
    ])
    label = Text(label_text, font_size=24, color=color)
    return curve, dots, label


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 1 — Title
# ═══════════════════════════════════════════════════════════════════════════════
class TitleScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text(
            "How Do Transformers Encode Go Syntax?",
            font_size=40, color=C_TEXT, weight=BOLD
        ).shift(UP * 1.2)

        subtitle = Text(
            "A Layer-wise Analysis: UniXcoder vs CodeBERT",
            font_size=28, color="#8b949e"
        ).next_to(title, DOWN, buff=0.5)

        uni_dot  = Dot(color=C_UNI, radius=0.12)
        uni_lbl  = Text("UniXcoder  (AST-augmented)", font_size=24, color=C_UNI)
        cb_dot   = Dot(color=C_CB, radius=0.12)
        cb_lbl   = Text("CodeBERT   (code-only)", font_size=24, color=C_CB)

        legend = VGroup(
            VGroup(uni_dot, uni_lbl).arrange(RIGHT, buff=0.2),
            VGroup(cb_dot,  cb_lbl ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(DOWN * 1.5)

        metrics = VGroup(
            Text("RQ2  ·  Attention–AST Alignment",  font_size=20, color="#8b949e"),
            Text("RQ3a ·  Structural Probing  (ρ)",  font_size=20, color="#8b949e"),
            Text("RQ3b ·  Tree Induction  (F1)",     font_size=20, color="#8b949e"),
        ).arrange(RIGHT, buff=0.6).shift(DOWN * 2.6)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=0.8)
        self.play(FadeIn(legend),  run_time=0.8)
        self.play(FadeIn(metrics), run_time=0.8)
        self.wait(2)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 2 — RQ2: Attention-AST Alignment
# ═══════════════════════════════════════════════════════════════════════════════
class RQ2Scene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("RQ2 · Attention–AST Alignment", font_size=32,
                     color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.35)

        ax, xl, yl = make_axes(
            x_range=[0, 11, 1], y_range=[0.3, 0.75, 0.1],
            x_label="Layer", y_label="Mean Alignment Score"
        )
        chart = VGroup(ax, xl, yl).shift(DOWN * 0.3)

        # Layer-7 vertical guide
        x7 = ax.c2p(7, 0.3)[0]
        y_bot, y_top = ax.c2p(7, 0.3)[1], ax.c2p(7, 0.75)[1]
        guide = DashedLine(
            start=[x7, y_bot, 0], end=[x7, y_top, 0],
            color=C_HIGH, stroke_width=2, dash_length=0.12
        )
        guide_lbl = Text("Layer 7", font_size=20, color=C_HIGH)\
            .move_to([x7, y_top + 0.25, 0])

        # Curves
        uni_curve, uni_dots, _ = draw_curve(ax, RQ2_UNIXCODER, C_UNI, "UniXcoder")
        cb_curve,  cb_dots,  _ = draw_curve(ax, RQ2_CODEBERT,  C_CB,  "CodeBERT")

        # Legend
        uni_leg = VGroup(Line(ORIGIN, RIGHT*0.5, color=C_UNI, stroke_width=3),
                         Text("UniXcoder", font_size=22, color=C_UNI)).arrange(RIGHT, buff=0.2)
        cb_leg  = VGroup(Line(ORIGIN, RIGHT*0.5, color=C_CB,  stroke_width=3),
                         Text("CodeBERT",  font_size=22, color=C_CB )).arrange(RIGHT, buff=0.2)
        legend  = VGroup(uni_leg, cb_leg).arrange(DOWN, aligned_edge=LEFT, buff=0.2)\
                       .to_corner(UR, buff=0.7).shift(DOWN * 0.8)

        # Peak dots highlighted
        uni_peak = Dot(ax.c2p(7, RQ2_UNIXCODER[7]), radius=0.1, color=C_HIGH).set_z_index(3)
        cb_peak  = Dot(ax.c2p(7, RQ2_CODEBERT[7]),  radius=0.1, color=C_HIGH).set_z_index(3)

        finding = Text(
            "Both models peak at Layer 7",
            font_size=24, color=C_HIGH
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(title), run_time=0.8)
        self.play(Create(ax), Write(xl), Write(yl), run_time=1.0)
        self.play(
            Create(uni_curve), FadeIn(uni_dots),
            Create(cb_curve),  FadeIn(cb_dots),
            run_time=2.5
        )
        self.play(FadeIn(legend), run_time=0.5)
        self.play(Create(guide), Write(guide_lbl), run_time=0.6)
        self.play(
            uni_peak.animate.scale(1.6),
            cb_peak.animate.scale(1.6),
            FadeIn(uni_peak), FadeIn(cb_peak),
            run_time=0.6
        )
        self.play(FadeIn(finding), run_time=0.6)
        self.wait(2.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 3 — RQ3a: Structural Probing
# ═══════════════════════════════════════════════════════════════════════════════
class RQ3ProbeScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("RQ3a · Structural Probing  (Spearman ρ)", font_size=32,
                     color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.35)

        ax, xl, yl = make_axes(
            x_range=[0, 12, 1], y_range=[0.25, 0.70, 0.1],
            x_label="Layer", y_label="Spearman ρ"
        )
        chart = VGroup(ax, xl, yl).shift(DOWN * 0.3)

        uni_curve, uni_dots, _ = draw_curve(ax, RQ3_PROBE_UNIXCODER, C_UNI, "UniXcoder")
        cb_curve,  cb_dots,  _ = draw_curve(ax, RQ3_PROBE_CODEBERT,  C_CB,  "CodeBERT")

        # Highlight UniXcoder's Layer 5 peak
        x5 = ax.c2p(5, 0.25)[0]
        y_bot, y_top = ax.c2p(5, 0.25)[1], ax.c2p(5, 0.70)[1]
        uni_guide = DashedLine([x5, y_bot, 0], [x5, y_top, 0],
                               color=C_UNI, stroke_width=2, dash_length=0.12)
        uni_guide_lbl = Text("Uni peak\nL5–L6", font_size=18, color=C_UNI)\
            .move_to([x5, y_top + 0.35, 0])

        # Highlight CodeBERT's Layer 0 peak
        x0 = ax.c2p(0, 0.25)[0]
        cb_guide = DashedLine([x0, y_bot, 0], [x0, y_top, 0],
                              color=C_CB, stroke_width=2, dash_length=0.12)
        cb_guide_lbl = Text("CB peak\nL0", font_size=18, color=C_CB)\
            .move_to([x0, y_top + 0.35, 0])

        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT*0.5, color=C_UNI, stroke_width=3),
                   Text("UniXcoder", font_size=22, color=C_UNI)).arrange(RIGHT, buff=0.2),
            VGroup(Line(ORIGIN, RIGHT*0.5, color=C_CB,  stroke_width=3),
                   Text("CodeBERT",  font_size=22, color=C_CB )).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UR, buff=0.7).shift(DOWN * 0.8)

        finding = Text(
            "UniXcoder refines syntax through layers  ·  CodeBERT encodes it from embeddings",
            font_size=20, color=C_HIGH
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(title), run_time=0.8)
        self.play(Create(ax), Write(xl), Write(yl), run_time=1.0)
        self.play(FadeIn(legend), run_time=0.4)

        # Animate UniXcoder first, then CodeBERT
        self.play(Create(uni_curve), FadeIn(uni_dots), run_time=2.0)
        self.play(Create(uni_guide), Write(uni_guide_lbl), run_time=0.6)
        self.play(Create(cb_curve),  FadeIn(cb_dots),  run_time=2.0)
        self.play(Create(cb_guide),  Write(cb_guide_lbl), run_time=0.6)
        self.play(FadeIn(finding), run_time=0.6)
        self.wait(2.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 4 — RQ3b: Tree Induction
# ═══════════════════════════════════════════════════════════════════════════════
class RQ3TreeScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("RQ3b · Tree Induction  (F1)", font_size=32,
                     color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.35)

        ax, xl, yl = make_axes(
            x_range=[0, 12, 1], y_range=[0.08, 0.38, 0.1],
            x_label="Layer", y_label="F1 Score"
        )
        chart = VGroup(ax, xl, yl).shift(DOWN * 0.3)

        uni_curve, uni_dots, _ = draw_curve(ax, RQ3_TREE_UNIXCODER, C_UNI, "UniXcoder")
        cb_curve,  cb_dots,  _ = draw_curve(ax, RQ3_TREE_CODEBERT,  C_CB,  "CodeBERT")

        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT*0.5, color=C_UNI, stroke_width=3),
                   Text("UniXcoder", font_size=22, color=C_UNI)).arrange(RIGHT, buff=0.2),
            VGroup(Line(ORIGIN, RIGHT*0.5, color=C_CB,  stroke_width=3),
                   Text("CodeBERT",  font_size=22, color=C_CB )).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UR, buff=0.7).shift(DOWN * 0.8)

        # Shaded region showing UniXcoder consistently higher
        fill_pts = []
        for i, (u, c) in enumerate(zip(RQ3_TREE_UNIXCODER, RQ3_TREE_CODEBERT)):
            fill_pts.append(ax.c2p(i, u))
        for i, (u, c) in reversed(list(enumerate(zip(RQ3_TREE_UNIXCODER, RQ3_TREE_CODEBERT)))):
            fill_pts.append(ax.c2p(i, c))
        fill = Polygon(*fill_pts, fill_color=C_UNI, fill_opacity=0.12,
                       stroke_width=0, color=C_UNI)

        finding = Text(
            "UniXcoder outperforms CodeBERT across all layers  (+AST pre-training advantage)",
            font_size=20, color=C_HIGH
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(title), run_time=0.8)
        self.play(Create(ax), Write(xl), Write(yl), run_time=1.0)
        self.play(FadeIn(legend), run_time=0.4)
        self.play(
            Create(uni_curve), FadeIn(uni_dots),
            Create(cb_curve),  FadeIn(cb_dots),
            run_time=2.5
        )
        self.play(FadeIn(fill), run_time=0.6)
        self.play(FadeIn(finding), run_time=0.6)
        self.wait(2.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 5 — Summary: The Divergence Story
# ═══════════════════════════════════════════════════════════════════════════════
class SummaryScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("Key Finding: Divergent Encoding Strategies",
                     font_size=34, color=C_TEXT, weight=BOLD).to_edge(UP, buff=0.4)

        # Two-column layout
        left_box  = RoundedRectangle(width=5.8, height=5.2, corner_radius=0.3,
                                     color=C_UNI, stroke_width=2, fill_opacity=0.06,
                                     fill_color=C_UNI).shift(LEFT * 3.2 + DOWN * 0.3)
        right_box = RoundedRectangle(width=5.8, height=5.2, corner_radius=0.3,
                                     color=C_CB,  stroke_width=2, fill_opacity=0.06,
                                     fill_color=C_CB ).shift(RIGHT * 3.2 + DOWN * 0.3)

        uni_header = Text("UniXcoder", font_size=28, color=C_UNI, weight=BOLD)\
            .next_to(left_box,  UP, buff=0).shift(DOWN * 0.5)
        cb_header  = Text("CodeBERT",  font_size=28, color=C_CB,  weight=BOLD)\
            .next_to(right_box, UP, buff=0).shift(DOWN * 0.5)

        uni_pts = VGroup(
            Text("✦  AST-augmented pre-training",   font_size=19, color=C_TEXT),
            Text("✦  Syntax builds layer-by-layer",  font_size=19, color=C_TEXT),
            Text("✦  RQ2 & RQ3 peak → Layer 5–7",   font_size=19, color=C_UNI),
            Text("✦  Progressive syntactic deepening",font_size=19, color=C_TEXT),
            Text("✦  Tree F1 consistently higher",   font_size=19, color=C_UNI),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)\
         .move_to(left_box.get_center()).shift(LEFT * 0.3)

        cb_pts = VGroup(
            Text("✦  Code-only pre-training",         font_size=19, color=C_TEXT),
            Text("✦  Syntax encoded in embeddings",   font_size=19, color=C_TEXT),
            Text("✦  RQ3 probing peaks → Layer 0",    font_size=19, color=C_CB),
            Text("✦  Attention aligns at Layer 7",    font_size=19, color=C_TEXT),
            Text("✦  No progressive refinement",      font_size=19, color=C_CB),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)\
         .move_to(right_box.get_center()).shift(LEFT * 0.3)

        conclusion = Text(
            "AST pre-training drives layer-wise syntactic refinement",
            font_size=23, color=C_HIGH
        ).to_edge(DOWN, buff=0.35)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(left_box), FadeIn(right_box), run_time=0.6)
        self.play(Write(uni_header), Write(cb_header), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(p) for p in uni_pts], lag_ratio=0.15), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(p) for p in cb_pts],  lag_ratio=0.15), run_time=1.5)
        self.play(FadeIn(conclusion), run_time=0.8)
        self.wait(3)


# ═══════════════════════════════════════════════════════════════════════════════
# Render all scenes in sequence
# Run with:
#   manim -pql go_syntax_animation.py TitleScene
#   manim -pql go_syntax_animation.py RQ2Scene
#   ...or render all:
#   manim -pql go_syntax_animation.py TitleScene RQ2Scene RQ3ProbeScene RQ3TreeScene SummaryScene
# For high quality replace -pql with -pqh
# ═══════════════════════════════════════════════════════════════════════════════
