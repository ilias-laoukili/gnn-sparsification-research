"""
Graph Laplacian Animation - Manim Community Edition
====================================================
Visualizes the Graph Laplacian (x^T L x) using Spring/Energy analogy.
Outputs PNG frames for LaTeX animate package integration.

Author: Ilias Laoukili
Project: GNN Sparsification Research - TREMPLIN 2025/2026

RENDER COMMANDS:
----------------
# Output PNG frames for LaTeX animate package:
manim -qh --format=png -o laplacian_frames laplacian_scene.py GraphLaplacianEnglish

# Or use the save_last_frame for individual frames:
manim -qh -s laplacian_scene.py GraphLaplacianEnglish

# For GIF output:
manim -qh --format=gif laplacian_scene.py GraphLaplacianEnglish

ALT-TEXT (Accessibility):
-------------------------
"This animation illustrates the Graph Laplacian concept through a physical spring 
analogy. A circular graph with 5 nodes is displayed, where each node has a scalar 
value represented by a colored vertical bar.

HIGH ENERGY PHASE: Nodes alternate between extreme values (+1, -1), creating large 
differences between neighbors. Edges glow magenta and vibrate like taut springs. 
The quadratic form x^T L x shows high energy.

DIFFUSION PHASE: Node values gradually equilibrate through neighbor averaging, 
simulating heat diffusion on the graph.

LOW ENERGY PHASE: All nodes converge to similar values. Edges relax to cyan, 
indicating no tension. The Laplacian energy approaches zero.

Key insight: The Graph Laplacian measures signal 'roughness' - smooth signals 
(similar neighbor values) have low quadratic energy."
"""

from manim import *
import numpy as np
import os

# =============================================================================
# CYBERPUNK COLOR PALETTE
# =============================================================================
DARK_BG = "#0A0E14"        # Deep blue-black background
CYAN_NEON = "#00F0FF"      # Stable/Low energy
MAGENTA_NEON = "#FF0055"   # Tension/High energy
LIGHT_TEXT = "#E6E6E6"     # Primary text
VIOLET_ACCENT = "#9D4EDD"  # Tertiary accent
GOLD_ACCENT = "#FFD700"    # Highlight


class GraphLaplacianEnglish(Scene):
    """
    Manim scene explaining Graph Laplacian and Quadratic Form
    via a spring/energy physical analogy.
    
    SLOW animation designed for PNG frame export and LaTeX animate integration.
    """
    
    def construct(self):
        # Set background color
        self.camera.background_color = DARK_BG
        
        # =================================================================
        # PHASE 0: SETUP - Create the graph structure
        # =================================================================
        
        # Title
        title = Text(
            "Graph Laplacian: L = D - A",
            font_size=42,
            color=CYAN_NEON,
            weight=BOLD
        ).to_edge(UP, buff=0.4)
        
        subtitle = Text(
            "Physical Analogy: Springs & Energy",
            font_size=28,
            color=LIGHT_TEXT
        ).next_to(title, DOWN, buff=0.15)
        
        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(1)
        
        # Create circular graph with 5 nodes
        n_nodes = 5
        radius = 2.0
        angles = [2 * PI * i / n_nodes - PI/2 for i in range(n_nodes)]
        
        # Node positions (circular layout)
        positions = [
            np.array([radius * np.cos(a), radius * np.sin(a) - 0.3, 0])
            for a in angles
        ]
        
        # Initial node values (HIGH ENERGY STATE: alternating +1, -1)
        initial_values = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        
        # Create node circles
        nodes = VGroup()
        node_labels = VGroup()
        value_bars = VGroup()
        value_texts = VGroup()
        
        for i, pos in enumerate(positions):
            # Node circle
            node = Circle(
                radius=0.35,
                color=CYAN_NEON,
                fill_opacity=0.3,
                stroke_width=3
            ).move_to(pos)
            nodes.add(node)
            
            # Node label (index)
            label = Text(
                f"v{i+1}",
                font_size=20,
                color=LIGHT_TEXT,
                weight=BOLD
            ).move_to(pos)
            node_labels.add(label)
            
            # Value bar (vertical bar showing the scalar value)
            bar_height = initial_values[i] * 0.6  # Scale for visibility
            bar_color = MAGENTA_NEON if initial_values[i] > 0 else CYAN_NEON
            
            bar = Rectangle(
                width=0.18,
                height=abs(bar_height),
                fill_color=bar_color,
                fill_opacity=0.85,
                stroke_width=1,
                stroke_color=WHITE
            )
            
            # Position bar above/below node based on sign
            bar_pos = pos + np.array([0.5, bar_height/2, 0])
            bar.move_to(bar_pos)
            value_bars.add(bar)
            
            # Value text
            val_text = Text(
                f"{initial_values[i]:+.1f}",
                font_size=16,
                color=bar_color,
                weight=BOLD
            ).next_to(bar, UP if initial_values[i] > 0 else DOWN, buff=0.08)
            value_texts.add(val_text)
        
        # Create edges (circular connectivity: each node connected to neighbors)
        edges = VGroup()
        edge_list = [(0,1), (1,2), (2,3), (3,4), (4,0)]  # Ring graph
        
        for i, j in edge_list:
            edge = Line(
                positions[i],
                positions[j],
                color=MAGENTA_NEON,
                stroke_width=5
            )
            edges.add(edge)
        
        # Animate graph appearance (SLOW)
        self.play(
            *[Create(edge) for edge in edges],
            run_time=2
        )
        self.play(
            *[Create(node) for node in nodes],
            run_time=2
        )
        self.play(
            *[FadeIn(label) for label in node_labels],
            run_time=1
        )
        self.play(
            *[GrowFromCenter(bar) for bar in value_bars],
            *[FadeIn(txt) for txt in value_texts],
            run_time=2
        )
        
        self.wait(1)
        
        # =================================================================
        # PHASE 1: HIGH ENERGY STATE - "CHAOS"
        # =================================================================
        
        phase1_label = Text(
            "⚡ HIGH LAPLACIAN ENERGY",
            font_size=32,
            color=MAGENTA_NEON,
            weight=BOLD
        ).to_edge(DOWN, buff=1.2)
        
        self.play(Write(phase1_label), run_time=1.5)
        
        # Quadratic form equation
        equation = MathTex(
            r"x^\top L x = \sum_{(u,v) \in E} (x_u - x_v)^2",
            font_size=36,
            color=LIGHT_TEXT
        ).next_to(phase1_label, UP, buff=0.25)
        
        self.play(Write(equation), run_time=2)
        
        # Calculate actual energy
        energy_high = sum((initial_values[i] - initial_values[j])**2 
                         for i, j in edge_list)
        
        energy_text = MathTex(
            f"= {energy_high:.0f}",
            font_size=42,
            color=MAGENTA_NEON
        ).next_to(equation, RIGHT, buff=0.3)
        
        self.play(Write(energy_text), run_time=1.5)
        
        # Emphasize high energy state
        self.play(
            *[edge.animate.set_stroke(width=8, color=MAGENTA_NEON) for edge in edges],
            equation.animate.scale(1.2),
            energy_text.animate.scale(1.3),
            run_time=1.5
        )
        
        # Vibration effect (SLOW pulsing)
        for _ in range(3):
            self.play(
                *[edge.animate.set_stroke(width=10) for edge in edges],
                rate_func=there_and_back,
                run_time=0.8
            )
        
        self.wait(2)
        
        # Reset scaling
        self.play(
            equation.animate.scale(1/1.2),
            energy_text.animate.scale(1/1.3),
            run_time=0.5
        )
        
        # =================================================================
        # PHASE 2: DIFFUSION - "EQUILIBRATION"
        # =================================================================
        
        # Transition label
        self.play(FadeOut(phase1_label), run_time=0.5)
        
        phase2_label = Text(
            "🌊 DIFFUSION: Heat Spreading",
            font_size=32,
            color=VIOLET_ACCENT,
            weight=BOLD
        ).to_edge(DOWN, buff=1.2)
        
        self.play(Write(phase2_label), run_time=1.5)
        
        # Diffusion process: values converge to mean (SLOW - many steps)
        n_steps = 8
        values = initial_values.copy()
        
        # Adjacency for ring graph
        adj = np.zeros((n_nodes, n_nodes))
        for i, j in edge_list:
            adj[i, j] = 1
            adj[j, i] = 1
        
        for step in range(n_steps):
            # Compute new values (averaging with neighbors)
            new_values = values.copy()
            for i in range(n_nodes):
                neighbors = np.where(adj[i] > 0)[0]
                # Diffusion: move toward neighbor average
                new_values[i] = 0.6 * values[i] + 0.4 * np.mean(values[neighbors])
            values = new_values
            
            # Animate value bar changes
            animations = []
            for i in range(n_nodes):
                bar_height = values[i] * 0.6
                bar_color = self.interpolate_color(values[i])
                
                new_bar = Rectangle(
                    width=0.18,
                    height=max(abs(bar_height), 0.03),  # Min height for visibility
                    fill_color=bar_color,
                    fill_opacity=0.85,
                    stroke_width=1,
                    stroke_color=WHITE
                )
                bar_pos = positions[i] + np.array([0.5, bar_height/2 if bar_height > 0 else bar_height/2, 0])
                new_bar.move_to(bar_pos)
                
                new_text = Text(
                    f"{values[i]:+.2f}",
                    font_size=16,
                    color=bar_color,
                    weight=BOLD
                ).next_to(new_bar, UP if values[i] >= 0 else DOWN, buff=0.08)
                
                animations.append(Transform(value_bars[i], new_bar))
                animations.append(Transform(value_texts[i], new_text))
            
            # Update edge colors (interpolate based on tension)
            for idx, (i, j) in enumerate(edge_list):
                tension = abs(values[i] - values[j])
                edge_color = self.interpolate_edge_color(tension, max_tension=2.0)
                edge_width = 5 + 3 * (tension / 2.0)  # Thicker = more tension
                animations.append(
                    edges[idx].animate.set_color(edge_color).set_stroke(width=edge_width)
                )
            
            # Update energy display
            energy = sum((values[i] - values[j])**2 for i, j in edge_list)
            new_energy_text = MathTex(
                f"= {energy:.2f}",
                font_size=42,
                color=self.interpolate_color(energy / 20)  # Normalize
            ).next_to(equation, RIGHT, buff=0.3)
            animations.append(Transform(energy_text, new_energy_text))
            
            # SLOW animation per step
            self.play(*animations, run_time=1.5)
        
        self.wait(1)
        
        # =================================================================
        # PHASE 3: LOW ENERGY STATE - "HARMONY"
        # =================================================================
        
        self.play(FadeOut(phase2_label), run_time=0.5)
        
        phase3_label = Text(
            "✨ LOW LAPLACIAN ENERGY",
            font_size=32,
            color=CYAN_NEON,
            weight=BOLD
        ).to_edge(DOWN, buff=1.2)
        
        self.play(Write(phase3_label), run_time=1.5)
        
        # Final convergence to uniform state
        final_value = np.mean(initial_values)  # Should be ~0.2
        
        final_animations = []
        for i in range(n_nodes):
            bar_height = final_value * 0.6
            
            new_bar = Rectangle(
                width=0.18,
                height=max(abs(bar_height), 0.03),
                fill_color=CYAN_NEON,
                fill_opacity=0.85,
                stroke_width=1,
                stroke_color=WHITE
            )
            bar_pos = positions[i] + np.array([0.5, bar_height/2, 0])
            new_bar.move_to(bar_pos)
            
            new_text = Text(
                f"{final_value:+.2f}",
                font_size=16,
                color=CYAN_NEON,
                weight=BOLD
            ).next_to(new_bar, UP, buff=0.08)
            
            final_animations.append(Transform(value_bars[i], new_bar))
            final_animations.append(Transform(value_texts[i], new_text))
        
        # All edges become cyan (relaxed)
        for edge in edges:
            final_animations.append(
                edge.animate.set_color(CYAN_NEON).set_stroke(width=4)
            )
        
        # Energy goes to 0
        final_energy = MathTex(
            r"\to 0",
            font_size=48,
            color=CYAN_NEON
        ).next_to(equation, RIGHT, buff=0.3)
        
        final_animations.append(Transform(energy_text, final_energy))
        
        # SLOW final transition
        self.play(*final_animations, run_time=3)
        
        # Glow effect on stable state
        self.play(
            *[edge.animate.set_stroke(width=6) for edge in edges],
            *[node.animate.set_fill(CYAN_NEON, opacity=0.5) for node in nodes],
            run_time=1.5
        )
        
        self.wait(1)
        
        # =================================================================
        # CONCLUSION
        # =================================================================
        
        conclusion = Text(
            "The Laplacian measures signal smoothness on the graph",
            font_size=26,
            color=GOLD_ACCENT,
            weight=BOLD
        ).to_edge(DOWN, buff=0.4)
        
        self.play(
            FadeOut(phase3_label),
            Write(conclusion),
            run_time=2
        )
        
        self.wait(3)
        
        # Fade out everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=2
        )
    
    def interpolate_color(self, value, min_val=-1, max_val=1):
        """Interpolate between cyan (low) and magenta (high) based on value."""
        t = (value - min_val) / (max_val - min_val)
        t = np.clip(t, 0, 1)
        
        cyan = np.array([0, 240/255, 255/255])
        magenta = np.array([255/255, 0, 85/255])
        
        rgb = (1 - t) * cyan + t * magenta
        return rgb_to_color(rgb)
    
    def interpolate_edge_color(self, tension, max_tension=2.0):
        """Interpolate edge color based on tension."""
        t = min(tension / max_tension, 1.0)
        
        cyan = np.array([0, 240/255, 255/255])
        magenta = np.array([255/255, 0, 85/255])
        
        rgb = (1 - t) * cyan + t * magenta
        return rgb_to_color(rgb)


# =============================================================================
# FRAME-BY-FRAME EXPORT VERSION (for LaTeX animate package)
# =============================================================================
class GraphLaplacianFrames(Scene):
    """
    Simplified version that outputs clean frames for animate package.
    Designed to produce ~60 frames at 10fps = 6 seconds of animation.
    """
    
    def construct(self):
        self.camera.background_color = DARK_BG
        
        # Graph setup
        n_nodes = 5
        radius = 2.2
        angles = [2 * PI * i / n_nodes - PI/2 for i in range(n_nodes)]
        positions = [
            np.array([radius * np.cos(a), radius * np.sin(a), 0])
            for a in angles
        ]
        
        # Create nodes
        nodes = VGroup()
        for pos in positions:
            node = Circle(radius=0.4, color=CYAN_NEON, fill_opacity=0.3, stroke_width=4)
            node.move_to(pos)
            nodes.add(node)
        
        # Create edges
        edges = VGroup()
        edge_list = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for i, j in edge_list:
            edge = Line(positions[i], positions[j], color=MAGENTA_NEON, stroke_width=6)
            edges.add(edge)
        
        # Value indicators (height bars)
        initial_values = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        bars = VGroup()
        
        for i, pos in enumerate(positions):
            height = initial_values[i] * 0.8
            color = MAGENTA_NEON if initial_values[i] > 0 else CYAN_NEON
            bar = Rectangle(
                width=0.2, 
                height=abs(height),
                fill_color=color,
                fill_opacity=0.9,
                stroke_width=0
            )
            bar.move_to(pos + np.array([0.6, height/2, 0]))
            bars.add(bar)
        
        # Title
        title = Text("Graph Laplacian Energy", font_size=36, color=CYAN_NEON, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        # Energy label
        energy_label = Text("HIGH ENERGY", font_size=28, color=MAGENTA_NEON, weight=BOLD)
        energy_label.to_edge(DOWN, buff=0.8)
        
        # Show initial state
        self.add(title, nodes, edges, bars, energy_label)
        self.wait(2)  # Hold for frames
        
        # Diffusion animation
        adj = np.zeros((n_nodes, n_nodes))
        for i, j in edge_list:
            adj[i, j] = 1
            adj[j, i] = 1
        
        values = initial_values.copy()
        n_steps = 20
        
        for step in range(n_steps):
            # Update values
            new_values = values.copy()
            for i in range(n_nodes):
                neighbors = np.where(adj[i] > 0)[0]
                new_values[i] = 0.7 * values[i] + 0.3 * np.mean(values[neighbors])
            values = new_values
            
            # Update bars
            new_bars = VGroup()
            for i, pos in enumerate(positions):
                height = values[i] * 0.8
                t = abs(values[i])
                color = self.blend_color(t)
                bar = Rectangle(
                    width=0.2,
                    height=max(abs(height), 0.05),
                    fill_color=color,
                    fill_opacity=0.9,
                    stroke_width=0
                )
                bar.move_to(pos + np.array([0.6, height/2 if height > 0 else height/2, 0]))
                new_bars.add(bar)
            
            # Update edges
            new_edges = VGroup()
            for idx, (i, j) in enumerate(edge_list):
                tension = abs(values[i] - values[j]) / 2.0
                color = self.blend_color(tension)
                edge = Line(positions[i], positions[j], color=color, stroke_width=6)
                new_edges.add(edge)
            
            # Update energy label
            progress = step / n_steps
            if progress < 0.3:
                label_text = "HIGH ENERGY"
                label_color = MAGENTA_NEON
            elif progress < 0.7:
                label_text = "DIFFUSING..."
                label_color = VIOLET_ACCENT
            else:
                label_text = "LOW ENERGY"
                label_color = CYAN_NEON
            
            new_label = Text(label_text, font_size=28, color=label_color, weight=BOLD)
            new_label.to_edge(DOWN, buff=0.8)
            
            # Animate transition
            self.play(
                Transform(bars, new_bars),
                Transform(edges, new_edges),
                Transform(energy_label, new_label),
                run_time=0.3
            )
        
        # Final stable state
        final_label = Text("EQUILIBRIUM REACHED", font_size=28, color=CYAN_NEON, weight=BOLD)
        final_label.to_edge(DOWN, buff=0.8)
        
        self.play(
            *[node.animate.set_fill(CYAN_NEON, opacity=0.5) for node in nodes],
            Transform(energy_label, final_label),
            run_time=1
        )
        
        self.wait(2)
    
    def blend_color(self, t):
        """Blend from cyan (t=0) to magenta (t=1)."""
        t = np.clip(t, 0, 1)
        cyan = np.array([0, 240/255, 255/255])
        magenta = np.array([255/255, 0, 85/255])
        rgb = (1 - t) * cyan + t * magenta
        return rgb_to_color(rgb)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              GRAPH LAPLACIAN ANIMATION - ENGLISH VERSION                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FOR VIDEO OUTPUT (MP4):                                                    ║
║    manim -qh laplacian_scene.py GraphLaplacianEnglish                       ║
║                                                                              ║
║  FOR GIF OUTPUT:                                                            ║
║    manim -qh --format=gif laplacian_scene.py GraphLaplacianEnglish          ║
║                                                                              ║
║  FOR PNG FRAMES (LaTeX animate package):                                    ║
║    manim -qh --format=png laplacian_scene.py GraphLaplacianFrames           ║
║                                                                              ║
║  The PNG frames will be in: media/images/laplacian_scene/                   ║
║  Copy them to presentation/laplacian_frames/ for LaTeX integration.         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
