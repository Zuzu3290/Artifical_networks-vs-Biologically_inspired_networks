"""
snn_visualisation.py
Three independent matplotlib figures, each in its own window.

Figure 1 — Neuron anatomy diagrams (ANN vs LIF)
            - ANN: pixel grid → weight matrix → soma → dense activation
            - LIF: spike trains on dendrites → integrate → threshold → reset
Figure 2 — Population activation bar chart (cumulative firing)
Figure 3 — Information flow + fixed LIF membrane that actually crosses threshold
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

np.random.seed(7)

BG      = "#0d1117"
CARD_L  = "#161b22"
CARD_R  = "#0d2018"
C_ANN   = "#4a9eff"
C_SNN   = "#22c55e"
C_SPIKE = "#f97316"
C_MEM   = "#a78bfa"
C_TEXT  = "#e6edf3"
C_DIM   = "#8b949e"
C_W_POS = "#f97316"   # warm orange for positive weights
C_W_NEG = "#4a9eff"   # cool blue for negative weights


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Neuron anatomy: information format illustrated
# ══════════════════════════════════════════════════════════════════
def fig1_neurons():
    fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle("Neuron Architecture & Information Format",
                 color=C_TEXT, fontsize=15, fontweight="bold", y=0.98)

    for ax in (ax_a, ax_s):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_aspect("equal")
        ax.axis("off")

    ax_a.set_facecolor(CARD_L)
    ax_s.set_facecolor(CARD_R)

    ax_a.text(5, 11.5, "Traditional ANN Neuron",
              ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    ax_a.text(5, 11.1, "processes dense pixel values → real-valued activation",
              ha="center", color=C_DIM, fontsize=8)

    ax_s.text(5, 11.5, "Biologically Inspired LIF Neuron (SNN)",
              ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    ax_s.text(5, 11.1, "processes sparse spike trains → binary spike events",
              ha="center", color=C_DIM, fontsize=8)


    # ──────────────────────────────────────────────────────────────
    # ANN SIDE
    # ──────────────────────────────────────────────────────────────

    # ── 1. Pixel grid input (6×6) ─────────────────────────────────
    pixel_values = np.array([
        [0.12, 0.65, 0.82, 0.90, 0.45, 0.10],
        [0.20, 0.78, 0.95, 0.88, 0.55, 0.18],
        [0.08, 0.50, 0.72, 0.60, 0.35, 0.08],
        [0.15, 0.30, 0.42, 0.38, 0.22, 0.12],
        [0.65, 0.80, 0.55, 0.40, 0.70, 0.50],
        [0.88, 0.60, 0.30, 0.20, 0.55, 0.78],
    ])
    grid_x0, grid_y0 = 0.3, 7.8
    cell = 0.45
    for r in range(6):
        for c in range(6):
            v = pixel_values[r, c]
            rect = Rectangle(
                (grid_x0 + c * cell, grid_y0 - r * cell),
                cell, cell,
                linewidth=0.4, edgecolor=C_ANN,
                facecolor=plt.cm.Blues(0.25 + 0.75 * v),
                alpha=0.92, zorder=3
            )
            ax_a.add_patch(rect)

    # Outer border
    ax_a.add_patch(Rectangle(
        (grid_x0, grid_y0 - 5 * cell), 6 * cell, 6 * cell,
        linewidth=1.0, edgecolor=C_ANN, facecolor="none",
        alpha=0.7, zorder=4
    ))
    ax_a.text(grid_x0 + 3 * cell, grid_y0 + 0.35, "pixel patch",
              ha="center", color=C_ANN, fontsize=8)

    # ── 2. Weight matrix (4×4) ────────────────────────────────────
    weights = np.array([
        [ 0.75, -0.20,  0.50, -0.35],
        [-0.15,  0.90,  0.80, -0.10],
        [ 0.30,  0.65, -0.40,  0.45],
        [-0.25, -0.50,  0.55, -0.30],
    ])
    wm_x0, wm_y0 = 3.6, 7.7
    wc = 0.46
    for r in range(4):
        for c in range(4):
            w = weights[r, c]
            color = C_W_POS if w > 0 else C_W_NEG
            rect = Rectangle(
                (wm_x0 + c * wc, wm_y0 - r * wc),
                wc, wc,
                linewidth=0.4, edgecolor="#333",
                facecolor=color, alpha=abs(w) * 0.9 + 0.05,
                zorder=3
            )
            ax_a.add_patch(rect)
    ax_a.add_patch(Rectangle(
        (wm_x0, wm_y0 - 3 * wc), 4 * wc, 4 * wc,
        linewidth=1.0, edgecolor=C_ANN, facecolor="none",
        alpha=0.6, zorder=4
    ))
    ax_a.text(wm_x0 + 2 * wc, wm_y0 + 0.35, "weight matrix",
              ha="center", color=C_ANN, fontsize=8)
    # Legend: warm/cool
    ax_a.add_patch(Rectangle((wm_x0, wm_y0 - 4 * wc - 0.28), 0.25, 0.18,
                              facecolor=C_W_POS, alpha=0.75,
                              edgecolor="none", zorder=5))
    ax_a.text(wm_x0 + 0.30, wm_y0 - 4 * wc - 0.19, "pos", color=C_DIM, fontsize=7)
    ax_a.add_patch(Rectangle((wm_x0 + 0.75, wm_y0 - 4 * wc - 0.28), 0.25, 0.18,
                              facecolor=C_W_NEG, alpha=0.55,
                              edgecolor="none", zorder=5))
    ax_a.text(wm_x0 + 1.05, wm_y0 - 4 * wc - 0.19, "neg", color=C_DIM, fontsize=7)

    # Arrow: pixel grid → weight matrix
    ax_a.annotate("", xy=(wm_x0 - 0.05, wm_y0 - 1.5 * wc),
                  xytext=(grid_x0 + 6 * cell + 0.05, grid_y0 - 2.5 * cell),
                  arrowprops=dict(arrowstyle="-|>", color=C_ANN,
                                 lw=1.4, mutation_scale=10))

    # ── 3. Dendrites from weight matrix → soma ────────────────────
    soma_cx, soma_cy, soma_r = 7.2, 7.4, 0.55
    dend_x_start = wm_x0 + 4 * wc + 0.05
    dend_y_starts = [wm_y0 - 0.25 * wc, wm_y0 - 1.25 * wc,
                     wm_y0 - 2.25 * wc, wm_y0 - 3.25 * wc]
    dend_y_ends   = [soma_cy + 0.22, soma_cy + 0.10,
                     soma_cy - 0.10, soma_cy - 0.22]
    for ys, ye in zip(dend_y_starts, dend_y_ends):
        ax_a.annotate("", xy=(soma_cx - soma_r, ye),
                      xytext=(dend_x_start, ys),
                      arrowprops=dict(arrowstyle="-|>", color=C_ANN,
                                     lw=1.5, mutation_scale=9, alpha=0.8))

    # ── 4. ANN soma ───────────────────────────────────────────────
    soma_a = Circle((soma_cx, soma_cy), soma_r,
                    fc="#1a3a5c", ec=C_ANN, lw=2.0, alpha=0.95, zorder=5)
    ax_a.add_patch(soma_a)
    ax_a.text(soma_cx, soma_cy + 0.15, "∑", ha="center", va="center",
              fontsize=16, color="white", fontweight="bold", zorder=6)
    ax_a.text(soma_cx, soma_cy - 0.22, "ReLU(z)", ha="center", va="center",
              fontsize=8, color=C_ANN, alpha=0.85, zorder=6)

    # ── 5. Axon → dense activation bar ───────────────────────────
    ax_a.annotate("", xy=(soma_cx + soma_r + 0.55, soma_cy),
                  xytext=(soma_cx + soma_r, soma_cy),
                  arrowprops=dict(arrowstyle="-|>", color=C_ANN,
                                 lw=1.8, mutation_scale=12))
    # Dense bar: full height = dense continuous output
    bar_x = soma_cx + soma_r + 0.6
    ax_a.add_patch(Rectangle((bar_x, soma_cy - 0.38), 0.22, 0.75,
                              fc=C_ANN, ec="white", lw=0.8, alpha=0.85, zorder=5))
    ax_a.text(bar_x + 0.11, soma_cy + 0.50, "a = ReLU(z)",
              ha="center", color=C_ANN, fontsize=7.5, zorder=6)
    ax_a.text(bar_x + 0.11, soma_cy - 0.55, "dense\nactivation",
              ha="center", color=C_DIM, fontsize=7, zorder=6)

    # ── 6. ANN output signal over time (dense waveform) ──────────
    t = np.linspace(0, 10, 200)
    ann_sig = 0.62 + 0.20 * np.sin(t * 0.9) + 0.10 * np.sin(t * 2.1)
    ann_sig = np.clip(ann_sig, 0.1, 1.0)

    # Place the mini-axes as an inset
    ax_inset_a = ax_a.inset_axes([0.03, 0.03, 0.94, 0.25])
    ax_inset_a.set_facecolor("#0d1117")
    ax_inset_a.fill_between(t, 0, ann_sig, color=C_ANN, alpha=0.25)
    ax_inset_a.plot(t, ann_sig, color=C_ANN, lw=1.8)
    ax_inset_a.axhline(0, color=C_DIM, lw=0.5, alpha=0.4)
    ax_inset_a.set_yticks([0, 0.5, 1.0])
    ax_inset_a.set_yticklabels(["0", "0.5", "1.0"], fontsize=7, color=C_DIM)
    ax_inset_a.set_xticks([])
    ax_inset_a.tick_params(axis="y", colors=C_DIM, length=2)
    for sp in ax_inset_a.spines.values():
        sp.set_color(C_DIM); sp.set_linewidth(0.4)
    ax_inset_a.set_ylabel("activation", fontsize=7, color=C_DIM, labelpad=2)
    ax_inset_a.set_xlabel("timestep →", fontsize=7, color=C_DIM, labelpad=2)
    ax_inset_a.text(5, 0.85, "fires every timestep (dense)",
                    ha="center", color=C_ANN, fontsize=7.5, fontstyle="italic")
    ax_inset_a.set_ylim(-0.05, 1.15)

    # ── 7. ANN annotation caption ────────────────────────────────
    ax_a.text(5, 3.05,
              "Input: continuous pixel intensities   |   "
              "Format: real-valued numbers\n"
              "z = Σ wᵢxᵢ (dot product)   →   a = ReLU(z)   →   always outputs a value",
              ha="center", color=C_DIM, fontsize=7.8,
              bbox=dict(fc=BG, ec=C_ANN, lw=0.8, pad=5, alpha=0.6))


    # ──────────────────────────────────────────────────────────────
    # SNN SIDE
    # ──────────────────────────────────────────────────────────────

    # ── 1. Spike trains on 5 dendrites ───────────────────────────
    # Each dendrite: horizontal baseline + vertical tick marks for spikes
    # e1: 2 spikes, e2: silent, e3: 1 spike, e4: silent, e5: 3 spikes
    spike_patterns = [
        [2.0, 5.5],          # e1
        [],                   # e2  (silent)
        [4.2],               # e3
        [],                   # e4  (silent)
        [1.0, 4.8, 7.5],    # e5
    ]
    dend_ys   = [9.6, 8.9, 8.2, 7.5, 6.8]
    dend_x0_s = 0.4
    dend_x1_s = 4.8
    tick_h    = 0.28

    for i, (dy, spks) in enumerate(zip(dend_ys, spike_patterns)):
        active = len(spks) > 0
        color  = C_SNN if active else C_DIM
        alpha  = 0.85 if active else 0.30
        # Baseline rail
        ax_s.plot([dend_x0_s, dend_x1_s], [dy, dy],
                  color=color, lw=0.8, alpha=alpha * 0.6, zorder=2)
        # Spike ticks
        for st in spks:
            sx = dend_x0_s + st / 9.0 * (dend_x1_s - dend_x0_s)
            ax_s.plot([sx, sx], [dy - tick_h, dy + tick_h],
                      color=C_SNN, lw=2.0, alpha=0.9, zorder=4)
            ax_s.plot(sx, dy + tick_h, "o",
                      ms=2.5, color=C_SNN, alpha=0.8, zorder=5)
        # Label
        label = f"e{i+1}"
        ax_s.text(dend_x0_s - 0.15, dy, label, ha="right", va="center",
                  color=color, fontsize=8.5, alpha=alpha)

    # Label for spike train region
    ax_s.text(dend_x0_s + (dend_x1_s - dend_x0_s) / 2, 10.1,
              "spike trains  (binary, sparse)", ha="center",
              color=C_SNN, fontsize=8)
    # time axis label
    ax_s.annotate("", xy=(dend_x1_s, 6.4), xytext=(dend_x0_s, 6.4),
                  arrowprops=dict(arrowstyle="-|>", color=C_DIM,
                                 lw=0.8, mutation_scale=8, alpha=0.4))
    ax_s.text((dend_x0_s + dend_x1_s) / 2, 6.2, "time →",
              ha="center", color=C_DIM, fontsize=7, alpha=0.5)

    # ── 2. Convergence lines → SNN soma ──────────────────────────
    soma_sx, soma_sy, soma_sr = 6.4, 8.2, 0.58
    for i, dy in enumerate(dend_ys):
        active = len(spike_patterns[i]) > 0
        col = C_SNN if active else C_DIM
        alp = 0.75 if active else 0.20
        lw  = 1.5  if active else 0.8
        ax_s.annotate("", xy=(soma_sx - soma_sr, soma_sy + (dy - 8.2) * 0.25),
                      xytext=(dend_x1_s, dy),
                      arrowprops=dict(arrowstyle="-|>", color=col,
                                     lw=lw, mutation_scale=8, alpha=alp))

    # ── 3. LIF soma ───────────────────────────────────────────────
    soma_s = Circle((soma_sx, soma_sy), soma_sr,
                    fc="#0a2a1a", ec=C_SNN, lw=2.0, alpha=0.95, zorder=5)
    ax_s.add_patch(soma_s)
    ax_s.text(soma_sx, soma_sy + 0.16, "∫ Uₘ", ha="center", va="center",
              fontsize=13, color="white", fontweight="bold", zorder=6)
    ax_s.text(soma_sx, soma_sy - 0.22, "≥ θ → spike", ha="center", va="center",
              fontsize=7.5, color=C_SNN, alpha=0.85, zorder=6)

    # ── 4. Output: single spike tick ─────────────────────────────
    ax_s.annotate("", xy=(soma_sx + soma_sr + 0.55, soma_sy),
                  xytext=(soma_sx + soma_sr, soma_sy),
                  arrowprops=dict(arrowstyle="-|>", color=C_SNN,
                                 lw=1.8, mutation_scale=12))
    out_x = soma_sx + soma_sr + 0.60
    ax_s.plot([out_x, out_x], [soma_sy - 0.38, soma_sy + 0.38],
              color=C_SNN, lw=3.0, alpha=0.9, zorder=5)
    ax_s.plot(out_x, soma_sy + 0.38, "o", ms=5, color=C_SNN, zorder=6)
    ax_s.text(out_x + 0.18, soma_sy + 0.52, "spike event",
              color=C_SNN, fontsize=7.5, zorder=6)
    ax_s.text(out_x + 0.18, soma_sy - 0.55, "binary\noutput",
              ha="left", color=C_DIM, fontsize=7, zorder=6)

    # ── 5. LIF membrane potential trace (biologically grounded) ──
    # −70 mV = resting, −55 mV = threshold
    # Integrate → cross threshold → spike → reset → repeat
    T_lif  = 200
    dt     = 0.1       # ms per step
    tau    = 10.0      # ms membrane time constant
    V_rest = -70.0     # mV
    V_thr  = -55.0     # mV
    V_spk  =  30.0     # mV (spike peak)
    V_rst  = -75.0     # mV (after reset, brief undershoot)

    # Input current pulses: burst at t=5–15 ms, t=30–40 ms, t=60–75 ms
    time_ms = np.arange(T_lif) * dt
    I_ext   = np.zeros(T_lif)
    # Burst 1 (crosses threshold)
    I_ext[50:150]  += 1.85    # 5–15 ms
    # Brief gap (sub-threshold)
    I_ext[160:195] += 0.90    # partial
    # Burst 2
    I_ext[350:500] += 2.00
    # Burst 3
    I_ext[600:750] += 1.80

    V = np.zeros(T_lif)
    V[0] = V_rest
    spike_times_idx = []
    in_reset = 0   # refractory counter (steps)

    for i in range(1, T_lif):
        if in_reset > 0:
            V[i] = V_rst + (V_rest - V_rst) * (1 - in_reset / 15.0)
            in_reset -= 1
        else:
            dV = (-(V[i-1] - V_rest) + I_ext[i] * 10) / tau
            V[i] = V[i-1] + dV * dt
            if V[i] >= V_thr:
                spike_times_idx.append(i)
                V[i] = V_spk
                in_reset = 15

    ax_inset_s = ax_s.inset_axes([0.03, 0.03, 0.94, 0.32])
    ax_inset_s.set_facecolor("#0d1117")

    # Plot membrane in segments (suppress spike artifacts as vertical lines)
    seg_start = 0
    for i in range(1, T_lif):
        if i in spike_times_idx:
            ax_inset_s.plot(time_ms[seg_start:i], V[seg_start:i],
                            color=C_MEM, lw=1.6, alpha=0.9)
            # Draw spike as narrow vertical line
            ax_inset_s.plot([time_ms[i], time_ms[i]], [V_thr, V_spk],
                            color=C_SPIKE, lw=1.8, alpha=0.85)
            ax_inset_s.plot(time_ms[i], V_spk, "o",
                            ms=4, color=C_SPIKE, zorder=5)
            seg_start = i + 1
        elif i == T_lif - 1:
            ax_inset_s.plot(time_ms[seg_start:], V[seg_start:],
                            color=C_MEM, lw=1.6, alpha=0.9)

    # Threshold and resting lines
    ax_inset_s.axhline(V_thr, color=C_SNN, lw=1.0, ls="--",
                        alpha=0.8, label=f"θ = {V_thr} mV")
    ax_inset_s.axhline(V_rest, color=C_DIM, lw=0.8, ls=":",
                        alpha=0.5, label=f"rest = {V_rest} mV")

    # Spike ticks on top of plot
    for idx in spike_times_idx:
        ax_inset_s.axvline(time_ms[idx], ymin=0.88, ymax=1.0,
                           color=C_SPIKE, lw=1.5, alpha=0.7)

    ax_inset_s.set_yticks([V_rst, V_rest, V_thr])
    ax_inset_s.set_yticklabels([f"{V_rst:.0f}", f"{V_rest:.0f}", f"{V_thr:.0f}"],
                                fontsize=6.5, color=C_DIM)
    ax_inset_s.set_xticks([])
    ax_inset_s.tick_params(axis="y", colors=C_DIM, length=2)
    for sp in ax_inset_s.spines.values():
        sp.set_color(C_DIM); sp.set_linewidth(0.4)
    ax_inset_s.set_ylabel("Uₘ (mV)", fontsize=7, color=C_DIM, labelpad=2)
    ax_inset_s.set_xlabel("timestep →", fontsize=7, color=C_DIM, labelpad=2)
    ax_inset_s.set_ylim(V_rst - 5, V_spk + 10)

    n_spikes = len(spike_times_idx)
    ax_inset_s.text(time_ms[-1] * 0.5, V_spk + 2,
                    f"fires only {n_spikes}× — most time: silent",
                    ha="center", color=C_SNN, fontsize=7.5, fontstyle="italic")

    # Right-side labels for threshold and resting
    ax_inset_s.text(time_ms[-1] * 1.01, V_thr, "θ = −55 mV",
                    va="center", color=C_SNN, fontsize=6.5)
    ax_inset_s.text(time_ms[-1] * 1.01, V_rest, "−70 mV (rest)",
                    va="center", color=C_DIM, fontsize=6.5)

    # ── 6. SNN annotation caption ─────────────────────────────────
    ax_s.text(5, 3.05,
              "Input: binary spike trains (0 or 1 per timestep)   |   "
              "Format: timing of events\n"
              "Uₘ integrates spikes → fires when Uₘ ≥ −55 mV → resets to −70 mV",
              ha="center", color=C_DIM, fontsize=7.8,
              bbox=dict(fc=BG, ec=C_SNN, lw=0.8, pad=5, alpha=0.6))

    # ── Shared legend ─────────────────────────────────────────────
    legend_els = [
        mpatches.Patch(fc=plt.cm.Blues(0.7), ec=C_ANN, label="pixel intensity"),
        mpatches.Patch(fc=C_W_POS, alpha=0.75,  ec="none", label="positive weight"),
        mpatches.Patch(fc=C_W_NEG, alpha=0.55,  ec="none", label="negative weight"),
        Line2D([0], [0], color=C_SNN, lw=2.0, label="spike event"),
        Line2D([0], [0], color=C_MEM, lw=2.0, label="membrane Uₘ"),
        Line2D([0], [0], color=C_SPIKE, lw=0, marker="o",
               markersize=6, label="spike peak + reset"),
        Line2D([0], [0], color=C_SNN, lw=1.2, ls="--", label="threshold θ"),
        Line2D([0], [0], color=C_DIM,  lw=0.8, ls=":", label="resting potential"),
    ]
    fig.legend(handles=legend_els, loc="lower center",
               ncol=4, facecolor=BG, edgecolor=C_DIM,
               labelcolor=C_TEXT, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(pad=1.5, rect=[0, 0.06, 1, 1])
    fig.savefig("figure1_neuron_anatomy.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("Figure 1 saved.")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Population activation bar chart
# ══════════════════════════════════════════════════════════════════
def fig2_population():
    N = 10
    neurons = [f"N{i+1}" for i in range(N)]

    ann_acts = np.clip(0.90 + 0.07 * np.random.randn(N), 0.70, 1.0)
    firing   = [2, 5]                      # N3 and N6 fire
    snn_acts = np.zeros(N)
    for i in firing:
        snn_acts[i] = np.random.uniform(0.90, 1.0)

    fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(14, 6),
                                      facecolor=BG)
    fig.suptitle("Neuron Population Activation — Cumulative View",
                 color=C_TEXT, fontsize=15, fontweight="bold", y=0.97)

    for ax, acts, col, bg, title, note in [
        (ax_a, ann_acts, C_ANN,   CARD_L,
         "Traditional Neural Network",
         f"All {N} neurons active every step, always"),
        (ax_s, snn_acts, C_SNN,   CARD_R,
         "Biologically Inspired Network (SNN)",
         f"Only {len(firing)} out of {N} neurons fire — "
         f"{int((1-len(firing)/N)*100)}% of work skipped"),
    ]:
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_color(C_DIM); sp.set_linewidth(0.5)
        ax.tick_params(colors=C_DIM, labelsize=9)

        bar_cols  = [col if v > 0 else C_DIM for v in acts]
        bar_alpha = [0.9 if v > 0 else 0.2  for v in acts]
        bars = ax.bar(neurons, acts * 100, color=bar_cols, edgecolor="white",
                      linewidth=0.6, width=0.6)
        for bar, alph in zip(bars, bar_alpha):
            bar.set_alpha(alph)

        for bar, val in zip(bars, acts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.5,
                        f"{val*100:.0f}%",
                        ha="center", va="bottom",
                        color=col, fontsize=8, fontweight="bold")

        ax.set_ylim(0, 120)
        ax.set_xlabel("Neuron", color=C_DIM, fontsize=10)
        ax.set_ylabel("Activation (%)", color=C_DIM, fontsize=10)
        ax.set_title(title, color=C_TEXT, fontsize=12,
                     fontweight="bold", pad=8)
        ax.text(0.5, -0.14, note, transform=ax.transAxes,
                ha="center", color=col, fontsize=9, fontstyle="italic")

    plt.tight_layout(pad=2.0)
    fig.savefig("figure2_population.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("Figure 2 saved.")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Information flow + fixed LIF membrane (crosses threshold)
# ══════════════════════════════════════════════════════════════════
def fig3_info_flow():
    T    = 50
    t    = np.arange(T)

    ann_out = 0.65 + 0.18 * np.sin(t * 0.35) + 0.07 * np.random.randn(T)
    ann_out = np.clip(ann_out, 0.1, 1.0)

    thresh = 0.75
    beta   = 0.80

    cur = np.zeros(T)
    burst_starts = [3, 13, 22, 32, 42]
    for bs in burst_starts:
        for k in range(min(4, T - bs)):
            cur[bs + k] += 0.28

    mem      = np.zeros(T)
    spikes   = np.zeros(T)
    u = 0.0
    for i in range(T):
        u = beta * u + cur[i]
        if u >= thresh:
            spikes[i] = 1.0
            u = 0.0
        mem[i] = u

    spike_times = np.where(spikes == 1)[0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), facecolor=BG)
    fig.suptitle("Information Flow: Frame Camera → ANN   vs   Event Camera → SNN",
                 color=C_TEXT, fontsize=14, fontweight="bold", y=0.98)

    (ax_ann_pipe, ax_snn_pipe,
     ax_ann_sig,  ax_snn_sig) = axes.flat

    def clean_ax(ax, bg):
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_color(C_DIM); sp.set_linewidth(0.5)
        ax.tick_params(colors=C_DIM, labelsize=8)

    def box(ax, x, y, w, h, txt, sub="", col=C_ANN):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                            fc=col, ec="white", lw=1.2, alpha=0.88, zorder=3)
        ax.add_patch(p)
        ax.text(x+w/2, y+h*(0.62 if sub else 0.5), txt,
                ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold", zorder=4)
        if sub:
            ax.text(x+w/2, y+h*0.26, sub, ha="center", va="center",
                    color="white", fontsize=7, alpha=0.8, zorder=4)

    def arr(ax, x1, x2, y, col):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=1.8, mutation_scale=12), zorder=5)

    # ── top-left: ANN pipeline ────────────────────────────────────
    ax_ann_pipe.set_xlim(0, 10); ax_ann_pipe.set_ylim(0, 5)
    ax_ann_pipe.axis("off"); ax_ann_pipe.set_facecolor(CARD_L)
    ax_ann_pipe.set_title("Frame Camera  →  ANN", color=C_TEXT,
                           fontsize=11, fontweight="bold", pad=6)

    box(ax_ann_pipe, 0.1, 1.8, 1.6, 1.2, "FRAME\nCAMERA", "30 fps", col="#374151")
    arr(ax_ann_pipe, 1.7, 2.4, 2.4, C_ANN)
    dense = np.random.rand(8, 8)
    ax_ann_pipe.imshow(dense, extent=(2.4,4.2,1.4,3.4),
                        cmap="Blues", alpha=0.9, zorder=3, aspect="auto")
    ax_ann_pipe.text(3.3, 3.65, "Dense Frame\n(all pixels)", ha="center",
                      color=C_ANN, fontsize=7.5)
    arr(ax_ann_pipe, 4.2, 5.0, 2.4, C_ANN)
    box(ax_ann_pipe, 5.0, 1.8, 1.6, 1.2, "Conv\nLayers", col=C_ANN)
    arr(ax_ann_pipe, 6.6, 7.3, 2.4, C_ANN)
    box(ax_ann_pipe, 7.3, 1.8, 1.4, 1.2, "Dense\nOutput", col="#1d4ed8")
    ax_ann_pipe.text(5.0, 0.5,
                      "Every pixel → every neuron → every step",
                      ha="center", color=C_ANN, fontsize=8,
                      bbox=dict(fc=BG, ec=C_ANN, lw=0.8, pad=3, alpha=0.6),
                      zorder=5)

    # ── top-right: SNN pipeline ───────────────────────────────────
    ax_snn_pipe.set_xlim(0, 10); ax_snn_pipe.set_ylim(0, 5)
    ax_snn_pipe.axis("off"); ax_snn_pipe.set_facecolor(CARD_R)
    ax_snn_pipe.set_title("Event Camera  →  SNN", color=C_TEXT,
                            fontsize=11, fontweight="bold", pad=6)

    box(ax_snn_pipe, 0.1, 1.8, 1.6, 1.2, "EVENT\nCAMERA", "µs res", col="#064e3b")
    arr(ax_snn_pipe, 1.7, 2.4, 2.4, C_SNN)

    ev_x = np.random.rand(18) * 1.8 + 2.4
    ev_y = np.random.rand(18) * 2.0 + 1.4
    ax_snn_pipe.add_patch(FancyBboxPatch((2.4,1.4), 1.8, 2.0,
                           boxstyle="round,pad=0.05",
                           fc="#0a1f14", ec=C_SNN, lw=1, zorder=2))
    ax_snn_pipe.scatter(ev_x, ev_y, s=22, c=C_SPIKE, zorder=4, alpha=0.9)
    ax_snn_pipe.text(3.3, 3.65, "Sparse Events\n(changes only)", ha="center",
                      color=C_SPIKE, fontsize=7.5)
    arr(ax_snn_pipe, 4.2, 5.0, 2.4, C_SNN)
    box(ax_snn_pipe, 5.0, 1.8, 1.6, 1.2, "Spiking\nConv", col=C_SNN)
    arr(ax_snn_pipe, 6.6, 7.3, 2.4, C_SNN)
    box(ax_snn_pipe, 7.3, 1.8, 1.4, 1.2, "Sparse\nSpikes", col="#166534")
    ax_snn_pipe.text(5.0, 0.5,
                      "Only changed pixels → only active neurons fire",
                      ha="center", color=C_SNN, fontsize=8,
                      bbox=dict(fc=BG, ec=C_SNN, lw=0.8, pad=3, alpha=0.6),
                      zorder=5)

    # ── bottom-left: ANN signal ───────────────────────────────────
    clean_ax(ax_ann_sig, CARD_L)
    ax_ann_sig.set_title("ANN — Output Signal over Time",
                          color=C_TEXT, fontsize=10, fontweight="bold", pad=5)
    ax_ann_sig.fill_between(t, 0, ann_out, color=C_ANN, alpha=0.25)
    ax_ann_sig.plot(t, ann_out, color=C_ANN, lw=2.0)
    ax_ann_sig.set_ylim(0, 1.2)
    ax_ann_sig.set_xlabel("Timestep", color=C_DIM, fontsize=9)
    ax_ann_sig.set_ylabel("Activation", color=C_DIM, fontsize=9)
    ax_ann_sig.text(T/2, 1.10, "Dense — active every timestep",
                    ha="center", color=C_ANN, fontsize=8, fontstyle="italic")

    # ── bottom-right: LIF membrane + spikes ───────────────────────
    clean_ax(ax_snn_sig, CARD_R)
    ax_snn_sig.set_title("LIF Neuron — Membrane Potential & Spikes",
                          color=C_TEXT, fontsize=10, fontweight="bold", pad=5)

    ax_snn_sig.fill_between(t, 0, mem, color=C_MEM, alpha=0.18)
    ax_snn_sig.plot(t, mem, color=C_MEM, lw=2.0, label="Uₘ (membrane)", zorder=4)
    ax_snn_sig.axhline(thresh, color=C_SNN, lw=1.5, ls="--",
                        alpha=0.9, label=f"Threshold θ = {thresh}", zorder=3)

    for st in spike_times:
        ax_snn_sig.axvline(st, color=C_SPIKE, lw=1.5, alpha=0.7, zorder=2)
        ax_snn_sig.scatter(st, thresh, color=C_SPIKE, s=60, zorder=6)

    ax_snn_sig.set_ylim(-0.05, 1.15)
    ax_snn_sig.set_xlabel("Timestep", color=C_DIM, fontsize=9)
    ax_snn_sig.set_ylabel("Membrane Potential Uₘ", color=C_DIM, fontsize=9)
    ax_snn_sig.text(T/2, 1.07,
                    f"Sparse — {len(spike_times)} spikes in {T} steps "
                    f"({int(len(spike_times)/T*100)}% active)",
                    ha="center", color=C_SNN, fontsize=8, fontstyle="italic")

    legend_els = [
        Line2D([0],[0], color=C_MEM,   lw=2,   label="Membrane Uₘ"),
        Line2D([0],[0], color=C_SNN,   lw=1.5, ls="--", label=f"Threshold θ={thresh}"),
        Line2D([0],[0], color=C_SPIKE, lw=0,   marker="o", markersize=7,
               label="Spike event"),
    ]
    ax_snn_sig.legend(handles=legend_els, loc="upper right",
                       facecolor=BG, edgecolor=C_DIM,
                       labelcolor=C_TEXT, fontsize=8)

    plt.tight_layout(pad=2.0)
    fig.savefig("figure3_info_flow.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("Figure 3 saved.")


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_neurons()
    fig2_population()
    fig3_info_flow()
    plt.show()