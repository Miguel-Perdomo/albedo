"""Figures of the simulated geometry.

``section`` draws the x-z cross-section with matplotlib, for papers and
posters. ``view3d`` builds an interactive 3D view with plotly that can be
rotated and zoomed in a notebook or saved as a standalone HTML file. Both
are drawn from the same ``Setup`` used by the transport code, so the
figure shows exactly the simulated geometry. Colours come from
``plotting.MATERIAL_COLORS``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.patches import Patch, Rectangle

from .geometry import Setup, Volume
from .plotting import MATERIAL_COLORS, SOURCE_COLOR, FONT_FAMILY_HTML


def _color(volume: Volume) -> str:
    if volume.material is None:
        return "white"
    return to_hex(MATERIAL_COLORS.get(volume.material, "lightgray"))


def _draw_volumes(ax, setup: Setup) -> None:
    # Drawing from the last volume to the first reproduces the priority
    # rule: earlier (inner) volumes are painted over later ones.
    for v in reversed(setup.volumes):
        ax.add_patch(
            Rectangle((-v.radius, v.z_min), 2.0 * v.radius, v.z_max - v.z_min,
                      facecolor=_color(v), edgecolor="none")
        )
    x, _, z = setup.source
    ax.plot(x, z, marker="*", ms=12, color=SOURCE_COLOR, ls="none", zorder=5)


def section(setup: Setup, zoom: bool = True, width: float = 22.0, height: float = 16.0):
    """Cross-section of the setup in the x-z plane. Returns ``(fig, ax)``.

    With ``zoom`` a side panel magnifies the upper wall corner, where the
    liner layers (too thin to see at full scale) are labelled.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 2.54, height / 2.54), layout="constrained")
    if zoom:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.6])
        ax = fig.add_subplot(gs[:, 0])
        legend_ax = fig.add_subplot(gs[0, 1])
        zoom_ax = fig.add_subplot(gs[1, 1])
    else:
        ax = fig.add_subplot()
        legend_ax = ax

    _draw_volumes(ax, setup)
    outer = setup.volumes[-1]
    ax.set_xlim(-outer.radius, outer.radius)
    ax.set_ylim(outer.z_min, outer.z_max)
    ax.set_aspect("equal")
    ax.grid(False, which="both")
    ax.set_xlabel(r"$x\,(\mathrm{cm})$")
    ax.set_ylabel(r"$z\,(\mathrm{cm})$")

    handles = [Patch(facecolor=to_hex(MATERIAL_COLORS.get(m, "lightgray")), label=m)
               for m in setup.material_names]
    handles.append(plt.Line2D([], [], marker="*", ms=12, color=SOURCE_COLOR,
                              ls="none", label="Source"))

    if not zoom:
        ax.legend(handles=handles, loc="upper right", framealpha=1.0)
        return fig, ax

    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", frameon=False)

    cavity = setup.volumes[setup.volume_index("cavity")]
    r, h = cavity.radius, cavity.z_max
    _draw_volumes(zoom_ax, setup)
    zoom_ax.set_xlim(r - 0.3, r + 0.5)
    zoom_ax.set_ylim(h - 0.3, h + 0.5)
    zoom_ax.set_aspect("equal")
    zoom_ax.grid(False, which="both")
    zoom_ax.tick_params(which="both", left=False, right=False, top=False, bottom=False,
                        labelleft=False, labelbottom=False)

    # Label each liner layer with its thickness, half-way across the layer
    inner = r
    for v in setup.volumes:
        if v.name.startswith("liner_"):
            mid = 0.5 * (inner + v.radius)
            mm = 10.0 * (v.radius - inner)
            zoom_ax.text(mid, h - 0.15, f"{v.material}\n{mm:.1f} mm", rotation=90,
                         ha="center", va="center", fontsize="small",
                         color="white" if v.material == "Sn" else "black")
            inner = v.radius
    ax.indicate_inset_zoom(zoom_ax, edgecolor="black")
    return fig, ax


# --- interactive 3D view ----------------------------------------------------

def _surface(x, y, z, color, name, show_legend, opacity):
    import plotly.graph_objects as go

    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(z),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        name=name,
        legendgroup=name,
        showlegend=show_legend,
        hoverinfo="name",
    )


def _hole(setup: Setup, i: int) -> Volume | None:
    """Largest earlier volume contained in volume i (the cavity of a shell)."""
    v = setup.volumes[i]
    inner = [
        w for w in setup.volumes[:i]
        if w.radius <= v.radius and w.z_min >= v.z_min and w.z_max <= v.z_max
    ]
    return max(inner, key=lambda w: w.radius, default=None)


def view3d(setup: Setup, cutaway: float = 120.0, opacity: float = 1.0, n_theta: int = 72):
    """Interactive 3D view with a wedge of ``cutaway`` degrees removed."""
    import plotly.graph_objects as go

    t0 = np.radians(cutaway)
    theta = np.linspace(t0, 2.0 * np.pi, n_theta)
    traces = []
    seen = set()

    for i, v in enumerate(setup.volumes):
        if v.material is None:
            continue
        color = _color(v)
        first = v.material not in seen
        seen.add(v.material)

        # Outer side surface
        th, zz = np.meshgrid(theta, [v.z_min, v.z_max])
        traces.append(_surface(v.radius * np.cos(th), v.radius * np.sin(th), zz,
                               color, v.material, first, opacity))
        first = False

        # Top and bottom caps
        rr, th = np.meshgrid([0.0, v.radius], theta)
        for zc in (v.z_min, v.z_max):
            traces.append(_surface(rr * np.cos(th), rr * np.sin(th), np.full_like(rr, zc),
                                   color, v.material, False, opacity))

        # Faces exposed by the cut: the r-z section of the volume minus its hole
        hole = _hole(setup, i)
        if hole is None:
            rects = [(0.0, v.radius, v.z_min, v.z_max)]
        else:
            rects = [
                (hole.radius, v.radius, v.z_min, v.z_max),
                (0.0, hole.radius, v.z_min, hole.z_min),
                (0.0, hole.radius, hole.z_max, v.z_max),
            ]
        for angle in (t0, 2.0 * np.pi):
            for r0, r1, z0, z1 in rects:
                rr, zz = np.meshgrid([r0, r1], [z0, z1])
                traces.append(_surface(rr * np.cos(angle), rr * np.sin(angle), zz,
                                       color, v.material, False, opacity))

    x, y, z = setup.source
    traces.append(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers", name="Source",
                               marker=dict(size=5, color=to_hex(SOURCE_COLOR))))

    axis = dict(showbackground=False, gridcolor="lightgray", zerolinecolor="gray")
    fig = go.Figure(traces)
    fig.update_layout(
        font=dict(family=FONT_FAMILY_HTML, size=16, color="black"),
        paper_bgcolor="white",
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="x (cm)", **axis),
            yaxis=dict(title="y (cm)", **axis),
            zaxis=dict(title="z (cm)", **axis),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=setup.name, x=0.5),
    )
    return fig
