from math import pi
from pathlib import Path
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import openpyxl.cell.cell
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from mmpose.utils.logger import MMLogger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys


matplotlib.use("tkAgg")
logger = MMLogger.get_instance("Accuracy")


WHOLEBODY_SECTIONS: dict[str, str] = {
    "body": "keypoints_body",
    "foot": "keypoints_foot",
    "face": "keypoints_face",
    "left_hand": "keypoints_lefthand",
    "right_hand": "keypoints_righthand",
    "all": "keypoints_wholebody",
}

SECTION_DISPLAY_LABELS: dict[str, str] = {
    "body": "Body (17 kps)",
    "foot": "Foot (6 kps)",
    "face": "Face (68 kps)",
    "left_hand": "Left Hand (21 kps)",
    "right_hand": "Right Hand (21 kps)",
    "all": "Wholebody (133 kps)",
}

SERIES_PALETTE = [
    "#6366F1",  # indigo
    "#14B8A6",  # teal
    "#F43F5E",  # rose
    "#F59E0B",  # amber
    "#8B5CF6",  # violet
    "#06B6D4",  # cyan
]

INK = "#1E1B2E"
MUTED = "#585568"
PAPER = "#F7F7FB"
SURFACE = "#FFFFFF"
HAIRLINE = "#C7C4D6"

FONT_BODY = "Inter, -apple-system, sans-serif"
FONT_DISPLAY = "'Space Grotesk', Inter, sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"


def _split_wholebody_metrics(raw: dict) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    for section_key, metrics_key in WHOLEBODY_SECTIONS.items():
        if metrics_key in raw and isinstance(raw[metrics_key], dict):
            sections[section_key] = dict(raw[metrics_key])
    return sections


def _stat_names_from(data: dict[str, dict[str, float]]) -> list[str]:
    return list(next(iter(data.values())).keys())


def save_metrics_xlsx(
    exp_metrics: dict[str, dict[str, float]],
    save_path: str,
    is_wholebody: bool = True,
) -> None:
    wb = Workbook()
    default_sheet = wb.active

    header_font = Font(name="TimesNewRoman", bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", start_color="2F5496")
    sub_header_fill = PatternFill("solid", start_color="D6E4F7")
    center_align = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin")
    thin_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def _style_header(cell: openpyxl.cell.cell.Cell, main: bool = True) -> None:
        cell.font = header_font if main else Font(name="TimesNewRoman", bold=True)
        cell.fill = header_fill if main else sub_header_fill
        cell.alignment = center_align
        cell.border = thin_border

    def _write_section_sheet(
        sheet_name: str,
        section_data: dict[str, dict[str, float]],
    ) -> None:
        ws = wb.create_sheet(sheet_name)
        legends = list(section_data.keys())
        stat_names = _stat_names_from(section_data)

        _style_header(ws.cell(1, 1, "Metric"))
        for col_idx, legend in enumerate(legends, start=2):
            _style_header(ws.cell(1, col_idx, legend))

        for row_idx, stat in enumerate(stat_names, start=2):
            cell = ws.cell(row_idx, 1, stat)
            cell.font = Font(name="TimesNewRoman", bold=True)
            cell.fill = sub_header_fill
            cell.border = thin_border

            for col_idx, legend in enumerate(legends, start=2):
                val = section_data[legend].get(stat)
                cell = ws.cell(row_idx, col_idx, round(val, 4) if val else "—")
                cell.alignment = center_align
                cell.border = thin_border
                cell.font = Font(name="TimesNewRoman")

        ws.column_dimensions["A"].width = 14
        for col_idx in range(2, len(legends) + 2):
            ws.column_dimensions[get_column_letter(col_idx)].width = 22

    if is_wholebody:
        by_section: dict[str, dict[str, dict[str, float]]] = {
            s: {} for s in WHOLEBODY_SECTIONS
        }
        for legend, raw in exp_metrics.items():
            split = _split_wholebody_metrics(raw)
            for section, stats_dict in split.items():
                by_section[section][legend] = stats_dict

        for section, label in SECTION_DISPLAY_LABELS.items():
            if by_section.get(section):
                _write_section_sheet(label, by_section[section])
    else:
        _write_section_sheet("Metrics", exp_metrics)

    wb.remove(default_sheet)
    wb.save(save_path)
    logger.info(f"Metrics XLSX saved: '{save_path}'")


def _draw_radar(
    ax: matplotlib.projections.polar.PolarAxes,
    labels: list[str],
    datasets: list[tuple[str, list[float]]],
    xticks: list[float],
    colors: list[tuple[float, float, float]]
) -> None:
    N = len(labels)
    angles = [n / N * 2 * pi for n in range(N)] + [0]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7)

    ax.set_yticks(xticks)
    ax.set_yticklabels([f"{t:.2f}" for t in xticks], size=6)

    margin = (xticks[-1] - xticks[0]) * 0.08
    ax.set_ylim(min(xticks), max(xticks) + margin)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))

    for idx, (name, values) in enumerate(datasets):
        vals = values + [values[0]]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, linewidth=1.8, linestyle="solid", label=name, color=color)
        ax.fill(angles, vals, color=color, alpha=0.12)


def _dynamic_ticks(values_flat: list[float], n_steps: int = 5, upper: float = 1.0) -> list[float]:
    positive = [v for v in values_flat if v > 0]
    if not positive:
        return [0.0, upper]

    lo = min(positive)
    span = max(upper - lo, 1e-6)
    pad = max(0.01, span * 0.08)

    lo = max(0.0, lo - pad)
    lo = math.floor(lo * 20) / 20  # round down to nearest 0.05

    if lo >= upper:
        lo = max(0.0, upper - 0.05)

    step = max(0.01, round((upper - lo) / n_steps * 100) / 100)
    ticks = sorted({round(lo + step * i, 3) for i in range(n_steps)} | {upper})
    return ticks


def generate_radar_plot(
    exp_metrics: dict[str, dict[str, float]],
    is_wholebody: bool = True,
    xticks: list[float] | None = None,
    save_path: str | None = None,
    show: bool = True,
    title: str = "Pose models comparison",
) -> plt.Figure | None:
    if not exp_metrics:
        logger.warning("Param 'exp_metrics' is empty — skip radar.")
        return None

    colors = plt.cm.tab10.colors

    if is_wholebody:
        by_section: dict[str, dict[str, dict[str, float]]] = {
            s: {} for s in WHOLEBODY_SECTIONS
        }
        for legend, raw in exp_metrics.items():
            split = _split_wholebody_metrics(raw)
            for section, stats_dict in split.items():
                if section in by_section:
                    by_section[section][legend] = stats_dict

        non_empty = [(s, lbl) for s, lbl in SECTION_DISPLAY_LABELS.items() if by_section.get(s)]
        if not non_empty:
            logger.warning("Doesn't have enough data for whole body radar.")
            return None

        ncols = 3
        nrows = (len(non_empty) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, 7 * nrows),
            subplot_kw=dict(polar=True),
        )
        axes_flat = np.array(axes).flatten()

        for plot_idx, (section, section_label) in enumerate(non_empty):
            ax = axes_flat[plot_idx]
            sec_data = by_section[section] # {legend: {stat: val}}
            stat_names = _stat_names_from(sec_data)
            datasets = [
                (legend, [sec_data[legend].get(s, 0.0) for s in stat_names])
                for legend in sec_data
            ]

            all_vals = [v for _, vals in datasets for v in vals]
            section_ticks = xticks if xticks is not None else _dynamic_ticks(all_vals)

            _draw_radar(ax, stat_names, datasets, section_ticks, colors)
            ax.set_title(section_label, pad=2, size=11, weight="bold")
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)

        for extra_idx in range(len(non_empty), len(axes_flat)):
            axes_flat[extra_idx].set_visible(False)

        fig.suptitle(title, size=15, weight="bold", y=1.)

    else:
        stat_names = _stat_names_from(exp_metrics)
        if not stat_names:
            logger.warning("Metrics is empty - skip radar creation.")
            return None

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        datasets = [
            (legend, [m.get(s, 0.0) for s in stat_names])
            for legend, m in exp_metrics.items()
        ]

        all_vals = [v for _, vals in datasets for v in vals]
        section_ticks = xticks if xticks is not None else _dynamic_ticks(all_vals)

        _draw_radar(ax, stat_names, datasets, section_ticks, colors)
        ax.set_title(title, pad=2, size=13, weight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Plot saved → {save_path}")

    if show:
        plt.show(block=True)

    return fig


def _saturate(hex_color: str, amount: float = 0.28) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Increase saturation towards 1.0
    s = min(1.0, s + (1.0 - s) * amount)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def _darken(hex_color: str, amount: float = 0.28) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, int(c * (1 - amount))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def generate_radar_plot_html(
    exp_metrics: dict[str, dict[str, float]],
    is_wholebody: bool = True,
    save_path: str | None = None,
    title: str = "Pose Model Comparison",
) -> go.Figure | None:
    if not exp_metrics:
        logger.warning("exp_metrics is empty — skip interactive radar.")
        return None

    legends = list(exp_metrics.keys())
    legend_color = {legend: SERIES_PALETTE[i % len(SERIES_PALETTE)] for i, legend in enumerate(legends)}

    def _traces_for(stat_names: list[str], data: dict[str, dict[str, float]], show_legend: bool):
        traces = []
        for legend, stats in data.items():
            values = [stats.get(s, 0.0) for s in stat_names]
            color = legend_color[legend]
            contour_color = _saturate(color, 0.8)
            contour_color = _darken(contour_color, 0.25)
            traces.append(go.Scatterpolar(
                r=values + [values[0]],
                theta=stat_names + [stat_names[0]],
                fill="toself",
                fillcolor=color,
                mode="lines+markers",
                marker=dict(size=6, color=contour_color, line=dict(color=SURFACE, width=1)),
                name=legend,
                opacity=0.5,
                line=dict(color=contour_color, width=2.5),
                showlegend=show_legend,
                hoveron="points",
                hovertemplate="<b>%{theta}</b>: %{r:.4f}<extra>%{fullData.name}</extra>",
            ))
        return traces

    axis_style = dict(
        gridcolor=HAIRLINE,
        linecolor=MUTED,
        linewidth=1.5,
        tickfont=dict(family=FONT_MONO, size=11, color=MUTED, weight="bold"),
    )
    angular_style = dict(
        gridcolor=HAIRLINE,
        linecolor=MUTED,
        linewidth=1.5,
        tickfont=dict(family=FONT_BODY, size=12.5, color=INK, weight="bold"),
    )

    if is_wholebody:
        by_section: dict[str, dict[str, dict[str, float]]] = {
            s: {} for s in WHOLEBODY_SECTIONS
        }
        for legend, raw in exp_metrics.items():
            for section, stats_dict in _split_wholebody_metrics(raw).items():
                if section in by_section:
                    by_section[section][legend] = stats_dict

        non_empty = [(s, lbl) for s, lbl in SECTION_DISPLAY_LABELS.items() if by_section.get(s)]
        if not non_empty:
            logger.warning("Doesn't have enough data for wholebody radar.")
            return None

        ncols = 3 if len(non_empty) > 1 else 1
        nrows = math.ceil(len(non_empty) / ncols)
        fig = make_subplots(
            rows=nrows, cols=ncols,
            specs=[[{"type": "polar"}] * ncols for _ in range(nrows)],
            subplot_titles=[lbl for _, lbl in non_empty],
            horizontal_spacing=0.08,
            vertical_spacing=0.14,
        )
        for ann in fig.layout.annotations:
            ann.font = dict(family=FONT_DISPLAY, size=13, color=INK, weight="bold")

        for i, (section, _) in enumerate(non_empty):
            row, col = divmod(i, ncols)
            sec_data = by_section[section]
            stat_names = _stat_names_from(sec_data)
            for trace in _traces_for(stat_names, sec_data, show_legend=(i == 0)):
                fig.add_trace(trace, row=row + 1, col=col + 1)

            all_vals = [v for stats in sec_data.values() for v in stats.values()]
            ticks = _dynamic_ticks(all_vals)
            ticks.append(1.1)
            fig.update_polars(
                bgcolor=SURFACE,
                radialaxis=dict(
                    range=[ticks[0], ticks[-1]],
                    tickvals=ticks,
                    ticktext=[f"{t:.2f}" for t in ticks],
                    **axis_style,
                ),
                angularaxis=angular_style,
                row=row + 1, col=col + 1,
            )

    else:
        stat_names = _stat_names_from(exp_metrics)
        if not stat_names:
            logger.warning("Metrics is empty - skip interactive radar creation.")
            return None

        fig = go.Figure(data=_traces_for(stat_names, exp_metrics, show_legend=True))

        all_vals = [v for stats in exp_metrics.values() for v in stats.values()]
        ticks = _dynamic_ticks(all_vals)
        ticks.append(1.1)

        fig.update_layout(
            polar=dict(
                bgcolor=SURFACE,
                radialaxis=dict(
                    visible=True,
                    range=[ticks[0], ticks[-1]],
                    tickvals=ticks,
                    ticktext=[f"{t:.2f}" for t in ticks],
                    **axis_style,
                ),
                angularaxis=angular_style,
            ),
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=FONT_DISPLAY, size=22, color=INK),
            x=0.03, xanchor="left",
            y=0.98, yanchor="top",
        ),
        autosize=True,
        margin=dict(l=50, r=50, t=90, b=30),
        hovermode="closest",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_BODY, color=INK),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=HAIRLINE,
            font=dict(family=FONT_MONO, size=12, color=INK),
        ),
        legend=dict(
            orientation="v",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(family=FONT_BODY, size=12, color=INK),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )

    if not save_path:
        return fig

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        save_path,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )
    fig.write_json(Path(save_path).with_suffix(".json"))
    logger.info(f"Interactive HTML plot saved → {save_path}")

    return fig
