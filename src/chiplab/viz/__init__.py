"""ChipLab Visualization Module — floorplanning, die visualization, and thermal/power heatmaps.

This package provides tools for floorplan allocation, SVG/JSON visualization,
thermal/power density heatmaps, and 3D die cross-section rendering.
"""

from .floorplan import allocate_floorplan, floorplan_to_svg, floorplan_to_json, check_floorplan_drc
from .heatmap import power_heatmap, thermal_heatmap, heatmap_to_svg, heatmap_to_html
from .die_renderer import render_die_3d

__all__ = [
    "allocate_floorplan",
    "floorplan_to_svg",
    "floorplan_to_json",
    "check_floorplan_drc",
    "power_heatmap",
    "thermal_heatmap",
    "heatmap_to_svg",
    "heatmap_to_html",
    "render_die_3d",
]