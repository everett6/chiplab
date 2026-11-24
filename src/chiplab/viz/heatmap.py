"""Thermal & power density heatmap generation.

Implements 2D power density and thermal heatmaps as JSON grids,
with SVG and HTML visualization output including color gradients
and interactive hover tooltips.
"""

from __future__ import annotations

from typing import Any


# Import TrackedValue and Provenance from chiplab schemas if available
try:
    from chiplab.schemas.common import TrackedValue, Provenance
except ImportError:
    TrackedValue = None  # type: ignore
    Provenance = None  # type: ignore


def power_heatmap(
    power_map: dict,
    floorplan: dict,
    grid_size: int = 50,
) -> dict:
    """Generate a 2D power density heatmap as JSON (grid of W/mm²).

    Creates a grid-based power density heatmap by distributing chiplet power
    across the die area. Each grid cell contains the power density in W/mm².

    Args:
        power_map: Dict mapping chiplet id to power dict with keys:
            - "power_watts": float — chiplet power in watts
            - "area_mm2": float — chiplet area in square millimeters
        floorplan: Floorplan dict as returned by allocate_floorplan()
        grid_size: Number of grid cells per dimension (default 50 = 50x50 grid)

    Returns:
        dict with keys:
            - "grid_size": int — grid dimension (grid_size x grid_size)
            - "cell_size_mm": float — size of each grid cell in millimeters
            - "power_density_w_mm2": 2D list of float — power density per cell
            - "total_power_w": float — total power across all chiplets
            - "die_width_mm": float — die width
            - "die_height_mm": float — die height

    Raises:
        ValueError: If power_map is empty or floorplan has no placed chiplets

    Example:
        >>> power_map = {
        ...     "cpu0": {"power_watts": 25.0, "area_mm2": 50.0},
        ...     "mem0": {"power_watts": 10.0, "area_mm2": 25.0},
        ... }
        >>> fp = allocate_floorplan([{"id": "cpu0", "area_mm2": 50.0}, {"id": "mem0", "area_mm2": 25.0}], 100.0)
        >>> hm = power_heatmap(power_map, fp, grid_size=10)
        >>> len(hm["power_density_w_mm2"])
        10
        >>> hm["total_power_w"]
        35.0
    """
    if not power_map:
        raise ValueError("power_map must not be empty")

    die_width = floorplan["die_width_mm"]
    die_height = floorplan["die_height_mm"]
    placed = floorplan["placed"]

    # Build a lookup of chiplet id -> (x, y, width, height) from floorplan
    chiplet_positions = {}
    for block in placed:
        chiplet_positions[block["id"]] = {
            "x": block["x_mm"],
            "y": block["y_mm"],
            "w": block["width_mm"],
            "h": block["height_mm"],
        }

    # Calculate total power
    total_power_w = 0.0
    power_areas = {}  # chiplet_id -> power_per_area

    for cid, pinfo in power_map.items():
        power_w = pinfo.get("power_watts", 0)
        area = pinfo.get("area_mm2", 0)
        power_areas[cid] = power_w / area if area > 0 else 0
        total_power_w += power_w

    # Initialize grid
    grid = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    cell_size_w = die_width / grid_size
    cell_size_h = die_height / grid_size

    # Distribute power density across grid cells based on chiplet positions
    for cid, ppw in power_areas.items():
        if cid not in chiplet_positions:
            continue

        pos = chiplet_positions[cid]
        x_start = pos["x"]
        y_start = pos["y"]
        w = pos["w"]
        h = pos["h"]

        # Determine which grid cells this chipet covers
        col_start = max(0, int(x_start / cell_size_w))
        col_end = min(grid_size, int((x_start + w) / cell_size_w))
        row_start = max(0, int(y_start / cell_size_h))
        row_end = min(grid_size, int((y_start + h) / cell_size_h))

        # Distribute power density evenly across the chipet's area
        if row_end > row_start and col_end > col_start:
            cells_in_region = (row_end - row_start) * (col_end - col_start)
            density_per_cell = ppw / cells_in_region

            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    grid[r][c] += density_per_cell

    # Normalize: ensure total grid power approximates total chiplet power
    # Compute total grid power
    grid_total = sum(sum(row) for row in grid)

    if grid_total > 0 and abs(grid_total - total_power_w) / total_power_w > 0.1:
        # Scale to match total power
        scale = total_power_w / grid_total
        for r in range(grid_size):
            for c in range(grid_size):
                grid[r][c] *= scale

    return {
        "grid_size": grid_size,
        "cell_size_mm": round(die_width / grid_size, 4),
        "power_density_w_mm2": grid,
        "total_power_w": round(total_power_w, 4),
        "die_width_mm": die_width,
        "die_height_mm": die_height,
    }


def thermal_heatmap(
    power_map: dict,
    floorplan: dict,
    ambient_temp: float = 25.0,
    cooling: str = "natural_convection",
) -> dict:
    """Generate a 2D thermal heatmap as JSON (grid of °C).

    Uses a simple thermal resistance model to convert power density into
    temperature rise. The model assumes steady-state heat dissipation
    from each grid cell to ambient temperature.

    Thermal model: T_cell = T_ambient + R_th * P_cell
    where R_th is the thermal resistance per cell (depends on cooling method).

    Args:
        power_map: Dict mapping chiplet id to power dict (same format as power_heatmap)
        floorplan: Floorplan dict as returned by allocate_floorplan()
        ambient_temp: Ambient temperature in °C (default 25.0)
        cooling: Cooling method. One of:
            - "natural_convection": ~5-10 °C/W thermal resistance
            - "forced_air": ~2-5 °C/W thermal resistance
            - "liquid_cooling": ~0.5-1.5 °C/W thermal resistance

    Returns:
        dict with keys:
            - "grid_size": int — grid dimension
            - "cell_size_mm": float — size of each grid cell in millimeters
            - "temperature_c": 2D list of float — temperature per cell in °C
            - "ambient_temp_c": float — ambient temperature
            - "cooling": str — cooling method used
            - "max_temp_c": float — maximum temperature in the grid
            - "total_power_w": float — total power

    Raises:
        ValueError: If cooling method is unknown

    Example:
        >>> power_map = {
        ...     "cpu0": {"power_watts": 25.0, "area_mm2": 50.0},
        ...     "mem0": {"power_watts": 10.0, "area_mm2": 25.0},
        ... }
        >>> fp = allocate_floorplan([{"id": "cpu0", "area_mm2": 50.0}, {"id": "mem0", "area_mm2": 25.0}], 100.0)
        >>> hm = thermal_heatmap(power_map, fp, ambient_temp=25.0, cooling="forced_air")
        >>> hm["ambient_temp_c"]
        25.0
        >>> hm["max_temp_c"] > 25.0
        True
    """
    # Validate cooling method
    cooling_lower = cooling.lower()
    if cooling_lower == "natural_convection":
        r_th_per_w = 8.0  # °C/W
    elif cooling_lower == "forced_air":
        r_th_per_w = 3.0  # °C/W
    elif cooling_lower == "liquid_cooling":
        r_th_per_w = 0.8  # °C/W
    else:
        raise ValueError(
            f"Unknown cooling method: {cooling}. "
            f"Choose from: natural_convection, forced_air, liquid_cooling"
        )

    # Generate power heatmap first
    power_hm = power_heatmap(power_map, floorplan, grid_size=50)

    grid_size = power_hm["grid_size"]
    cell_size_mm = power_hm["cell_size_mm"]
    grid = power_hm["power_density_w_mm2"]

    # Compute temperature grid using thermal model
    temp_grid = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]

    for r in range(grid_size):
        for c in range(grid_size):
            density = grid[r][c]
            # Power in this cell = density * cell_area
            cell_area_mm2 = cell_size_mm ** 2
            cell_power_w = density * cell_area_mm2
            # Temperature rise = R_th * power
            temp_rise = r_th_per_w * cell_power_w
            temp_grid[r][c] = round(ambient_temp + temp_rise, 2)

    # Find max temperature
    max_temp = max(max(row) for row in temp_grid)

    return {
        "grid_size": grid_size,
        "cell_size_mm": cell_size_mm,
        "temperature_c": temp_grid,
        "ambient_temp_c": ambient_temp,
        "cooling": cooling,
        "max_temp_c": round(max_temp, 2),
        "total_power_w": power_hm["total_power_w"],
    }


def _color_for_value(value: float, min_val: float, max_val: float) -> str:
    """Generate a color from blue→green→yellow→red gradient based on value.

    Args:
        value: The value to map to a color
        min_val: Minimum value in range
        max_val: Maximum value in range

    Returns:
        CSS color string (hex format)
    """
    # Clamp value to range
    if max_val == min_val:
        ratio = 0.0
    else:
        ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))

    # Color gradient: blue(0) -> green(0.33) -> yellow(0.66) -> red(1.0)
    # RGB interpolation
    if ratio < 1 / 3:
        # Blue to Green: ratio 0->0.33
        # R: 0, G: 3*ratio, B: 1-3*ratio
        r = 0
        g = int(255 * (ratio * 3))
        b = int(255 * (1 - ratio * 3))
    elif ratio < 2 / 3:
        # Green to Yellow: ratio 0.33->0.66
        # R: 3*ratio-1, G: 1, B: 0
        r = int(255 * ((ratio - 1 / 3) * 3))
        g = 255
        b = 0
    else:
        # Yellow to Red: ratio 0.66->1.0
        # R: 1, G: 3*-ratio+3, B: 0
        r = 255
        g = int(255 * (1 - (ratio - 2 / 3) * 3))
        b = 0

    return f"rgb({r},{g},{b})"


def heatmap_to_svg(
    heatmap: dict,
    title: str = "Heatmap",
    unit: str = "W/mm²",
) -> str:
    """Convert a heatmap dict to an SVG string with color gradient.

    Generates an SVG representation where each grid cell is colored
    according to its value using a blue→green→yellow→red color map.

    Args:
        heatmap: Heatmap dict as returned by power_heatmap() or thermal_heatmap()
        title: Title for the SVG
        unit: Unit string for the colorbar (e.g., "W/mm²", "°C")

    Returns:
        SVG string with color gradient visualization and colorbar

    Example:
        >>> hm = power_heatmap({"cpu0": {"power_watts": 25.0}}, {"die_width_mm": 10, "die_height_mm": 10, "placed": []}, grid_size=10)
        >>> svg = heatmap_to_svg(hm, "Power Heatmap", "W/mm²")
        >>> "<svg" in svg
        True
    """
    grid_size = heatmap["grid_size"]
    cell_size = heatmap["cell_size_mm"]
    data = heatmap["power_density_w_mm2"] if "power_density_w_mm2" in heatmap else heatmap["temperature_c"]

    # Determine min/max for color scaling
    # Flatten grid to find min/max, ignoring zeros if possible
    flat_vals = [val for row in data for val in row]
    if not flat_vals:
        min_val, max_val = 0.0, 1.0
    else:
        # Use non-zero values if available, otherwise all
        non_zero = [v for v in flat_vals if v > 0]
        if non_zero:
            min_val = min(non_zero)
            max_val = max(flat_vals)
        else:
            min_val = 0.0
            max_val = max(flat_vals) if max(flat_vals) > 0 else 1.0

    die_width = heatmap.get("die_width_mm", grid_size * cell_size)
    die_height = heatmap.get("die_height_mm", grid_size * cell_size)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{die_width}" height="{die_height}" viewBox="0 0 {die_width} {die_height}" style="background:#f0f0f0">',
        f'  <title>{title}</title>',
    ]

    # Draw each grid cell with color based on value
    for r in range(grid_size):
        for c in range(grid_size):
            value = data[r][c]
            color = _color_for_value(value, min_val, max_val)

            x = c * cell_size
            y = r * cell_size

            # Add rect with color
            svg_parts.append(
                f'  <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{color}" fill-opacity="0.8"/>'
            )

            # Add value label if significant
            if abs(value) > 0.01 and unit != "°C":
                svg_parts.append(
                    f'  <text x="{x + cell_size/2}" y="{y + cell_size/2}" text-anchor="middle" '
                    f'dominant-baseline="middle" fill="#000000" font-size="5" font-family="sans-serif">'
                    f"{value:.1f}</text>"
                )

    # Add colorbar
    colorbar_width = 20
    svg_parts.append(
        f'  <rect x="{die_width - colorbar_width}" y="5" width="{colorbar_width}" height="20" fill="none" stroke="#666666"/>'
    )
    # Draw gradient rect for colorbar
    svg_parts.append(
        f'  <linearGradient id="grad" x1="0%" y1="0%" x2="0%" y2="100%">'
    )
    # Add color stops
    num_stops = 10
    for i in range(num_stops + 1):
        stop_ratio = i / num_stops
        # Compute the value at this ratio
        if min_val == max_val:
            test_val = min_val
        else:
            test_val = min_val + (max_val - min_val) * stop_ratio
        if i == 0:
            c = _color_for_value(min_val, min_val, max_val)
        elif i == num_stops:
            c = _color_for_value(max_val, min_val, max_val)
        else:
            c = _color_for_value(
                min_val + (max_val - min_val) * stop_ratio, min_val, max_val
            )
        svg_parts.append(
            f'    <stop offset="{stop_ratio*100}%" stop-color="{c}"/>'
        )
    svg_parts.append("</linearGradient>")

    svg_parts.append(
        f'  <rect x="{die_width - colorbar_width}" y="5" width="{colorbar_width}" height="20" fill="url(#grad)"/>'
    )
    svg_parts.append(
        f'  <text x="{die_width - colorbar_width/2}" y="4" text-anchor="middle" fill="#000000" font-size="6">{min_val:.1f}</text>'
    )
    svg_parts.append(
        f'  <text x="{die_width - colorbar_width/2}" y="24" text-anchor="middle" fill="#000000" font-size="6">{max_val:.1f} {unit}</text>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def heatmap_to_html(
    heatmap: dict,
    title: str = "Heatmap",
    unit: str = "W/mm²",
) -> str:
    """Convert a heatmap dict to an interactive HTML string with hover tooltips.

    Generates a self-contained HTML page with embedded CSS/JS that displays
    the heatmap as a grid with hover tooltips showing the exact value for
    each cell. Uses the color gradient blue→green→yellow→red.

    Args:
        heatmap: Heatmap dict as returned by power_heatmap() or thermal_heatmap()
        title: Title for the HTML page
        unit: Unit string (e.g., "W/mm²", "°C")

    Returns:
        Self-contained HTML string with embedded CSS and JS for interactivity

    Example:
        >>> hm = power_heatmap({"cpu0": {"power_watts": 25.0}}, {"die_width_mm": 10, "die_height_mm": 10, "placed": []}, grid_size=10)
        >>> html = heatmap_to_html(hm, "Power Heatmap", "W/mm²")
        >>> "<!DOCTYPE html>" in html
        True
    """
    grid_size = heatmap["grid_size"]
    cell_size = heatmap["cell_size_mm"]
    data = heatmap["power_density_w_mm2"] if "power_density_w_mm2" in heatmap else heatmap["temperature_c"]

    # Determine min/max for color scaling
    flat_vals = [val for row in data for val in row]
    if not flat_vals:
        min_val, max_val = 0.0, 1.0
    else:
        non_zero = [v for v in flat_vals if v > 0]
        if non_zero:
            min_val = min(non_zero)
            max_val = max(flat_vals)
        else:
            min_val = 0.0
            max_val = max(flat_vals) if max(flat_vals) > 0 else 1.0

    die_width = heatmap.get("die_width_mm", grid_size * cell_size)
    die_height = heatmap.get("die_height_mm", grid_size * cell_size)

    # Generate cell rows with tooltips
    def color_for_value(value: float) -> str:
        if max_val == min_val:
            ratio = 0.0
        else:
            ratio = (value - min_val) / (max_val - min_val)
        ratio = max(0.0, min(1.0, ratio))

        if ratio < 1 / 3:
            r, g, b = 0, int(255 * (ratio * 3)), int(255 * (1 - ratio * 3))
        elif ratio < 2 / 3:
            r, g, b = int(255 * ((ratio - 1 / 3) * 3)), 255, 0
        else:
            r, g, b = 255, int(255 * (1 - (ratio - 2 / 3) * 3)), 0
        return f"rgb({r},{g},{b})"

    cell_rows = []
    for r in range(grid_size):
        cells = []
        for c in range(grid_size):
            value = data[r][c]
            color = color_for_value(value)
            tooltip = f"{value:.2f} {unit}"
            cells.append(
                f'<div style="width:{cell_size}mm;height:{cell_size}mm;background:{color};'
                f'float:left;position:relative;" title="{tooltip}"></div>'
            )
        row_div = '<div style="clear:both;"></div>'
        cell_rows.append(f'<div style="overflow:hidden;">{"".join(cells)}{row_div}</div>')

    cells_html = "\n".join(cell_rows)

    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "    <meta charset=\"UTF-8\">\n"
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        f"    <title>{title}</title>\n"
        "    <style>\n"
        "        body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 20px; }\n"
        "        h1 { color: #333; }\n"
        "        .heatmap-container { border: 1px solid #ddd; padding: 10px; background: white; }\n"
        "        .tooltip { position: absolute; padding: 5px; background: #fff; border: 1px solid #ddd; "
        "display: none; font-size: 12px; min-width: 80px; }\n"
        "    </style>\n"
        "</head>\n"
        "<body>\n"
        f"    <h1>{title}</h1>\n"
        "    <div class=\"heatmap-container\">\n"
        f"        {cells_html}\n"
        "    </div>\n"
        "    <script>\n"
        "        // Tooltip functionality\n"
        "        const cells = document.querySelectorAll(\".heatmap-container div div div\");\n"
        "        cells.forEach(cell => {\n"
        "            cell.addEventListener(\"mouseover\", function() {\n"
        "                const tooltip = document.createElement(\"div\");\n"
        "                tooltip.className = \"tooltip\";\n"
        "                tooltip.textContent = cell.getAttribute(\"title\");\n"
        "                document.body.appendChild(tooltip);\n"
        "                const rect = cell.getBoundingClientRect();\n"
        "                tooltip.style.left = (rect.left) + \"px\";\n"
        "                tooltip.style.top = (rect.top - 25) + \"px\";\n"
        "                tooltip.style.display = \"block\";\n"
        "            });\n"
        "            cell.addEventListener(\"mouseout\", function() {\n"
        "                const tooltip = document.querySelector(\".tooltip\");\n"
        "                if (tooltip) {\n"
        "                    tooltip.remove();\n"
        "                }\n"
        "            });\n"
        "        });\n"
        "    </script>\n"
        "</body>\n"
        "</html>"
    )

    return html