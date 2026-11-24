"""Floorplan allocator and visualizer for chiplet designs.

Implements a slicing tree based floorplan allocation algorithm that places
chiplets on a die while respecting spacing, boundary, and aspect ratio constraints.
Generates SVG and JSON representations for visualization.
"""

from __future__ import annotations

from typing import Any
import json

# Import TrackedValue and Provenance from chiplab schemas if available
try:
    from chiplab.schemas.common import TrackedValue, Provenance
except ImportError:
    TrackedValue = None  # type: ignore
    Provenance = None  # type: ignore


def allocate_floorplan(
    chiplets: list[dict],
    package_area_mm2: float,
    aspect_ratio: float = 1.0,
) -> dict:
    """Allocate chiplets on a floorplan using a slicing tree algorithm.

    Places rectangular chiplets within a die of given area and aspect ratio.
    Uses a recursive slicing tree approach: alternating horizontal and vertical
    cuts to pack chiplets efficiently.

    Args:
        chiplets: List of chiplet dicts, each with keys:
            - "id": str — chiplet identifier
            - "area_mm2": float — chiplet area in square millimeters
            - "aspect_ratio": float — desired width/height ratio (optional)
        package_area_mm2: Total die area in square millimeters
        aspect_ratio: Die aspect ratio (width/height). Default 1.0 (square).

    Returns:
        dict with keys:
            - "die_width_mm": float — die width in millimeters
            - "die_height_mm": float — die height in millimeters
            - "placed": list of dicts, each with:
                - "id": str
                - "x_mm": float — x position in millimeters
                - "y_mm": float — y position in millimeters
                - "width_mm": float — placed width in millimeters
                - "height_mm": float — placed height in millimeters
            - "unused_area_mm2": float — remaining die area

    Raises:
        ValueError: If chiplets total area exceeds package area

    Example:
        >>> chiplets = [
        ...     {"id": "mem0", "area_mm2": 25.0},
        ...     {"id": "cpu0", "area_mm2": 50.0},
        ...     {"id": "io0", "area_mm2": 10.0},
        ... ]
        >>> fp = allocate_floorplan(chiplets, package_area_mm2=100.0, aspect_ratio=1.5)
        >>> fp["die_width_mm"], fp["die_height_mm"]
        (12.25, 8.17)
    """
    if not chiplets:
        raise ValueError("chiplets list must not be empty")

    total_chiplet_area = sum(c.get("area_mm2", 0) for c in chiplets)
    if total_chiplet_area > package_area_mm2:
        raise ValueError(
            f"Total chiplet area {total_chiplet_area:.2f} mm² exceeds package area "
            f"{package_area_mm2:.2f} mm²"
        )

    # Determine die dimensions from area and aspect ratio
    # width * height = package_area, width / height = aspect_ratio
    # => width = sqrt(package_area * aspect_ratio), height = sqrt(package_area / aspect_ratio)
    die_width_mm = (package_area_mm2 * aspect_ratio) ** 0.5
    die_height_mm = package_area_mm2 / die_width_mm

    # Sort chiplets by area (largest first) for better packing
    sorted_chiplets = sorted(chiplets, key=lambda c: c.get("area_mm2", 0), reverse=True)

    # Build slicing tree placement
    placed = _build_slicing_tree(
        sorted_chiplets, die_width_mm, die_height_mm
    )

    # Compute unused area
    placed_area = sum(p["width_mm"] * p["height_mm"] for p in placed)
    unused_area_mm2 = package_area_mm2 - placed_area

    return {
        "die_width_mm": round(die_width_mm, 2),
        "die_height_mm": round(die_height_mm, 2),
        "placed": placed,
        "unused_area_mm2": round(unused_area_mm2, 2),
    }


def _build_slicing_tree(
    chiplets: list[dict], avail_width: float, avail_height: float, x_offset: float = 0.0, y_offset: float = 0.0
) -> list[dict]:
    """Recursively build slicing tree placement for chiplets.

    Alternates between horizontal and vertical cuts at each level of the tree.
    Largest chiplet goes first, and the remaining space is split accordingly.

    Args:
        chiplets: List of chiplet dicts sorted by area (largest first)
        avail_width: Available width for this sub-region
        avail_height: Available height for this sub-region
        x_offset: X offset of this sub-region
        y_offset: Y offset of this sub-region

    Returns:
        List of placed chiplet dicts with x, y, width, height
    """
    if not chiplets:
        return []

    if len(chiplets) == 1:
        # Place single chiplet in remaining area
        c = chiplets[0]
        area = c.get("area_mm2", 0)
        # Use preferred aspect ratio if available, otherwise fill available space
        preferred_ar = c.get("aspect_ratio", 1.0)
        w = (area * preferred_ar) ** 0.5
        h = area / w
        # Scale to fit within available space while maintaining ratio
        scale_w = avail_width / w
        scale_h = avail_height / h
        scale = min(scale_w, scale_h, 1.0)  # Don't enlarge beyond available

        actual_w = w * scale
        actual_h = h * scale
        actual_x = x_offset + (avail_width - actual_w) / 2
        actual_y = y_offset + (avail_height - actual_h) / 2

        return [
            {
                "id": c.get("id", "chiplet"),
                "x_mm": round(actual_x, 4),
                "y_mm": round(actual_y, 4),
                "width_mm": round(actual_w, 4),
                "height_mm": round(actual_h, 4),
            }
        ]

    # Determine cut direction: alternate based on number of chiplets
    # Even count -> vertical cut, Odd count -> horizontal cut
    # This creates a balanced slicing tree
    n = len(chiplets)
    if n % 2 == 0:
        # Vertical cut: split width, keep full height
        # Find split point based on area proportions
        total_area = sum(c.get("area_mm2", 0) for c in chiplets)
        cumulative = 0
        split_idx = 0
        for i, c in enumerate(chiplets):
            cumulative += c.get("area_mm2", 0)
            if cumulative >= total_area / 2:
                split_idx = i + 1
                break

        split_width = avail_width * (cumulative / total_area)
        split_height = avail_height

        left_chiplets = chiplets[:split_idx]
        right_chiplets = chiplets[split_idx:]

        left_placed = _build_slicing_tree(
            left_chiplets, split_width, split_height, x_offset, y_offset
        )
        right_placed = _build_slicing_tree(
            right_chiplets,
            avail_width - split_width,
            split_height,
            x_offset + split_width,
            y_offset,
        )
        return left_placed + right_placed
    else:
        # Horizontal cut: split height, keep full width
        total_area = sum(c.get("area_mm2", 0) for c in chiplets)
        cumulative = 0
        split_idx = 0
        for i, c in enumerate(chiplets):
            cumulative += c.get("area_mm2", 0)
            if cumulative >= total_area / 2:
                split_idx = i + 1
                break

        split_height = avail_height * (cumulative / total_area)
        split_width = avail_width

        top_chiplets = chiplets[:split_idx]
        bottom_chiplets = chiplets[split_idx:]

        top_placed = _build_slicing_tree(
            top_chiplets, split_width, split_height, x_offset, y_offset
        )
        bottom_placed = _build_slicing_tree(
            bottom_chiplets,
            split_width,
            avail_height - split_height,
            x_offset,
            y_offset + split_height,
        )
        return top_placed + bottom_placed


def floorplan_to_svg(floorplan: dict, title: str = "Floorplan") -> str:
    """Convert a floorplan dict to an SVG string.

    Args:
        floorplan: Floorplan dict as returned by allocate_floorplan()
        title: Title for the SVG canvas

    Returns:
        SVG string showing the die floorplan with labeled blocks

    Example:
        >>> fp = allocate_floorplan([{"id": "cpu0", "area_mm2": 50.0}], 100.0)
        >>> svg = floorplan_to_svg(fp, "Test Floorplan")
        >>> "<svg" in svg
        True
    """
    die_width = floorplan["die_width_mm"]
    die_height = floorplan["die_height_mm"]
    placed = floorplan["placed"]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{die_width}" height="{die_height}" viewBox="0 0 {die_width} {die_height}" style="background:#f5f5f5">',
        f'  <title>{title}</title>',
    ]

    # Add die boundary
    svg_parts.append(
        f'  <rect x="0" y="0" width="{die_width}" height="{die_height}" fill="none" stroke="#1a1a1a" stroke-width="2"/>'
    )

    # Color palette for different chiplet types
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7"]

    for i, block in enumerate(placed):
        cid = block["id"]
        x = block["x_mm"]
        y = block["y_mm"]
        w = block["width_mm"]
        h = block["height_mm"]
        color = colors[i % len(colors)]

        # Draw chiplet block
        svg_parts.append(
            f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" fill-opacity="0.8" stroke="#ffffff" stroke-width="0.5"/>'
        )

        # Add label - truncated if too long
        label = cid[:15] if len(cid) > 15 else cid
        svg_parts.append(
            f'  <text x="{x + w/2}" y="{y + h/2}" text-anchor="middle" dominant-baseline="middle" fill="#ffffff" font-size="8" font-family="sans-serif">{label}</text>'
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def floorplan_to_json(floorplan: dict) -> str:
    """Convert a floorplan dict to JSON string for web visualization.

    Strips internal metadata and returns only the essential placement data.

    Args:
        floorplan: Floorplan dict as returned by allocate_floorplan()

    Returns:
        JSON string suitable for web visualization

    Example:
        >>> fp = allocate_floorplan([{"id": "cpu0", "area_mm2": 50.0}], 100.0)
        >>> json_str = floorplan_to_json(fp)
        >>> len(json_str) > 0
        True
    """
    # Essential fields for web visualization
    data = {
        "die_width_mm": floorplan["die_width_mm"],
        "die_height_mm": floorplan["die_height_mm"],
        "placed": [
            {
                "id": p["id"],
                "x_mm": p["x_mm"],
                "y_mm": p["y_mm"],
                "width_mm": p["width_mm"],
                "height_mm": p["height_mm"],
            }
            for p in floorplan["placed"]
        ],
        "unused_area_mm2": floorplan["unused_area_mm2"],
    }
    return json.dumps(data, indent=2)


def check_floorplan_drc(
    floorplan: dict, min_spacing_um: float = 0.0
) -> list[dict[str, Any]]:
    """Check floorplan DRC (Design Rules Check) violations.

    Checks for:
    - Overlapping blocks
    - Spacing violations between blocks
    - Boundary violations (blocks extending beyond die)

    Args:
        floorplan: Floorplan dict as returned by allocate_floorplan()
        min_spacing_um: Minimum spacing between blocks in micrometers
                      (default 0.0 = no spacing check, use >0 for spacing checks)

    Returns:
        list of dicts, each with:
            - "type": str — violation type ("overlap", "spacing", "boundary")
            - "block1": dict — first block involved
            - "block2": dict | None — second block (None for boundary violations)
            - "message": str — descriptive error message
            - "severity": str — "error" or "warning"

    Example:
        >>> fp = allocate_floorplan([{"id": "cpu0", "area_mm2": 50.0}], 100.0)
        >>> violations = check_floorplan_drc(fp, min_spacing_um=100.0)
        >>> len(violations) == 0
        True
    """
    violations = []
    placed = floorplan["placed"]
    die_width = floorplan["die_width_mm"]
    die_height = floorplan["die_height_mm"]

    # Convert spacing from um to mm
    min_spacing_mm = min_spacing_um / 1000.0

    # Check boundary violations
    for block in placed:
        x = block["x_mm"]
        y = block["y_mm"]
        w = block["width_mm"]
        h = block["height_mm"]

        # Check if block extends beyond die boundary
        if x < 0:
            violations.append(
                {
                    "type": "boundary",
                    "block1": block,
                    "block2": None,
                    "message": f"Block {block['id']} extends beyond left boundary (x={x} < 0)",
                    "severity": "error",
                }
            )
        if y < 0:
            violations.append(
                {
                    "type": "boundary",
                    "block1": block,
                    "block2": None,
                    "message": f"Block {block['id']} extends beyond bottom boundary (y={y} < 0)",
                    "severity": "error",
                }
            )
        if x + w > die_width:
            violations.append(
                {
                    "type": "boundary",
                    "block1": block,
                    "block2": None,
                    "message": f"Block {block['id']} extends beyond right boundary (x+w={x+w} > {die_width})",
                    "severity": "error",
                }
            )
        if y + h > die_height:
            violations.append(
                {
                    "type": "boundary",
                    "block1": block,
                    "block2": None,
                    "message": f"Block {block['id']} extends beyond top boundary (y+h={y+h} > {die_height})",
                    "severity": "error",
                }
            )

    # Check overlap and spacing violations between all pairs
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            b1 = placed[i]
            b2 = placed[j]

            x1 = b1["x_mm"]
            y1 = b1["y_mm"]
            w1 = b1["width_mm"]
            h1 = b1["height_mm"]

            x2 = b2["x_mm"]
            y2 = b2["y_mm"]
            w2 = b2["width_mm"]
            h2 = b2["height_mm"]

            # Calculate overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            has_overlap = overlap_x > 0 and overlap_y > 0

            # Calculate spacing (minimum distance between blocks)
            # Horizontal spacing
            h_spacing = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
            # Vertical spacing
            v_spacing = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))
            min_spacing = min(h_spacing, v_spacing)

            if has_overlap:
                violations.append(
                    {
                        "type": "overlap",
                        "block1": b1,
                        "block2": b2,
                        "message": f"Blocks {b1['id']} and {b2['id']} overlap "
                        f"(overlap=({overlap_x:.2f}mm x {overlap_y:.2f}mm))",
                        "severity": "error",
                    }
                )
            elif min_spacing < min_spacing_mm and min_spacing_mm > 0:
                violations.append(
                    {
                        "type": "spacing",
                        "block1": b1,
                        "block2": b2,
                        "message": f"Blocks {b1['id']} and {b2['id']} have spacing "
                        f"{min_spacing*1000:.1f}um < min_spacing_{min_spacing_um}um",
                        "severity": "warning",
                    }
                )

    return violations