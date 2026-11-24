"""3D die cross-section renderer using Three.js.

Generates interactive HTML with Three.js for 3D die visualization showing
all layers, chiplets, and optional power heatmap overlay.
"""

from __future__ import annotations

from typing import Any


# Import TrackedValue and Provenance from chiplab schemas if available
try:
    from chiplab.schemas.common import TrackedValue, Provenance
except ImportError:
    TrackedValue = None  # type: ignore
    Provenance = None  # type: ignore


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def _value_to_color(value: float, min_val: float, max_val: float) -> str:
    """Map a value to a color in the blue→green→yellow→red gradient.

    Args:
        value: The value to map
        min_val: Minimum value in range
        max_val: Maximum value in range

    Returns:
        CSS color string (rgb format)
    """
    ratio = _clamp((value - min_val) / (max_val - min_val) if max_val > min_val else 0.0, 0.0, 1.0)

    # Color gradient: blue(0) -> green(0.333) -> yellow(0.667) -> red(1.0)
    if ratio < 1 / 3:
        r = 0
        g = int(255 * (ratio * 3))
        b = int(255 * (1 - ratio * 3))
    elif ratio < 2 / 3:
        r = int(255 * ((ratio - 1 / 3) * 3))
        g = 255
        b = 0
    else:
        r = 255
        g = int(255 * (1 - (ratio - 2 / 3) * 3))
        b = 0

    return f"rgb({r},{g},{b})"


def render_die_3d(
    floorplan: dict,
    layer_stack: list[dict],
    power_map: dict | None = None,
) -> str:
    """Generate interactive HTML with Three.js 3D die visualization.

    Creates a 3D cross-section of the die showing all layers, chiplets positioned
    according to the floorplan, and optional power heatmap overlay color-coded by
    power density or temperature.

    Args:
        floorplan: Floorplan dict as returned by allocate_floorplan()
        layer_stack: List of layer dicts, each with keys:
            - "name": str — layer name (e.g., "substrate", "metal_1", "active")
            - "thickness_mm": float — layer thickness
            - "material": str — material type (affects rendering)
        power_map: Optional dict mapping chiplet id to power dict with:
            - "power_watts": float
            - "area_mm2": float
        Returns:
            Self-contained HTML string with embedded Three.js scene

    Raises:
        ValueError: If floorplan has no placed chiplets or layer_stack is empty

    Example:
        >>> fp = allocate_floorplan(
        ...     [{"id": "cpu0", "area_mm2": 50.0, "aspect_ratio": 1.0}],
        ...     100.0, aspect_ratio=1.0
        ... )
        >>> layers = [
        ...     {"name": "substrate", "thickness_mm": 0.5, "material": "silicon"},
        ...     {"name": "active", "thickness_mm": 0.1, "material": "silicon"},
        ... ]
        >>> html = render_die_3d(fp, layers, power_map={"cpu0": {"power_watts": 25.0, "area_mm2": 50.0}})
        >>> "<three" in html.lower()
        True
    """
    if not layer_stack:
        raise ValueError("layer_stack must not be empty")

    if not floorplan or not floorplan.get("placed"):
        raise ValueError("floorplan must have placed chiplets")

    die_width = floorplan["die_width_mm"]
    die_height = floorplan["die_height_mm"]
    placed = floorplan["placed"]

    # Determine power density range if power_map provided
    has_power = power_map is not None and len(power_map) > 0
    power_min = 0.0
    power_max = 0.0

    if has_power:
        power_values = []
        for cid, pinfo in power_map.items():
            power_w = pinfo.get("power_watts", 0)
            area = pinfo.get("area_mm2", 0)
            if area > 0:
                power_values.append(power_w / area)
        if power_values:
            power_min = min(power_values)
            power_max = max(power_values)

    # Compute z-layer positions based on stack
    total_thickness = sum(layer.get("thickness_mm", 0.1) for layer in layer_stack)
    layer_positions = []
    current_z = 0.0
    for layer in layer_stack:
        layer_positions.append(current_z)
        current_z += layer.get("thickness_mm", 0.1)
    # Normalize so center is at z=0
    offset = total_thickness / 2
    layer_positions = [z - offset for z in layer_positions]

    # Build chiplet positions lookup
    chiplet_positions = {}
    for block in placed:
        chiplet_positions[block["id"]] = {
            "x": block["x_mm"],
            "y": block["y_mm"],
            "w": block["width_mm"],
            "h": block["height_mm"],
        }

    # Generate HTML - build parts manually to avoid f-string issues with curly braces
    html_parts = []

    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang=\"en\">")
    html_parts.append("<head>")
    html_parts.append("    <meta charset=\"UTF-8\">")
    html_parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html_parts.append("    <title>3D Die Visualization</title>")
    html_parts.append("    <style>")
    html_parts.append("        body { margin: 0; overflow: hidden; font-family: sans-serif; }")
    html_parts.append('        #canvas { width: 100%; height: 100%; }')
    html_parts.append("        .info-panel {")
    html_parts.append("            position: absolute;")
    html_parts.append("            top: 20px;")
    html_parts.append("            left: 20px;")
    html_parts.append("            background: rgba(255,255,255,0.9);")
    html_parts.append("            padding: 15px;")
    html_parts.append("            border-radius: 8px;")
    html_parts.append("            box-shadow: 0 2px 10px rgba(0,0,0,0.3);")
    html_parts.append("            max-width: 300px;")
    html_parts.append("        }")
    html_parts.append("        .info-panel h3 { margin-top: 0; }")
    html_parts.append("        .layer-info {")
    html_parts.append("            margin: 10px 0;")
    html_parts.append("            padding: 5px;")
    html_parts.append("            background: #f0f0f0;")
    html_parts.append("            border-radius: 4px;")
    html_parts.append("        }")
    html_parts.append("    </style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("    <div class=\"info-panel\">")
    tag = "    <h3>3D Die Cross-Section</h3>"
    html_parts.append(tag.replace("3D Die Cross-Section", f"3D Die Cross-Section"))
    tag2 = "    <p>Die size: {} mm × {} mm</p>"
    html_parts.append(tag2.format(die_width, die_height))
    tag3 = "    <p>Layers: {}</p>"
    html_parts.append(tag3.format(len(layer_stack)))
    html_parts.append("    </div>")
    html_parts.append('    <canvas id="canvas"></canvas>')

    # Import map
    html_parts.append("    <script type=\"importmap\">")
    html_parts.append("    {")
    html_parts.append('        "imports": {')
    html_parts.append('            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",')
    html_parts.append('            "three/controls/OrbitControls": "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js"')
    html_parts.append("        }")
    html_parts.append("    }")
    html_parts.append("    </script>")

    html_parts.append("    <script>")
    html_parts.append("        const scene = new THREE.Scene();")
    html_parts.append("        scene.background = new THREE.Color(0xf0f0f0);")
    html_parts.append("        const camera = new THREE.PerspectiveCamera(")
    html_parts.append("            60,")
    html_parts.append("            window.innerWidth / window.innerHeight,")
    html_parts.append("            0.1,")
    html_parts.append("            1000")
    html_parts.append("        );")
    tag4 = "        camera.position.set({0}, {1}, {2});"
    html_parts.append(tag4.format(die_width * 0.8, die_height * 0.8, die_width * 1.2))
    html_parts.append("        const renderer = new THREE.WebGLRenderer({ antialias: true });")
    html_parts.append("        renderer.setSize(window.innerWidth, window.innerHeight);")
    html_parts.append("        document.body.appendChild(renderer.domElement);")
    html_parts.append("        const controls = new THREE.OrbitControls(camera, renderer.domElement);")
    html_parts.append("        controls.enableDamping = true;")
    html_parts.append("        // Add lighting")
    html_parts.append("        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);")
    html_parts.append("        scene.add(ambientLight);")
    html_parts.append("        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);")
    html_parts.append("        dirLight.position.set(50, 50, 50);")
    html_parts.append("        scene.add(dirLight);")

    # Draw Layers section header
    html_parts.append("        // === Draw Layers ===")
    html_parts.append("        const layerThicknesses = [")

    # Add layer thicknesses
    for i, layer in enumerate(layer_stack):
        thickness = layer.get("thickness_mm", 0.1)
        if i < len(layer_stack) - 1:
            html_parts.append("            " + str(thickness) + ",")
        else:
            html_parts.append("            " + str(thickness))

    html_parts.append("        ];")

    # Draw each layer as semi-transparent plane
    html_parts.append("        for (let li = 0; li < layerThicknesses.length; li++) {")
    html_parts.append("            const thickness = layerThicknesses[li];")
    html_parts.append("            const zPos = layerPositions[li];")
    html_parts.append("            const geometry = new THREE.PlaneGeometry(")
    html_parts.append("                " + str(die_width) + ",")
    html_parts.append("                " + str(die_height) + "")
    html_parts.append("            );")
    html_parts.append("            const material = new THREE.MeshStandardMaterial({")
    html_parts.append("                color: 0xdddddd,")
    html_parts.append("                opacity: 0.3,")
    html_parts.append("                transparent: true,")
    html_parts.append("                side: THREE.DoubleSide")
    html_parts.append("            };")
    html_parts.append("            const layerMesh = new THREE.Mesh(geometry, material);")
    html_parts.append("            layerMesh.rotation.x = -Math.PI / 2;")
    html_parts.append("            layerMesh.position.z = zPos;")
    html_parts.append("            scene.add(layerMesh);")
    html_parts.append("        }")

    # Draw Chiplets on Active Layer
    html_parts.append("        const activeZ = layerPositions[layerPositions.length - 1] || 0;")

    for i, chiplet in enumerate(placed):
        x = chiplet["x_mm"]
        y = chiplet["y_mm"]
        w = chiplet["width_mm"]
        h = chiplet["height_mm"]
        cid = chiplet["id"]

        html_parts.append("        // Chiplet: " + cid)

        html_parts.append("        const geometry" + str(i) + " = new THREE.BoxGeometry(w, h, 0.2);")

        # Color based on index for variety
        html_parts.append("        const colors = [0x4e79a7, 0xf28e2b, 0xe15759, 0x76b7b2, 0x59a14f];")
        html_parts.append("        const colorIndex = " + str(i) + " % colors.length;")
        html_parts.append("        const material = new THREE.MeshStandardMaterial({")
        html_parts.append("            color: colors[colorIndex],")
        html_parts.append("            metalness: 0.3,")
        html_parts.append("            roughness: 0.6")
        html_parts.append("        });")

        html_parts.append("        const chipletMesh" + str(i) + " = new THREE.Mesh(geometry" + str(i) + ", material);")
        html_parts.append(
            "        chipletMesh" + str(i) + ".position.set(x + w / 2 - die_width / 2, -y - h / 2 + die_height / 2, activeZ);"
        )
        html_parts.append("        chipletMesh" + str(i) + ".rotation.x = Math.PI / 2;")
        html_parts.append("        scene.add(chipletMesh" + str(i) + ");")

        # Add label (simple cube placeholder)
        html_parts.append("        const labelMaterial" + str(i) + " = new THREE.MeshBasicMaterial({")
        html_parts.append("            color: 0xffffff,")
        html_parts.append("            transparent: true,")
        html_parts.append("            opacity: 0.8")
        html_parts.append("        };")
        html_parts.append("        const labelGeometry" + str(i) + " = new THREE.PlaneGeometry(w * 0.8, h * 0.2);")
        html_parts.append("        const labelMesh" + str(i) + " = new THREE.Mesh(labelGeometry" + str(i) + ", labelMaterial" + str(i) + ");")
        html_parts.append("        labelMesh" + str(i) + ".position.set(0, 0, activeZ + 0.1);")
        html_parts.append("        labelMesh" + str(i) + ".rotation.x = Math.PI / 2;")
        html_parts.append("        scene.add(labelMesh" + str(i) + ");")

    # Draw Power Heatmap Overlay (if provided)
    if has_power:
        html_parts.append("        // === Draw Power Heatmap Overlay ===")
        html_parts.append("        const gridSize = 50;")
        html_parts.append("        const cellWidth = " + str(die_width) + " / gridSize;")
        html_parts.append("        const cellHeight = " + str(die_height) + " / gridSize;")
        html_parts.append("        const powerDensity = new Float32Array(gridSize * gridSize);")
        html_parts.append("        powerDensity.fill(0);")

        html_parts.append("        for (const chiplet of placed) {")
        html_parts.append("            const cid = chiplet.id;")
        html_parts.append("            if (powerMap && cid in powerMap) {")
        html_parts.append("                const pInfo = powerMap[cid];")
        html_parts.append("                const powerW = pInfo.power_watts || 0;")
        html_parts.append("                const areaMM2 = pInfo.area_mm2 || 1;")
        html_parts.append("                const density = powerW / areaMM2;")

        html_parts.append("                const xStart = chiplet.x;")
        html_parts.append("                const yStart = chiplet.y;")
        html_parts.append("                const w = chiplet.w;")
        html_parts.append("                const h = chiplet.h;")

        html_parts.append("                const colStart = Math.max(0, Math.floor(xStart / cellWidth));")
        html_parts.append("                const colEnd = Math.min(gridSize, Math.ceil((xStart + w) / cellWidth));")
        html_parts.append("                const rowStart = Math.max(0, Math.floor(yStart / cellHeight));")
        html_parts.append("                const rowEnd = Math.min(gridSize, Math.ceil((yStart + h) / cellHeight));")

        html_parts.append("                const cellsCount = (rowEnd - rowStart) * (colEnd - colStart);")
        html_parts.append("                const densityPerCell = cellsCount > 0 ? density / cellsCount : 0;")

        html_parts.append("                for (let r = rowStart; r < rowEnd; r++) {")
        html_parts.append("                    for (let c = colStart; c < colEnd; c++) {")
        html_parts.append("                    powerDensity[r * gridSize + c] += densityPerCell;")
        html_parts.append("                    }")
        html_parts.append("                }")
        html_parts.append("            }")
        html_parts.append("        }")

        html_parts.append("        const maxDensity = Math.max(...powerDensity);")
        html_parts.append("        const minDensity = Math.min(...powerDensity);")

        html_parts.append("        for (let i = 0; i < gridSize * gridSize; i++) {")
        html_parts.append("            const density = powerDensity[i];")
        html_parts.append("            const ratio = (density - minDensity) / (maxDensity - minDensity + 0.001);")
        html_parts.append("            const color = _valueToColor(ratio, 0, 1);")

        html_parts.append("            const c = Math.floor(i % gridSize);")
        html_parts.append("            const r = Math.floor(i / gridSize);")

        html_parts.append("            const x = c * cellWidth;")
        html_parts.append("            const y = r * cellHeight;")

        html_parts.append("            const geometry = new THREE.PlaneGeometry(cellWidth, cellHeight);")
        html_parts.append("            const material = new THREE.MeshStandardMaterial({")
        html_parts.append("                color: new THREE.Color(color),")
        html_parts.append("                opacity: 0.4,")
        html_parts.append("                transparent: true,")
        html_parts.append("                side: THREE.DoubleSide")
        html_parts.append("            };")
        html_parts.append("            const cellMesh = new THREE.Mesh(geometry, material);")
        html_parts.append(
            "            cellMesh.position.set(x + cellWidth / 2 - " + str(die_width) + " / 2,"
        )
        html_parts.append(
            "                -y - cellHeight / 2 + " + str(die_height) + " / 2,"
        )
        html_parts.append("                activeZ + 0.15")
        html_parts.append("            );")
        html_parts.append("            cellMesh.rotation.x = -Math.PI / 2;")
        html_parts.append("            scene.add(cellMesh);")
        html_parts.append("        }")
    # End power overlay section

    # Animation loop
    html_parts.append("        // Animation loop")
    html_parts.append("        function animate() {")
    html_parts.append("            requestAnimationFrame(animate);")
    html_parts.append("            controls.update();")
    html_parts.append("            renderer.render(scene, camera);")
    html_parts.append("        }")
    html_parts.append("        animate();")

    html_parts.append("    </script>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)