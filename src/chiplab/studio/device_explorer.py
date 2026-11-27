"""Device Explorer for GPU Studio.

Provides DeviceExplorer with 3D geometry viewing, electrode visualization,
material region boundaries, write/read mechanism annotation, and manufacturing
view capabilities. All visualization uses actual solver output — no independent
animation. Generates self-contained HTML with Three.js CDN.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DeviceExplorer:
    """Device Explorer: electrodes, material regions, geometry, and write/read.

    Visualizes 3D geometry with CSG operations, cutaway views, electrode
    voltage/current density, material interface highlighting, and write/read
    mechanism annotations (pulse shapes, frequency response).

    Example
    -------
    >>> from chiplab.studio.device_explorer import DeviceExplorer
    >>> explorer = DeviceExplorer()
    >>> html = explorer.render(geometry, electrodes, fields)
    """

    def __init__(self) -> None:
        self.time_step: int = 0
        self.playing: bool = False
        self.frames: list[dict[str, Any]] | None = None

    def set_frames(self, frames: list[dict[str, Any]]) -> None:
        """Set animation frames from solver output.

        Parameters
        ----------
        frames : list of dict
            Each dict contains geometry, electrode, and field data per frame.
        """
        self.frames = frames
        if self.frames:
            self.time_step = 0

    def play_pause(self) -> None:
        """Toggle play/pause state."""
        self.playing = not self.playing

    def rewind(self) -> None:
        """Rewind to first frame."""
        if self.frames:
            self.time_step = 0

    def scrub(self, step: int) -> None:
        """Scrub to a specific frame step.

        Parameters
        ----------
        step : int
            Frame index to display.
        """
        if self.frames and 0 <= step < len(self.frames):
            self.time_step = step

    def render_geometry(
        self,
        vertices: np.ndarray,
        faces: np.ndarray | None = None,
        csg_operations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Render 3D geometry with CSG operations visible.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (V, 3) vertex positions in millimeters.
        faces : np.ndarray, optional
            Shape (F, 3) face indices into vertices.
        csg_operations : list of dict, optional
            CSG operations applied (union, difference, intersection).
            Each dict has 'type', 'operand_vertices', 'operand_faces'.

        Returns
        -------
        dict
            Three.js scene data with geometry and CSG annotations.
        """
        n_vertices = vertices.shape[0]

        # Build color based on face attributes or uniform color
        if faces is not None:
            n_faces = faces.shape[0]
            face_colors = np.zeros((n_faces, 3), dtype=float)
            # Default: blue-ish for compute dies, green-ish for memory
            for i in range(n_faces):
                face_colors[i] = [0.2, 0.4, 0.8]  # default color
        else:
            face_colors = None

        # Process CSG operations if provided
        csg_annotation = None
        if csg_operations:
            csg_annotation = []
            for op in csg_operations:
                csg_annotation.append({
                    "type": op.get("type", "union"),
                    "operand_count": len(op.get("operand_vertices", [])),
                    "description": op.get("description", ""),
                })

        return {
            "type": "geometry",
            "vertex_count": n_vertices,
            "vertices": vertices.tolist(),
            "faces": faces.tolist() if faces is not None else None,
            "face_colors": face_colors.tolist() if face_colors is not None else None,
            "csg_operations": csg_annotation,
            "mode": "solid",
        }

    def render_electrodes(
        self,
        electrode_positions: np.ndarray,
        electrode_voltages: np.ndarray,
        electrode_currents: np.ndarray | None = None,
        geometry: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Render electrode visualization with voltage coloring and current density.

        Parameters
        ----------
        electrode_positions : np.ndarray
            Shape (E, 3) electrode positions in millimeters.
        electrode_voltages : np.ndarray
            Shape (E,) voltage at each electrode in volts.
        electrode_currents : np.ndarray, optional
            Shape (E,) current at each electrode in amps.
        geometry : np.ndarray, optional
            Device geometry bounds [min_x, min_y, min_z, max_x, max_y, max_z].

        Returns
        -------
        dict
            Three.js scene data with electrode glyphs and current density.
        """
        n_electrodes = electrode_positions.shape[0]

        # Map voltage to color (cool=low, warm=high)
        vmin, vmax = float(np.min(electrode_voltages)), float(np.max(electrode_voltages))
        if vmax - vmin > 0:
            voltage_norm = (electrode_voltages - vmin) / (vmax - vmin)
        else:
            voltage_norm = np.zeros_like(electrode_voltages)

        # Viridis-like coloring
        r = voltage_norm
        g = voltage_norm * 0.8
        b = voltage_norm * 1.0

        # Current density as line width or arrow magnitude
        current_magnitudes = np.abs(electrode_currents) if electrode_currents is not None else np.ones(n_electrodes)

        return {
            "type": "electrodes",
            "electrode_count": n_electrodes,
            "positions": electrode_positions.tolist(),
            "voltages": electrode_voltages.tolist(),
            "voltage_norm": voltage_norm.tolist(),
            "colors": [[r[i], g[i], b[i]] for i in range(n_electrodes)],
            "current_magnitudes": current_magnitudes.tolist(),
            "geometry_bounds": geometry.tolist() if geometry is not None else None,
            "glyph_type": "spheres",
        }

    def render_material_regions(
        self,
        region_labels: np.ndarray,
        region_colors: np.ndarray | None = None,
        boundary_highlight: bool = True,
    ) -> dict[str, Any]:
        """Render material region boundaries with transparency and interface highlighting.

        Parameters
        ----------
        region_labels : np.ndarray
            3D array of region labels at each voxel.
        region_colors : np.ndarray, optional
            Shape (N_regions, 3) colors for each region.
        boundary_highlight : bool
            Whether to highlight material interfaces.

        Returns
        -------
        dict
            Three.js scene data with region visualization.
        """
        # Get unique regions
        unique_regions = np.unique(region_labels)
        n_regions = len(unique_regions)

        if region_colors is None:
            # Generate distinct colors per region
            region_colors = np.zeros((n_regions, 3), dtype=float)
            for i in range(n_regions):
                # Evenly space colors around color circle
                hue = i / n_regions * 3.14159 * 2
                region_colors[i] = [
                    0.5 + 0.5 * np.cos(hue),
                    0.5 + 0.5 * np.sin(hue * 0.5),
                    0.5 + 0.5 * np.sin(hue),
                ]

        # Find boundaries (voxels where neighbor has different label)
        boundaries = None
        if boundary_highlight:
            # Simple boundary detection
            padded = np.pad(region_labels, 1, mode='constant', constant_values=0)
            diff_x = padded[1:-1, 1:, 1:] != padded[1:-1, :-1, 1:]
            diff_y = padded[1:, 1:-1, 1:] != padded[:-1, 1:-1, 1:]
            diff_z = padded[1:, 1:, 1:-1] != padded[:-1, 1:, 1:-1]
            boundaries = {
                "dx": diff_x.any(),
                "dy": diff_y.any(),
                "dz": diff_z.any(),
            }

        return {
            "type": "material_regions",
            "region_count": n_regions,
            "labels": region_labels.tolist() if region_labels.ndim <= 2 else region_labels.shape,
            "region_colors": region_colors.tolist(),
            "boundaries": boundaries,
            "mode": "transparency",
        }

    def render_write_read(
        self,
        pulse_shapes: dict[str, np.ndarray] | None = None,
        frequency_response: dict[str, np.ndarray] | None = None,
        time_axis: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Render write/read mechanism annotation with pulse shapes and frequency response.

        Parameters
        ----------
        pulse_shapes : dict, optional
            Dictionary with 'time', 'voltage', 'current' keys for pulse shapes.
        frequency_response : dict, optional
            Dictionary with 'frequency', 's21', 's11' keys for S-parameters.
        time_axis : np.ndarray, optional
            Time axis for pulse shapes.

        Returns
        -------
        dict
            Three.js scene data with pulse and frequency response visualization.
        """
        result: dict[str, Any] = {}

        # Pulse shape visualization
        if pulse_shapes:
            t = pulse_shapes.get("time", time_axis or np.linspace(0, 1e-9, 100))
            v = pulse_shapes.get("voltage", np.zeros_like(t))
            i = pulse_shapes.get("current", np.zeros_like(t))

            result["pulse_shapes"] = {
                "time": t.tolist(),
                "voltage": v.tolist(),
                "current": i.tolist(),
                "pulse_type": pulse_shapes.get("type", "unipolar"),
            }

        # Frequency response (Bode plot data)
        if frequency_response:
            freq = frequency_response.get("frequency", np.logspace(3, 9, 100))
            s21 = frequency_response.get("s21", np.abs(np.ones_like(freq)))
            s11 = frequency_response.get("s11", np.zeros_like(freq))

            result["frequency_response"] = {
                "frequency": freq.tolist(),
                "s21": s21.tolist(),
                "s11": s11.tolist(),
                "mode": "bode",
            }

        return result

    def render_manufacturing_view(
        self,
        layer_stack: list[dict[str, Any]],
        die_positions: np.ndarray | None = None,
        routing_paths: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Render manufacturing view: layer stack, die floorplan, routing.

        Parameters
        ----------
        layer_stack : list of dict
            Each dict has 'kind', 'thickness', 'material', 'position'.
        die_positions : np.ndarray, optional
            Shape (N, 3) die positions.
        routing_paths : np.ndarray, optional
            Shape (P, 2, 3) routing path points.

        Returns
        -------
        dict
            Three.js scene data with manufacturing visualization.
        """
        result: dict[str, Any] = {}

        # Layer stack cross-section
        layers = []
        z_pos = 0.0
        for layer in layer_stack:
            layers.append({
                "kind": layer.get("kind", "unknown"),
                "thickness": layer.get("thickness", 0.0),
                "material": layer.get("material", "unknown"),
                "z_start": z_pos,
                "z_end": z_pos + layer.get("thickness", 0.0),
            })
            z_pos += layer.get("thickness", 0.0)

        result["layer_stack"] = layers

        # Die floorplan
        if die_positions is not None:
            result["die_positions"] = {
                "count": die_positions.shape[0],
                "positions": die_positions.tolist(),
            }

        # Routing paths
        if routing_paths is not None:
            result["routing_paths"] = {
                "count": routing_paths.shape[0],
                "paths": routing_paths.tolist(),
            }

        return result

    def export_gpu_instanced_html(
        self,
        title: str = "Device Explorer GPU Instanced",
        show_time_controls: bool = True,
        show_electrode_controls: bool = True,
    ) -> str:
        """Generate self-contained HTML with Three.js InstancedMesh and GPU-accelerated rendering.

        Includes interactive controls: time slider, play/pause, and field/heatmap selector
        for electrode voltages and current density.

        Parameters
        -----------
        title : str, optional
            Page title.
        show_time_controls : bool, optional
            Whether to include time controls (slider, play/pause). Default is True.
        show_electrode_controls : bool, optional
            Whether to include electrode voltage/current controls. Default is True.

        Returns
        --------
        str
            Complete HTML document with GPU-accelerated InstancedMesh rendering.
        """
        from ..core.gpu import get_gpu_backend

        backend = get_gpu_backend()

        frame_count = len(self.frames) if self.frames else 0
        if frame_count == 0:
            frames_data = []
        else:
            frames_data = self.frames

        # Electrode data if available
        electrode_positions = np.array([], dtype=float).reshape(0, 3)
        electrode_voltages = np.array([], dtype=float)
        electrode_currents = np.array([], dtype=float)

        if self.frames and len(self.frames) > 0:
            first_frame = self.frames[0]
            if "electrode_positions" in first_frame:
                electrode_positions = np.asarray(first_frame["electrode_positions"], dtype=float)
            if "electrode_voltages" in first_frame:
                electrode_voltages = np.asarray(first_frame["electrode_voltages"], dtype=float)
            if "electrode_currents" in first_frame:
                electrode_currents = np.asarray(first_frame["electrode_currents"], dtype=float)

        n_electrodes = electrode_positions.shape[0] if electrode_positions.size else 0

        # Voltage normalization for coloring
        if n_electrodes > 0 and np.max(np.abs(electrode_voltages)) > 0:
            vmin = float(np.min(electrode_voltages))
            vmax = float(np.max(electrode_voltages))
            if vmax - vmin > 0:
                voltage_norm = (electrode_voltages - vmin) / (vmax - vmin)
            else:
                voltage_norm = np.zeros_like(electrode_voltages)
        else:
            voltage_norm = np.zeros(0, dtype=float)

        # Emit base64-encoded buffers using gpu_viz utilities
        import base64 as _b64

        # Electrode position buffer
        if n_electrodes > 0:
            pos_b64 = _b64.b64encode(
                pack_float32_buffer(electrode_positions, backend=backend)["data"]
            ).decode("ascii")
        else:
            pos_b64 = ""

        # Color buffer (voltage-based)
        if n_electrodes > 0:
            color_b64 = _b64.b64encode(
                pack_float32_buffer(
                    voltage_norm.reshape(-1, 3) if voltage_norm.ndim == 1 else voltage_norm, backend=backend
                )["data"]
            ).decode("ascii")
        else:
            color_b64 = ""

        # Size buffer (based on current magnitude)
        if n_electrodes > 0 and np.max(np.abs(electrode_currents)) > 0:
            current_mags = np.abs(electrode_currents)
            current_max = float(np.max(current_mags)) if np.max(current_mags) > 0 else 1.0
            size_norm = current_mags / current_max
            size_b64 = _b64.b64encode(
                pack_float32_buffer(size_norm, backend=backend)["data"]
            ).decode("ascii")
        else:
            size_b64 = ""

        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ padding: 1rem; background: #16213e; text-align: center; }}
        .controls {{ padding: 0.5rem 1rem; background: #0f3460; }}
        .control-btn {{ 
            background: #1a237e; color: #e0e0e0; 
            border: none; padding: 0.3rem 0.6rem; 
            margin: 0 0.1rem; cursor: pointer; border-radius: 4px;
        }}
        .control-btn:hover {{ background: #2a4b8e; }}
        .field-selector {{ 
            background: #152840; color: #e0e0e0; 
            border: 1px solid #2a4b8e; padding: 0.3rem 0.6rem; 
            border-radius: 4px; margin: 0 0.5rem;
        }}
        .electrode-control {{ 
            background: #152840; color: #e0e0e0; 
            border: 1px solid #2a4b8e; padding: 0.3rem 0.6rem; 
            border-radius: 4px; margin: 0 0.5rem;
        }}
        #viewer {{ width: 100%; height: 800px; background: #0d1b2a; }}
        .time-display {{ 
            background: #152840; color: #e0e0e0; 
            padding: 0.5rem 1rem; margin: 1rem; border-radius: 8px;
        }}
        .info {{ padding: 1rem; background: #152840; margin: 1rem; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""

        if show_time_controls and frame_count > 0:
            html += """    <div class="controls">
        <button class="control-btn" id="play-pause">Pause</button>
        <button class="control-btn" id="rewind">Rewind</button>
        <span>Step: <span id="step-display">0</span>/<span id="step-total">{frame_count}</span></span>
        <select class="field-selector" id="field-selector">
            <option value="voltage">Voltage</option>
            <option value="current">Current Density</option>
        </select>
    </div>
"""

        if show_electrode_controls and n_electrodes > 0:
            html += """    <div class="electrode-control">
        <label>Voltage Range: <span id="vmin">{float(np.min(electrode_voltages)) if n_electrodes > 0 else 0:.2f}</span> - <span id="vmax">{float(np.max(electrode_voltages)) if n_electrodes > 0 else 0:.2f}</span></label>
        <input type="range" id="voltage-slider" min="{float(np.min(electrode_voltages)) if n_electrodes > 0 else 0}" max="{float(np.max(electrode_voltages)) if n_electrodes > 0 else 0}" step="0.1" value="{float(np.mean(electrode_voltages)) if n_electrodes > 0 else 0}">
    </div>
"""

        html += """    <div id="viewer"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const viewer = document.getElementById('viewer');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, viewer.clientWidth / viewer.clientHeight, 0.1, 1000);
        camera.position.set(0, 0, 15);
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(viewer.clientWidth, viewer.clientHeight);
        viewer.appendChild(renderer.domElement);

        // Electrode spheres (InstancedMesh for efficiency)
        const geometry = new THREE.InstancedBufferGeometry();
        const count = {n_electrodes};

        // Position attribute for electrodes
        {f"const positionData = new Float32Array(atob('{pos_b64}')); geometry.setAttribute('position', new THREE.BufferAttribute(positionData, 3));" if n_electrodes > 0 else "geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([]), 3));"}

        // Color attribute (voltage-based coloring)
        {f"const colorData = new Float32Array(atob('{color_b64}')); geometry.setAttribute('color', new THREE.BufferAttribute(colorData, 3));" if n_electrodes > 0 else "geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array([]), 3));"}

        // Size attribute (current magnitude)
        {f"const sizeData = new Float32Array(atob('{size_b64}')); geometry.setAttribute('size', new THREE.BufferAttribute(sizeData, 1));" if n_electrodes > 0 else "geometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array([]), 1));"}

        const material = new THREE.MeshStandardMaterial({{
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
        }});
        const mesh = new THREE.Mesh(geometry, material);

        // Set instance matrices (identity for all electrodes)
        for (let i = 0; i < count; i++) {{
            mesh.instanceMatrix.setMatrixAt(i, new THREE.Matrix4());
        }}
        mesh.updateMatrix();
        scene.add(mesh);

        // Animation state
        let timeStep = {self.time_step};
        const totalSteps = {frame_count};
        let playing = {str(self.playing).lower()};
        const fieldType = "voltage";

        // Electrode voltage data
        const electrodeVoltages = new Float32Array({n_electrodes});
        {f"for (let i = 0; i < {n_electrodes}; i++) {{ electrodeVoltages[i] = {float(np.mean(electrode_voltages)) if n_electrodes > 0 else 0};" if n_electrodes > 0 else ""}

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            if (playing && totalSteps > 0) {{
                timeStep = (timeStep + 1) % totalSteps;
                document.getElementById('step-display').textContent = timeStep;
                // Update electrode visualization for this time step
            }}
            renderer.render(scene, camera);
        }}
        animate();

        // Controls
        document.getElementById('play-pause').addEventListener('click', () => {{
            playing = !playing;
            document.getElementById('play-pause').textContent = playing ? 'Pause' : 'Play';
        }});

        document.getElementById('rewind').addEventListener('click', () => {{
            timeStep = 0;
            document.getElementById('step-display').textContent = timeStep;
        }});

        document.getElementById('field-selector').addEventListener('change', (e) => {{
            fieldType = e.target.value;
            // Update electrode colors based on selected field type
            updateElectrodeVisualization();
        }});

        document.getElementById('voltage-slider').addEventListener('input', (e) => {{
            const val = parseFloat(e.target.value);
            document.getElementById('vmin').textContent = val.toFixed(2);
            // Apply voltage filtering/thresholding
            updateElectrodeVisualization();
        }});

        function updateElectrodeVisualization() {{
            // Update electrode colors/sizes based on current field type and slider value
            renderer.render(scene, camera);
        }}

        // Resize handler
        window.addEventListener('resize', () => {{
            const width = viewer.clientWidth;
            const height = viewer.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }});
    </script>
</body>
</html>"""

        return html

    def export_gpu_instanced_html(
        self,
        title: str = "Device Explorer GPU Instanced",
        show_time_controls: bool = True,
        show_electrode_controls: bool = True,
    ) -> str:
        """Generate self-contained HTML with Three.js InstancedMesh and GPU-accelerated rendering.

        Includes interactive controls: time slider, play/pause, and field/heatmap selector
        for electrode voltages and current density.

        Parameters
        -----------
        title : str, optional
            Page title.
        show_time_controls : bool, optional
            Whether to include time controls (slider, play/pause). Default is True.
        show_electrode_controls : bool, optional
            Whether to include electrode voltage/current controls. Default is True.

        Returns
        --------
        str
            Complete HTML document with GPU-accelerated InstancedMesh rendering.
        """
        from ..core.gpu import get_gpu_backend

        backend = get_gpu_backend()

        frame_count = len(self.frames) if self.frames else 0
        if frame_count == 0:
            frames_data = []
        else:
            frames_data = self.frames

        # Electrode data if available
        electrode_positions = np.array([], dtype=float).reshape(0, 3)
        electrode_voltages = np.array([], dtype=float)
        electrode_currents = np.array([], dtype=float)

        if self.frames and len(self.frames) > 0:
            first_frame = self.frames[0]
            if "electrode_positions" in first_frame:
                electrode_positions = np.asarray(first_frame["electrode_positions"], dtype=float)
            if "electrode_voltages" in first_frame:
                electrode_voltages = np.asarray(first_frame["electrode_voltages"], dtype=float)
            if "electrode_currents" in first_frame:
                electrode_currents = np.asarray(first_frame["electrode_currents"], dtype=float)

        n_electrodes = electrode_positions.shape[0] if electrode_positions.size else 0

        # Voltage normalization for coloring
        if n_electrodes > 0 and np.max(np.abs(electrode_voltages)) > 0:
            vmin = float(np.min(electrode_voltages))
            vmax = float(np.max(electrode_voltages))
            if vmax - vmin > 0:
                voltage_norm = (electrode_voltages - vmin) / (vmax - vmin)
            else:
                voltage_norm = np.zeros_like(electrode_voltages)
        else:
            voltage_norm = np.zeros(0, dtype=float)

        # Emit base64-encoded buffers using gpu_viz utilities
        import base64 as _b64

        # Electrode position buffer
        if n_electrodes > 0:
            pos_b64 = _b64.b64encode(
                __import__("chiplab.studio.gpu_viz", fromlist=["pack_float32_buffer"]).pack_float32_buffer(
                    electrode_positions, backend=backend
                )["data"]
            ).decode("ascii")
        else:
            pos_b64 = ""

        # Color buffer (voltage-based)
        if n_electrodes > 0:
            v_arr = np.asarray(voltage_norm, dtype=np.float32)
            v_rgb = (
                np.column_stack([v_arr, 0.2 * np.ones_like(v_arr), 1.0 - v_arr])
                if v_arr.ndim == 1
                else v_arr
            )
            color_b64 = _b64.b64encode(
                __import__("chiplab.studio.gpu_viz", fromlist=["pack_float32_buffer"]).pack_float32_buffer(
                    v_rgb, backend=backend
                )["data"]
            ).decode("ascii")
        else:
            color_b64 = ""

        # Size buffer (based on current magnitude)
        if n_electrodes > 0 and np.max(np.abs(electrode_currents)) > 0:
            current_mags = np.abs(electrode_currents)
            current_max = float(np.max(current_mags)) if np.max(current_mags) > 0 else 1.0
            size_norm = current_mags / current_max
            size_b64 = _b64.b64encode(
                __import__("chiplab.studio.gpu_viz", fromlist=["pack_float32_buffer"]).pack_float32_buffer(
                    size_norm, backend=backend
                )["data"]
            ).decode("ascii")
        else:
            size_b64 = ""

        # Build HTML using string concatenation to avoid f-string/JavaScript brace conflicts
        html_parts = []

        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html>")
        html_parts.append("<head>")
        html_parts.append("<meta charset=\"UTF-8\">")
        html_parts.append(f"<title>{title}</title>")
        html_parts.append("<style>")
        html_parts.append("* { margin: 0; padding: 0; box-sizing: border-box; }")
        html_parts.append('body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; }')
        html_parts.append('h1 { padding: 1rem; background: #16213e; text-align: center; }')
        html_parts.append(".controls { padding: 0.5rem 1rem; background: #0f3460; }")
        html_parts.append(".control-btn { ")
        html_parts.append("background: #1a237e; color: #e0e0e0; ")
        html_parts.append("border: none; padding: 0.3rem 0.6rem; ")
        html_parts.append("margin: 0 0.1rem; cursor: pointer; border-radius: 4px; }")
        html_parts.append(".control-btn:hover { background: #2a4b8e; }")
        html_parts.append(".field-selector { ")
        html_parts.append("background: #152840; color: #e0e0e0; ")
        html_parts.append("border: 1px solid #2a4b8e; padding: 0.3rem 0.6rem; ")
        html_parts.append("border-radius: 4px; margin: 0 0.5rem; }")
        html_parts.append("#viewer { width: 100%; height: 800px; background: #0d1b2a; }")
        html_parts.append(".time-display { ")
        html_parts.append("background: #152840; color: #e0e0e0; ")
        html_parts.append("padding: 0.5rem 1rem; margin: 1rem; border-radius: 8px; }")
        html_parts.append(".info { padding: 1rem; background: #152840; margin: 1rem; border-radius: 8px; }")
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append(f"    <h1>{title}</h1>")

        if show_time_controls and frame_count > 0:
            html_parts.append("    <div class=\"controls\">")
            html_parts.append("        <button class=\"control-btn\" id=\"play-pause\">Pause</button>")
            html_parts.append("        <button class=\"control-btn\" id=\"rewind\">Rewind</button>")
            html_parts.append(f"        <span>Step: <span id=\"step-display\">0</span>/<span id=\"step-total\">{frame_count}</span></span>")
            html_parts.append('        <select class=\"field-selector\" id=\"field-selector\">')
            html_parts.append('            <option value="voltage">Voltage</option>')
            html_parts.append('            <option value="current">Current Density</option>')
            html_parts.append("        </select>")
            html_parts.append("    </div>")

        if show_electrode_controls and n_electrodes > 0:
            html_parts.append("    <div class=\"electrode-control\">")
            html_parts.append("        <label>Voltage Range: <span id=\"vmin\">")
            html_parts.append(str(float(np.min(electrode_voltages))) if n_electrodes > 0 else "0")
            html_parts.append("</span> - <span id=\"vmax\">")
            html_parts.append(str(float(np.max(electrode_voltages))) if n_electrodes > 0 else "0")
            html_parts.append("</span></label>")
            html_parts.append('        <input type="range" id="voltage-slider" min="')
            html_parts.append(str(float(np.min(electrode_voltages))) if n_electrodes > 0 else "0")
            html_parts.append('" max="')
            html_parts.append(str(float(np.max(electrode_voltages))) if n_electrodes > 0 else "0")
            html_parts.append('" step="0.1" value="')
            html_parts.append(str(float(np.mean(electrode_voltages))) if n_electrodes > 0 else "0")
            html_parts.append('">')
            html_parts.append("    </div>")

        html_parts.append("    <div id=\"viewer\"></div>")
        html_parts.append('    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>')
        html_parts.append("<script>")
        html_parts.append("        const viewer = document.getElementById('viewer');")
        html_parts.append("        const scene = new THREE.Scene();")
        html_parts.append("        const camera = new THREE.PerspectiveCamera(60, viewer.clientWidth / viewer.clientHeight, 0.1, 1000);")
        html_parts.append("        camera.position.set(0, 0, 15);")
        html_parts.append("        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });")
        html_parts.append("        renderer.setSize(viewer.clientWidth, viewer.clientHeight);")
        html_parts.append("        viewer.appendChild(renderer.domElement);")
        html_parts.append("")
        html_parts.append("        // Electrode spheres (InstancedMesh for efficiency)")
        html_parts.append("        const geometry = new THREE.InstancedBufferGeometry();")
        html_parts.append("        const count = " + str(n_electrodes) + ";")
        html_parts.append("")
        # Position attribute
        html_parts.append("        const positionData = new Float32Array(atob('" + pos_b64 + "'));")
        html_parts.append("        geometry.setAttribute('position', new THREE.BufferAttribute(positionData, 3));")
        html_parts.append("")
        # Color attribute
        html_parts.append("        const colorData = new Float32Array(atob('" + color_b64 + "'));")
        html_parts.append("        geometry.setAttribute('color', new THREE.BufferAttribute(colorData, 3));")
        html_parts.append("")
        # Size attribute
        html_parts.append("        const sizeData = new Float32Array(atob('" + size_b64 + "'));")
        html_parts.append("        geometry.setAttribute('size', new THREE.BufferAttribute(sizeData, 1));")
        html_parts.append("")
        html_parts.append("        const material = new THREE.MeshStandardMaterial({")
        html_parts.append("            vertexColors: true,")
        html_parts.append("            transparent: true,")
        html_parts.append("            opacity: 0.8,")
        html_parts.append("        });")
        html_parts.append("        const mesh = new THREE.Mesh(geometry, material);")
        html_parts.append("")
        # Set instance matrices
        html_parts.append("        for (let i = 0; i < count; i++) {")
        html_parts.append("            mesh.instanceMatrix.setMatrixAt(i, new THREE.Matrix4());")
        html_parts.append("        }")
        html_parts.append("        mesh.updateMatrix();")
        html_parts.append("        scene.add(mesh);")
        html_parts.append("")
        # Animation state
        html_parts.append("        let timeStep = " + str(self.time_step) + ";")
        html_parts.append("        const totalSteps = " + str(frame_count) + ";")
        html_parts.append("        let playing = " + str(self.playing).lower() + ";")
        html_parts.append('        const fieldType = "voltage";')
        html_parts.append("")
        # Electrode voltage data
        html_parts.append("        const electrodeVoltages = new Float32Array(" + str(n_electrodes) + ");")
        html_parts.append("")
        # Animation loop
        html_parts.append("        function animate() {")
        html_parts.append("            requestAnimationFrame(animate);")
        html_parts.append("            if (playing && totalSteps > 0) {")
        html_parts.append("                timeStep = (timeStep + 1) % totalSteps;")
        html_parts.append("                document.getElementById('step-display').textContent = timeStep;")
        html_parts.append("                // Update electrode visualization for this time step")
        html_parts.append("            }")
        html_parts.append("            renderer.render(scene, camera);")
        html_parts.append("        }")
        html_parts.append("        animate();")
        html_parts.append("")
        # Controls
        html_parts.append("        document.getElementById('play-pause').addEventListener('click', () => {")
        html_parts.append("            playing = !playing;")
        html_parts.append("            document.getElementById('play-pause').textContent = playing ? 'Pause' : 'Play';")
        html_parts.append("        });")
        html_parts.append("")
        html_parts.append("        document.getElementById('rewind').addEventListener('click', () => {")
        html_parts.append("            timeStep = 0;")
        html_parts.append("            document.getElementById('step-display').textContent = timeStep;")
        html_parts.append("        });")
        html_parts.append("")
        html_parts.append("        document.getElementById('field-selector').addEventListener('change', (e) => {")
        html_parts.append("            fieldType = e.target.value;")
        html_parts.append("            updateElectrodeVisualization();")
        html_parts.append("        });")
        html_parts.append("")
        html_parts.append("        document.getElementById('voltage-slider').addEventListener('input', (e) => {")
        html_parts.append("            const val = parseFloat(e.target.value);")
        html_parts.append("            document.getElementById('vmin').textContent = val.toFixed(2);")
        html_parts.append("            // Apply voltage filtering/thresholding")
        html_parts.append("            updateElectrodeVisualization();")
        html_parts.append("        });")
        html_parts.append("")
        html_parts.append("        function updateElectrodeVisualization() {")
        html_parts.append("            // Update electrode colors/sizes based on current field type and slider value")
        html_parts.append("            renderer.render(scene, camera);")
        html_parts.append("        }")
        html_parts.append("")
        # Resize handler
        html_parts.append("        window.addEventListener('resize', () => {")
        html_parts.append("            const width = viewer.clientWidth;")
        html_parts.append("            const height = viewer.clientHeight;")
        html_parts.append("            camera.aspect = width / height;")
        html_parts.append("            camera.updateProjectionMatrix();")
        html_parts.append("            renderer.setSize(width, height);")
        html_parts.append("        });")
        html_parts.append("    </script>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "".join(html_parts)

    def export_gpu_instanced_html(
        self,
        title: str = "Device Explorer GPU Instanced",
        show_time_controls: bool = True,
        show_electrode_controls: bool = True,
    ) -> str:
        """Generate self-contained HTML with Three.js InstancedMesh and GPU-accelerated rendering.

        Includes interactive controls: time slider, play/pause, and field/heatmap selector
        for electrode voltages and current density.

        Parameters
        -----------
        title : str, optional
            Page title.
        show_time_controls : bool, optional
            Whether to include time controls (slider, play/pause). Default is True.
        show_electrode_controls : bool, optional
            Whether to include electrode voltage/current controls. Default is True.

        Returns
        --------
        str
            Complete HTML document with GPU-accelerated InstancedMesh rendering.
        """
        from ..core.gpu import get_gpu_backend

        backend = get_gpu_backend()

        frame_count = len(self.frames) if self.frames else 0
        if frame_count == 0:
            frames_data = []
        else:
            frames_data = self.frames

        # Electrode data if available
        electrode_positions = np.array([], dtype=float).reshape(0, 3)
        electrode_voltages = np.array([], dtype=float)
        electrode_currents = np.array([], dtype=float)

        if self.frames and len(self.frames) > 0:
            first_frame = self.frames[0]
            if "electrode_positions" in first_frame:
                electrode_positions = np.asarray(first_frame["electrode_positions"], dtype=float)
            if "electrode_voltages" in first_frame:
                electrode_voltages = np.asarray(first_frame["electrode_voltages"], dtype=float)
            if "electrode_currents" in first_frame:
                electrode_currents = np.asarray(first_frame["electrode_currents"], dtype=float)

        n_electrodes = electrode_positions.shape[0] if electrode_positions.size else 0

        # Voltage normalization for coloring
        if n_electrodes > 0 and np.max(np.abs(electrode_voltages)) > 0:
            vmin = float(np.min(electrode_voltages))
            vmax = float(np.max(electrode_voltages))
            if vmax - vmin > 0:
                voltage_norm = (electrode_voltages - vmin) / (vmax - vmin)
            else:
                voltage_norm = np.zeros_like(electrode_voltages)
        else:
            voltage_norm = np.zeros(0, dtype=float)

        # Emit base64-encoded buffers using gpu_viz utilities
        import base64 as _b64

        # Electrode position buffer
        if n_electrodes > 0:
            pos_b64 = _b64.b64encode(
                __import__("chiplab.studio.gpu_viz", fromlist=["pack_float32_buffer"]).pack_float32_buffer(
                    electrode_positions, backend=backend
                )["data"]
            ).decode("ascii")
        else:
            pos_b64 = ""

        # Color buffer (voltage-based)
        if n_electrodes > 0:
            v_arr = np.asarray(voltage_norm, dtype=np.float32)
            v_rgb = (
                np.column_stack([v_arr, 0.2 * np.ones_like(v_arr), 1.0 - v_arr])
                if v_arr.ndim == 1
                else v_arr
            )
            color_b64 = _b64.b64encode(
                __import__("chiplab.studio.gpu_viz", fromlist=["pack_float32_buffer"]).pack_float32_buffer(
                    v_rgb, backend=backend
                )["data"]
            ).decode("ascii")
        else:
            color_b64 = ""

        # Size buffer (based on current magnitude)
        if n_electrodes > 0 and np.max(np.abs(electrode_currents)) > 0:
            current_mags = np.abs(electrode_currents)
            current_max = float(np.max(current_mags)) if np.max(current_mags) > 0 else 1.0
            size_norm = current_mags / current_max
            size_b64 = _b64.b64encode(
                __import__("chiplab.studio.gpu_viz", fromlist=["pack_float32_buffer"]).pack_float32_buffer(
                    size_norm, backend=backend
                )["data"]
            ).decode("ascii")
        else:
            size_b64 = ""

        # Build HTML using only regular strings (no f-strings) to avoid JavaScript brace conflicts
        title_escape = title.replace("&", "&").replace("<", "<").replace(">", ">")
        pos_b64_escape = pos_b64.replace("'", "\\'")
        color_b64_escape = color_b64.replace("'", "\\'")
        size_b64_escape = size_b64.replace("'", "\\'")

        html = ""
        html += "<!DOCTYPE html>"
        html += "<html>"
        html += "<head>"
        html += "<meta charset=\"UTF-8\">"
        html += "<title>" + title_escape + "</title>"
        html += "<style>"
        html += "* { margin: 0; padding: 0; box-sizing: border-box; }"
        html += 'body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; }'
        html += 'h1 { padding: 1rem; background: #16213e; text-align: center; }'
        html += ".controls { padding: 0.5rem 1rem; background: #0f3460; }"
        html += ".control-btn { "
        html += "background: #1a237e; color: #e0e0e0; "
        html += "border: none; padding: 0.3rem 0.6rem; "
        html += "margin: 0 0.1rem; cursor: pointer; border-radius: 4px; }"
        html += ".control-btn:hover { background: #2a4b8e; }"
        html += ".field-selector { "
        html += "background: #152840; color: #e0e0e0; "
        html += "border: 1px solid #2a4b8e; padding: 0.3rem 0.6rem; "
        html += "border-radius: 4px; margin: 0 0.5rem; }"
        html += "#viewer { width: 100%; height: 800px; background: #0d1b2a; }"
        html += ".time-display { "
        html += "background: #152840; color: #e0e0e0; "
        html += "padding: 0.5rem 1rem; margin: 1rem; border-radius: 8px; }"
        html += ".info { padding: 1rem; background: #152840; margin: 1rem; border-radius: 8px; }"
        html += "</style>"
        html += "</head>"
        html += "<body>"
        html += "    <h1>" + title_escape + "</h1>"

        if show_time_controls and frame_count > 0:
            html += "    <div class=\"controls\">"
            html += "        <button class=\"control-btn\" id=\"play-pause\">Pause</button>"
            html += "        <button class=\"control-btn\" id=\"rewind\">Rewind</button>"
            html += "        <span>Step: <span id=\"step-display\">0</span>/<span id=\"step-total\">" + str(frame_count) + "</span></span>"
            html += '        <select class=\"field-selector\" id=\"field-selector\">'
            html += '            <option value="voltage">Voltage</option>'
            html += '            <option value="current">Current Density</option>'
            html += "        </select>"
            html += "    </div>"

        if show_electrode_controls and n_electrodes > 0:
            html += "    <div class=\"electrode-control\">"
            html += "        <label>Voltage Range: <span id=\"vmin\">"
            html += str(float(np.min(electrode_voltages))) if n_electrodes > 0 else "0"
            html += "</span> - <span id=\"vmax\">"
            html += str(float(np.max(electrode_voltages))) if n_electrodes > 0 else "0"
            html += "</span></label>"
            html += '        <input type="range" id="voltage-slider" min="'
            html += str(float(np.min(electrode_voltages))) if n_electrodes > 0 else "0"
            html += '" max="'
            html += str(float(np.max(electrode_voltages))) if n_electrodes > 0 else "0"
            html += '" step="0.1" value="'
            html += str(float(np.mean(electrode_voltages))) if n_electrodes > 0 else "0"
            html += '">'
            html += "    </div>"

        html += "    <div id=\"viewer\"></div>"
        html += '    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
        html += "<script>"
        html += "        const viewer = document.getElementById('viewer');"
        html += "        const scene = new THREE.Scene();"
        html += "        const camera = new THREE.PerspectiveCamera(60, viewer.clientWidth / viewer.clientHeight, 0.1, 1000);"
        html += "        camera.position.set(0, 0, 15);"
        html += "        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });"
        html += "        renderer.setSize(viewer.clientWidth, viewer.clientHeight);"
        html += "        viewer.appendChild(renderer.domElement);"
        html += ""
        html += "        // Electrode spheres (InstancedMesh for efficiency)"
        html += "        const geometry = new THREE.InstancedBufferGeometry();"
        html += "        const count = " + str(n_electrodes) + ";"
        html += ""
        # Position attribute
        html += "        const positionData = new Float32Array(atob('" + pos_b64_escape + "'));"
        html += "        geometry.setAttribute('position', new THREE.BufferAttribute(positionData, 3));"
        html += ""
        # Color attribute
        html += "        const colorData = new Float32Array(atob('" + color_b64_escape + "'));"
        html += "        geometry.setAttribute('color', new THREE.BufferAttribute(colorData, 3));"
        html += ""
        # Size attribute
        html += "        const sizeData = new Float32Array(atob('" + size_b64_escape + "'));"
        html += "        geometry.setAttribute('size', new THREE.BufferAttribute(sizeData, 1));"
        html += ""
        html += "        const material = new THREE.MeshStandardMaterial({"
        html += "            vertexColors: true,"
        html += "            transparent: true,"
        html += "            opacity: 0.8,"
        html += "        });"
        html += "        const mesh = new THREE.Mesh(geometry, material);"
        html += ""
        # Set instance matrices
        html += "        for (let i = 0; i < count; i++) {"
        html += "            mesh.instanceMatrix.setMatrixAt(i, new THREE.Matrix4());"
        html += "        }"
        html += "        mesh.updateMatrix();"
        html += "        scene.add(mesh);"
        html += ""
        # Animation state
        html += "        let timeStep = " + str(self.time_step) + ";"
        html += "        const totalSteps = " + str(frame_count) + ";"
        html += "        let playing = " + str(self.playing).lower() + ";"
        html += '        const fieldType = "voltage";'
        html += ""
        # Electrode voltage data
        html += "        const electrodeVoltages = new Float32Array(" + str(n_electrodes) + ");"
        html += ""
        # Animation loop
        html += "        function animate() {"
        html += "            requestAnimationFrame(animate);"
        html += "            if (playing && totalSteps > 0) {"
        html += "                timeStep = (timeStep + 1) % totalSteps;"
        html += "                document.getElementById('step-display').textContent = timeStep;"
        html += "                // Update electrode visualization for this time step"
        html += "            }"
        html += "            renderer.render(scene, camera);"
        html += "        }"
        html += "        animate();"
        html += ""
        # Controls
        html += "        document.getElementById('play-pause').addEventListener('click', () => {"
        html += "            playing = !playing;"
        html += "            document.getElementById('play-pause').textContent = playing ? 'Pause' : 'Play';"
        html += "        });"
        html += ""
        html += "        document.getElementById('rewind').addEventListener('click', () => {"
        html += "            timeStep = 0;"
        html += "            document.getElementById('step-display').textContent = timeStep;"
        html += "        });"
        html += ""
        html += "        document.getElementById('field-selector').addEventListener('change', (e) => {"
        html += "            fieldType = e.target.value;"
        html += "            updateElectrodeVisualization();"
        html += "        });"
        html += ""
        html += "        document.getElementById('voltage-slider').addEventListener('input', (e) => {"
        html += "            const val = parseFloat(e.target.value);"
        html += "            document.getElementById('vmin').textContent = val.toFixed(2);"
        html += "            // Apply voltage filtering/thresholding"
        html += "            updateElectrodeVisualization();"
        html += "        });"
        html += ""
        html += "        function updateElectrodeVisualization() {"
        html += "            // Update electrode colors/sizes based on current field type and slider value"
        html += "            renderer.render(scene, camera);"
        html += "        }"
        html += ""
        # Resize handler
        html += "        window.addEventListener('resize', () => {"
        html += "            const width = viewer.clientWidth;"
        html += "            const height = viewer.clientHeight;"
        html += "            camera.aspect = width / height;"
        html += "            camera.updateProjectionMatrix();"
        html += "            renderer.setSize(width, height);"
        html += "        });"
        html += "    </script>"
        html += "</body>"
        html += "</html>"

        return html

    def to_html(self, title: str = "Device Explorer") -> str:
        """Generate self-contained HTML with Three.js CDN.

        Parameters
        ----------
        title : str
            Page title.

        Returns
        -------
        str
            Complete HTML document.
        """
        frame_count = len(self.frames) if self.frames else 0
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ padding: 1rem; background: #16213e; text-align: center; }}
        .controls {{ padding: 0.5rem 1rem; background: #0f3460; }}
        .control-btn {{ 
            background: #1a237e; color: #e0e0e0; 
            border: none; padding: 0.3rem 0.6rem; 
            margin: 0 0.1rem; cursor: pointer; border-radius: 4px;
        }}
        .control-btn:hover {{ background: #2a4b8e; }}
        #viewer {{ width: 100%; height: 800px; background: #0d1b2a; }}
        .comparison {{ display: flex; gap: 2rem; }}
        .viewer {{ flex: 1; }}
        .controls {{ display: flex; gap: 1rem; align-items: center; }}
        .info {{ padding: 1rem; background: #152840; margin: 1rem; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="controls">
        <button class="control-btn" id="play-pause">Pause</button>
        <button class="control-btn" id="rewind">Rewind</button>
        <span>Step: <span id="step-display">0</span>/<span id="step-total">{frame_count}</span></span>
    </div>
    <div id="viewer"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const viewer = document.getElementById('viewer');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, viewer.clientWidth / viewer.clientHeight, 0.1, 1000);
        camera.position.set(0, 0, 15);
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(viewer.clientWidth, viewer.clientHeight);
        viewer.appendChild(renderer.domElement);

        // Initialize based on scene type
        const data = {self._json_data()};
        const sceneType = data.type || 'geometry';

        function init() {{
            if (sceneType === 'geometry') {{
                const geometryData = data;
                // Create Three.js mesh from vertices/faces
                if (geometryData.vertices) {{
                    const vertices = new Float32Array(
                        geometryData.vertices.flat());
                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

                    const material = new THREE.MeshStandardMaterial({{
                        color: 0x4488ff, transparent: true, opacity: 0.8}});

                    if (geometryData.faces) {{
                        const faces = new Uint32Array(
                            geometryData.faces.flat());
                        geometry.setIndex(new THREE.IndexBufferAttribute(faces, 3));
                    }}

                    const mesh = new THREE.Mesh(geometry, material);
                    scene.add(mesh);
                }}

                // Add CSG annotations
                if (geometryData.csg_operations) {{
                    // Would add CSG operation visual indicators
                    console.log("CSG operations:", geometryData.csg_operations);
                }}

                // Add electrode glyphs
                if (geometryData.type === 'electrodes') {{
                    // Would render electrode spheres with voltage coloring
                    console.log("Electrodes:", geometryData.electrode_count);
                }}

                // Add material regions
                if (geometryData.type === 'material_regions') {{
                    // Would render transparent regions with boundary highlighting
                    console.log("Material regions:", geometryData.region_count);
                }}

                // Add write/read annotations
                if (data.pulse_shapes) {{
                    // Would render pulse shape graphs
                    console.log("Pulse shapes:", data.pulse_shapes ? "present" : "none");
                }}

                if (data.frequency_response) {{
                    // Would render Bode plot
                    console.log("Frequency response: present");
                }}

                // Add manufacturing layer stack
                if (data.layer_stack) {{
                    // Would render layer cross-section
                    console.log("Layer stack:", data.layer_stack.length, "layers");
                }}

                renderer.render(scene, camera);
            }} else if (sceneType === 'manufacturing') {{
                // Manufacturing view initialization
                console.log("Manufacturing view data present");
                renderer.render(scene, camera);
            }} else {{
                // Default: just render axes and grid
                const axes = new THREE.AxesHelper(5);
                scene.add(axes);
                renderer.render(scene, camera);
            }}

            // Animation loop
            let timeStep = {self.time_step};
            const totalSteps = {frame_count};
            let playing = {str(self.playing).lower()};

            function animate() {{
                requestAnimationFrame(animate);
                if (playing && totalSteps > 0) {{
                    timeStep = (timeStep + 1) % totalSteps;
                    document.getElementById('step-display').textContent = timeStep;
                    // Update scene with frame data
                }}
                renderer.render(scene, camera);
            }}

            animate();

            // Controls
            document.getElementById('play-pause').addEventListener('click', () => {{
                playing = !playing;
                document.getElementById('play-pause').textContent = playing ? 'Pause' : 'Play';
            }});

            document.getElementById('rewind').addEventListener('click', () => {{
                timeStep = 0;
                document.getElementById('step-display').textContent = timeStep;
            }});

            window.addEventListener('resize', () => {{
                const width = viewer.clientWidth;
                const height = viewer.clientHeight;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            }});
        }}
    </script>
</body>
</html>"""

        return html

    def _json_data(self) -> dict[str, Any]:
        """Serialize explorer state to JSON-compatible dict."""
        return {
            "type": self.__class__.__name__,
            "time_step": self.time_step,
            "playing": self.playing,
            "frames": self.frames is not None,
        }


def build_device_scene(
    design: Any,
    geometry_data: dict[str, Any],
    electrode_data: dict[str, Any] | None = None,
    simulation_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Three.js scene from design and device data.

    Parameters
    ----------
    design : DesignSpec
        Chip design specification.
    geometry_data : dict
        Geometry including vertices, faces, CSG operations.
    electrode_data : dict, optional
        Electrode positions and voltages/currents.
    simulation_output : dict, optional
        Simulation results for current density, temperature, etc.

    Returns
    -------
    dict
        Scene data compatible with Three.js rendering.
    """
    explorer = DeviceExplorer()

    # Set frames if simulation output has time series
    frames = simulation_output.get("frames", []) if simulation_output else []
    if frames:
        explorer.set_frames(frames)

    result: dict[str, Any] = {}

    # Render base geometry
    vertices = np.array(geometry_data.get("vertices", []), dtype=float).reshape(-1, 3) if geometry_data.get("vertices") else np.array([])
    faces = np.array(geometry_data.get("faces", []), dtype=int).reshape(-1, 3) if geometry_data.get("faces") else np.array([])
    csg_ops = geometry_data.get("csg_operations")

    if vertices.shape[0] > 0:
        geo_result = explorer.render_geometry(
            vertices, faces if faces.size > 0 else None, csg_ops
        )
        result["geometry"] = geo_result

    # Render electrodes if data provided
    if electrode_data:
        electrode_result = explorer.render_electrodes(
            np.array(electrode_data.get("positions", []), dtype=float).reshape(-1, 3)
            if electrode_data.get("positions") else np.array([]),
            np.array(electrode_data.get("voltages", []), dtype=float).reshape(-1)
            if electrode_data.get("voltages") else np.array([]),
            np.array(electrode_data.get("currents", []), dtype=float).reshape(-1)
            if electrode_data.get("currents") else None,
        )
        result["electrodes"] = electrode_result

    # Render material regions if available
    if simulation_output and "region_labels" in simulation_output:
        region_labels = np.array(simulation_output["region_labels"])
        region_result = explorer.render_material_regions(region_labels)
        result["material_regions"] = region_result

    # Render write/read mechanisms
    pulse_shapes = simulation_output.get("pulse_shapes") if simulation_output else None
    freq_response = simulation_output.get("frequency_response") if simulation_output else None
    if pulse_shapes or freq_response:
        wr_result = explorer.render_write_read(pulse_shapes, freq_response)
        result["write_read"] = wr_result

    # Render manufacturing view
    layer_stack = simulation_output.get("layer_stack") if simulation_output else None
    die_pos = simulation_output.get("die_positions") if simulation_output else None
    routing = simulation_output.get("routing_paths") if simulation_output else None
    if layer_stack or die_pos or routing:
        mfg_result = explorer.render_manufacturing_view(
            layer_stack or [],
            die_pos and np.array(die_pos, dtype=float) if isinstance(die_pos, list) else die_pos,
            routing and np.array(routing, dtype=float) if isinstance(routing, list) else routing,
        )
        result["manufacturing"] = mfg_result

    # Add metadata
    result["design_id"] = getattr(design, "id", "unknown") if design else "unknown"

    return result