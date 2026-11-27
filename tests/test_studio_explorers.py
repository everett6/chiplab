"""Tests for MaterialExplorer and DeviceExplorer with GPU instanced HTML export."""

import numpy as np
import pytest

from chiplab.studio.material_explorer import MaterialExplorer
from chiplab.studio.device_explorer import DeviceExplorer


class TestMaterialExplorer:
    """Tests for MaterialExplorer class."""

    def test_initialization(self):
        """Test MaterialExplorer initializes correctly."""
        explorer = MaterialExplorer()
        assert explorer.time_step == 0
        assert explorer.playing is False
        assert explorer.frames is None

    def test_set_frames(self):
        """Test setting frames updates time step."""
        explorer = MaterialExplorer()
        frames = [
            {"positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
            {"positions": [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]},
        ]
        explorer.set_frames(frames)
        assert explorer.frames == frames
        assert explorer.time_step == 0

    def test_play_pause(self):
        """Test play/pause toggle."""
        explorer = MaterialExplorer()
        assert explorer.playing is False
        explorer.play_pause()
        assert explorer.playing is True
        explorer.play_pause()
        assert explorer.playing is False

    def test_rewind(self):
        """Test rewind to first frame."""
        explorer = MaterialExplorer()
        explorer.frames = [{"positions": [[1.0, 2.0, 3.0]]}, {"positions": [[4.0, 5.0, 6.0]]}]
        explorer.rewind()
        assert explorer.time_step == 0

    def test_scrub(self):
        """Test scrub to specific frame."""
        explorer = MaterialExplorer()
        explorer.frames = [{"positions": [[1.0, 2.0, 3.0]]}, {"positions": [[4.0, 5.0, 6.0]]}]
        explorer.scrub(1)
        assert explorer.time_step == 1
        explorer.scrub(0)
        assert explorer.time_step == 0

    def test_render_particles(self):
        """Test render_particles produces expected output structure."""
        explorer = MaterialExplorer()
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
        moments = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        orientations = np.array([0.5, 1.0], dtype=float)
        temperature = np.array([300.0, 310.0], dtype=float)

        result = explorer.render_particles(positions, moments, orientations, temperature)
        assert result["type"] == "instanced_particles"
        assert result["count"] == 2
        assert "positions" in result
        assert "moments" in result
        assert "colors" in result
        assert "orientations" in result
        assert result["size"] == 0.5

    def test_export_gpu_instanced_html(self):
        """Test export_gpu_instanced_html generates valid HTML."""
        explorer = MaterialExplorer()
        frames = [
            {
                "positions": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float).tolist(),
                "moments": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float).tolist(),
                "colors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "orientations": [0.5, 1.0],
                "size": 0.5,
            }
        ]
        explorer.set_frames(frames)
        html = explorer.export_gpu_instanced_html()
        assert "<!DOCTYPE html>" in html
        assert "three.min.js" in html
        assert "play-pause" in html
        assert "step-display" in html

    def test_export_gpu_instanced_html_with_field_selector(self):
        """Test export_gpu_instanced_html includes field type selector."""
        explorer = MaterialExplorer()
        frames = [
            {
                "positions": np.array([[1.0, 2.0, 3.0]], dtype=float).tolist(),
                "moments": np.array([[1.0, 0.0, 0.0]], dtype=float).tolist(),
            }
        ]
        explorer.set_frames(frames)
        html = explorer.export_gpu_instanced_html(field_type="temperature")
        assert 'field-selector' in html or 'field_type' in html

    def test_export_gpu_instanced_html_time_controls(self):
        """Test export_gpu_instanced_html includes time controls."""
        explorer = MaterialExplorer()
        explorer.frames = [
            {"positions": [[1.0, 2.0, 3.0]]}
        ]
        html = explorer.export_gpu_instanced_html(show_time_controls=True)
        assert 'play-pause' in html
        assert 'step-display' in html

    def test_export_gpu_instanced_html_no_time_controls(self):
        """Test export_gpu_instanced_html can omit time controls."""
        explorer = MaterialExplorer()
        explorer.frames = [
            {"positions": [[1.0, 2.0, 3.0]]}
        ]
        html = explorer.export_gpu_instanced_html(show_time_controls=False)
        assert 'play-pause' not in html or html.count('play-pause') == 0


class TestDeviceExplorer:
    """Tests for DeviceExplorer class."""

    def test_initialization(self):
        """Test DeviceExplorer initializes correctly."""
        explorer = DeviceExplorer()
        assert explorer.time_step == 0
        assert explorer.playing is False
        assert explorer.frames is None

    def test_set_frames(self):
        """Test setting frames updates time step."""
        explorer = DeviceExplorer()
        frames = [
            {"geometry": {"vertices": [[0, 0, 0], [1, 1, 1]]}},
            {"geometry": {"vertices": [[2, 2, 2], [3, 3, 3]]}},
        ]
        explorer.set_frames(frames)
        assert explorer.frames == frames
        assert explorer.time_step == 0

    def test_play_pause(self):
        """Test play/pause toggle."""
        explorer = DeviceExplorer()
        assert explorer.playing is False
        explorer.play_pause()
        assert explorer.playing is True
        explorer.play_pause()
        assert explorer.playing is False

    def test_rewind(self):
        """Test rewind to first frame."""
        explorer = DeviceExplorer()
        explorer.frames = [{"geometry": {"vertices": [[1.0, 2.0, 3.0]]}}]
        explorer.rewind()
        assert explorer.time_step == 0

    def test_scrub(self):
        """Test scrub to specific frame."""
        explorer = DeviceExplorer()
        explorer.frames = [{"geometry": {"vertices": [[1.0, 2.0, 3.0]]}}, {"geometry": {"vertices": [[4.0, 5.0, 6.0]]}}]
        explorer.scrub(1)
        assert explorer.time_step == 1
        explorer.scrub(0)
        assert explorer.time_step == 0

    def test_render_geometry(self):
        """Test render_geometry produces expected output structure."""
        explorer = DeviceExplorer()
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)

        result = explorer.render_geometry(vertices, faces)
        assert result["type"] == "geometry"
        assert result["vertex_count"] == 3
        assert "vertices" in result
        assert "faces" in result

    def test_render_electrodes(self):
        """Test render_electrodes produces expected output structure."""
        explorer = DeviceExplorer()
        electrode_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
        electrode_voltages = np.array([1.5, 3.0], dtype=float)
        electrode_currents = np.array([0.1, 0.2], dtype=float)

        result = explorer.render_electrodes(
            electrode_positions, electrode_voltages, electrode_currents
        )
        assert result["type"] == "electrodes"
        assert result["electrode_count"] == 2
        assert "positions" in result
        assert "voltages" in result
        assert "colors" in result
        assert "current_magnitudes" in result

    def test_export_gpu_instanced_html(self):
        """Test export_gpu_instanced_html generates valid HTML."""
        explorer = DeviceExplorer()
        frames = [
            {
                "electrode_positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "electrode_voltages": [1.5, 3.0],
                "electrode_currents": [0.1, 0.2],
            }
        ]
        explorer.set_frames(frames)
        html = explorer.export_gpu_instanced_html()
        assert "<!DOCTYPE html>" in html
        assert "three.min.js" in html
        assert "play-pause" in html
        assert "step-display" in html

    def test_export_gpu_instanced_html_electrode_controls(self):
        """Test export_gpu_instanced_html includes electrode controls."""
        explorer = DeviceExplorer()
        explorer.frames = [
            {
                "electrode_positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "electrode_voltages": [1.5, 3.0],
                "electrode_currents": [0.1, 0.2],
            }
        ]
        html = explorer.export_gpu_instanced_html(show_electrode_controls=True)
        assert 'voltage-slider' in html or 'electrode' in html.lower()

    def test_export_gpu_instanced_html_no_electrode_controls(self):
        """Test export_gpu_instanced_html can omit electrode controls."""
        explorer = DeviceExplorer()
        explorer.frames = [
            {
                "electrode_positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "electrode_voltages": [1.5, 3.0],
                "electrode_currents": [0.1, 0.2],
            }
        ]
        html = explorer.export_gpu_instanced_html(show_electrode_controls=False)
        # Should still have basic HTML structure
        assert "<!DOCTYPE html>" in html


class TestGPUInstancedExport:
    """Tests for GPU instanced HTML export functionality."""

    def test_material_explorer_preserves_api(self):
        """Test that adding export_gpu_instanced_html doesn't break existing API."""
        explorer = MaterialExplorer()
        # Existing methods should still work
        html = explorer.to_html()
        assert "<!DOCTYPE html>" in html
        # New method should also work
        html2 = explorer.export_gpu_instanced_html()
        assert "<!DOCTYPE html>" in html2

    def test_device_explorer_preserves_api(self):
        """Test that adding export_gpu_instanced_html doesn't break existing API."""
        explorer = DeviceExplorer()
        # Existing methods should still work
        html = explorer.to_html()
        assert "<!DOCTYPE html>" in html
        # New method should also work
        html2 = explorer.export_gpu_instanced_html()
        assert "<!DOCTYPE html>" in html2

    def test_empty_explorer_html(self):
        """Test HTML generation with no frames set."""
        material_explorer = MaterialExplorer()
        device_explorer = DeviceExplorer()

        # Both should generate valid HTML even with no frames
        html_m = material_explorer.export_gpu_instanced_html(show_time_controls=False)
        assert "<!DOCTYPE html>" in html_m

        html_d = device_explorer.export_gpu_instanced_html(show_electrode_controls=False)
        assert "<!DOCTYPE html>" in html_d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])