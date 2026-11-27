"""Photonic computing models for chiplab - MZI meshes, microrings, and link budget."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import GPUBackend, get_gpu_backend


class PhotonicMeshState:
    """State container for an MZI mesh, holding the current phase settings and metadata."""

    def __init__(
        self,
        n_mzi: int,
        phases: np.ndarray | None = None,
        provenance: Provenance = Provenance.ASSUMED,
        source: str | None = None,
    ):
        self.n_mzi = n_mzi
        if phases is None:
            phases = np.zeros(n_mzi, dtype=np.float64)
        elif len(phases) != n_mzi:
            raise ValueError(f"phases must have length {n_mzi}, got {len(phases)}")
        self.phases = TrackedValue(value=phases, provenance=provenance, source=source)

    def update_phase(self, index: int, phase: float, provenance: Provenance = Provenance.ASSUMED) -> None:
        """Update a single phase setting in the mesh."""
        phases = np.array(self.phases.value, dtype=np.float64)
        phases[index] = phase
        self.phases = TrackedValue(value=phases, provenance=provenance, source=self.phases.source)

    def get_phases(self) -> np.ndarray:
        """Return the current phase settings."""
        return np.array(self.phases.value, dtype=np.float64)


def mzi_transfer_matrix(
    wavelength: float,
    length: float,
    gap: float,
    index_profile: Literal["symmetric", "asymmetric"] = "symmetric",
    phase_shift: float | TrackedValue | None = None,
) -> np.ndarray:
    """
    Compute the S-parameter (transfer) matrix of a single MZI interferometer.

    Parameters
    ----------
    wavelength : float
        Operating wavelength in meters.
    length : float
        Physical length of the MZI arms in meters.
    gap : float
        Coupling gap between waveguides in meters.
    index_profile : {"symmetric", "asymmetric"}
        Refractive index profile type.
    phase_shift : float or TrackedValue | None
        External phase shift applied to one arm (in radians). If TrackedValue,
        provenance is preserved.

    Returns
    -------
    np.ndarray
        2x2 complex transfer matrix [[S11, S12], [S21, S22]].
    """
    # Effective index depends on wavelength and index profile
    if index_profile == "symmetric":
        neff = 2.4  # typical silicon photonics effective index
    else:
        neff = 2.35

    # Free spectral range related phase
    phi = 2 * np.pi * neff * length / wavelength

    # Additional phase shift (from heaters, etc.)
    if phase_shift is not None:
        if isinstance(phase_shift, TrackedValue):
            phi += float(phase_shift)
            actual_phase = phase_shift.value
        else:
            phi += phase_shift
            actual_phase = phase_shift
    else:
        actual_phase = 0.0

    # MZI transfer matrix (unitary, lossless symmetric coupler)
    phase_half = phi / 2.0
    c = np.cos(phase_half)
    s = np.sin(phase_half)

    # Symmetric MZI transfer matrix
    S = np.array(
        [
            [c, -1j * s],
            [-1j * s, c],
        ],
        dtype=np.complex128,
    )

    # If there's an additional phase shift, apply it to the appropriate element
    if phase_shift is not None and not isinstance(phase_shift, TrackedValue):
        # Rotate phase - this is a simplified model
        phase_factor = np.exp(1j * actual_phase)
        S = S * phase_factor

    return S


def _extract_array(input_vector: np.ndarray | TrackedValue) -> np.ndarray:
    """Extract the numpy array from a TrackedValue or return the array itself."""
    if isinstance(input_vector, TrackedValue):
        val = input_vector.value
        if np.iscomplexobj(val):
            return np.array(val, dtype=np.result_type(val, np.float64))
        else:
            return np.array(val, dtype=np.float64)
    return np.asarray(input_vector)


def mzi_mesh_vmm(
    input_vector: np.ndarray | TrackedValue,
    n_mzi: int,
    phases: np.ndarray | TrackedValue | None = None,
    device: str = "auto",
    provenance: Provenance = Provenance.ASSUMED,
) -> np.ndarray:
    """
    Perform matrix-multiply using an MZI mesh (optical VMM).

    This implements a mesh of MZIs that can realize any unitary matrix.
    The mesh applies the unitary defined by the phase settings to the input vector.

    Parameters
    ----------
    input_vector : np.ndarray or TrackedValue
        Input vector (complex amplitudes). If TrackedValue, provenance is preserved.
    n_mzi : int
        Number of MZIs in the mesh (for an NxN mesh, use n_mzi=N*N-N or similar).
    phases : np.ndarray or TrackedValue | None
        Phase settings for each MZI. If TrackedValue, provenance is preserved.
        If None, random/zero phases are used.
    device : str
        Device target: "auto", "cpu", "cuda", or "gpu". Default "auto" uses GPUBackend.
    provenance : Provenance
        Provenance tracking for the output.

    Returns
    -------
    np.ndarray
        Output vector after unitary transformation.
    """
    backend = get_gpu_backend(prefer=device)

    # Determine phases
    if phases is None:
        # Default: random phases (or zeros)
        phases_arr = np.random.rand(n_mzi) * 2 * np.pi
    elif isinstance(phases, TrackedValue):
        phases_arr = np.array(phases.value, dtype=np.float64)
        if phases_arr.ndim == 0 or len(phases_arr) != n_mzi:
            raise ValueError(
                f"phases must have length {n_mzi}, got {len(phases_arr)}"
            )
    else:
        phases_arr = np.array(phases, dtype=np.float64)
        if len(phases_arr) != n_mzi:
            raise ValueError(
                f"phases must have length {n_mzi}, got {len(phases_arr)}"
            )

    # Build the unitary matrix from the MZI mesh
    # For a triangular mesh (NxN), we use N*(N-1)/2 MZIs
    # Each MZI applies a Givens rotation in the unitary space
    N = int(np.ceil((1 + np.sqrt(1 + 8 * n_mzi)) / 2))  # solve N*(N-1)/2 = n_mzi
    actual_n_mzi = N * (N - 1) // 2

    # Start with identity unitary
    U = np.eye(N, dtype=np.complex128)

    # Apply each MZI's rotation based on its phase
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            if k < actual_n_mzi:
                phi = phases_arr[k] if k < len(phases_arr) else 0.0
                c = np.cos(phi / 2.0)
                s = np.sin(phi / 2.0)

                # Givens rotation matrix for unitary transformation
                # This rotates in the (j,i) plane
                G = np.eye(N, dtype=np.complex128)
                G[i, i] = c
                G[j, j] = c
                G[i, j] = -1j * s
                G[j, i] = 1j * s

                U = G @ U
            k += 1

    # Apply the unitary to the input vector
    # Extract raw array from TrackedValue if needed
    x_raw = _extract_array(input_vector)
    x = backend.to_device(x_raw, dtype="complex128")
    # Use appropriate matrix multiply based on backend
    if backend.backend_name in ("torch_cuda", "torch_cpu"):
        import torch
        if not isinstance(x, torch.Tensor):
            x_t = torch.as_tensor(x, device=backend._torch_device if backend._torch_device else torch.device("cpu"))
        else:
            x_t = x
        result_t = torch.matmul(U, x_t)
        result = backend.to_numpy(result_t)
    elif backend.backend_name == "cupy":
        import cupy as cp
        if not isinstance(x, cp.ndarray):
            x_cp = cp.asarray(x, dtype=cp.complex128)
        else:
            x_cp = x
        result_cp = cp.matmul(U, x_cp)
        result = backend.to_numpy(result_cp)
    else:
        # CPU numpy fallback (including numpy_cpu)
        result = U @ x

    # Convert back to numpy and preserve provenance if input was TrackedValue
    # result is already numpy from to_numpy

    if isinstance(input_vector, TrackedValue):
        result = TrackedValue(
            value=result,
            provenance=input_vector.provenance,  # Preserve input provenance
            source=input_vector.source,
        )

    return result


def microring_weight_bank(
    num_rings: int,
    resonance_wavelengths: np.ndarray,
    coupling_coeffs: np.ndarray | None = None,
    provenance: Provenance = Provenance.ASSUMED,
) -> dict[str, TrackedValue]:
    """
    Create a weight bank from an array of microrings.

    Each microring provides a complex weight based on its detuning from resonance.
    The weight is modeled as a Lorentzian lineshape.

    Parameters
    ----------
    num_rings : int
        Number of microrings in the bank.
    resonance_wavelengths : np.ndarray
        Array of resonance wavelengths for each ring (in meters).
    coupling_coeffs : np.ndarray | None
        Coupling coefficients (power coupling, 0 to 1). If None, defaults to 0.1.
    provenance : Provenance
        Provenance tracking for the weights.

    Returns
    -------
    dict[str, TrackedValue]
        Dictionary mapping ring index to its complex weight TrackedValue.
    """
    if len(resonance_wavelengths) != num_rings:
        raise ValueError(
            f"resonance_wavelengths must have length {num_rings}, got {len(resonance_wavelengths)}"
        )

    if coupling_coeffs is None:
        coupling_coeffs = np.ones(num_rings) * 0.1
    elif len(coupling_coeffs) != num_rings:
        raise ValueError(
            f"coupling_coeffs must have length {num_rings}, got {len(coupling_coeffs)}"
        )

    # Reference wavelength (e.g., 1550 nm)
    lambda_ref = 1550e-9

    weights: dict[str, TrackedValue] = {}
    for i in range(num_rings):
        detuning = resonance_wavelengths[i] - lambda_ref
        # Lorentzian lineshape: weight ~ 1/(1 + (2*delta*lambda/gamma)^2)
        # gamma is the linewidth (~0.1 nm typical)
        gamma = 0.1e-9  # linewidth in meters
        # Normalized detuning
        norm_detuning = 2 * np.pi * detuning / (lambda_ref ** 2) * 1e-9  # approximate
        # Lorentzian response
        lorentzian = 1.0 / (1.0 + (norm_detuning / (gamma / lambda_ref * 1e9)) ** 2)
        # Complex weight with phase
        weight_magnitude = coupling_coeffs[i] * lorentzian
        weight_phase = 2 * np.pi * detuning / lambda_ref
        complex_weight = weight_magnitude * np.exp(1j * weight_phase)

        weights[f"ring_{i}"] = TrackedValue(
            value=complex_weight,
            provenance=provenance,
            source=f"microring_weight_bank ring {i}",
        )

    return weights


def photonic_link_budget(
    tx_power_dbm: float | TrackedValue = 0.0,
    insertion_loss_dbm: float | TrackedValue | None = None,
    fiber_length_km: float = 10.0,
    fiber_attenuation_dbm_km: float = 0.2,
    photodetector_responsivity_ama: float = 0.8,
    osnr_target_db: float = 15.0,
    provenance: Provenance = Provenance.ASSUMED,
) -> dict[str, TrackedValue]:
    """
    Compute a photonic link budget: insertion loss, thermal tuning power, and OSNR.

    Parameters
    ----------
    tx_power_dbm : float or TrackedValue
        Transmitter output power in dBm. If TrackedValue, provenance is preserved.
    insertion_loss_dbm : float or TrackedValue | None
        Total insertion loss in dBm. If None, computed from components.
        If TrackedValue, provenance is preserved.
    fiber_length_km : float
        Length of fiber in kilometers.
    fiber_attenuation_dbm_km : float
        Fiber attenuation in dB/km.
    photodetector_responsivity_ama : float
        Photodetector responsivity in A/W.
    osnr_target_db : float
        Target optical signal-to-noise ratio in dB.
    provenance : Provenance
        Provenance tracking.

    Returns
    -------
    dict[str, TrackedValue]
        Dictionary with 'total_loss_dbm', 'thermal_tuning_power_mw', and 'osnr_db'
        as TrackedValue entries.
    """
    # Handle TrackedValue for tx_power
    if isinstance(tx_power_dbm, TrackedValue):
        tx_power = float(tx_power_dbm)
        tx_power_source = tx_power_dbm.source
    else:
        tx_power = float(tx_power_dbm)
        tx_power_source = None

    # Compute insertion loss if not provided
    if insertion_loss_dbm is None:
        # Component losses: connectors (0.5 dB each, 2 total), splices (0.3 dB each, 2 total)
        # Fiber loss, waveguide loss (~2 dB/cm typical)
        n_connectors = 2
        n_splices = 2
        connector_loss = n_connectors * 0.5  # dB
        splice_loss = n_splices * 0.3  # dB
        fiber_loss = fiber_length_km * fiber_attenuation_dbm_km  # dB
        # Waveguide loss estimate (assume 5 cm total path)
        waveguide_loss = 0.2  # dB/cm * 5 cm = 1 dB (typical)
        total_loss_dbm = connector_loss + splice_loss + fiber_loss + waveguide_loss
    elif isinstance(insertion_loss_dbm, TrackedValue):
        total_loss_dbm = float(insertion_loss_dbm)
        total_loss_source = insertion_loss_dbm.source
    else:
        total_loss_dbm = float(insertion_loss_dbm)
        total_loss_source = None

    # Thermal tuning power estimate
    # Typical heater energy to tune one wavelength: ~10-100 µJ per tuning event
    # Power depends on repetition rate; assuming 1 GHz modulation: ~1 mW per heater
    # For a typical link with 4 heaters: ~4 mW
    n_heaters = 4
    heater_energy_per_tuning_j = 50e-6  # 50 µJ typical
    # Assume tuning occurs at the data rate; simplified power estimate
    thermal_tuning_power_mw = n_heaters * heater_energy_per_tuning_j * 1e3  # convert J to mW

    # OSNR computation
    # OSNR = signal_power / (noise_power density)
    # Noise figure of amplifier(s) + spontaneous emission
    # Simplified: OSNR_db = tx_power_dbm - total_loss_dbm - amplifier_noise_figure_dB + reference bandwidth factor
    amplifier_noise_figure_db = 5.0  # typical EDFA NF in dB
    reference_bandwidth_hz = 12.5e9  # 12.5 GHz reference bandwidth for OSNR measurement
    # Noise floor in reference BW = -174 dBm/Hz + 10*log10(BW)
    noise_floor_dbm = -174.0 + 10 * np.log10(reference_bandwidth_hz)
    signal_power_after_loss_dbm = tx_power - total_loss_dbm
    osnr_db = signal_power_after_loss_dbm - noise_floor_dbm

    # Build result with provenance tracking
    result: dict[str, TrackedValue] = {}

    # total_loss_dbm
    if insertion_loss_dbm is None:
        result["total_loss_dbm"] = TrackedValue(
            value=total_loss_dbm,
            provenance=provenance,
            source="photonic_link_budget computed",
        )
    else:
        if isinstance(insertion_loss_dbm, TrackedValue):
            result["total_loss_dbm"] = TrackedValue(
                value=total_loss_dbm,
                provenance=insertion_loss_dbm.provenance,
                source=insertion_loss_dbm.source,
            )
        else:
            result["total_loss_dbm"] = TrackedValue(
                value=total_loss_dbm,
                provenance=provenance,
                source=total_loss_source,
            )

    # thermal_tuning_power_mw
    result["thermal_tuning_power_mw"] = TrackedValue(
        value=thermal_tuning_power_mw,
        provenance=provenance,
        source="photonic_link_budget computed",
    )

    # osnr_db
    result["osnr_db"] = TrackedValue(
        value=osnr_db,
        provenance=provenance,
        source="photonic_link_budget computed",
    )

    return result
