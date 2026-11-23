"""Schemas for the zero-token digital twin: search spaces, uncertainty priors, Pareto results,
calibration reports and black-box IP macros."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .common import FidelityLevel, Provenance

TWIN_DISCLAIMER = (
    "Digital-twin results come from a parametric analytical PPA model (alpha-power-law "
    "timing, CV^2f + leakage power, SRAM/logic density scaling, yield-based cost). They rank "
    "design points and expose sensitivities; they are not signoff. Calibrate with "
    "calibrate_from_eda_report / ingest_byo_pdk against real synthesis and P&R runs."
)


class SearchParameter(BaseModel):
    """One tunable knob. ``int``/``float`` use [low, high]; ``choice`` picks from choices."""

    name: str
    kind: Literal["int", "float", "choice"] = "float"
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False
    unit: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SearchParameter:
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"{self.name}: choice parameter needs choices")
        else:
            if self.low is None or self.high is None or self.high < self.low:
                raise ValueError(f"{self.name}: numeric parameter needs low <= high")
            if self.log_scale and self.low <= 0:
                raise ValueError(f"{self.name}: log_scale needs low > 0")
        return self


TSMC_TWIN_NODES = ["N16", "N7", "N5", "N3"]
LADDER_TWIN_NODES = ["GF180", "SKY130", "IHP_SG13G2", "N65", "N28", "FREEPDK15", "ASAP7",
                     "N16", "N7", "N5", "N3", "N2"]


def default_search_space(nodes: Literal["tsmc", "ladder"] = "tsmc",
                         memory: bool = False) -> list[SearchParameter]:
    """RISC-V SoC knobs the twin explores by default.

    ``nodes='ladder'`` searches the 180nm-to-2nm PDK ladder; supply is then a ratio of each
    node's nominal Vdd (``vdd_rel``) and the clock range is log-scaled from 50 MHz.
    ``memory=True`` adds the on-chip memory knobs (needs ``DigitalTwinConfig.memory``).
    """
    ladder = nodes == "ladder"
    space = [
        SearchParameter(name="cores", kind="int", low=1, high=64, log_scale=True),
        SearchParameter(name="vlen", kind="choice", choices=[128, 256, 512, 1024], unit="bit"),
        SearchParameter(name="dlen", kind="choice", choices=[64, 128, 256, 512], unit="bit"),
        SearchParameter(name="l1_kb", kind="choice", choices=[16, 32, 64, 128], unit="KiB"),
        SearchParameter(name="l2_kb", kind="choice", choices=[256, 512, 1024, 2048],
                        unit="KiB"),
        SearchParameter(name="llc_mb", kind="float", low=0.0, high=64.0, unit="MiB"),
        SearchParameter(name="vdd_rel", kind="float", low=0.8, high=1.15, unit="x Vnom")
        if ladder else SearchParameter(name="vdd", kind="float", low=0.55, high=0.95, unit="V"),
        SearchParameter(name="freq_ghz", kind="float", low=0.05, high=3.5, log_scale=True,
                        unit="GHz") if ladder else
        SearchParameter(name="freq_ghz", kind="float", low=0.5, high=3.5, unit="GHz"),
        SearchParameter(name="node", kind="choice",
                        choices=list(LADDER_TWIN_NODES if ladder else TSMC_TWIN_NODES)),
        SearchParameter(name="ucie_lanes", kind="choice", choices=[0, 16, 32, 64]),
        SearchParameter(name="utilization", kind="float", low=0.55, high=0.85),
    ]
    if memory:
        space += [
            SearchParameter(name="mem_tech", kind="choice", choices=["sram6t", "gc2t", "gc3t"]),
            SearchParameter(name="mem_banks", kind="choice", choices=[2, 4, 8, 16, 32, 64]),
            SearchParameter(name="mem_interleave", kind="choice",
                            choices=["low_order", "xor", "block"]),
            SearchParameter(name="pim_lanes", kind="choice", choices=[0, 4, 8, 16]),
        ]
    return space


class TwinMemory(BaseModel):
    """Shared on-chip memory (scratchpad / LLC data array) explored by the twin.

    Technology, bank count, interleaving and PIM lanes are search knobs (or pinned via
    ``fixed``); this block holds what the workload and floorplan decide.
    """

    capacity_kib: int = Field(2048, ge=16, le=1_048_576)
    word_bits: int = Field(64, ge=8, le=1024)
    num_requesters: int = Field(8, ge=1, le=32, description="Ports into the bank crossbar")
    requests_per_cycle: float = Field(2.0, gt=0.0, le=32.0,
                                      description="Offered load, all requesters (assumed)")
    strided_fraction: float = Field(0.5, ge=0.0, le=1.0,
                                    description="Share of traffic that is strided (rest random)")
    stride: int = Field(4, ge=1, le=4096)
    ecc: Literal["none", "secded"] = "secded"
    refresh_margin: float = Field(0.8, gt=0.0, le=1.0)
    mac_bits: Literal[4, 8, 16] = 8
    pim_fraction: float = Field(0.25, ge=0.0, le=1.0,
                                description="Share of idle bank cycles PIM actually uses")


class UncertaintyPrior(BaseModel):
    """How an unknown parameter is perturbed in Monte Carlo (a multiplicative factor)."""

    name: str
    provenance: Provenance = Provenance.ASSUMED
    distribution: Literal["normal", "uniform", "triangular"] = "normal"
    spread_pct: float = Field(20.0, ge=0.0, le=100.0,
                              description="1-sigma (normal) or half-width (uniform/tri) in %")
    description: str = ""


class Objective(BaseModel):
    name: Literal["perf", "power_w", "area_mm2", "cost_usd", "perf_per_watt", "energy_pj_op",
                  "pim_gmacs", "mem_bandwidth_gbs", "refresh_overhead"]
    goal: Literal["max", "min"]


class TwinConstraints(BaseModel):
    min_slack_ps: float = 0.0
    max_temp_c: float = 95.0
    max_power_w: float | None = None
    max_area_mm2: float | None = None
    max_cost_usd: float | None = None
    max_refresh_overhead: float = Field(0.25, gt=0.0, le=1.0,
                                        description="Max share of bank cycles spent refreshing")


class TwinWorkload(BaseModel):
    """Workload knobs for the twin's performance model (unknowns default to 'assumed')."""

    name: str = "generic_riscv"
    vector_fraction: float = Field(0.4, ge=0.0, le=1.0)
    parallel_fraction: float = Field(0.95, ge=0.0, le=1.0)
    base_cpi: float = 1.2
    working_set_mb: float = 8.0
    mem_refs_per_inst: float = 0.35
    activity_factor: float = 0.15


class DigitalTwinConfig(BaseModel):
    search_space: list[SearchParameter] = Field(default_factory=default_search_space)
    objectives: list[Objective] = Field(default_factory=lambda: [
        Objective(name="perf", goal="max"), Objective(name="power_w", goal="min"),
        Objective(name="cost_usd", goal="min")])
    constraints: TwinConstraints = Field(default_factory=TwinConstraints)
    workload: TwinWorkload = Field(default_factory=TwinWorkload)
    strategy: Literal["nsga2", "lhs", "bayes"] = "nsga2"
    population: int = Field(64, ge=8, le=2048)
    generations: int = Field(30, ge=1, le=1000)
    lhs_samples: int = Field(256, ge=8, le=200_000)
    bayes_iterations: int = Field(40, ge=1, le=500)
    mc_trials: int = Field(256, ge=0, le=100_000)
    top_k: int = Field(5, ge=1, le=50)
    seed: int = 42
    priors: list[UncertaintyPrior] | None = None
    fixed: dict[str, Any] = Field(default_factory=dict,
                                  description="Parameters pinned to a value (removed from search)")
    calibration: dict[str, float] | None = None
    blackboxes: list[BlackBoxMacro] = Field(default_factory=list)
    ambient_c: float = 25.0
    theta_ja_c_per_w: float | None = None
    memory: TwinMemory | None = Field(None, description="Enable the on-chip memory model "
                                      "(knobs mem_tech, mem_banks, mem_interleave, pim_lanes)")


class ParetoPoint(BaseModel):
    rank: int
    params: dict[str, Any]
    objectives: dict[str, float]
    metrics: dict[str, float]
    feasible: bool
    robustness: RobustnessReport | None = None


class RobustnessReport(BaseModel):
    trials: int
    slack_yield: float
    thermal_violation_prob: float
    worst_slack_ps_p5: float
    perf_p5_ratio: float
    power_p95_ratio: float
    refresh_violation_prob: float = 0.0
    accuracy_margin_pct: float | None = Field(
        None, description="PDK accuracy margin applied to technology factors in Monte Carlo")
    robustness_score: float = Field(description="0-100: slack yield x (1 - thermal viol.) x "
                                                "(1 - refresh viol.) x perf p5 retention "
                                                "(capped at 1)")


class SensitivityEntry(BaseModel):
    parameter: str
    provenance: Provenance
    impact: dict[str, float] = Field(description="|Spearman rho| vs each metric")
    score: float


class DigitalTwinSummary(BaseModel):
    run_id: str
    strategy: str
    seed: int
    evaluated: int = Field(description="design points explored by the search")
    mc_evaluations: int = Field(0, description="extra model runs spent on Monte Carlo")
    feasible: int
    elapsed_s: float
    designs_per_sec: float = Field(description="model evaluations per second (search + MC)")
    pareto_size: int
    top: list[ParetoPoint]
    sensitivity: list[SensitivityEntry]
    unknown_parameters: list[str]
    calibrated_parameters: list[str]
    fidelity: FidelityLevel = FidelityLevel.ANALYTICAL
    fidelity_disclaimer: str = TWIN_DISCLAIMER
    notes: list[str] = Field(default_factory=list)


# --- Calibration ---------------------------------------------------------------------------

class CalibrationFactor(BaseModel):
    name: str
    value: float
    provenance: Provenance = Provenance.MEASURED
    source: str
    detail: str = ""


class CalibrationReport(BaseModel):
    calibration_id: str
    node: str | None = None
    factors: list[CalibrationFactor]
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    def factor_map(self) -> dict[str, float]:
        return {f.name: f.value for f in self.factors}


# --- Black-box IP ---------------------------------------------------------------------------

class PinTiming(BaseModel):
    name: str
    direction: Literal["input", "output", "inout"]
    clock: str | None = None
    setup_ps: float | None = None
    hold_ps: float | None = None
    clk_to_q_ps: float | None = None
    max_capacitance_ff: float | None = None
    width: int = 1


class PowerRail(BaseModel):
    name: str
    voltage_v: float
    typical_mw: float = 0.0
    max_mw: float | None = None


class BlackBoxMacro(BaseModel):
    """Third-party PHY/controller described only by its datasheet (no RTL needed)."""

    name: str
    vendor: str = "third_party"
    kind: Literal["phy", "controller", "pll", "sram", "analog", "io", "other"] = "other"
    width_um: float
    height_um: float
    keepout_um: float = 0.0
    pins: list[PinTiming] = Field(default_factory=list)
    rails: list[PowerRail] = Field(default_factory=list)
    max_freq_ghz: float | None = None
    count: int = 1
    provenance: Provenance = Provenance.PUBLISHED
    datasheet: str | None = None

    @property
    def area_mm2(self) -> float:
        return self.width_um * self.height_um / 1e6

    @property
    def area_with_keepout_mm2(self) -> float:
        k = self.keepout_um
        return (self.width_um + 2 * k) * (self.height_um + 2 * k) / 1e6

    @property
    def power_mw(self) -> float:
        return sum(r.typical_mw for r in self.rails)


DigitalTwinConfig.model_rebuild()
ParetoPoint.model_rebuild()


# --- On-chip memory (SRAM / gain-cell eDRAM / PIM) ------------------------------------------

class PimConfig(BaseModel):
    enabled: bool = False
    mac_bits: Literal[4, 8, 16] = 8
    lanes: int | None = Field(None, ge=1, le=256,
                              description="MAC lanes per bank (default word_bits/mac_bits)")


class OnChipMemoryConfig(BaseModel):
    """A multi-bank on-chip memory: 6T SRAM or 2T/3T gain-cell eDRAM, optional near-bank PIM."""

    name: str = "onchip_mem"
    technology: Literal["sram6t", "gc2t", "gc3t"] = "sram6t"
    pdk: str = "sky130"
    capacity_kib: int = Field(64, ge=1, le=1_048_576)
    banks: int = Field(8, ge=1, le=256)
    word_bits: int = Field(64, ge=8, le=1024)
    ports: Literal["1rw", "1r1w", "2rw"] = "1rw"
    interleave: Literal["low_order", "xor", "block"] = "low_order"
    arbiter: Literal["round_robin", "priority"] = "round_robin"
    ecc: Literal["none", "secded"] = "none"
    num_requesters: int = Field(4, ge=1, le=16)
    freq_mhz: float | None = Field(None, gt=0, description="default: 80% of the macro fmax")
    temp_c: float = Field(85.0, ge=-40, le=150)
    refresh_margin: float = Field(0.8, gt=0.1, le=1.0,
                                  description="refresh every rows within margin x retention")
    col_mux: int = Field(4, ge=1, le=32)
    pim: PimConfig = Field(default_factory=PimConfig)
    axi4_lite_bridge: bool = True

    @model_validator(mode="after")
    def _check(self) -> OnChipMemoryConfig:
        def pow2(n: int) -> bool:
            return n > 0 and n & (n - 1) == 0

        if not pow2(self.banks):
            raise ValueError("banks must be a power of two")
        if self.word_bits % 8:
            raise ValueError("word_bits must be a multiple of 8")
        words = self.capacity_kib * 1024 * 8 // self.word_bits
        if words % self.banks or not pow2(words // self.banks):
            raise ValueError("capacity / (banks x word_bits) must give a power-of-two word "
                             "count per bank")
        if words // self.banks < self.col_mux:
            raise ValueError("each bank needs at least col_mux words")
        if self.pim.enabled and self.word_bits % self.pim.mac_bits:
            raise ValueError("word_bits must be a multiple of pim.mac_bits")
        return self

    @property
    def words_per_bank(self) -> int:
        return self.capacity_kib * 1024 * 8 // self.word_bits // self.banks

    @property
    def rows_per_bank(self) -> int:
        return max(1, self.words_per_bank // self.col_mux)

    @property
    def pim_lanes(self) -> int:
        if not self.pim.enabled:
            return 0
        return min(self.pim.lanes or self.word_bits // self.pim.mac_bits,
                   self.word_bits // self.pim.mac_bits)


class MemWorkload(BaseModel):
    """Traffic offered to the memory. Defaults are 'assumed' and perturbed in Monte Carlo."""

    requests_per_cycle: float = Field(1.0, ge=0.0, le=16.0)
    read_fraction: float = Field(0.7, ge=0.0, le=1.0)
    pattern: Literal["sequential", "random", "strided"] = "random"
    stride: int = Field(1, ge=1)
    pim_fraction: float = Field(0.0, ge=0.0, le=1.0)
