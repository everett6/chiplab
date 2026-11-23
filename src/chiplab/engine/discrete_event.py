"""Level 2 discrete-event simulation engine for ChipLab (tile-level, heap-driven).

What is simulated
-----------------
Each request is one pass of the workload: its operators run in order, and every operator is
split into tiles. A tile moves through real FIFO resources, one event per stage:

    memory bank (row access, ``latency_ns``) -> memory channel bus (bytes / channel BW)
    -> interconnect links along the topology route (bytes / link BW + hop latency each)
    -> compute engine (FLOPs / effective rate) -> links back -> channel bus (write-back)

Resources are individual memory channels and banks of each chiplet *instance*, individual
directed links of the package interconnect (mesh XY routes, ring shortest direction,
crossbar/PCIe switch ports, one shared bus), and one compute engine per compute instance.
Requests are issued closed-loop with a fixed concurrency, so later requests queue behind
earlier ones. Queueing, contention and utilization come out of the event simulation, not out
of fixed overhead factors. Bank selection is seeded, so a seed reproduces a run exactly.

What is not simulated
---------------------
DRAM command timing (tRCD/tRP/refresh), caches and on-chip SRAM reuse, flow control and
cut-through routing (links are store-and-forward per tile), and operator-internal dataflow.
Area, power, thermal and cost come from the analytical models. It is not cycle-accurate.
"""

from __future__ import annotations

import heapq
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..schemas.design import DesignSpec
from ..schemas.simulation import (
    AssumptionRecord,
    BottleneckReport,
    SimulationConfig,
    SimulationMetrics,
    SimulationResult,
)
from ..schemas.workload import WorkloadSpec

DISCLAIMER = (
    "Level 2 discrete-event simulation (tile-level): operator tiles queue FIFO for memory "
    "banks and channels, interconnect links along topology routes, and compute engines; "
    "contention and queueing are simulated. First-order timing only (no DRAM command timing, "
    "no caches, store-and-forward links); not cycle-accurate."
)
MAX_TILES = 150_000         # total tile budget per run (about a second of simulation)
MAX_TILES_PER_OP = 256      # fine enough that tile granularity costs < ~1% of makespan
MIN_TILE_BYTES = 4096       # a tile moves at least a DRAM page worth of data ...
MIN_TILE_FLOPS = 1e6        # ... or does at least 1 MFLOP of work
DEFAULT_CONCURRENCY = 4     # requests in flight (closed loop)


# --- platform ------------------------------------------------------------------------------

@dataclass(slots=True)
class _Res:
    """A FIFO server: the time it is next free, and accumulated busy time."""
    name: str
    free: float = 0.0
    busy: float = 0.0
    wait: float = 0.0
    jobs: int = 0
    pending: float = 0.0     # service reserved by jobs routed here but not yet arrived

    def serve(self, now: float, service: float) -> float:
        start = now if now > self.free else self.free
        self.wait += start - now
        self.free = start + service
        self.busy += service
        self.jobs += 1
        return self.free


@dataclass(slots=True)
class _Channel:
    bus: _Res
    banks: list[_Res]
    node: int
    bw: float            # bytes / s
    latency: float       # s


@dataclass(slots=True)
class _Engine:
    res: _Res
    node: int
    rate: float          # FLOP / s (effective)
    kind: str
    pending: float = 0.0


@dataclass
class _Platform:
    nodes: int
    channels: list[_Channel]
    engines: list[_Engine]
    links: dict[Any, _Res]
    link_bw: float
    hop_latency: float
    topology: str
    cols: int
    tags: list[str] = field(default_factory=list)          # node index -> instance tag
    warnings: list[str] = field(default_factory=list)
    assumptions: list[AssumptionRecord] = field(default_factory=list)

    def route(self, src: int, dst: int) -> list[_Res]:
        if src == dst or not self.links:
            return []
        t = self.topology
        if t == "bus":
            return [self.links["bus"]]
        if t in ("crossbar", "pcie"):
            return [self.links[("out", src)], self.links[("in", dst)]]
        if t == "ring":
            n = self.nodes
            fwd = (dst - src) % n
            step = 1 if fwd <= n - fwd else -1
            path, cur = [], src
            while cur != dst:
                nxt = (cur + step) % n
                path.append(self.links[(cur, nxt)])
                cur = nxt
            return path
        # mesh / chiplet_link / unknown: dimension-ordered routing on a grid. XY first; if
        # the last grid row is partial and XY would cross a missing node, route YX instead.
        return self._grid(src, dst, x_first=True) or self._grid(src, dst, x_first=False)

    def _grid(self, src: int, dst: int, x_first: bool) -> list[_Res]:
        c = self.cols
        x, y, dx, dy = src % c, src // c, dst % c, dst // c
        path: list[_Res] = []
        for axis in ((0, 1) if x_first else (1, 0)):
            while (x != dx) if axis == 0 else (y != dy):
                nx, ny = (x + (1 if dx > x else -1), y) if axis == 0 else (x, y + (1 if dy > y else -1))
                link = self.links.get((y * c + x, ny * c + nx))
                if link is None:
                    return []
                path.append(link)
                x, y = nx, ny
        return path


def _build_platform(design: DesignSpec, allow_defaults: bool) -> _Platform:
    from chiplab.engine.analytical import _estimate_compute_chiplet, _estimate_memory_chiplet
    from chiplab.models.interconnect import estimate_interconnect

    channels: list[_Channel] = []
    engines: list[_Engine] = []
    warnings: list[str] = []
    node = 0
    tags: list[str] = []
    for ch in design.chiplets:
        count = int(ch.count)
        mem, _, _ = _estimate_memory_chiplet(ch, allow_defaults)
        comp, _, _ = _estimate_compute_chiplet(ch, allow_defaults)
        n_ch_total = int(mem.get("channels") or 0) if mem.get("bandwidth_tbps", 0) > 0 else 0
        per_inst_ch = n_ch_total // count if count else 0
        bw_ch = (mem["bandwidth_tbps"] * 1e12 / n_ch_total) if n_ch_total else 0.0
        banks = max(1, int(mem.get("banks_per_channel") or 1))
        lat = float(mem.get("latency_ns") or 0.0) * 1e-9
        rate = float(comp.get("effective_tops") or 0.0) * 1e12
        kind = f"{ch.type} {getattr(ch.compute, 'architecture', '') or ''}".lower()
        for i in range(count):
            tag = f"{ch.id or ch.type}[{i}]"
            tags.append(tag)
            for c in range(per_inst_ch):
                channels.append(_Channel(
                    bus=_Res(f"{tag}.ch{c}"), node=node, bw=bw_ch, latency=lat,
                    banks=[_Res(f"{tag}.ch{c}.b{b}") for b in range(banks)]))
            if rate > 0:
                engines.append(_Engine(res=_Res(f"{tag}.compute"), node=node, rate=rate,
                                       kind=kind))
            node += 1

    inter, _, _ = estimate_interconnect(design, node, allow_defaults=allow_defaults)
    topology = str(design.interconnect.topology).lower()
    link_bw = float(inter.get("per_link_bw_gbps") or 0.0) * 1e9
    hop_lat = float(inter.get("hop_latency_ns") or 0.0) * 1e-9
    cols = max(1, math.ceil(math.sqrt(node)))
    links: dict[Any, _Res] = {}
    assumptions = [AssumptionRecord(
        parameter="interconnect.lanes_per_chiplet", value=design.interconnect.lanes_per_chiplet,
        unit="lanes", provenance="assumed",
        notes="Not modeled separately: each link carries link_bandwidth_gbps (as in the "
              "analytical bisection estimate).")]
    if str(design.package.type).lower() == "monolithic" or node <= 1:
        pass  # one die: no package interconnect
    elif link_bw <= 0:
        warnings.append("Interconnect link bandwidth is 0/unknown: transfers between chiplets "
                        "are treated as free (interconnect not simulated).")
    elif topology == "bus":
        links["bus"] = _Res("bus")
    elif topology in ("crossbar", "pcie"):
        for n in range(node):
            links[("out", n)] = _Res(f"n{n}->switch")
            links[("in", n)] = _Res(f"switch->n{n}")
    elif topology == "ring":
        for n in range(node):
            for m in ((n + 1) % node, (n - 1) % node):
                links[(n, m)] = _Res(f"n{n}->n{m}")
    else:
        if topology not in ("mesh", "chiplet_link"):
            warnings.append(f"Unknown topology '{topology}': routed as a 2D mesh.")
        for n in range(node):
            x, y = n % cols, n // cols
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < cols and ny >= 0 and ny * cols + nx < node:
                    links[(n, ny * cols + nx)] = _Res(f"n{n}->n{ny * cols + nx}")
    return _Platform(nodes=node, channels=channels, engines=engines, links=links,
                     link_bw=link_bw, hop_latency=hop_lat, topology=topology, cols=cols,
                     tags=tags, warnings=warnings, assumptions=assumptions)


# --- event simulation ----------------------------------------------------------------------

@dataclass(slots=True)
class _Tile:
    req: int
    op: int
    rd_bytes: float
    wr_bytes: float
    flops: float
    stage: int = 0
    rd_ch: _Channel | None = None
    wr_ch: _Channel | None = None
    engine: _Engine | None = None
    path: list[_Res] = field(default_factory=list)
    hop: int = 0


def _simulate(p: _Platform, ops: list[Any], num_requests: int, concurrency: int,
              tiles: list[int], seed: int, pin_preferred: bool = False) -> dict[str, Any]:
    rng = random.Random(seed)
    heap: list[tuple[float, int, _Tile]] = []
    seq = 0
    ch_rr = 0
    n_ch = len(p.channels)
    remaining: dict[tuple[int, int], int] = {}
    req_start: dict[int, float] = {}
    latencies: list[float] = []
    flops_done = 0.0
    bytes_moved = 0.0
    events = 0
    pools: list[list[_Engine]] = []
    for op in ops:
        pref = (op.preferred_chiplet_type or "").lower()
        match = [e for e in p.engines if pin_preferred and pref and pref in e.kind]
        pools.append(match or p.engines)

    route_cache: dict[tuple[int, int], list[_Res]] = {}

    def route(a: int, b: int) -> list[_Res]:
        k = (a, b)
        if k not in route_cache:
            route_cache[k] = p.route(a, b)
        return route_cache[k]

    def reserve(path: list[_Res], nbytes: float) -> list[_Res]:
        svc = nbytes / p.link_bw if path else 0.0
        for link in path:
            link.pending += svc
        return path

    def push(t: float, tile: _Tile) -> None:
        nonlocal seq
        heapq.heappush(heap, (t, seq, tile))
        seq += 1

    def next_channel() -> _Channel | None:
        nonlocal ch_rr
        if not n_ch:
            return None
        c = p.channels[ch_rr % n_ch]   # address interleaving across all channels
        ch_rr += 1
        return c

    def issue_op(req: int, k: int, now: float) -> None:
        op, n = ops[k], tiles[k]
        remaining[(req, k)] = n
        rd, wr, fl = op.bytes_read / n, op.bytes_write / n, op.flops / n
        for _ in range(n):
            push(now, _Tile(req=req, op=k, rd_bytes=rd, wr_bytes=wr, flops=fl))

    def start_request(req: int, now: float) -> None:
        req_start[req] = now
        issue_op(req, 0, now)

    issued = min(concurrency, num_requests)
    for r in range(issued):
        start_request(r, 0.0)

    # Stages: 0 read-bank, 1 read-bus, 2 route-to-engine, 3 compute, 4 route-to-memory,
    # 5 write-bank/bus, 6 done.
    now = 0.0
    while heap:
        now, _, t = heapq.heappop(heap)
        events += 1
        s = t.stage
        if s == 0:
            t.rd_ch = next_channel()
            if t.rd_ch is None or t.rd_bytes <= 0:
                t.stage = 2
                push(now, t)
                continue
            bank = t.rd_ch.banks[rng.randrange(len(t.rd_ch.banks))]
            t.stage = 1
            push(bank.serve(now, t.rd_ch.latency), t)
        elif s == 1:
            bytes_moved += t.rd_bytes
            t.stage = 2
            push(t.rd_ch.bus.serve(now, t.rd_bytes / t.rd_ch.bw), t)
        elif s == 2:
            if t.hop == 0 and t.engine is None:
                if t.flops > 0 and pools[t.op]:
                    # Earliest projected finish: data arrival (unloaded transfer over the
                    # route) vs engine backlog, plus this tile's compute. This prefers the
                    # engine next to the data unless it is already overloaded.
                    src0 = t.rd_ch.node if t.rd_ch is not None else -1

                    def _finish(e: _Engine) -> float:
                        arrive = now
                        if src0 >= 0 and t.rd_bytes > 0 and p.links:
                            xfer = t.rd_bytes / p.link_bw
                            for link in route(src0, e.node):   # backlog incl. reservations
                                arrive = max(arrive, link.free + link.pending) + xfer \
                                    + p.hop_latency
                        return max(arrive, e.res.free + e.pending) + t.flops / e.rate

                    eng = min(pools[t.op], key=_finish)
                    eng.pending += t.flops / eng.rate
                    t.engine = eng
                src = t.rd_ch.node if t.rd_ch is not None else (t.engine.node if t.engine else 0)
                dst = t.engine.node if t.engine else src
                t.path = reserve(route(src, dst), t.rd_bytes) if t.rd_bytes > 0 else []
            if t.hop < len(t.path):
                link = t.path[t.hop]
                t.hop += 1
                svc = t.rd_bytes / p.link_bw
                link.pending -= svc
                push(link.serve(now, svc) + p.hop_latency, t)
            else:
                t.stage, t.hop = 3, 0
                push(now, t)
        elif s == 3:
            if t.engine is not None:
                svc = t.flops / t.engine.rate
                t.engine.pending -= svc
                flops_done += t.flops
                t.stage = 4
                push(t.engine.res.serve(now, svc), t)
            else:
                t.stage = 4
                push(now, t)
        elif s == 4:
            if t.hop == 0:
                t.wr_ch = next_channel() if t.wr_bytes > 0 else None
                src = t.engine.node if t.engine else (t.rd_ch.node if t.rd_ch else 0)
                t.path = (reserve(route(src, t.wr_ch.node), t.wr_bytes)
                          if t.wr_ch is not None else [])
            if t.hop < len(t.path):
                link = t.path[t.hop]
                t.hop += 1
                svc = t.wr_bytes / p.link_bw
                link.pending -= svc
                push(link.serve(now, svc) + p.hop_latency, t)
            else:
                t.stage = 5
                push(now, t)
        elif s == 5:
            t.stage = 6
            if t.wr_ch is not None:
                bytes_moved += t.wr_bytes
                bank = t.wr_ch.banks[rng.randrange(len(t.wr_ch.banks))]
                done = t.wr_ch.bus.serve(bank.serve(now, t.wr_ch.latency), t.wr_bytes / t.wr_ch.bw)
                push(done, t)
            else:
                push(now, t)
        else:
            key = (t.req, t.op)
            remaining[key] -= 1
            if remaining[key]:
                continue
            del remaining[key]
            if t.op + 1 < len(ops):
                issue_op(t.req, t.op + 1, now)
            else:
                latencies.append(now - req_start.pop(t.req))
                if issued < num_requests:
                    start_request(issued, now)
                    issued += 1
    return {"makespan_s": now, "latencies_s": latencies, "flops_done": flops_done,
            "bytes_moved": bytes_moved, "events": events}


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def _util(res: list[_Res], span: float) -> tuple[float, float]:
    if not res or span <= 0:
        return 0.0, 0.0
    u = [r.busy / span for r in res]
    return sum(u) / len(u), max(u)


def _resource_map(p: _Platform, span: float) -> dict[str, Any]:
    """Busy fraction of every simulated server, keyed by chiplet instance (for the studio)."""
    def u(r: _Res) -> float:
        return round(r.busy / span, 4) if span > 0 else 0.0

    engines = {p.tags[e.node]: u(e.res) for e in p.engines}
    channels: dict[str, list[float]] = {}
    banks: dict[str, list[list[float]]] = {}
    access: dict[str, list[list[float]]] = {}
    for c in p.channels:
        channels.setdefault(p.tags[c.node], []).append(u(c.bus))
        banks.setdefault(p.tags[c.node], []).append([u(b) for b in c.banks])
        # traffic share: channel busy fraction x this bank's share of the channel's accesses
        n = sum(b.jobs for b in c.banks)
        access.setdefault(p.tags[c.node], []).append(
            [round(u(c.bus) * b.jobs * len(c.banks) / n, 4) if n else 0.0 for b in c.banks])
    links = []
    for key, r in p.links.items():
        if key == "bus":
            src, dst = "bus", "bus"
        elif isinstance(key[0], str):          # crossbar / PCIe switch ports
            src, dst = ((p.tags[key[1]], "switch") if key[0] == "out"
                        else ("switch", p.tags[key[1]]))
        else:
            src, dst = p.tags[key[0]], p.tags[key[1]]
        links.append({"src": src, "dst": dst, "util": u(r), "jobs": r.jobs})
    return {"nodes": p.tags, "topology": p.topology, "engines": engines,
            "channels": channels, "banks": banks, "bank_access": access, "links": links}


# --- public entry point --------------------------------------------------------------------

def run_discrete_event_simulation(
    design: DesignSpec,
    workload: WorkloadSpec | None,
    config: SimulationConfig,
    design_id: str = "unsaved",
) -> SimulationResult:
    """Run the Level 2 tile-level discrete-event simulation (see module docstring)."""
    from ..engine.analytical import run_analytical_simulation
    from ..engine.validator import validate_design
    from ..schemas.workload import expand_workload_operators, is_llm_decode

    run_id = f"discrete_event_{design_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    from ..backends.ecore import apply_ecore_to_design
    from ..schemas.design import plain_design
    original_design = design  # with TrackedValues: the analytical pass records their provenance
    design, _ = plain_design(design)
    design, _, _ = apply_ecore_to_design(design)  # eCore PIM numbers; recorded by analytical
    validation = validate_design(design, config.allow_estimated_defaults)

    def _fail(msg: str, errors: list[str], warnings: list[str]) -> SimulationResult:
        return SimulationResult(
            run_id=run_id, design_id=design_id, design_name=design.name,
            fidelity=config.fidelity, fidelity_disclaimer=f"Simulation skipped: {msg}",
            metrics=SimulationMetrics(),
            bottleneck=BottleneckReport(primary_bottleneck="validation_error",
                                        bound_type="capacity", severity_score=1.0,
                                        details=errors, recommendations=[msg]),
            assumptions=validation.missing_parameters, warnings=warnings,
            raw_details={"validation_errors": errors}, success=False,
            error_message="; ".join(errors))

    if not validation.valid:
        return _fail("design validation failed", validation.errors, validation.warnings)

    wl = workload if workload is not None else WorkloadSpec()
    base = run_analytical_simulation(original_design, wl, config, design_id)  # static metrics
    if not base.success:
        return _fail("analytical models failed", [base.error_message or "unknown"],
                     base.warnings)

    plat = _build_platform(design, config.allow_estimated_defaults)
    ops = [op for op in expand_workload_operators(wl)
           if op.flops > 0 or op.bytes_read > 0 or op.bytes_write > 0]
    if not ops:  # mirror the analytical default: one second of effective compute
        from ..schemas.workload import OperatorSpec
        ops = [OperatorSpec(name="default_compute", op_type="compute",
                            flops=base.metrics.effective_compute_tops * 1e12,
                            bytes_read=0.0, bytes_write=0.0)]
    if any(op.flops > 0 for op in ops) and not plat.engines:
        return _fail("no compute engines", ["Workload has FLOPs but no chiplet has compute "
                                            "(peak_tops) to run them."], base.warnings)
    if any(op.bytes_read > 0 or op.bytes_write > 0 for op in ops) and not plat.channels:
        plat.warnings.append("Workload moves bytes but no chiplet has memory channels with "
                             "bandwidth: memory traffic is treated as free.")

    bws = {round(c.bw) for c in plat.channels}
    if len(bws) > 1:
        plat.warnings.append(
            "Memory channels have different bandwidths and addresses interleave uniformly "
            "across them, so the slowest channels set the pace; the analytical model sums "
            "bandwidth as if it were all usable.")

    num_requests = max(1, int(config.num_requests or 1))
    concurrency = min(DEFAULT_CONCURRENCY, num_requests)
    # Tile count follows the physical amount of work, so it does not change with
    # num_requests; only an over-budget run is coarsened (and says so).
    tiles = [max(1, min(MAX_TILES_PER_OP,
                        max(math.ceil((op.bytes_read + op.bytes_write) / MIN_TILE_BYTES),
                            math.ceil(op.flops / MIN_TILE_FLOPS)))) for op in ops]
    total_tiles = sum(tiles) * num_requests
    if total_tiles > MAX_TILES:
        f = MAX_TILES / total_tiles
        tiles = [max(1, int(n * f)) for n in tiles]
        plat.warnings.append(f"Large run: tiles per operator coarsened by {1 / f:.1f}x to stay "
                             f"within {MAX_TILES:,} simulated tiles.")
    tiles_per_op = max(tiles)
    # User-written operators may pin themselves to a chiplet type (preferred_chiplet_type);
    # ChipLab's synthesized operators only carry it as a hint, so any engine may run them
    # (the analytical model makes the same assumption).
    pin = bool(wl.operators)
    sim = _simulate(plat, ops, num_requests, concurrency, tiles, int(config.seed), pin)

    span = sim["makespan_s"]
    lat = sim["latencies_s"]
    mean_lat_ms = (sum(lat) / len(lat) * 1e3) if lat else 0.0
    cap = sum(e.rate for e in plat.engines)
    compute_util = sim["flops_done"] / (cap * span) if cap and span else 0.0
    mem_avg, mem_max = _util([c.bus for c in plat.channels], span)
    bank_avg, bank_max = _util([b for c in plat.channels for b in c.banks], span)
    link_avg, link_max = _util(list(plat.links.values()), span)
    eng_avg, eng_max = _util([e.res for e in plat.engines], span)
    tokens = wl.batch_size * (1 if is_llm_decode(wl) else wl.sequence_length)
    throughput = num_requests * tokens / span if span > 0 else 0.0
    achieved_tops = sim["flops_done"] / span / 1e12 if span > 0 else 0.0

    def _wait_us(res: list[_Res]) -> float:
        jobs = sum(r.jobs for r in res)
        return sum(r.wait for r in res) / jobs * 1e6 if jobs else 0.0

    waits = {"memory_bank_wait_us": _wait_us([b for c in plat.channels for b in c.banks]),
             "memory_channel_wait_us": _wait_us([c.bus for c in plat.channels]),
             "interconnect_wait_us": _wait_us(list(plat.links.values())),
             "compute_wait_us": _wait_us([e.res for e in plat.engines])}

    m = base.metrics
    total_bytes = sum(op.bytes_read + op.bytes_write for op in ops)
    total_flops = sum(op.flops for op in ops)
    metrics = m.model_copy(update={
        "effective_compute_tops": round(achieved_tops, 4),
        "estimated_latency_ms": round(mean_lat_ms, 6),
        "throughput_tokens_per_sec": round(throughput, 4) if throughput else None,
        "throughput_ops_per_sec": sim["flops_done"] / span if span > 0 else 0.0,
        "arithmetic_intensity": round(total_flops / total_bytes, 4) if total_bytes else 0.0,
        "ridge_point_ops_per_byte": (round(m.peak_compute_tops / m.aggregate_memory_bandwidth_tbps,
                                           4) if m.aggregate_memory_bandwidth_tbps else 0.0),
        "utilization_pct": round(min(100.0, compute_util * 100.0), 2),
        "energy_per_inference_mj": round(m.total_power_watts * mean_lat_ms, 6),
    })

    # Bottleneck = the most utilized resource class (busiest single server in it).
    stages = {"compute": eng_max, "memory_bandwidth": max(mem_max, bank_max),
              "interconnect": link_max}
    bound, top = max(stages.items(), key=lambda kv: kv[1])
    details = [f"{num_requests} requests ({concurrency} in flight), {len(ops)} operators x "
               f"up to {tiles_per_op} tiles, {sim['events']:,} events, makespan {span * 1e3:.4g} ms.",
               f"Busiest server utilization: compute {eng_max:.0%}, memory channel "
               f"{mem_max:.0%}, memory bank {bank_max:.0%}, interconnect link {link_max:.0%}."]
    if top < 0.5:
        bound = "balanced"
        details.append("No resource is above 50% busy: the run is latency/dependency bound "
                       "(operator barriers, access and hop latency).")
    recs = {"compute": "Add compute engines or raise efficiency; memory and links are idle "
                       "part of the time.",
            "memory_bandwidth": "Add memory channels/bandwidth or reduce bytes per token "
                                "(quantization, PIM).",
            "interconnect": "Widen links, add lanes, or use a higher-bisection topology; place "
                            "memory next to the compute that reads it.",
            "balanced": "Increase concurrency/batching or overlap operators to hide latency."}
    bottleneck = BottleneckReport(
        primary_bottleneck=bound, bound_type=bound, severity_score=round(top, 3),
        utilization_breakdown={
            "compute_util_pct": round(compute_util * 100, 2),
            "compute_engine_max_pct": round(eng_max * 100, 2),
            "memory_channel_avg_pct": round(mem_avg * 100, 2),
            "memory_channel_max_pct": round(mem_max * 100, 2),
            "memory_bank_max_pct": round(bank_max * 100, 2),
            "interconnect_link_avg_pct": round(link_avg * 100, 2),
            "interconnect_link_max_pct": round(link_max * 100, 2),
            **{k: round(v, 4) for k, v in waits.items()}},
        details=details, recommendations=[recs[bound]])

    assumptions = list(base.assumptions) + plat.assumptions + [
        AssumptionRecord(parameter="num_requests", value=num_requests, unit="requests",
                         provenance="assumed", notes=f"closed loop, {concurrency} in flight"),
        AssumptionRecord(parameter="tiles_per_operator", value=tiles, unit="tiles",
                         provenance="assumed",
                         notes=f"simulation granularity: >= {MIN_TILE_BYTES} B or "
                               f"{MIN_TILE_FLOPS:.0e} FLOP per tile, <= {MAX_TILES_PER_OP}"),
        AssumptionRecord(parameter="operator_mapping",
                         value="pinned to preferred_chiplet_type" if pin else "any engine",
                         provenance="assumed",
                         notes="tiles go to the engine with the earliest projected finish "
                               "(data transfer + backlog + compute)"),
        AssumptionRecord(parameter="memory_channel_selection", value="interleaved",
                         provenance="assumed",
                         notes="tiles round-robin over all channels; banks drawn with the seed"),
    ]
    return SimulationResult(
        run_id=run_id, design_id=design_id, design_name=design.name,
        fidelity=config.fidelity, fidelity_disclaimer=DISCLAIMER, metrics=metrics,
        bottleneck=bottleneck, assumptions=assumptions,
        warnings=list(dict.fromkeys(validation.warnings + base.warnings + plat.warnings)),
        raw_details={
            "engine": "tile_level_discrete_event",
            "num_requests_simulated": num_requests, "concurrency": concurrency,
            "operators": len(ops), "tiles_per_operator": tiles,
            "events": sim["events"], "makespan_ms": span * 1e3,
            "latency_ms": {"mean": mean_lat_ms, "p50": _pct(lat, 0.5) * 1e3,
                           "p95": _pct(lat, 0.95) * 1e3, "max": max(lat, default=0) * 1e3},
            "resources": {"compute_engines": len(plat.engines),
                          "memory_channels": len(plat.channels),
                          "interconnect_links": len(plat.links), "topology": plat.topology,
                          "chiplet_instances": plat.nodes},
            "total_flops": total_flops, "total_bytes": total_bytes,
            "bytes_moved": sim["bytes_moved"],
            "analytical_latency_ms": m.estimated_latency_ms,
            "resource_utilization": _resource_map(plat, span),
            **waits,
        },
        success=True,
    )
