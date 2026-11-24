// ChipLab ASIC Studio: competition ASIC, memory studio and component foundry.
// Every number shown comes from the API (runs of the model, Verilator, the memory compiler or
// the component SDK); steps that have not run are shown as not run.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (v, d = 2) => (v === null || v === undefined ? "–" : typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: d }) : esc(v));
let O = null;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const pill = (s) => {
  const t = String(s);
  const cls = /^(met|done|true|ok|passed|ready)/i.test(t) ? "ok" : /(fail|missing|not met|false)/i.test(t) ? "bad" : /(not run|unknown|partial|progress|blocked|not checked)/i.test(t) ? "warn" : "";
  return `<span class="pill ${cls}">${esc(t)}</span>`;
};
const table = (cols, rows) => `<table><tr>${cols.map((c) => `<th>${esc(c[0])}</th>`).join("")}</tr>${rows.map((r) => `<tr>${cols.map((c) => `<td class="${c[2] || ""}">${c[1](r)}</td>`).join("")}</tr>`).join("")}</table>`;

function md(s) { // tiny Markdown renderer: headings, tables, code, lists, paragraphs
  const out = []; let tbl = null, list = false, code = false;
  for (const raw of s.split("\n")) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) { out.push(code ? "</pre>" : "<pre>"); code = !code; continue; }
    if (code) { out.push(esc(line)); continue; }
    if (line.startsWith("|")) {
      const cells = line.split("|").slice(1, -1).map((c) => c.trim());
      if (cells.every((c) => /^-+$/.test(c))) continue;
      if (!tbl) { tbl = []; out.push("<table>"); out.push(`<tr>${cells.map((c) => `<th>${inl(c)}</th>`).join("")}</tr>`); } else out.push(`<tr>${cells.map((c) => `<td>${inl(c)}</td>`).join("")}</tr>`);
      continue;
    } else if (tbl) { out.push("</table>"); tbl = null; }
    if (/^- /.test(line)) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inl(line.slice(2))}</li>`); continue; } else if (list) { out.push("</ul>"); list = false; }
    const h = line.match(/^(#{1,3}) (.*)/);
    if (h) out.push(`<h${h[1].length + 1}>${inl(h[2])}</h${h[1].length + 1}>`); else if (line) out.push(`<p>${inl(line)}</p>`);
  }
  if (tbl) out.push("</table>"); if (list) out.push("</ul>");
  return out.join("\n");
}
function inl(s) { return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>"); }

// ---------------------------------------------------------------- tabs
$("tabs").onclick = (e) => {
  const b = e.target.closest("button"); if (!b) return;
  for (const x of $("tabs").children) x.classList.toggle("on", x === b);
  for (const s of document.querySelectorAll("section.tab")) s.classList.toggle("on", s.id === "t-" + b.dataset.t);
  if (b.dataset.t === "foundry" && !$("c-list").options.length) loadComponents();
};

// ---------------------------------------------------------------- overview
async function load() {
  O = await api("/api/asic/overview");
  const a = O.architecture, c = a.competition, R = O.report;
  $("comp").innerHTML = Object.entries({ Process: c.process, Flow: c.flow, "Max tiles": c.max_tiles, Deadline: c.deadline,
    Baseline: c.baseline_protocols.join(", "), Stretch: c.stretch_protocols.join(", "), Source: c.source }).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  const days = R.requirements.find((r) => r.id === "R11");
  $("deadline").textContent = days ? days.status : "";
  $("lanes").innerHTML = [0, 1, 2, 3].map((i) => `<div class="lane"><b>lane ${i}</b><br>PC · delay · divider<br>X Y · OSR ISR (16 b)<br>TX/RX FIFO ×2<br>NRZI · stuffing · CRC 1-32</div>`).join("");
  $("rbits").textContent = `${a.register_bits.total} flip-flops from the RTL declarations (${a.register_bits.provenance}); shared 32×16 instruction memory, 8 IRQ flags, 16-pin space, SPI host port.`;
  $("isa").innerHTML = md(a.isa_markdown);
  // protocol lab examples
  $("ex").innerHTML = Object.keys(a.examples).map((k) => `<option>${k}</option>`).join("");
  setExample("uart_tx");
  $("planned").innerHTML = table([["Protocol", (r) => esc(r.protocol)], ["Status", (r) => pill(r.status)], ["Note", (r) => esc(r.note)]], a.planned_protocols);
  renderResults();
  $("levels").innerHTML = table([["#", (r) => r.level, "n"], ["Level", (r) => esc(r.name)], ["Engine", (r) => esc(r.engine)], ["Here", (r) => pill(r.available ? "ready" : "not installed")]], O.levels);
  const pdks = ["sky130", "ihp_sg13g2", "gf180", "n65", "n28", "asap7"];
  for (const id of ["m-pdk", "c-pdk"]) $(id).innerHTML = pdks.map((p) => `<option${p === "ihp_sg13g2" ? " selected" : ""}>${p}</option>`).join("");
}

function setExample(k) {
  const e = O && O.architecture.examples[k];
  if (!e) return;
  $("src").value = e.source.replace(/^\n/, "");
  $("regs").value = JSON.stringify(e.regs);
  $("bench").value = e.bench;
  $("clkdiv").value = 2;
}
$("ex").onchange = (e) => setExample(e.target.value);

// ---------------------------------------------------------------- protocol lab
$("asm").onclick = async () => {
  try { const r = await api("/api/asic/assemble", { source: $("src").value });
    $("labout").textContent = r.ok ? r.words.map((w) => `${String(w.addr).padStart(2)}  ${w.word}  ${w.asm}`).join("\n") + `\nstart ${r.start}  wrap ${r.wrap}` : r.error;
  } catch (e) { $("labout").textContent = e.message; }
};
$("sim").onclick = async () => {
  $("labbusy").textContent = "simulating…";
  try {
    const r = await api("/api/asic/simulate", { source: $("src").value, lane_regs: JSON.parse($("regs").value || "{}"), cycles: +$("cycles").value,
      bench: $("bench").value, lane: +$("lane").value, clkdiv: +$("clkdiv").value });
    if (!r.ok) { $("labout").textContent = r.error; return; }
    drawWave(r.waveform, r.oe_waveform, r.cycles);
    $("reads").innerHTML = r.reads.length ? r.reads.map(([a, v]) => `<span class="pill">R ${a.toString(16)} = ${v.toString(16)}</span>`).join(" ") : "none";
    $("labout").textContent = `${r.fidelity}\ncycles ${r.cycles}\nlane state ${JSON.stringify(r.lane_state)}\n\n` + r.events.filter((x) => !x.startsWith("P")).join("\n");
  } catch (e) { $("labout").textContent = e.message; } finally { $("labbusy").textContent = ""; }
};
function drawWave(w, oe, total) {
  const cv = $("wave"), r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = r.width * dpr; cv.height = r.height * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  g.fillStyle = "#0e141b"; g.fillRect(0, 0, r.width, r.height);
  const L = 44, lane = (r.height - 20) / 8, X = (c) => L + (c / Math.max(total, 1)) * (r.width - L - 8);
  g.font = "11px ui-monospace"; g.lineWidth = 1.5;
  for (let b = 0; b < 8; b++) {
    const y0 = 10 + b * lane, hi = y0 + 4, lo = y0 + lane - 6;
    g.fillStyle = "#8b97a4"; g.fillText(`pin ${b}`, 4, y0 + lane / 2 + 4);
    g.strokeStyle = "#263140"; g.beginPath(); g.moveTo(L, lo + 3); g.lineTo(r.width, lo + 3); g.stroke();
    const oev = (c) => { let v = 0; for (const [cc, vv] of oe) { if (cc > c) break; v = vv; } return v >> b & 1; };
    g.beginPath(); let px = L, py = lo;
    for (let i = 0; i < w.length; i++) {
      const [c, v] = w[i], x = X(c), y = (v >> b & 1) ? hi : lo;
      g.strokeStyle = oev(c) ? "#6ea0ff" : "#4a5563";
      g.lineTo(x, py); g.lineTo(x, y); px = x; py = y;
    }
    g.lineTo(r.width - 8, py); g.strokeStyle = "#6ea0ff"; g.stroke();
  }
}

// ---------------------------------------------------------------- runs
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-run]"); if (!b) return;
  const kind = b.dataset.run, busy = kind === "protocols" ? $("labbusy") : kind === "synthesis_ihp" ? $("ihpbusy") : kind === "hardening_import" ? $("hardbusy") : $("verbusy");
  busy.textContent = `running ${kind}…`;
  document.querySelectorAll("[data-run]").forEach((x) => (x.disabled = true));
  try {
    const r = await api("/api/asic/run", { kind, backend: b.dataset.backend || "rtl", seeds: +$("seeds").value, cycles: kind === "formal" ? 20 : +$("vcycles").value });
    if (kind === "protocols") showProtocols(r);
    if (["tt_smoke", "lint", "synthesis", "synthesis_ihp", "icarus_tt", "formal", "gate_sim"].includes(kind)) busy.textContent = `${kind}: ${(r.ok ?? r.passed) ? "passed" : "failed"}`;
    if (kind === "hardening_import") busy.textContent = r.ok ? `run ${r.run_id}: ${r.conclusion}` : r.error;
    O = await api("/api/asic/overview"); renderResults();
    if (!["tt_smoke", "lint", "synthesis", "synthesis_ihp", "icarus_tt", "formal", "gate_sim", "hardening_import"].includes(kind)) busy.textContent = "";
  } catch (err) { busy.textContent = err.message; } finally { document.querySelectorAll("[data-run]").forEach((x) => (x.disabled = false)); }
});
function showProtocols(r) {
  $("prot").innerHTML = `<div class="row">${pill(`${r.passed}/${r.total} passed`)} <span class="faint">backend: ${esc(r.backend)}</span></div>` +
    table([["Test", (x) => esc(x.name)], ["Protocol", (x) => esc(x.protocol || "")], ["Case", (x) => esc(x.case || x.error || "")], ["Result", (x) => pill(x.passed ? "passed" : "failed")], ["Fidelity", (x) => esc(x.fidelity || "")]], r.results);
}
function renderResults() {
  const R = O.report, V = R.verification;
  if (R.protocol_tests.passed !== undefined && !$("prot").innerHTML) $("prot").innerHTML = `<div class="row">${pill(`${R.protocol_tests.passed}/${R.protocol_tests.total} passed (last RTL run)`)} ${R.protocol_tests.stale ? pill("stale: RTL changed") : ""}<span class="faint">${esc(R.protocol_tests.timestamp)}</span></div>`;
  const g = V.regression;
  $("reg").innerHTML = g.status ? pill("not run") : `<dl class="kv"><dt>Lockstep seeds</dt><dd>${g.passed}/${g.seeds} × ${g.cycles_per_seed} cycles</dd>
    <dt>Functional coverage</dt><dd>${g.functional.pct}% (${g.functional.covered}/${g.functional.bins} bins)</dd>
    <dt>Code coverage</dt><dd>${g.code.available ? g.code.pct + "% of " + g.code.points + " points" : "n/a"}</dd>
    <dt>Assertions</dt><dd>${g.assertions.defined} defined, ${g.assertions.failed} failed</dd><dt>RTL</dt><dd>${esc(g.rtl_hash)} ${g.stale ? "(stale)" : ""}</dd></dl>
    ${g.functional.missing.length ? `<div class="faint">Uncovered: ${esc(g.functional.missing.join(", "))}</div>` : ""}`;
  const m = V.mutation;
  $("mut").innerHTML = m.status ? pill("not run") : `<div class="row">${pill(`${m.killed}/${m.mutants} mutants killed`)}</div>` +
    table([["Injected bug", (x) => esc(x.mutation)], ["Caught", (x) => pill(x.killed ? "passed" : "missing")], ["By", (x) => esc(x.random_seed !== null && x.random_seed !== undefined ? `random seed ${x.random_seed}` : x.directed_test || "–")]], m.results);
  const sy = R.synthesis, fo = V.formal, gs = V.gate_sim;
  $("syn").innerHTML = sy.status ? pill("not run") : `<dl class="kv"><dt>Result</dt><dd>${pill(sy.ok ? "passed" : "failed")}</dd><dt>Cells</dt><dd>${f(sy.cells, 0)}</dd><dt>Flip-flops</dt><dd>${f(sy.flops, 0)}</dd>
    <dt>Mapping</dt><dd>${esc(sy.mapping)}</dd><dt>Tool</dt><dd>${esc(sy.tool)}</dd><dt>RTL</dt><dd>${esc(sy.rtl_hash)}${sy.stale ? " (stale)" : ""}</dd></dl><div class="faint">${esc(sy.note)}</div>`;
  $("frm").innerHTML = `<dl class="kv"><dt>Formal</dt><dd>${fo.status && !fo.mode ? pill("not run") : `${pill(fo.ok ? "passed" : fo.status)} ${esc(fo.mode)} depth ${fo.depth}`}</dd>
    <dt>Gate-level random</dt><dd>${gs.status ? pill("not run") : `${gs.passed}/${gs.seeds} seeds × ${gs.cycles_per_seed}`}</dd>
    <dt>Gate-level protocols</dt><dd>${gs.status ? "–" : `${gs.protocols.passed}/${gs.protocols.total}`}</dd>
    <dt>TT test (Icarus)</dt><dd>${V.icarus_tt.status ? pill("not run") : pill(V.icarus_tt.ok ? "passed" : "failed") + " " + esc(V.icarus_tt.detail)}</dd></dl>
    <div class="faint">${esc(fo.meaning || "")}${gs.fidelity ? " · " + esc(gs.fidelity) : ""}</div>`;
  const ih = R.synthesis_ihp;
  $("ihp").innerHTML = ih.status ? pill("not run") : `<div class="grid"><dl class="kv"><dt>Result</dt><dd>${pill(ih.ok ? "passed" : "failed")} ${ih.stale ? pill("stale: RTL changed") : ""}</dd>
    <dt>Corner</dt><dd>${esc(ih.liberty)}</dd><dt>Cell area</dt><dd>${f(ih.cell_area_um2, 0)} µm²</dd><dt>Sequential</dt><dd>${f(ih.sequential_area_um2, 0)} µm²</dd>
    <dt>Cells / flops</dt><dd>${f(ih.cells, 0)} / ${f(ih.flops, 0)}</dd><dt>Logic delay</dt><dd>${ih.logic_delay_ns} ns (slack ${ih.logic_slack_ns} ns at ${ih.clock_mhz} MHz)</dd>
    <dt>PDK</dt><dd>${esc((ih.pdk_source || "").slice(0, 28))}…</dd></dl>
    <div>${table([["Tiles", (t) => t[0]], ["Area µm²", (t) => f(t[1].area_um2, 0), "n"], ["Cell density", (t) => (t[1].cell_density * 100).toFixed(0) + "%", "n"], ["Fit", (t) => pill(t[1].verdict === "likely" ? "ok" : t[1].verdict)]],
      Object.entries(ih.tiles).filter(([, t]) => t.cell_density < 2))}</div></div>
    <div class="faint">${esc(ih.fidelity)}. Tile outlines: ${esc(ih.tile_source)}.</div>`;
  const H = R.hardening, yn = (v) => pill(v ? "passed" : "failed");
  $("hard").innerHTML = H.status ? pill("not run") : `<dl class="kv"><dt>Run</dt><dd><a href="${esc(H.url)}" target="_blank" rel="noopener">${esc(H.run_id)}</a> ${pill(H.hardened ? "passed" : H.jobs && H.jobs.gds === "success" ? "GDS built" : "failed")} ${H.stale || !H.rtl_matches_repo ? pill("stale: RTL changed") : ""}</dd>
    <dt>Clock</dt><dd>${H.clock_mhz ?? "?"} MHz</dd><dt>Placed and routed</dt><dd>${yn(H.routed)}</dd><dt>Magic DRC / LVS</dt><dd>${yn(H.drc_clear)} ${yn(H.lvs_clear)}</dd>
    <dt>Setup</dt><dd>${H.setup_violation_corners.length ? pill("failed") + " " + esc(H.setup_violation_corners.join(", ")) : pill("passed")}${H.setup_wns_ns !== null && H.setup_wns_ns !== undefined ? ` WNS ${H.setup_wns_ns.toFixed(2)} ns` : ""}</dd>
    <dt>Hold</dt><dd>${H.hold_violation_corners.length ? pill("failed") + " " + esc(H.hold_violation_corners.join(", ")) : pill("passed")}${H.hold_wns_ns != null ? ` WNS ${H.hold_wns_ns.toFixed(3)} ns` : ""}</dd>
    <dt>Cell area</dt><dd>${H.instance_area_um2 != null ? f(H.instance_area_um2, 0) + " µm²" : "?"}</dd><dt>Power</dt><dd>${H.power_w != null ? (H.power_w * 1000).toFixed(2) + " mW" : "?"}</dd>
    <dt>Antenna</dt><dd>${H.antenna_pins ?? "?"} pin / ${H.antenna_nets ?? "?"} net</dd><dt>Utilization</dt><dd>${H.utilization !== null && H.utilization !== undefined ? (H.utilization * 100).toFixed(1) + "%" : "?"}</dd>
    <dt>Jobs</dt><dd>${Object.entries(H.jobs || {}).map(([k, v]) => esc(k) + " " + pill(v === "success" ? "passed" : v)).join(" ")}</dd></dl><div class="faint">${esc(H.fidelity)}</div>`;
  $("hist").innerHTML = table([["When", (x) => esc(x.timestamp)], ["Kind", (x) => esc(x.kind)], ["Result", (x) => esc(x.passed !== undefined ? `${x.passed}/${x.total ?? x.seeds}` : x.killed !== undefined ? `${x.killed}/${x.mutants}` : x.ok !== undefined ? String(x.ok) : "")], ["RTL", (x) => esc(x.rtl_hash)]], O.history.slice().reverse().slice(0, 15));
  // physical
  const F = R.flow;
  $("flow").innerHTML = table([["Step", (s) => esc(s.step)], ["Tool", (s) => esc(s.tool)], ["What", (s) => esc(s.description)], ["Status", (s) => pill(s.status)]], F.steps);
  $("tools").innerHTML = table([["Tool", (t) => esc(t[0])], ["Installed", (t) => pill(t[1].installed ? "ready" : "not installed")]], Object.entries(F.tools.tools)) + `<div class="faint" style="margin-top:6px">${esc(F.tools.install_hint)}</div>`;
  const rb = R.design.register_bits;
  $("rb").innerHTML = `<dl class="kv">${Object.entries(rb.core).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}<dt>per lane × ${rb.lanes}</dt><dd>${Object.values(rb.per_lane).reduce((a, b) => a + b, 0)}</dd><dt>SPI host</dt><dd>${rb.spi_host}</dd><dt><b>total</b></dt><dd><b>${rb.total}</b></dd></dl>`;
  // evaluation
  const met = [["Tile utilization", R.tile_utilization], ["Mapped cell area", R.mapped_cell_area_um2], ["Timing margin", R.timing_margin_ns], ["Max clock", R.fmax_mhz], ["Routing congestion", R.routing_congestion], ["Power", R.power_mw]];
  $("metrics").innerHTML = table([["Metric", (x) => esc(x[0])], ["Value", (x) => x[1].value ?? "–", "n"], ["Status", (x) => pill(x[1].status)], ["Needs", (x) => esc(x[1].needs || "")]], met) +
    `<dl class="kv" style="margin-top:8px"><dt>Instruction memory</dt><dd>${R.memory_utilization.largest_concurrent_program_words}/${R.memory_utilization.imem_words} words (${esc(R.memory_utilization.note)})</dd>
    <dt>Protocol tests</dt><dd>${R.protocol_tests.passed ?? "not run"}/${R.protocol_tests.total ?? "–"}</dd><dt>Verification</dt><dd>${g.status ? "not run" : `${g.passed}/${g.seeds} seeds, ${g.functional.pct}% functional`}</dd></dl>`;
  $("reqs").innerHTML = table([["ID", (q) => q.id], ["Requirement", (q) => esc(q.requirement)], ["Status", (q) => pill(q.status)], ["Evidence", (q) => esc(q.evidence)]], R.requirements);
  $("check").innerHTML = table([["Item", (c) => esc(c.item)], ["Status", (c) => pill(c.status)], ["Evidence", (c) => esc(c.evidence)]], O.checklist);
  $("lims").innerHTML = R.limitations.map((l) => `<li>${esc(l)}</li>`).join("");
}
$("export").onclick = async () => {
  try { const r = await api("/api/asic/export", { name: "jane_street_pem" }); $("expout").textContent = `${r.files.length} files → ${r.dest}`; } catch (e) { $("expout").textContent = e.message; }
};

// ---------------------------------------------------------------- memory studio
const CELLS = ["rom_nor_contact", "rom_nor_active", "rom_nand", "rom_logic", "sram6t", "sram8t", "dram1t1c", "gc2t", "gc3t", "rf_flop", "rf_latch"];
$("m-cell").innerHTML += CELLS.map((c) => `<option>${c}</option>`).join("");
function romData(n, w, mode) {
  const out = [], M = 2 ** w - 1; let s = 12345;
  for (let i = 0; i < n; i++) {
    if (mode === "zeros") out.push(0);
    else if (mode === "count") out.push(i & M);
    else if (mode === "sine") out.push(Math.round((Math.sin(2 * Math.PI * i / n) + 1) / 2 * M));
    else { s = (s * 1103515245 + 12345) % 2147483648; out.push(s % (M + 1)); }
  }
  return out;
}
function memSpec(over = {}) {
  const kind = over.kind || $("m-kind").value, words = +$("m-words").value, width = +$("m-width").value;
  const s = { name: (over.name || kind) + "_m", kind, words, width, pdk: $("m-pdk").value, freq_mhz: +$("m-f").value, objective: $("m-obj").value,
    read_ports: +$("m-r").value, write_ports: +$("m-w").value, rw_ports: +$("m-rw").value, ...over };
  if (kind === "mask_rom") { s.read_ports = Math.max(1, s.read_ports + s.rw_ports); s.rw_ports = 0; s.write_ports = 0; if (words <= 65536) s.contents = romData(words, width, $("m-data").value); }
  if (kind === "fifo") { s.read_ports = 1; s.rw_ports = 0; s.write_ports = 1; }
  if (!over.cell && $("m-cell").value && !over.kind) s.cell = $("m-cell").value;
  return s;
}
$("m-compile").onclick = async () => {
  $("membusy").textContent = "compiling…";
  try {
    const r = await api("/api/memory/compile", { spec: memSpec(), views: true });
    const b = r.best;
    $("m-best").innerHTML = `<dl class="kv"><dt>Cell</dt><dd>${esc(b.cell)}</dd><dt>Circuit</dt><dd>${esc(r.circuit.desc)}</dd><dt>Organization</dt><dd>${b.banks} × ${b.rows} × ${b.cols} (mux ${b.column_mux}, ${esc(b.decoder)})</dd>
      <dt>Area</dt><dd>${f(b.area_um2, 0)} µm² (array ${f((b.array_efficiency || 0) * 100, 0)}%)</dd><dt>Access / cycle</dt><dd>${b.access_ns} / ${b.cycle_ns} ns · fmax ${f(b.fmax_mhz, 0)} MHz</dd>
      <dt>Read energy</dt><dd>${b.read_energy_pj} pJ</dd><dt>Leakage</dt><dd>${b.leakage_uw} µW</dd><dt>Power @ freq</dt><dd>${b.power_mw_at_freq} mW</dd><dt>Throughput</dt><dd>${b.throughput_gbps} Gb/s</dd>
      <dt>Transistors</dt><dd>${f(b.transistors, 0)}</dd>${b.refresh ? `<dt>Refresh</dt><dd>${f(b.refresh.retention_us, 0)} µs retention · ${b.refresh.power_uw} µW · busy ${f(b.refresh.busy_fraction * 100, 3)}%</dd>` : ""}
      <dt>Feasible</dt><dd>${pill(b.feasibility.feasible ? "ok" : "not feasible")}</dd></dl><div class="faint">${esc(b.feasibility.notes.join(" · "))}</div>
      <div class="faint" style="margin-top:6px">Fidelity: ${esc(r.fidelity)} · ±${r.provenance.uncertainty_pct}% · ${r.evaluated} organizations evaluated</div>`;
    $("m-checks").innerHTML = "<h3>PDK checks</h3>" + table([["Check", (c) => esc(c.check)], ["Result", (c) => pill(c.ok === null ? "not run" : c.ok ? "ok" : "failing")], ["Note", (c) => esc(c.note || "")]], r.checks);
    $("m-cands").innerHTML = table([["", (c) => (c.pareto ? "◆" : "")], ["Cell", (c) => esc(c.cell)], ["Banks", (c) => c.banks, "n"], ["Rows×Cols", (c) => `${c.rows}×${c.cols}`, "n"], ["Mux", (c) => c.column_mux, "n"], ["Decoder", (c) => esc(c.decoder)],
      ["Area µm²", (c) => f(c.area_um2, 0), "n"], ["Access ns", (c) => c.access_ns, "n"], ["pJ/read", (c) => c.read_energy_pj, "n"], ["Meets f", (c) => pill(c.meets_freq ? "ok" : "failing")], ["Feasible", (c) => pill(c.feasibility.feasible ? "ok" : "not feasible")]], r.candidates);
    for (const k of ["physical", "circuit", "logical", "block"]) $("v-" + k).innerHTML = r.views[k];
  } catch (e) { $("m-best").textContent = e.message; } finally { $("membusy").textContent = ""; }
};
$("m-rtl").onclick = async () => {
  $("membusy").textContent = "generating RTL and running Verilator…";
  try {
    const s = memSpec(); const dyn = s.kind === "dram" || s.kind === "edram";
    if (s.words > 4096) throw new Error("use ≤ 4096 words for the RTL test");
    const r = await api("/api/memory/rtl", { spec: s, n_ops: 400, retention_cycles: dyn ? 400 : 0, refresh_interval: dyn ? 2 : 0 });
    $("m-extra").innerHTML = r.ran ? `<div class="row">${pill(r.passed ? "passed" : "failed")} <span class="faint">${r.checks} checks, ${r.errors} errors · ${esc(r.fidelity)}${r.retention_error !== null ? " · retention error flag " + r.retention_error : ""}</span></div><pre class="out">${esc(r.rtl.slice(0, 3000))}</pre>` : pill(r.reason || "not run");
  } catch (e) { $("m-extra").textContent = e.message; } finally { $("membusy").textContent = ""; }
};
$("m-cmp").onclick = async () => {
  $("membusy").textContent = "comparing…";
  try {
    const base = { words: +$("m-words").value, width: +$("m-width").value, pdk: $("m-pdk").value, freq_mhz: +$("m-f").value };
    const specs = [["mask_rom", "rom_nor_contact"], ["mask_rom", "rom_nand"], ["mask_rom", "rom_logic"], ["sram", ""], ["edram", ""], ["regfile", ""]].map(([k, c], i) => {
      const s = { ...base, name: `${k}_${c || "auto"}`, kind: k, read_ports: 0, write_ports: 0, rw_ports: 1 };
      if (k === "mask_rom") { s.read_ports = 1; s.rw_ports = 0; s.cell = c; if (s.words <= 65536) s.contents = romData(s.words, s.width, $("m-data").value); }
      return s;
    });
    const rows = await api("/api/memory/compare", { specs });
    $("m-extra").innerHTML = table([["Implementation", (r) => esc(r.name)], ["Cell", (r) => esc(r.cell)], ["Area µm²", (r) => f(r.area_um2, 0), "n"], ["Access ns", (r) => r.access_ns, "n"], ["pJ/read", (r) => r.read_energy_pj, "n"],
      ["mW", (r) => r.power_mw, "n"], ["Gb/s", (r) => r.throughput_gbps, "n"], ["Transistors", (r) => f(r.transistors, 0), "n"], ["Feasible", (r) => pill(r.feasible ? "ok" : "not feasible")]], rows) + `<div class="faint">Analytical estimates (±40%); same function, different implementations.</div>`;
  } catch (e) { $("m-extra").textContent = e.message; } finally { $("membusy").textContent = ""; }
};

// ---------------------------------------------------------------- component foundry
async function loadComponents() {
  const list = await api("/api/components");
  const byCat = {};
  for (const c of list) (byCat[c.category] ||= []).push(c);
  $("c-list").innerHTML = Object.entries(byCat).map(([cat, cs]) => `<optgroup label="${esc(cat)}">${cs.map((c) => `<option value="${esc(c.key)}">${esc(c.name)}${c.composite ? " ▣" : ""}${c.builtin ? "" : " (user)"} — ${esc(c.description).slice(0, 60)}</option>`).join("")}</optgroup>`).join("");
  $("c-list").value = "mac_unit_pipelined@1.0";
  $("c-def").value = JSON.stringify({ name: "rom_bitline_demo", version: "1.0", category: "memory", description: "NOR ROM bitline: N access transistors on one bitline",
    params: { N: { default: 8, unit: "cells", provenance: "assumed" }, W: { default: 0.3, unit: "um", provenance: "assumed" } },
    ports: [{ name: "bl", dir: "inout", domain: "analog" }, { name: "wl0", dir: "inout", domain: "analog" }],
    instances: [{ name: "m0", component: "nmos", params: { W: "W" } }, { name: "sa", component: "sense_amp" }],
    connections: [["bl", "m0.d"], ["wl0", "m0.g"], ["bl", "sa.bl"]],
    models: { area_um2: "N * (W * 0.13 * 6 + 0.2) + gate_um2 * 12" }, rules: [{ expr: "N <= 1024", message: "bitline too long" }], backend: "spice" }, null, 1);
  elab();
}
async function elab() {
  try {
    const r = await api("/api/components/elaborate", { ref: $("c-list").value, params: JSON.parse($("c-params").value || "{}"), pdk: $("c-pdk").value });
    const ru = r.rollup;
    $("c-info").innerHTML = `<dl class="kv"><dt>Component</dt><dd>${esc(r.component)}</dd><dt>Area</dt><dd>${f(ru.area_um2)} µm²</dd><dt>Transistors</dt><dd>${f(ru.transistors, 0)}</dd><dt>Power</dt><dd>${f(ru.power_uw)} µW</dd><dt>Delay</dt><dd>${f(ru.delay_ns, 3)} ns</dd>
      <dt>Ports</dt><dd>${r.ports.map((p) => `${esc(p.name)}[${p.width}] ${p.dir}${p.domain !== "digital" ? " " + p.domain : ""}`).join(", ")}</dd></dl>
      ${r.errors.map((e) => `<div class="pill bad">${esc(e)}</div>`).join("")}${r.warnings.map((e) => `<div class="faint">⚠ ${esc(e)}</div>`).join("")}<div class="faint">Roll-ups: ${esc(r.fidelity.rollup)} · simulation: ${esc(r.fidelity.simulation)}</div>`;
    $("c-svg").innerHTML = r.svg;
  } catch (e) { $("c-info").textContent = e.message; $("c-svg").innerHTML = ""; }
}
$("c-list").onchange = elab;
$("c-elab").onclick = elab;
$("c-pdk").onchange = elab;
$("c-sim").onclick = async () => {
  try { const r = await api("/api/components/simulate", { ref: $("c-list").value, params: JSON.parse($("c-params").value || "{}"), stimulus: JSON.parse($("c-stim").value) });
    $("c-out").textContent = r.ok ? `${r.fidelity}\n` + r.trace.map((t, i) => `cycle ${i}: ${JSON.stringify(t)}`).join("\n") : r.errors.join("\n");
  } catch (e) { $("c-out").textContent = e.message; }
};
$("c-v").onclick = async () => {
  try { const r = await api("/api/components/verilog", { ref: $("c-list").value, params: JSON.parse($("c-params").value || "{}") }); $("c-out").textContent = r.verilog; } catch (e) { $("c-out").textContent = e.message; }
};
$("c-create").onclick = async () => {
  try { const r = await api("/api/components/create", { definition: JSON.parse($("c-def").value) }); $("c-cres").textContent = `registered ${r.created}`; await loadComponents(); $("c-list").value = r.created; elab(); }
  catch (e) { $("c-cres").textContent = e.message; }
};

load().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="pill bad">${esc(e.message)}</div>`); });
