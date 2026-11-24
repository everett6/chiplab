// ChipLab GPU Studio: external-tool views (Circuit, Board, Layout, CFD).
// Each view shows output from the real tool: ngspice waveforms and Xschem schematics,
// KiCad's own SVG plot / 3D render / DRC, KLayout's layout renderer, OpenFOAM results.
// Tools that are not installed get a clear note instead of a view.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (v, d = 2) => (v === null || v === undefined || Number.isNaN(+v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
const dark = matchMedia("(prefers-color-scheme: dark)").matches;

const VIEWS = [
  ["circuit", "Circuit Studio", "ngspice + Xschem", ["ngspice"]],
  ["board", "Board Studio", "KiCad PCB", ["kicad"]],
  ["layout", "Layout Studio", "KLayout GDS/OASIS", ["klayout"]],
  ["cfd", "CFD", "OpenFOAM heatsink", ["openfoam"]],
];
const T = { status: null, view: null, templates: null, tpl: "dram_sense_amp", params: {}, run: null, layout: null, layers: null };
const pane = $("toolpane"), side = $("toolside");
const design = () => $("design").value;
const workload = () => $("workload").value || "llm8_decode";

async function api(url, body) {
  const r = await fetch(url, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) { const e = new Error(j.error || r.statusText); e.missing = j.missing_tool; throw e; }
  return j;
}
function kv(rows) { return `<dl class="kv">${rows.filter(Boolean).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`; }
function missingNote(tool, what) {
  const s = T.status?.[tool] || {};
  return `<div class="missing"><b>${esc(tool)} not installed.</b> ${esc(what)} ${s.note ? `<br><span style="color:var(--muted)">${esc(s.note)}</span>` : ""}
    <br><span style="color:var(--muted)">ChipLab only drives the real tool; it never imitates its output. Installing it needs your OK (no-sudo local install under ~/.local/opt works).</span></div>`;
}
let plotlyP = null;
function plotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  plotlyP ||= new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    s.onload = () => res(window.Plotly); s.onerror = () => rej(new Error("could not load Plotly"));
    document.head.appendChild(s);
  });
  return plotlyP;
}

// ---------------------------------------------------------------- rail + switching
function renderRail() {
  $("toolmodes").innerHTML = VIEWS.map(([k, t, s, tools]) => {
    const ok = T.status ? tools.every((x) => T.status[x]?.available) : true;
    const ver = T.status && ok ? tools.map((x) => T.status[x].version).filter(Boolean).join(" ") : "";
    return `<button data-v="${k}" class="${T.view === k ? "on" : ""} ${ok ? "" : "na"}">${t}<small>${ok ? esc(s + (ver ? " · " + ver : "")) : "not installed"}</small></button>`;
  }).join("");
}
function open(view) {
  T.view = view;
  pane.hidden = false; side.hidden = false;
  for (const id of ["inspector", "mps", "summary"]) { const el = $(id); if (el) el.style.display = "none"; }
  for (const b of $("modes").children) b.classList.remove("on");
  renderRail();
  ({ circuit: circuitView, board: boardView, layout: layoutView, cfd: cfdView })[view]();
}
function close() {
  if (!T.view) return;
  T.view = null; pane.hidden = true; side.hidden = true;
  for (const id of ["inspector", "mps", "summary"]) { const el = $(id); if (el) el.style.display = ""; }
  renderRail();
}
$("toolmodes").onclick = (e) => { const b = e.target.closest("button"); if (b) open(b.dataset.v); };
$("modes").addEventListener("click", () => close(), true);
$("design").addEventListener("change", () => { if (T.view && T.view !== "circuit" && T.view !== "layout") open(T.view); });
$("workload").addEventListener("change", () => { if (T.view === "board" || T.view === "cfd") open(T.view); });

// ---------------------------------------------------------------- Circuit Studio
async function circuitView() {
  T.templates ||= await api("/api/tools/circuit/templates");
  const tpl = T.templates.find((t) => t.key === T.tpl) || T.templates[0];
  T.tpl = tpl.key;
  T.params[tpl.key] ||= Object.fromEntries(tpl.params.map((p) => [p.name, p.default]));
  const have = T.status?.ngspice?.available;
  pane.innerHTML = `<h2>Circuit Studio</h2>
    <div class="sub">SPICE netlists from fixed templates, simulated by ngspice ${esc(T.status?.ngspice?.version || "")} in batch mode; schematic ${T.status?.xschem?.available ? "exported by Xschem " + esc(T.status.xschem.version || "") : "generated (Xschem not installed)"}. Node labels light up with the simulated voltage at the cursor time.</div>
    <div class="tabs" id="ctabs">${T.templates.map((t) => `<button data-t="${t.key}" class="${t.key === tpl.key ? "on" : ""}">${esc(t.title)}</button>`).join("")}</div>
    ${have ? "" : missingNote("ngspice", "Waveforms need ngspice; the schematic and netlist below are still generated.")}
    <div class="tgrid">
      <div class="tbox"><h4 id="schem-h">Schematic</h4><div class="schem" id="schem"><div class="empty">Generating…</div></div>
        <div class="scrub" id="scrub" hidden><span>t</span><input type="range" id="tcur" min="0" max="1000" value="1000"><span id="tval">–</span></div></div>
      <div class="tbox"><h4>Node voltages (V)</h4><div class="plot" id="vplot"></div><h4 style="margin-top:8px">Currents <span style="text-transform:none">(µA)</span></h4><div class="plot" id="iplot"></div></div>
    </div>`;
  $("ctabs").onclick = (e) => { const b = e.target.closest("button"); if (b) { T.tpl = b.dataset.t; T.run = null; circuitView(); } };
  side.innerHTML = `<h2>${esc(tpl.title)} <span class="badge">${have ? "ngspice" : "no simulator"}</span></h2>
    <div class="empty" style="margin-bottom:6px">${esc(tpl.description)}</div>
    <div class="form" id="cform">${tpl.params.map((p) => p.bit
      ? `<label for="cp-${p.name}">${esc(p.label)}</label><input type="checkbox" id="cp-${p.name}" data-p="${p.name}" ${T.params[tpl.key][p.name] ? "checked" : ""}>`
      : `<label for="cp-${p.name}" title="${p.min}–${p.max} ${esc(p.unit)}">${esc(p.label)} <span style="color:var(--faint)">${esc(p.unit)}</span></label><input id="cp-${p.name}" data-p="${p.name}" type="number" step="any" min="${p.min}" max="${p.max}" value="${T.params[tpl.key][p.name]}">`).join("")}</div>
    <div class="row"><button class="btn primary" id="crun">${have ? "Run ngspice" : "Generate netlist"}</button></div>
    <div class="msg" id="cmsg"></div><div id="cmetrics"></div>
    <details style="margin-top:8px"><summary>SPICE netlist</summary><pre class="code" id="cnet"></pre></details>
    <details><summary>ngspice log</summary><pre class="code" id="clog"></pre></details>`;
  $("cform").onchange = (e) => {
    const k = e.target.dataset.p; if (!k) return;
    T.params[tpl.key][k] = e.target.type === "checkbox" ? (e.target.checked ? 1 : 0) : +e.target.value;
  };
  $("crun").onclick = () => runCircuit();
  if (T.run && T.run.template === tpl.key) drawCircuit(T.run); else runCircuit();
}

async function runCircuit() {
  const msg = $("cmsg"); msg.className = "msg"; msg.textContent = "Running…";
  $("crun").disabled = true;
  try {
    T.run = await api("/api/tools/circuit/run", { template: T.tpl, params: T.params[T.tpl] });
    msg.className = "msg ok";
    msg.textContent = T.run.simulator ? `ngspice ${T.run.simulator.version || ""}: ${T.run.simulator.points} time points` : "Netlist generated (ngspice not installed).";
    drawCircuit(T.run);
  } catch (e) { msg.className = "msg err"; msg.textContent = e.message; }
  $("crun").disabled = false;
}

function ramp(t) {   // 0..1 -> blue..red, same feel as the 3D views
  const stops = [[0, [56, 108, 230]], [0.5, [46, 190, 120]], [0.75, [240, 190, 40]], [1, [225, 50, 50]]];
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], u = (t - a) / (b - a);
    return `rgb(${ca.map((c, k) => Math.round(c + (cb[k] - c) * u)).join(",")})`;
  }
  return "rgb(225,50,50)";
}

async function drawCircuit(r) {
  const sch = r.schematic;
  $("schem-h").textContent = `Schematic · ${sch.source === "xschem" ? "Xschem SVG export" : "generated"}`;
  $("schem").innerHTML = (sch.note ? `<div class="warn">${esc(sch.note)}</div>` : "") + sch.svg;
  $("cnet").textContent = r.netlist || "";
  $("clog").textContent = (r.log || []).join("\n");
  const m = r.metrics || {};
  $("cmetrics").innerHTML = r.simulator ? `<h3>Result (from the waveforms)</h3>` + kv(Object.entries(m).map(([k, v]) => [k.replace(/_/g, " "), typeof v === "boolean" ? (v ? '<span style="color:var(--ok)">yes</span>' : '<span style="color:var(--bad)">NO</span>') : esc(f(v, 3))])) : "";
  if (!r.simulator) { $("vplot").innerHTML = $("iplot").innerHTML = '<div class="empty">No waveforms without ngspice.</div>'; return; }
  const P = await plotly().catch((e) => { $("vplot").textContent = e.message; return null; });
  if (!P) return;
  const t = r.time_ns, css = getComputedStyle(document.documentElement);
  const fg = css.getPropertyValue("--muted").trim(), grid = css.getPropertyValue("--line").trim();
  const lay = (ytitle) => ({
    margin: { l: 48, r: 10, t: 6, b: 34 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg, size: 11 }, xaxis: { title: "time (ns)", gridcolor: grid, zeroline: false },
    yaxis: { title: ytitle, gridcolor: grid, zeroline: false }, legend: { orientation: "h", y: 1.12 },
    shapes: [...(r.events || []).map(([, te]) => ({ type: "line", x0: te, x1: te, yref: "paper", y0: 0, y1: 1, line: { color: grid, dash: "dot", width: 1 } })),
      { type: "line", x0: t[t.length - 1], x1: t[t.length - 1], yref: "paper", y0: 0, y1: 1, line: { color: "#e05252", width: 1.5 } }],
    annotations: (r.events || []).map(([name, te]) => ({ x: te, y: 1, yref: "paper", text: name, showarrow: false, font: { size: 9, color: fg }, xanchor: "left", yanchor: "top", textangle: -90 })),
  });
  const prim = new Set(r.primary || []);
  const vtr = Object.entries(r.voltages).map(([n, y]) => ({ x: t, y, name: n, mode: "lines", line: { width: prim.has(n) ? 2 : 1 }, visible: prim.has(n) ? true : "legendonly" }));
  const itr = Object.entries(r.currents_ua).map(([n, y]) => ({ x: t, y, name: n.replace(/^i\(@?/, "").replace(/\[id\]\)$|\)$/, "").toUpperCase() + (n.includes("[id]") ? " Id" : ""), mode: "lines", line: { width: 1.5 } }));
  const cfg = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  await P.react("vplot", vtr, lay("V"), cfg);
  await P.react("iplot", itr, lay("µA"), cfg);
  // time cursor -> node colors on the schematic + cursor line on the plots
  const vmax = Math.max(r.params.vdd + (r.params.wl_boost || r.params.wwl_boost || 0), 0.1);
  const scrub = $("scrub"); scrub.hidden = false;
  const act = r.activity_v || {};
  const labels = [...$("schem").querySelectorAll("[data-net]")];
  const halo = r.schematic.source === "xschem" || dark ? "#000" : "#fff";   // Xschem draws on black
  const apply = (k) => {
    const i = Math.round((k / 1000) * (t.length - 1));
    $("tval").textContent = `${f(t[i], 2)} ns`;
    for (const el of labels) {
      const n = el.dataset.net, y = r.voltages[n];
      if (!y) continue;
      const c = ramp(y[i] / vmax), active = (act[n] || 0) > 0.1 * r.params.vdd;
      if (el.tagName === "text") {
        el.style.fill = c; el.style.fontWeight = active ? "700" : "400";
        el.style.paintOrder = "stroke"; el.style.stroke = active ? halo : "none"; el.style.strokeWidth = active ? "3px" : "0";
      } else el.style.stroke = c;
      el.setAttribute("data-v", `${n} = ${f(y[i], 3)} V${active ? " (active)" : ""}`);
    }
    const sh = (id) => { const g = $(id); if (!g?.layout) return; const s = g.layout.shapes.slice(); s[s.length - 1] = { ...s[s.length - 1], x0: t[i], x1: t[i] }; P.relayout(g, { shapes: s }); };
    sh("vplot"); sh("iplot");
  };
  for (const el of labels) {
    el.addEventListener("mouseenter", () => { el.querySelector("title")?.remove(); const ti = document.createElementNS("http://www.w3.org/2000/svg", "title"); ti.textContent = el.getAttribute("data-v") || el.dataset.net; el.appendChild(ti); });
    el.addEventListener("click", () => {
      const idx = vtr.findIndex((tr) => tr.name === el.dataset.net);
      if (idx >= 0) P.restyle("vplot", { visible: true, "line.width": 3.5 }, [idx]);
    });
  }
  let pend = null;
  $("tcur").oninput = (e) => { const k = +e.target.value; if (pend) cancelAnimationFrame(pend); pend = requestAnimationFrame(() => apply(k)); };
  $("tcur").value = 1000; apply(1000);
}

// ---------------------------------------------------------------- Board Studio
async function boardView() {
  const have = T.status?.kicad?.available;
  const dl = `/api/tools/board/kicad_pcb?design=${encodeURIComponent(design())}&workload=${encodeURIComponent(workload())}`;
  pane.innerHTML = `<h2>Board Studio</h2>
    <div class="sub">The studio board (VRM phases, PCIe lanes, 12 V connectors, GPU package BGA) exported as a 4-layer <code>.kicad_pcb</code>${have ? `, plotted, rendered and DRC-checked by KiCad ${esc(T.status.kicad.version || "")} (kicad-cli)` : ""}.</div>
    ${have ? "" : missingNote("kicad", "The .kicad_pcb export still works; KiCad's own renders and DRC need kicad-cli.")}
    <div class="tabs" id="btabs"><button data-b="3d" class="on">KiCad 3D render</button><button data-b="svg">KiCad copper plot (SVG)</button></div>
    <div class="tbox" id="bbody"><div class="empty">${have ? "Asking KiCad to render…" : "No KiCad output."}</div></div>`;
  side.innerHTML = `<h2>Board <span class="badge">${have ? "kicad-cli" : "export only"}</span></h2>
    <div class="row"><a class="btn" href="${dl}" download>Download .kicad_pcb</a><button class="btn" id="brun" ${have ? "" : "disabled"}>Re-render</button></div>
    <div class="msg" id="bmsg"></div><div id="bsum"></div>`;
  $("brun").onclick = () => boardView();
  try {
    const r = await api("/api/tools/board", { design: design(), workload: workload() });
    const s = r.summary;
    $("bsum").innerHTML = kv([["board", `${f(s.board_mm[0], 1)} × ${f(s.board_mm[1], 1)} × ${s.board_mm[2]} mm`], ["copper layers", s.layers],
      ...Object.entries(s.parts).map(([k, n]) => [k.replace(/_/g, " "), n]), ["traces", s.traces], ["trace length", `${f(s.trace_length_mm, 0)} mm`],
      ["VRM", `${s.vrm.phases} phases · ${f(s.vrm.amps, 0)} A · ${f(s.vrm.loss_w, 1)} W loss`]])
      + (r.drc ? `<h3>KiCad DRC (as reported)</h3>` + kv([["violations", r.drc.violations], ["unconnected items", r.drc.unconnected], ...Object.entries(r.drc.by_type).sort((a, b) => b[1] - a[1]).map(([k, n]) => [k.replace(/_/g, " "), n])])
        + `<div class="empty">Coarse automatic placement, not a routed board: KiCad flags the gaps a layout engineer would close.</div>` : "");
    if (r.errors?.length) $("bmsg").innerHTML = r.errors.map((x) => `<div class="warn">${esc(x)}</div>`).join("");
    const show = (k) => {
      for (const b of $("btabs").children) b.classList.toggle("on", b.dataset.b === k);
      if (k === "3d") $("bbody").innerHTML = r.png_b64 ? `<img class="toolimg" alt="KiCad 3D render" src="data:image/png;base64,${r.png_b64}">` : '<div class="empty">No 3D render.</div>';
      else $("bbody").innerHTML = r.svg ? `<img class="toolimg" alt="KiCad SVG plot" src="data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(r.svg)))}">` : '<div class="empty">No SVG plot.</div>';
    };
    $("btabs").onclick = (e) => { const b = e.target.closest("button"); if (b) show(b.dataset.b); };
    if (have) show("3d");
  } catch (e) { $("bmsg").className = "msg err"; $("bmsg").textContent = e.message; }
}

// ---------------------------------------------------------------- Layout Studio
function layoutView() {
  const have = T.status?.klayout?.available;
  pane.innerHTML = `<h2>Layout Studio</h2>
    <div class="sub">Read-only GDSII / OASIS viewing through KLayout ${esc(T.status?.klayout?.version || "")}'s own renderer. Files are stored under a content hash and never modified.</div>
    ${have ? "" : missingNote("klayout", "Layout viewing needs the klayout Python package.")}
    <div class="tbox" id="lbody"><div class="empty">Open a .gds / .oas file, or export this design's package floorplan to GDS.</div></div>`;
  side.innerHTML = `<h2>Layout <span class="badge">${have ? "KLayout" : "unavailable"}</span></h2>
    <div class="row"><input type="file" id="lfile" accept=".gds,.gds2,.gdsii,.oas,.oasis" ${have ? "" : "disabled"}></div>
    <div class="row"><button class="btn" id="lfp" ${have ? "" : "disabled"}>Floorplan of this design → GDS</button></div>
    <div class="msg" id="lmsg"></div><div id="linfo"></div><div id="llayers"></div>`;
  $("lfile").onchange = async (e) => {
    const file = e.target.files[0]; if (!file) return;
    const msg = $("lmsg"); msg.className = "msg"; msg.textContent = `Uploading ${file.name}…`;
    try {
      const r = await fetch("/api/tools/layout/upload", { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": file.name }, body: file });
      const j = await r.json(); if (!r.ok) throw new Error(j.error);
      T.layout = j; T.layers = null; msg.textContent = ""; showLayout();
    } catch (err) { msg.className = "msg err"; msg.textContent = err.message; }
  };
  $("lfp").onclick = async () => {
    const msg = $("lmsg"); msg.className = "msg"; msg.textContent = "Writing GDS with klayout.db…";
    try { T.layout = await api("/api/tools/layout/floorplan", { design: design(), workload: workload() }); T.layers = null; msg.textContent = ""; showLayout(); }
    catch (err) { msg.className = "msg err"; msg.textContent = err.message; }
  };
  if (T.layout) showLayout();
}
async function showLayout() {
  const L = T.layout;
  T.layers ||= new Set(L.layers.map((l) => `${l.layer}/${l.datatype}`));
  $("linfo").innerHTML = kv([["format", L.format], ["top cell", esc(L.top_cells.join(", "))], ["cells", L.cells], ["dbu", `${L.dbu_um} µm`],
    L.bbox_um && ["extent", `${f(L.bbox_um[2] - L.bbox_um[0], 2)} × ${f(L.bbox_um[3] - L.bbox_um[1], 2)} µm`]]);
  $("llayers").innerHTML = `<h3>Layers</h3>` + L.layers.map((l) => { const k = `${l.layer}/${l.datatype}`; return `<label class="ck"><input type="checkbox" data-l="${k}" ${T.layers.has(k) ? "checked" : ""}> ${k}${l.name ? " " + esc(l.name) : ""} <span style="color:var(--faint)">${l.shapes}${l.capped ? "+" : ""} shapes</span></label>`; }).join("");
  $("llayers").onchange = (e) => { const k = e.target.dataset.l; if (!k) return; e.target.checked ? T.layers.add(k) : T.layers.delete(k); renderLayout(); };
  renderLayout();
}
async function renderLayout() {
  const box = $("lbody"); box.style.opacity = 0.6;
  try {
    const w = Math.min(2000, Math.max(600, Math.round(box.clientWidth * devicePixelRatio)));
    const r = await api("/api/tools/layout/render", { id: T.layout.id, layers: [...T.layers], width: w, height: Math.round(w * 0.62) });
    box.innerHTML = `<img class="toolimg" alt="KLayout render" src="data:image/png;base64,${r.png_b64}">`;
  } catch (e) { box.innerHTML = `<div class="msg err">${esc(e.message)}</div>`; }
  box.style.opacity = 1;
}

// ---------------------------------------------------------------- CFD
async function cfdView() {
  const have = T.status?.openfoam?.available;
  pane.innerHTML = `<h2>CFD · heatsink fin channel</h2>
    <div class="sub">A coarse OpenFOAM case (simpleFoam laminar flow, then scalarTransportFoam air temperature) for one fin-to-fin channel of this design's heatsink, compared against the studio airflow model's heat-transfer coefficient.</div>
    ${have ? "" : missingNote("openfoam", "The case files are generated below, but nothing is solved and no CFD numbers are shown until OpenFOAM is installed.")}
    <div class="tgrid"><div class="tbox"><h4>Airflow model (studio/thermal.py)</h4><div id="cfdm"></div></div><div class="tbox"><h4>OpenFOAM</h4><div id="cfdr"><div class="empty">${have ? "Press Run OpenFOAM." : "Not available."}</div></div></div></div>
    <details style="margin-top:10px"><summary>Generated OpenFOAM case files</summary><div id="cfdf"></div></details>`;
  side.innerHTML = `<h2>CFD <span class="badge">${have ? "OpenFOAM" : "not installed"}</span></h2>
    <div class="row"><button class="btn primary" id="cfdrun" ${have ? "" : "disabled"}>Run OpenFOAM</button></div><div class="msg" id="cfdmsg"></div><div id="cfdch"></div>`;
  const go = async (run) => {
    const msg = $("cfdmsg"); if (run) { msg.className = "msg"; msg.textContent = "Solving (can take minutes)…"; }
    try {
      const r = await api("/api/tools/cfd", { design: design(), workload: workload(), run });
      const a = r.airflow_model, c = r.channel;
      $("cfdm").innerHTML = kv([["flow", `${f(a.flow_cfm, 0)} CFM`], ["channel velocity", `${f(a.velocity_ms, 2)} m/s`], ["Reynolds (length)", f(a.reynolds, 0)], ["h", `${f(a.h_w_m2k, 1)} W/m²K`], ["fin efficiency", f(a.fin_efficiency, 3)], ["R heatsink", `${f(a.r_heatsink_c_per_w, 4)} °C/W`], ["air rise", `${f(a.air_rise_c, 2)} °C`]]);
      $("cfdch").innerHTML = `<h3>Channel</h3>` + kv([["gap × height × length", `${f(c.gap_mm, 2)} × ${f(c.height_mm, 1)} × ${f(c.length_mm, 1)} mm`], ["inlet", `${f(c.velocity_ms, 2)} m/s at ${f(c.ambient_c, 1)} °C`], ["fin wall", `${f(c.wall_c, 1)} °C`]]);
      $("cfdf").innerHTML = Object.entries(r.files).map(([k, v]) => `<h3>${esc(k)}</h3><pre class="code">${esc(v)}</pre>`).join("");
      if (r.result) {
        const x = r.result;
        $("cfdr").innerHTML = kv([["outlet mean T", `${f(x.outlet_c, 2)} °C`], ["h (CFD)", `${f(x.h_cfd_w_m2k, 1)} W/m²K`], ["h (model)", `${f(x.h_model_w_m2k, 1)} W/m²K`], ["CFD / model", f(x.ratio_cfd_over_model, 3)]]);
        msg.className = "msg ok"; msg.textContent = "OpenFOAM finished.";
      }
    } catch (e) { msg.className = "msg err"; msg.textContent = e.message; }
  };
  $("cfdrun").onclick = () => go(true);
  go(false);
}

// ---------------------------------------------------------------- init
// The tool views need no WebGL: if the 3D studio could not start (WebGL disabled), fill the
// design list here so Board / Layout / CFD still work.
setTimeout(async () => {
  if ($("design").options.length) return;
  try {
    const j = await api("/api/designs");
    const list = Array.isArray(j) ? j : j.designs || [];
    $("design").innerHTML = list.map((d) => `<option value="${esc(d.design_id)}">${esc(d.name)} · v${esc(d.version)}</option>`).join("");
    const good = list.find((d) => /hbm|llm|accelerator/i.test(d.name));
    if (good) $("design").value = good.design_id;
    if (!$("workload").options.length) $("workload").innerHTML = '<option value="llm8_decode">llm8 decode</option>';
  } catch { /* the 3D studio reports load errors */ }
}, 1500);
renderRail();
api("/api/tools/status").then((s) => { T.status = s; renderRail(); if (T.view) open(T.view); }).catch(() => {});
