// ChipLab GPU Studio: interactive 3D package view driven by ChipLab's design + simulators.
// Every color, animation speed and highlight comes from the scene JSON
// (/api/studio/scene): analytical power, the package thermal/airflow solve and the
// discrete-event simulation's per-resource utilization. Nothing is decorative.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 }));
const dark = matchMedia("(prefers-color-scheme: dark)").matches;

const MODES = [
  ["assembly", "3D Assembly", "package, board, cooler"],
  ["die", "Die Explorer", "inside the selected die"],
  ["memory", "Memory Explorer", "banks, channels, stacks"],
  ["thermal", "Thermal", "temperature, hotspots, transient"],
  ["power", "Power", "watts per component"],
  ["interconnect", "Interconnect", "data movement, link load"],
  ["airflow", "Airflow", "fans, fin channels, air temp"],
];
const LAYERS = [
  ["cooling", "Cooler (fans + heatsink)"], ["ihs", "Heat spreader"], ["dies", "Dies & stacks"],
  ["interposer", "Interposer & routes"], ["substrate", "Package substrate"], ["pcb", "PCB & board parts"],
];
const DEFAULT_LAYERS = {
  assembly: { cooling: true, ihs: true, dies: true, interposer: true, substrate: true, pcb: true },
  die: { cooling: false, ihs: false, dies: true, interposer: true, substrate: false, pcb: false },
  memory: { cooling: false, ihs: false, dies: true, interposer: true, substrate: true, pcb: false },
  thermal: { cooling: false, ihs: false, dies: true, interposer: true, substrate: true, pcb: true },
  power: { cooling: false, ihs: false, dies: true, interposer: true, substrate: true, pcb: true },
  interconnect: { cooling: false, ihs: false, dies: true, interposer: true, substrate: true, pcb: false },
  airflow: { cooling: true, ihs: true, dies: true, interposer: true, substrate: true, pcb: true },
};

const S = {
  data: null, mode: "assembly", layers: { ...DEFAULT_LAYERS.assembly }, explode: 0, vscale: 6,
  wire: false, selected: null, move: false, playing: false, frame: -1, designs: [],
  lastEventId: 0, busy: false,
};

// ---------------------------------------------------------------- three.js setup
const vp = $("vp");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (e) {
  $("loading").textContent = "GPU Studio needs WebGL, which this browser has disabled. Open ChipLab in Chrome, Edge, Firefox or Safari with hardware acceleration on; the live feed is at /live.";
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
vp.prepend(renderer.domElement);
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--viewport").trim() || "#dfe4ea");
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 5000);
camera.position.set(140, 160, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.screenSpacePanning = true;
scene3.add(new THREE.HemisphereLight(0xffffff, dark ? 0x202530 : 0x8090a0, dark ? 1.1 : 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(120, 260, 140);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -260, right: 260, top: 260, bottom: -260, near: 10, far: 900 });
scene3.add(sun);
const fill = new THREE.DirectionalLight(0x9fb8ff, 0.6);
fill.position.set(-200, 120, -160);
scene3.add(fill);
let world = new THREE.Group();
scene3.add(world);

function resize() {
  const r = vp.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(vp);

// ---------------------------------------------------------------- color scales
const RAMP = [[0, [49, 54, 149]], [0.25, [69, 117, 180]], [0.45, [116, 173, 209]], [0.6, [254, 224, 144]], [0.78, [244, 109, 67]], [1, [165, 0, 38]]];
function ramp(t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, a] = RAMP[i - 1], [t1, b] = RAMP[i], k = (t - t0) / (t1 - t0);
      return a.map((v, j) => Math.round(v + (b[j] - v) * k));
    }
  }
  return RAMP[RAMP.length - 1][1];
}
const rampCss = (t) => `rgb(${ramp(t).join(",")})`;
const rampColor = (t) => new THREE.Color(rampCss(t));
const UTIL = [[0, [40, 48, 60]], [0.5, [20, 150, 160]], [1, [250, 210, 60]]];
function utilCss(t) {
  t = Math.min(1, Math.max(0, t));
  const i = t <= 0.5 ? 1 : 2, [t0, a] = UTIL[i - 1], [t1, b] = UTIL[i], k = (t - t0) / (t1 - t0);
  return `rgb(${a.map((v, j) => Math.round(v + (b[j] - v) * k)).join(",")})`;
}
const gradientCss = (fn) => `linear-gradient(90deg, ${[0, .2, .4, .6, .8, 1].map((t) => fn(t)).join(",")})`;

// ---------------------------------------------------------------- materials
const MAT = {
  pcb: () => new THREE.MeshStandardMaterial({ color: 0x0d4a2e, roughness: 0.75 }),
  copper: () => new THREE.MeshStandardMaterial({ color: 0xc27a3e, metalness: 0.85, roughness: 0.35 }),
  gold: () => new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.9, roughness: 0.25 }),
  substrate: () => new THREE.MeshStandardMaterial({ color: 0x3a3226, roughness: 0.7 }),
  interposer: () => new THREE.MeshStandardMaterial({ color: 0x6f7784, metalness: 0.55, roughness: 0.32 }),
  die: () => new THREE.MeshStandardMaterial({ color: 0x252b36, metalness: 0.5, roughness: 0.38 }),
  dram: () => new THREE.MeshStandardMaterial({ color: 0x343a46, metalness: 0.45, roughness: 0.4 }),
  base: () => new THREE.MeshStandardMaterial({ color: 0x4a515e, metalness: 0.45, roughness: 0.4 }),
  ihs: () => new THREE.MeshStandardMaterial({ color: 0xc9cdd2, metalness: 0.92, roughness: 0.28 }),
  alu: () => new THREE.MeshStandardMaterial({ color: 0xd5d9de, metalness: 0.75, roughness: 0.35 }),
  fan: () => new THREE.MeshStandardMaterial({ color: 0x1d2128, roughness: 0.6 }),
  part: () => new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.55 }),
  inductor: () => new THREE.MeshStandardMaterial({ color: 0x6b7078, metalness: 0.5, roughness: 0.45 }),
  cap: () => new THREE.MeshStandardMaterial({ color: 0x9aa3ad, metalness: 0.7, roughness: 0.3 }),
  bridge: () => new THREE.MeshStandardMaterial({ color: 0x8c96a6, metalness: 0.6, roughness: 0.3 }),
};
const pickables = [];
const animators = [];

function box(w, h, d, mat, pick) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.castShadow = m.receiveShadow = true;
  if (pick) { m.userData.pick = pick; pickables.push(m); }
  m.userData.baseColor = mat.color.clone();
  return m;
}

// ---------------------------------------------------------------- floorplan textures
function neighborsSide(c, comps) {
  // which edges of a compute die face memory (where its PHYs go)
  const sides = new Set();
  for (const o of comps) {
    if (o === c || !["hbm_stack", "memory_chiplet", "dram_chiplet"].includes(o.kind)) continue;
    const dx = o.x - c.x, dy = o.y - c.y;
    if (Math.abs(dx) > Math.abs(dy)) sides.add(dx < 0 ? "W" : "E"); else sides.add(dy < 0 ? "S" : "N");
  }
  return sides;
}

function floorplanCanvas(c, mode) {
  const W = 512, H = Math.max(64, Math.round(512 * c.d / c.w));
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  const it = c.internal || {};
  g.fillStyle = "#1b2029"; g.fillRect(0, 0, W, H);
  if (it.kind === "compute") {
    const sides = neighborsSide(c, S.data.components);
    const phy = 22;
    const x0 = sides.has("W") ? phy : 6, x1 = W - (sides.has("E") ? phy : 6);
    const y0 = sides.has("S") ? phy : 6, y1 = H - (sides.has("N") ? phy : 6);
    g.fillStyle = "#7a5c2e";
    if (sides.has("W")) g.fillRect(2, 6, phy - 6, H - 12);
    if (sides.has("E")) g.fillRect(W - phy + 4, 6, phy - 6, H - 12);
    if (sides.has("S")) g.fillRect(6, 2, W - 12, phy - 6);
    if (sides.has("N")) g.fillRect(6, H - phy + 4, W - 12, phy - 6);
    const sramH = (y1 - y0) * it.sram_fraction;
    const ym = (y0 + y1) / 2;
    g.fillStyle = "#3a4f7a"; g.fillRect(x0, ym - sramH / 2, x1 - x0, sramH);
    g.strokeStyle = "#5872a8"; g.lineWidth = 1;
    for (let x = x0; x < x1; x += 10) { g.beginPath(); g.moveTo(x, ym - sramH / 2); g.lineTo(x, ym + sramH / 2); g.stroke(); }
    const n = it.clusters, aspect = (x1 - x0) / Math.max(1, (y1 - y0 - sramH));
    const cols = Math.max(2, Math.round(Math.sqrt(n * aspect))), rows = Math.ceil(n / cols);
    const rowsTop = Math.ceil(rows / 2), rowsBot = rows - rowsTop;
    const cw = (x1 - x0) / cols;
    const u = mode === "power" || mode === "memory" ? null : it.utilization;
    const draw = (r, yA, hh) => {
      for (let q = 0; q < cols && r * cols + q < n; q++) {
        const x = x0 + q * cw;
        g.fillStyle = u === null || u === undefined || mode === "assembly" || mode === "die" ? "#2e6b5e" : utilCss(u);
        g.fillRect(x + 2, yA + 2, cw - 4, hh - 4);
        g.strokeStyle = "rgba(255,255,255,.14)";
        for (let k = 1; k < 4; k++) { g.beginPath(); g.moveTo(x + 2 + k * (cw - 4) / 4, yA + 2); g.lineTo(x + 2 + k * (cw - 4) / 4, yA + hh - 2); g.stroke(); }
      }
    };
    const hTop = (ym - sramH / 2 - y0) / Math.max(rowsTop, 1), hBot = (y1 - (ym + sramH / 2)) / Math.max(rowsBot, 1);
    let ri = 0;
    for (let r = 0; r < rowsTop; r++, ri++) draw(ri, y0 + r * hTop, hTop);
    for (let r = 0; r < rowsBot; r++, ri++) draw(ri, ym + sramH / 2 + r * hBot, hBot);
  } else {
    const ch = it.channels || 1, nb = it.banks_per_channel || 1;
    const cim = it.cim_fraction || 0;
    const yb = H * cim;
    if (cim) {
      g.fillStyle = mode === "memory" && it.compute_util != null ? utilCss(it.compute_util) : "#5b3d7a";
      g.fillRect(4, 4, W - 8, yb - 6);
      g.fillStyle = "rgba(255,255,255,.75)"; g.font = "bold 18px system-ui"; g.fillText("compute-in-memory", 12, yb / 2 + 6);
    }
    const cw = (W - 8) / ch, bh = (H - yb - 6) / nb;
    for (let a = 0; a < ch; a++) {
      for (let b = 0; b < nb; b++) {
        const x = 4 + a * cw, y = yb + 2 + b * bh;
        const u = (it.bank_access?.[a]?.[b]) ?? 0;
        g.fillStyle = mode === "memory" ? utilCss(Math.min(1, u)) : "#2b3f55";
        g.fillRect(x + 1.5, y + 1.5, cw - 3, bh - 3);
        // subarrays separated by sense-amp strips
        const sub = Math.min(8, Math.max(2, Math.floor(bh / 6)));
        for (let s = 1; s < sub; s++) {
          const yy = y + 1.5 + s * (bh - 3) / sub;
          g.fillStyle = "rgba(240,200,120,.55)"; g.fillRect(x + 2, yy - 0.6, cw - 4, 1.2);
        }
      }
    }
  }
  return cv;
}

function thermalTexture(grid, values) {
  const { nx, ny } = grid;
  const flat = values.flat();
  const lo = S.tmin, hi = S.tmax;
  const data = new Uint8Array(nx * ny * 4);
  for (let i = 0; i < nx * ny; i++) {
    const [r, g, b] = ramp((flat[i] - lo) / Math.max(hi - lo, 1e-6));
    data.set([r, g, b, 255], i * 4);
  }
  const t = new THREE.DataTexture(data, nx, ny);
  t.colorSpace = THREE.SRGBColorSpace;
  t.magFilter = t.minFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

// ---------------------------------------------------------------- build the scene
const H = {}; // handles: groups and per-component meshes for fast mode/explode updates

function build() {
  const D = S.data;
  scene3.remove(world);
  world.traverse((o) => { o.geometry?.dispose?.(); });
  world = new THREE.Group();
  scene3.add(world);
  pickables.length = 0;
  animators.length = 0;
  const V = S.vscale;
  const pkg = D.package, pcb = D.pcb;
  const px = pcb.package_x; // package center x on the board
  const groups = {};
  for (const k of ["pcb", "substrate", "interposer", "dies", "ihs", "cooling", "links", "air"]) {
    groups[k] = new THREE.Group(); world.add(groups[k]);
  }
  H.groups = groups;
  // PCB
  const board = box(pcb.w, pcb.t, pcb.d, MAT.pcb(), { type: "pcb", id: "PCB" });
  board.position.set(0, pcb.t / 2, 0);
  groups.pcb.add(board);
  for (const tr of pcb.traces) {
    for (let i = 1; i < tr.points.length; i++) {
      const [ax, ay] = tr.points[i - 1], [bx, by] = tr.points[i];
      const len = Math.hypot(bx - ax, by - ay);
      const m = box(len, 0.08, tr.width, MAT.copper());
      m.position.set((ax + bx) / 2, pcb.t + 0.04, -(ay + by) / 2);
      m.rotation.y = Math.atan2(by - ay, bx - ax);
      groups.pcb.add(m);
    }
  }
  for (const p of pcb.parts) {
    let m;
    const pick = { type: "pcb_part", id: p.id, part: p };
    if (p.kind === "capacitor") {
      m = new THREE.Mesh(new THREE.CylinderGeometry(p.w / 2, p.w / 2, p.h, 16), MAT.cap());
      m.userData.pick = pick; pickables.push(m);
    } else {
      m = box(p.w, p.h, p.d, p.kind === "inductor" ? MAT.inductor() : p.kind === "edge_connector" ? MAT.gold() : MAT.part(), pick);
    }
    m.position.set(p.x, pcb.t + p.h / 2, -p.y);
    groups.pcb.add(m);
  }
  // Package substrate (BGA)
  const subT = pkg.substrate.t;
  const sub = box(pkg.substrate.w, subT, pkg.substrate.d, MAT.substrate(), { type: "substrate", id: "Package substrate" });
  sub.position.set(px, subT / 2, 0);
  groups.substrate.add(sub);
  groups.substrate.position.y = pcb.t;
  // Interposer
  const ip = pkg.interposer;
  const ipT = ip.t * V;
  const ipm = box(ip.w, ipT, ip.d, MAT.interposer(), { type: "interposer", id: ip.kind.replace("_", " ") });
  ipm.position.set(px + ip.x, ipT / 2, -ip.y);
  groups.interposer.add(ipm);
  H.ipTop = ipT;
  // Dies and stacks
  H.comp = {};
  for (const c of D.components) {
    const g = new THREE.Group();
    g.position.set(px + c.x, ipT, -c.y);
    g.userData.cid = c.id;
    groups.dies.add(g);
    const parts = [];
    const ub = 0.03 * V; // microbump layer
    if (c.kind === "hbm_stack") {
      c.layers.forEach((L, i) => {
        const m = box(c.w, L.h * V, c.d, L.kind === "base_die" ? MAT.base() : MAT.dram(), { type: "die_layer", id: c.id, layer: i });
        m.userData.baseY = ub + L.z * V + (L.h * V) / 2;
        m.userData.stackIndex = i;
        m.position.y = m.userData.baseY;
        g.add(m); parts.push(m);
      });
      // TSV columns through the stack (sampled grid of the estimated TSV field)
      const cols = 10, rows = 6, h = c.h * V;
      const tsvG = new THREE.CylinderGeometry(0.12, 0.12, 1, 6);
      const tsv = new THREE.InstancedMesh(tsvG, MAT.copper(), cols * rows);
      const mtx = new THREE.Matrix4();
      let k = 0;
      for (let a = 0; a < cols; a++) for (let b = 0; b < rows; b++) {
        mtx.makeScale(1, h, 1).setPosition(-c.w * 0.28 + a * (c.w * 0.56) / (cols - 1), ub + h / 2, -c.d * 0.12 + b * (c.d * 0.24) / (rows - 1));
        tsv.setMatrixAt(k++, mtx);
      }
      tsv.userData.isTSV = true;
      g.add(tsv);
      H.comp[c.id] = { group: g, parts, top: parts[parts.length - 1], c, tsv };
    } else {
      const m = box(c.w, c.h * V, c.d, c.kind === "dram_chiplet" || c.kind === "memory_chiplet" ? MAT.dram() : MAT.die(), { type: "die", id: c.id });
      m.userData.baseY = ub + (c.h * V) / 2;
      m.position.y = m.userData.baseY;
      g.add(m); parts.push(m);
      H.comp[c.id] = { group: g, parts, top: m, c };
    }
    // top-face texture plane (floorplan / thermal / power)
    const top = new THREE.Mesh(new THREE.PlaneGeometry(c.w * 0.995, c.d * 0.995), new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.2 }));
    top.rotation.x = -Math.PI / 2;
    top.userData.pick = { type: c.kind === "hbm_stack" ? "die_layer" : "die", id: c.id, layer: c.kind === "hbm_stack" ? c.layers.length - 1 : undefined };
    pickables.push(top);
    g.add(top);
    H.comp[c.id].face = top;
    // microbumps under the die (sampled)
    const bx = 14, by = 10;
    const bump = new THREE.InstancedMesh(new THREE.SphereGeometry(0.22, 6, 4), MAT.copper(), bx * by);
    const mtx = new THREE.Matrix4();
    let k = 0;
    for (let a = 0; a < bx; a++) for (let b = 0; b < by; b++) {
      mtx.makeTranslation(-c.w * 0.45 + a * c.w * 0.9 / (bx - 1), ub / 2, -c.d * 0.45 + b * c.d * 0.9 / (by - 1));
      bump.setMatrixAt(k++, mtx);
    }
    g.add(bump);
  }
  // Interposer routes / bridges (from the simulated topology)
  H.links = [];
  const center = new THREE.Vector3(px + ip.x, 0, -ip.y);
  for (const L of D.links) {
    const a = D.components.find((c) => c.id === L.from);
    const b = L.to === "switch" ? null : D.components.find((c) => c.id === L.to);
    if (!a) continue;
    const pa = new THREE.Vector3(px + a.x, 0, -a.y), pb = b ? new THREE.Vector3(px + b.x, 0, -b.y) : center.clone();
    const path = [pa, new THREE.Vector3(pb.x, 0, pa.z), pb];
    // arc for the data-movement view (logical links may join non-adjacent dies)
    const span = pa.distanceTo(pb);
    const arc = new THREE.QuadraticBezierCurve3(pa.clone(), pa.clone().add(pb).multiplyScalar(0.5).setY(6 + span * 0.35), pb.clone());
    const segs = [];
    for (let i = 1; i < path.length; i++) {
      const s0 = path[i - 1], s1 = path[i], len = s0.distanceTo(s1);
      if (len < 0.01) continue;
      const m = box(Math.max(len, 0.01), 0.04 * V, 1.2, MAT.copper(), { type: "link", id: `${L.from} ↔ ${L.to}`, link: L });
      m.position.set((s0.x + s1.x) / 2, ipT + 0.02 * V, (s0.z + s1.z) / 2);
      m.rotation.y = -Math.atan2(s1.z - s0.z, s1.x - s0.x);
      groups.interposer.add(m); segs.push(m);
    }
    if (L.kind === "bridge" && b) {
      const br = box(4, 0.1 * V, 6, MAT.bridge(), { type: "link", id: `bridge ${L.from} ↔ ${L.to}`, link: L });
      br.position.set((pa.x + pb.x) / 2, ipT * 0.5, (pa.z + pb.z) / 2);
      groups.interposer.add(br);
    }
    H.links.push({ L, path, segs, arc });
  }
  // Heat spreader
  const ihsT = pkg.ihs.t;
  const ihs = box(pkg.ihs.w, ihsT, pkg.ihs.d, MAT.ihs(), { type: "ihs", id: "Integrated heat spreader" });
  ihs.position.set(px + ip.x, ihsT / 2, -ip.y);
  groups.ihs.add(ihs);
  // Heatsink + fans (flow runs along -z through the fin channels)
  const hs = pkg.heatsink;
  const base = box(hs.w, hs.base_t, hs.d, MAT.alu(), { type: "heatsink", id: "Heatsink" });
  base.position.set(px + ip.x, hs.base_t / 2, -ip.y);
  groups.cooling.add(base);
  const fins = new THREE.InstancedMesh(new THREE.BoxGeometry(hs.fin_t, hs.fin_h, hs.d), MAT.alu(), hs.fins);
  const mt = new THREE.Matrix4();
  for (let i = 0; i < hs.fins; i++) {
    mt.makeTranslation(px + ip.x - hs.w / 2 + (i + 0.5) * hs.w / hs.fins, hs.base_t + hs.fin_h / 2, -ip.y);
    fins.setMatrixAt(i, mt);
  }
  fins.castShadow = true;
  fins.userData.pick = { type: "heatsink", id: "Heatsink fins" };
  pickables.push(fins);
  groups.cooling.add(fins);
  H.fins = fins;
  const nf = pkg.fans.length;
  pkg.fans.forEach((fan, i) => {
    const fg = new THREE.Group();
    const r = Math.min(fan.d / 2, hs.w / (2 * nf) - 2);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 2.2, 8, 40), MAT.fan());
    fg.add(ring);
    const rotor = new THREE.Group();
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.3, r * 0.3, 6, 24), MAT.fan());
    hub.rotation.x = Math.PI / 2; rotor.add(hub);
    for (let b = 0; b < 9; b++) {
      const bl = new THREE.Mesh(new THREE.BoxGeometry(r * 0.66, r * 0.26, 0.8), MAT.fan());
      bl.position.x = r * 0.62;
      const arm = new THREE.Group(); arm.rotation.z = (b / 9) * Math.PI * 2; bl.rotation.x = 0.5;
      arm.add(bl); rotor.add(arm);
    }
    fg.add(rotor);
    fg.position.set(px + ip.x - hs.w / 2 + (i + 0.5) * hs.w / nf, hs.base_t + hs.fin_h / 2, -ip.y + hs.d / 2 + 6);
    fg.userData.pick = { type: "fan", id: `Fan ${i + 1}`, fan };
    ring.userData.pick = fg.userData.pick; pickables.push(ring);
    groups.cooling.add(fg);
    // rotation speed from the airflow model's fan rpm (slowed 1:40 so it is visible)
    animators.push((dt) => { rotor.rotation.z -= (fan.rpm / 60) * 2 * Math.PI * dt / 40; });
  });
  buildAir();
  buildLinkFlow();
  applyMode();
  layout();
}

// Air particles: speed = simulated channel velocity, color = air temperature rise.
function buildAir() {
  const D = S.data, hs = D.package.heatsink, air = D.thermal.airflow, ip = D.package.interposer;
  const n = 900;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), seed = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = D.pcb.package_x + ip.x - hs.w / 2 + Math.random() * hs.w;
    pos[i * 3 + 1] = hs.base_t + Math.random() * hs.fin_h;
    seed[i] = Math.random();
  }
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.6, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
  H.groups.air.add(pts);
  const amb = D.thermal.ambient_c, rise = air.air_rise_c;
  const lo = amb, hi = amb + Math.max(rise, 1);
  const speed = air.velocity_ms * 1000 / 60; // mm/s, slowed 1:60
  animators.push((dt) => {
    if (!H.groups.air.visible) return;
    const z0 = -ip.y + hs.d / 2, len = hs.d;
    for (let i = 0; i < n; i++) {
      seed[i] = (seed[i] + speed * dt / len) % 1;
      pos[i * 3 + 2] = z0 - seed[i] * len;
      const c = rampColor(((amb + rise * seed[i]) - lo) / (hi - lo) * 0.85 + 0.05);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
  });
}

// Data packets along links: count and speed scale with simulated link utilization.
function buildLinkFlow() {
  const maxU = Math.max(1e-9, ...S.data.links.map((l) => l.util));
  for (const h of H.links) {
    const rel = h.L.util / maxU;
    const col = new THREE.Color(utilCss(rel));
    const tube = new THREE.Mesh(new THREE.TubeGeometry(h.arc, 40, 0.25 + 0.9 * rel, 8, false),
      new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.35, transparent: true, opacity: 0.55 }));
    tube.userData.pick = { type: "link", id: `${h.L.from} ↔ ${h.L.to}`, link: h.L };
    pickables.push(tube);
    H.groups.links.add(tube);
    // packets: count and speed scale with the simulated link utilization
    const n = h.L.util > 0 ? Math.max(2, Math.round(3 + 30 * Math.sqrt(rel))) : 0;
    if (!n) continue;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3);
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    H.groups.links.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 2.4, color: 0xffffff, transparent: true, depthWrite: false })));
    const ph = Array.from({ length: n }, (_, i) => i / n);
    const v = 0.05 + 0.5 * rel;
    const p = new THREE.Vector3();
    animators.push((dt) => {
      if (!H.groups.links.visible) return;
      for (let i = 0; i < n; i++) {
        ph[i] = (ph[i] + v * dt) % 1;
        h.arc.getPoint(i % 2 ? 1 - ph[i] : ph[i], p);
        pos.set([p.x, p.y, p.z], i * 3);
      }
      geo.attributes.position.needsUpdate = true;
    });
  }
}

// Vertical placement of the layer groups (explode spreads them apart).
function layout() {
  const D = S.data, V = S.vscale, E = S.explode, g = H.groups;
  const pcbT = D.pcb.t, subT = D.package.substrate.t;
  const gap = 40 * E;
  g.substrate.position.y = pcbT + gap * 0.5;
  const ipY = pcbT + subT + gap * 1.1;
  g.interposer.position.y = ipY;
  g.links.position.y = ipY;
  g.dies.position.y = ipY + gap * 0.5;
  let maxTop = 0;
  for (const h of Object.values(H.comp)) {
    const c = h.c;
    const expand = c.kind === "hbm_stack" && (S.mode === "memory" || S.mode === "die" || E > 0) ? 1 + 10 * E + (S.selected?.id === c.id && S.mode !== "assembly" ? 6 : 0) : 1;
    let top = 0;
    for (const m of h.parts) {
      const i = m.userData.stackIndex;
      m.position.y = i === undefined ? m.userData.baseY : m.userData.baseY + i * (expand - 1) * 0.6;
      top = Math.max(top, m.position.y + m.geometry.parameters.height / 2);
    }
    if (h.tsv) h.tsv.visible = expand <= 1.05 || S.mode !== "assembly";
    h.face.position.y = top + 0.02;
    maxTop = Math.max(maxTop, top);
  }
  const ihsY = g.dies.position.y + maxTop + 0.05 * V + gap * 1.2;
  g.ihs.position.y = ihsY;
  g.cooling.position.y = ihsY + D.package.ihs.t + gap * 1.6;
  g.air.position.y = g.cooling.position.y;
  g.links.position.y = g.dies.position.y + maxTop;
  // shadow camera follows the board
  sun.shadow.camera.updateProjectionMatrix();
}

// ---------------------------------------------------------------- modes
function applyMode() {
  const D = S.data, mode = S.mode;
  for (const [k, on] of Object.entries(S.layers)) {
    if (k === "cooling") { H.groups.cooling.visible = on; }
    else if (H.groups[k]) H.groups[k].visible = on;
  }
  H.groups.air.visible = mode === "airflow" && S.layers.cooling;
  H.groups.links.visible = mode === "interconnect" && S.layers.interposer;
  if (H.fins) {
    H.fins.material.transparent = mode === "airflow";
    H.fins.material.opacity = mode === "airflow" ? 0.18 : 1;
    H.fins.material.depthWrite = mode !== "airflow";
    H.fins.material.needsUpdate = true;
  }
  // die faces
  const tgrid = D.thermal.grid;
  let ttex = null;
  if (mode === "thermal") {
    const fr = S.frame >= 0 ? D.thermal.transient.frames[S.frame] : null;
    ttex = fr ? thermalTexture(D.thermal.transient.frame_grid, fr) : thermalTexture(tgrid, tgrid.die_c);
  }
  const maxPD = Math.max(1e-9, ...D.components.map((c) => c.power_density_w_mm2));
  for (const h of Object.values(H.comp)) {
    const c = h.c, mat = h.face.material;
    mat.map?.dispose?.();
    mat.map = null; mat.emissive = new THREE.Color(0); mat.color = new THREE.Color(0xffffff);
    if (mode === "thermal") {
      const t = ttex.clone();
      t.needsUpdate = true;
      t.repeat.set(c.w / tgrid.w, c.d / tgrid.d);
      t.offset.set((c.x - c.w / 2 - tgrid.x0) / tgrid.w, (c.y - c.d / 2 - tgrid.y0) / tgrid.d);
      mat.map = t;
    } else if (mode === "power") {
      mat.color = rampColor(c.power_density_w_mm2 / maxPD);
      if (c.inefficient) mat.emissive = new THREE.Color(0x401010);
    } else {
      const dim = (mode === "memory" && c.internal.kind !== "memory") || (mode === "interconnect");
      const t = new THREE.CanvasTexture(floorplanCanvas(c, mode));
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      mat.map = t;
      if (dim) mat.color = new THREE.Color(0x555a62);
    }
    mat.needsUpdate = true;
    for (const m of h.parts) {
      if (mode === "memory" && c.kind === "hbm_stack" && m.userData.stackIndex !== undefined) {
        const L = c.layers[m.userData.stackIndex];
        m.material.color = L.kind === "base_die" ? new THREE.Color(0x4a515e) : new THREE.Color(utilCss(Math.min(1, c.utilization || 0)));
      } else if (mode === "thermal" && c.kind === "hbm_stack" && m.userData.stackIndex !== undefined) {
        const L = c.layers[m.userData.stackIndex];
        m.material.color = rampColor((L.temp_c - S.tmin) / Math.max(S.tmax - S.tmin, 1e-6));
      } else {
        m.material.color.copy(m.userData.baseColor);
      }
    }
  }
  // links colored by utilization in interconnect mode
  const maxU = Math.max(1e-9, ...D.links.map((l) => l.util));
  for (const h of H.links) {
    for (const s of h.segs) s.material.color = new THREE.Color(mode === "interconnect" ? utilCss(h.L.util / maxU) : 0xc27a3e);
    for (const s of h.segs) s.material.emissive = new THREE.Color(mode === "interconnect" && h.L.util === maxU && maxU > 0.05 ? 0x551100 : 0);
  }
  applyWire();
  legend();
  hud();
  $("timeline").style.display = mode === "thermal" ? "flex" : "none";
  if (mode === "thermal") drawTimeline();
}

function applyWire() {
  world.traverse((o) => {
    if (!o.isMesh) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    if (S.wire) {
      if (!o.userData.edges && !o.isInstancedMesh) {
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 20), new THREE.LineBasicMaterial({ color: dark ? 0x38bdf8 : 0x0369a1 }));
        e.userData.isEdge = true; o.add(e); o.userData.edges = e;
      }
      if (o.userData.edges) o.userData.edges.visible = true;
      for (const m of mats) { if (m.userData.op === undefined) m.userData.op = m.opacity; m.transparent = true; m.opacity = 0.06; m.depthWrite = false; }
    } else {
      if (o.userData.edges) o.userData.edges.visible = false;
      for (const m of mats) if (m.userData.op !== undefined) { m.opacity = m.userData.op; m.transparent = m.opacity < 1; m.depthWrite = true; }
    }
  });
}

function legend() {
  const D = S.data, el = $("legend"), m = S.mode;
  let html = "";
  if (m === "thermal") {
    html = `<b>Die temperature</b><div class="bar" style="background:${gradientCss(rampCss)}"></div><div class="ends"><span>${f(S.tmin)} °C</span><span>${f(S.tmax)} °C</span></div>
      <div class="note">Hotspot ${f(D.thermal.hotspot.temp_c)} °C · limit ${f(D.max_temp_c)} °C · heatsink ${f(D.thermal.r_heatsink_c_per_w, 3)} °C/W (${esc(D.thermal.r_source)}) · ${f(D.thermal.power_in_w)} W in, ${f(D.thermal.heat_out_w)} W out</div>`;
  } else if (m === "power") {
    const maxPD = Math.max(...D.components.map((c) => c.power_density_w_mm2));
    html = `<b>Power density</b><div class="bar" style="background:${gradientCss(rampCss)}"></div><div class="ends"><span>0</span><span>${f(maxPD, 3)} W/mm²</span></div>
      <div class="note">Package ${f(D.power.package_w)} W (dies ${f(D.power.dies_w)} W + PHY/links ${f(D.power.interconnect_w)} W) · VRM loss ${f(D.power.vrm_loss_w)} W · board input ${f(D.power.board_input_w)} W. Red glow: large power share but &lt;30% busy in the simulated workload.</div>`;
  } else if (m === "memory" || m === "interconnect" || m === "die") {
    html = `<b>${m === "interconnect" ? "Link utilization" : "Simulated activity"}</b><div class="bar" style="background:${gradientCss(utilCss)}"></div><div class="ends"><span>idle</span><span>${m === "interconnect" ? "busiest link" : "100% busy"}</span></div>
      <div class="note">${D.simulation ? `Discrete-event run of <b>${esc(S.data.workload)}</b>: bottleneck <b>${esc(D.simulation.bottleneck.bound_type)}</b>, ${f(D.simulation.events, 0)} events.` : "No simulation."}</div>`;
  } else if (m === "airflow") {
    const a = D.thermal.airflow;
    html = `<b>Air temperature</b><div class="bar" style="background:${gradientCss(rampCss)}"></div><div class="ends"><span>${f(D.thermal.ambient_c)} °C in</span><span>${f(D.thermal.ambient_c + a.air_rise_c)} °C out</span></div>
      <div class="note">${f(a.flow_cfm, 0)} CFM · ${f(a.velocity_ms)} m/s in fin channels (${esc(a.regime)}, Re ${f(a.reynolds, 0)}) · h ${f(a.h_w_m2k)} W/m²K · fin efficiency ${f(a.fin_efficiency * 100, 0)}% · fans ${f(a.fan_rpm, 0)} rpm. Particles move at the simulated velocity, slowed 1:60.</div>`;
  }
  el.hidden = !html;
  el.innerHTML = html;
}

function hud() {
  const D = S.data, sim = D.simulation;
  const cap = D.components.reduce((a, c) => a + c.capacity_gb, 0), bw = D.components.reduce((a, c) => a + c.bandwidth_gbps, 0);
  $("hud").innerHTML = `<b>${esc(D.design.name)}</b> v${esc(D.design.version)} · ${esc(D.design.package_type)} · ${esc(D.package.interposer.kind.replace(/_/g, " "))}<br>
    ${D.components.length} chiplets · ${f(cap)} GB · ${f(bw / 1000, 2)} TB/s · ${f(D.power.package_w)} W · peak ${f(D.thermal.hotspot.temp_c)} °C${sim ? ` · ${f(sim.throughput_tokens_per_sec)} tok/s (${esc(D.workload)})` : ""}`;
}

// ---------------------------------------------------------------- thermal transient
function drawTimeline() {
  const cv = $("tchart"), D = S.data, ser = D.thermal.transient.series;
  const r = cv.getBoundingClientRect();
  cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio;
  const g = cv.getContext("2d");
  g.scale(devicePixelRatio, devicePixelRatio);
  const W = r.width, Hh = r.height;
  const tmin = Math.log10(ser[0].t_s), tmax = Math.log10(ser[ser.length - 1].t_s);
  const lo = D.thermal.ambient_c, hi = Math.max(D.max_temp_c, ...ser.map((s) => s.max_c)) + 2;
  const X = (t) => 4 + (Math.log10(t) - tmin) / (tmax - tmin) * (W - 8), Y = (v) => Hh - 4 - (v - lo) / (hi - lo) * (Hh - 10);
  const css = getComputedStyle(document.body);
  g.strokeStyle = css.getPropertyValue("--bad"); g.setLineDash([4, 3]); g.beginPath(); g.moveTo(0, Y(D.max_temp_c)); g.lineTo(W, Y(D.max_temp_c)); g.stroke(); g.setLineDash([]);
  for (const [key, color] of [["max_c", css.getPropertyValue("--accent")], ["mean_die_c", css.getPropertyValue("--faint")]]) {
    g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
    ser.forEach((s, i) => (i ? g.lineTo(X(s.t_s), Y(s[key])) : g.moveTo(X(s.t_s), Y(s[key]))));
    g.stroke();
  }
  if (S.frame >= 0) {
    const s = ser[S.frame];
    g.fillStyle = css.getPropertyValue("--text"); g.beginPath(); g.arc(X(s.t_s), Y(s.max_c), 4, 0, 7); g.fill();
    $("tt").textContent = `t = ${s.t_s < 1 ? f(s.t_s * 1000, 0) + " ms" : f(s.t_s, 1) + " s"} · max ${f(s.max_c)} °C`;
  } else {
    $("tt").textContent = `steady state · τ ≈ ${f(D.thermal.time_constant_s, 0)} s`;
  }
}
$("play").onclick = () => {
  if (S.playing) return;
  S.playing = true;
  const n = S.data.thermal.transient.frames.length;
  let i = 0;
  const step = () => {
    S.frame = i; applyMode();
    if (++i < n) setTimeout(step, 380);
    else setTimeout(() => { S.frame = -1; S.playing = false; applyMode(); }, 900);
  };
  step();
};

// ---------------------------------------------------------------- picking + inspector
const ray = new THREE.Raycaster();
const ptr = new THREE.Vector2();
function pickAt(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hits = ray.intersectObjects(pickables.filter((m) => visibleDeep(m)), false);
  return hits[0] || null;
}
function visibleDeep(o) { for (let p = o; p; p = p.parent) if (!p.visible) return false; return true; }

let drag = null;
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
renderer.domElement.addEventListener("pointerdown", (ev) => {
  const h = pickAt(ev);
  S.downAt = [ev.clientX, ev.clientY];
  if (S.move && h && (h.object.userData.pick.type === "die" || h.object.userData.pick.type === "die_layer")) {
    const hc = H.comp[h.object.userData.pick.id];
    controls.enabled = false;
    dragPlane.constant = -h.point.y;
    drag = { id: hc.c.id, g: hc.group, off: hc.group.position.clone().sub(h.point), moved: false };
    renderer.domElement.setPointerCapture(ev.pointerId);
  }
});
renderer.domElement.addEventListener("pointermove", (ev) => {
  if (drag) {
    const r = renderer.domElement.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    const p = new THREE.Vector3();
    if (ray.ray.intersectPlane(dragPlane, p)) { drag.g.position.x = p.x + drag.off.x; drag.g.position.z = p.z + drag.off.z; drag.moved = true; }
    return;
  }
  const h = pickAt(ev), tip = $("tip");
  if (h) {
    const pk = h.object.userData.pick;
    tip.style.display = "block";
    tip.textContent = tipText(pk);
    const r = vp.getBoundingClientRect();
    tip.style.left = `${Math.min(ev.clientX - r.left + 14, r.width - 220)}px`;
    tip.style.top = `${ev.clientY - r.top + 14}px`;
    renderer.domElement.style.cursor = S.move && (pk.type === "die" || pk.type === "die_layer") ? "grab" : "pointer";
  } else { tip.style.display = "none"; renderer.domElement.style.cursor = ""; }
});
renderer.domElement.addEventListener("pointerup", async (ev) => {
  if (drag) {
    controls.enabled = true;
    const d = drag; drag = null;
    if (d.moved) {
      const x = d.g.position.x - S.data.pcb.package_x, y = -d.g.position.z;
      await edit({ placement: { [d.id]: { x: +x.toFixed(2), y: +y.toFixed(2) } } }, false, `Moved ${d.id}`);
    }
    return;
  }
  if (S.downAt && Math.hypot(ev.clientX - S.downAt[0], ev.clientY - S.downAt[1]) > 4) return; // orbit drag
  const h = pickAt(ev);
  select(h ? h.object.userData.pick : null);
});
renderer.domElement.addEventListener("dblclick", () => { if (S.selected) focusOn(S.selected.id); });

function tipText(pk) {
  const c = S.data.components.find((x) => x.id === pk.id);
  if (c) {
    if (pk.type === "die_layer" && c.layers) { const L = c.layers[pk.layer]; return `${c.id} · ${L.label} · ${f(L.temp_c)} °C`; }
    return `${c.id} · ${c.kind.replace(/_/g, " ")} · ${f(c.power_w)} W · ${f(c.temp_c)} °C`;
  }
  if (pk.type === "link") return `${pk.id} · ${f(pk.link.util * 100, 1)}% busy`;
  if (pk.type === "pcb_part") return `${pk.part.kind.replace(/_/g, " ")} ${pk.id}`;
  return pk.id;
}

function select(pk) {
  S.selected = pk;
  inspector();
  memoryStudio();
  if (S.mode === "die" || S.mode === "memory") { applyMode(); layout(); }
}

function kv(rows) { return `<dl class="kv">${rows.filter(Boolean).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`; }
function bars(label, vals) {
  return `<div class="empty" style="margin-top:8px">${esc(label)}</div><div class="bars">${vals.map((v, i) => `<div><span>${i}</span><span class="tr"><i style="width:${Math.min(100, v * 100)}%"></i></span><span>${f(v * 100, 0)}%</span></div>`).join("")}</div>`;
}

function inspector() {
  const el = $("inspector"), pk = S.selected, D = S.data;
  if (!pk) { el.innerHTML = `<h2>Inspector</h2><div class="empty">Click a die, memory stack, link or board part. Double-click to fly to it.</div>`; return; }
  const c = D.components.find((x) => x.id === pk.id);
  if (c) {
    const it = c.internal, th = D.thermal.components.find((t) => t.id === c.id);
    let html = `<h2>${esc(c.id)} <span class="badge">${esc(c.kind.replace(/_/g, " "))}</span></h2>`;
    html += kv([
      ["Footprint", `${f(c.w)} × ${f(c.d)} mm (${f(c.area_mm2, 0)} mm²)`],
      ["Height", `${f(c.h * 1000, 0)} µm`],
      ["Position", `${f(c.x)}, ${f(c.y)} mm${c.moved ? " (moved)" : ""}`],
      c.process_nm && ["Process", `${f(c.process_nm, 0)} nm`],
      ["Power", `${f(c.power_w, 2)} W (${f(c.power_density_w_mm2, 3)} W/mm²)`],
      c.phy_power_w && ["of which PHY/links", `${f(c.phy_power_w, 2)} W`],
      ["Temperature", `max ${f(c.temp_c)} °C · mean ${f(c.temp_mean_c)} °C`],
      th && ["Headroom", `<span style="color:var(${th.headroom_c < 0 ? "--bad" : th.headroom_c < 10 ? "--warn" : "--ok"})">${f(th.headroom_c)} °C</span>`],
      c.capacity_gb && ["Capacity", `${f(c.capacity_gb)} GB`],
      c.bandwidth_gbps && ["Bandwidth", `${f(c.bandwidth_gbps, 0)} GB/s (${c.channels} ch)`],
      c.memory_tech && ["Memory", `${esc(c.memory_tech)}${c.latency_ns ? `, ${f(c.latency_ns, 0)} ns` : ""}`],
      c.peak_tops && ["Compute", `${f(c.peak_tops, 0)} TOPS peak · ${f(c.effective_tops, 0)} effective (${esc(c.precision)})`],
      c.architecture && ["Architecture", esc(c.architecture)],
      c.utilization != null && ["Simulated busy", `${f(c.utilization * 100, 1)}%${c.inefficient ? ' <span style="color:var(--warn)">· inefficient</span>' : ""}`],
    ]);
    if (c.layers) {
      html += `<div class="stack-layers"><div><b>layer</b><b>°C</b><b>W</b></div>${c.layers.slice().reverse().map((L) => `<div><span>${esc(L.label)}</span><span>${f(L.temp_c, 2)}</span><span>${f(L.power_w, 2)}</span></div>`).join("")}</div>`;
      html += kv([["TSVs (est.)", `${f(c.tsv.count_estimate, 0)} @ ${f(c.tsv.pitch_um, 0)} µm`], ["Microbumps (est.)", `${f(c.microbumps.count_estimate, 0)} @ ${f(c.microbumps.pitch_um, 0)} µm`]]);
    }
    if (it.kind === "memory" && it.channel_util?.length) html += bars("Channel utilization (simulated)", it.channel_util);
    if (it.kind === "compute") html += `<div class="empty" style="margin-top:6px">${it.clusters} compute clusters · SRAM ${f(it.sram_mb)} MB · ${esc(it.note)}</div>`;
    else html += `<div class="empty" style="margin-top:6px">${it.channels} channels × ${it.banks_per_channel} banks · ${it.subarrays_per_bank} subarrays/bank${it.cim_fraction ? " · compute-in-memory region" : ""}. ${esc(it.note)}</div>`;
    el.innerHTML = html;
    return;
  }
  if (pk.type === "link") {
    const L = pk.link;
    el.innerHTML = `<h2>Link <span class="badge">${esc(L.kind.replace(/_/g, " "))}</span></h2>` + kv([["Between", `${esc(L.from)} ↔ ${esc(L.to)}`], ["Bandwidth", `${f(L.bandwidth_gbps, 0)} GB/s`], ["Simulated busy", `${f(L.util * 100, 2)}%`], ["Topology", esc(D.package.topology)]]);
    return;
  }
  if (pk.type === "pcb_part") {
    const p = pk.part;
    el.innerHTML = `<h2>${esc(p.kind.replace(/_/g, " "))} <span class="badge">${esc(p.id)}</span></h2>` + kv([["Size", `${f(p.w)} × ${f(p.d)} × ${f(p.h)} mm`], ...Object.entries(p.metrics).map(([k, v]) => [k, esc(v)])]) +
      (p.kind === "power_stage" || p.kind === "inductor" ? `<div class="empty" style="margin-top:6px">${D.pcb.vrm.phases} phases for ${f(D.pcb.vrm.amps, 0)} A at ${f(D.params.core_voltage_v, 2)} V; ${f(D.pcb.vrm.efficiency * 100, 0)}% efficient → ${f(D.pcb.vrm.loss_w)} W loss on the board.</div>` : "");
    return;
  }
  if (pk.type === "fan" || pk.type === "heatsink") {
    const a = D.thermal.airflow, hs = D.package.heatsink;
    el.innerHTML = `<h2>${esc(pk.id)}</h2>` + kv([["Heatsink", `${f(hs.w, 0)} × ${f(hs.d, 0)} mm, ${hs.fins} fins × ${f(hs.fin_h, 0)} mm`], ["Fans", `${D.package.fans.length} × ${f(a.flow_cfm / D.package.fans.length, 0)} CFM @ ${f(a.fan_rpm, 0)} rpm`], ["Channel velocity", `${f(a.velocity_ms)} m/s`], ["h", `${f(a.h_w_m2k)} W/m²K`], ["Fin efficiency", `${f(a.fin_efficiency * 100, 0)}%`], ["R heatsink", `${f(D.thermal.r_heatsink_c_per_w, 3)} °C/W`], ["Air rise", `${f(a.air_rise_c)} °C`]]);
    return;
  }
  el.innerHTML = `<h2>${esc(pk.id)}</h2>` + (pk.type === "interposer" ? kv([["Size", `${f(D.package.interposer.w)} × ${f(D.package.interposer.d)} mm`], ["Area", `${f(D.package.interposer.w * D.package.interposer.d, 0)} mm²`], ["Routes", D.links.length]]) : "");
}

// ---------------------------------------------------------------- Memory Packaging Studio
function memoryStudio() {
  const D = S.data, el = $("mps");
  const sel = D.components.find((c) => c.id === S.selected?.id && c.capacity_gb > 0) || D.components.find((c) => c.kind === "hbm_stack") || D.components.find((c) => c.capacity_gb > 0);
  const P = D.params, cl = P.cooling;
  const memRows = sel ? (() => {
    const n = D.components.filter((c) => c.chiplet_id === sel.chiplet_id).length;
    return `<div class="sec">${esc(sel.chiplet_id)} (${esc(sel.memory_tech || sel.kind)})</div>
      ${fld("Stacks / chiplets", "count", n, 1)}
      ${sel.kind === "hbm_stack" ? fld("Dies per stack", "stack_layers", sel.stack_layers, 1) : ""}
      ${fld("Capacity per stack, GB", "capacity_per_chiplet_gb", sel.capacity_gb, 0.5)}
      ${fld("Channels per stack", "channels", sel.channels, 1)}
      ${fld("GB/s per channel", "bandwidth_per_channel_gbps", sel.channels ? sel.bandwidth_gbps / sel.channels : 0, 1)}
      ${fld("Footprint, mm²", "area_mm2", sel.area_mm2, 1)}`;
  })() : `<div class="empty" style="grid-column:1/3">This design has no memory chiplets.</div>`;
  el.innerHTML = `<h2>Memory Packaging Studio</h2>
    <div class="form" id="mps-form">${memRows}
      <div class="sec">Stack & package</div>
      ${fld("DRAM die thickness, µm", "s.die_thickness_um", P.die_thickness_um, 1)}
      ${fld("Bond layer, µm", "s.bond_layer_um", P.bond_layer_um, 1)}
      ${fld("TSV pitch, µm", "s.tsv_pitch_um", P.tsv_pitch_um, 1)}
      ${fld("Microbump pitch, µm", "s.microbump_pitch_um", P.microbump_pitch_um, 1)}
      ${fld("Chiplet spacing, mm", "s.spacing_mm", P.spacing_mm, 0.1)}
      ${fld("Interposer margin, mm", "s.interposer_margin_mm", P.interposer_margin_mm, 0.1)}
      <div class="sec">Thermal</div>
      ${fld("Max temperature, °C", "c.max_temp_c", D.max_temp_c, 1)}
      ${fld("Ambient, °C", "c.ambient_temp_c", D.thermal.ambient_c, 1)}
      ${fld("Fans", "k.fans", cl.fans, 1)}
      ${fld("CFM per fan", "k.fan_cfm", cl.fan_cfm, 1)}
      ${fld("Fin count", "k.fin_count", cl.fin_count, 1)}
      ${fld("Fin height, mm", "k.fin_height_mm", cl.fin_height_mm, 1)}
    </div>
    <div class="row"><button class="btn primary" id="mps-apply">Apply</button><label class="ck"><input type="checkbox" id="mps-ver"> save as new version</label></div>
    <div class="msg" id="mps-msg"></div>`;
  $("mps-apply").onclick = async () => {
    const vals = {};
    for (const inp of el.querySelectorAll("input[data-k]")) if (inp.value !== inp.defaultValue) vals[inp.dataset.k] = Number(inp.value);
    if (!Object.keys(vals).length) { $("mps-msg").textContent = "Nothing changed."; return; }
    const patch = {}, chip = { chiplet_id: sel?.chiplet_id }, mem = {};
    for (const [k, v] of Object.entries(vals)) {
      if (k.startsWith("s.")) (patch.studio ??= {})[k.slice(2)] = v;
      else if (k.startsWith("k.")) ((patch.studio ??= {}).cooling ??= {})[k.slice(2)] = v;
      else if (k.startsWith("c.")) (patch.constraints ??= {})[k.slice(2)] = v;
      else if (k === "count" || k === "area_mm2") chip[k] = v;
      else mem[k] = v;
    }
    if (Object.keys(mem).length) chip.memory = mem;
    if (Object.keys(chip).length > 1) patch.chiplets = [chip];
    await edit(patch, $("mps-ver").checked, "Applied");
  };
}
function fld(label, key, val, step) {
  const v = Number.isFinite(val) ? +Number(val).toFixed(step < 1 ? 2 : 0) : "";
  return `<label for="f-${key}">${esc(label)}</label><input id="f-${key}" data-k="${key}" type="number" step="${step}" value="${v}">`;
}

async function edit(patch, newVersion, okText) {
  const msg = $("mps-msg");
  setBusy(true, "Re-simulating…");
  try {
    const r = await fetch("/api/studio/edit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ design_id: S.data.design.id, patch, new_version: newVersion, workload: $("workload").value }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    if (j.design_id !== S.data.design.id) { await loadDesigns(); $("design").value = j.design_id; }
    show(j, true);
    const m = $("mps-msg"); if (m) { m.className = "msg ok"; m.textContent = `${okText} · v${j.version} · peak ${f(j.thermal.hotspot.temp_c)} °C · ${f(j.power.package_w)} W`; }
  } catch (e) {
    if (msg) { msg.className = "msg err"; msg.textContent = e.message; }
    else status(e.message, true);
    build();
  } finally { setBusy(false); }
}

function summary() {
  const D = S.data, el = $("summary"), sim = D.simulation;
  el.innerHTML = `<h2>Package summary</h2>` + kv([
    ["Interposer", `${f(D.package.interposer.w)} × ${f(D.package.interposer.d)} mm`],
    ["Board", `${f(D.pcb.w, 0)} × ${f(D.pcb.d, 0)} mm · ${D.pcb.vrm.phases} VRM phases`],
    ["Package power", `${f(D.power.package_w)} W`],
    ["Peak temperature", `${f(D.thermal.hotspot.temp_c)} °C (limit ${f(D.max_temp_c)})`],
    ["Thermal τ", `${f(D.thermal.time_constant_s, 0)} s`],
    sim && ["Throughput", `${f(sim.throughput_tokens_per_sec)} tok/s`],
    sim && ["Latency", `${f(sim.latency_ms, 3)} ms`],
    sim && ["Bottleneck", esc(sim.bottleneck.bound_type)],
  ]) + D.warnings.map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("") +
    `<details style="margin-top:6px"><summary class="empty">Assumptions</summary>${D.assumptions.map((a) => `<div class="empty" style="margin:4px 0">${esc(a)}</div>`).join("")}${sim ? `<div class="empty">${esc(sim.disclaimer)}</div>` : ""}</details>`;
}

// ---------------------------------------------------------------- camera helpers
function frameAll() {
  const b = new THREE.Box3().setFromObject(H.groups.dies);
  const pcbB = new THREE.Box3().setFromObject(H.groups.pcb);
  const tgt = S.mode === "die" || S.mode === "memory" || S.mode === "interconnect" ? b : pcbB.union(b);
  const c = tgt.getCenter(new THREE.Vector3()), r = tgt.getSize(new THREE.Vector3()).length() / 2;
  controls.target.copy(c);
  const fit = r / Math.sin((camera.fov * Math.PI) / 360) / Math.min(1, camera.aspect);
  camera.position.copy(c).add(new THREE.Vector3(0.5, 0.8, 0.85).normalize().multiplyScalar(fit * 0.95 + 10));
}
function focusOn(id) {
  const h = H.comp[id];
  if (!h) return;
  const b = new THREE.Box3().setFromObject(h.group), c = b.getCenter(new THREE.Vector3()), r = b.getSize(new THREE.Vector3()).length() / 2;
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.3, 1, 0.6).normalize().multiplyScalar(r * 2.6 + 8));
}

// ---------------------------------------------------------------- UI wiring
$("modes").innerHTML = MODES.map(([k, t, s]) => `<button data-m="${k}"${k === S.mode ? ' class="on"' : ""}>${t}<small>${s}</small></button>`).join("");
$("modes").onclick = (e) => {
  const b = e.target.closest("button"); if (!b || !S.data) return;
  S.mode = b.dataset.m;
  S.layers = { ...DEFAULT_LAYERS[S.mode] };
  S.frame = -1;
  for (const x of $("modes").children) x.classList.toggle("on", x === b);
  renderLayers();
  if (S.mode === "die" && !S.selected) { const c = S.data.components.find((x) => x.kind === "compute_die") || S.data.components[0]; select({ type: "die", id: c.id }); }
  applyMode(); layout();
  if (S.mode === "die" && S.selected) focusOn(S.selected.id); else frameAll();
};
function renderLayers() {
  $("layers").innerHTML = LAYERS.map(([k, t]) => `<label class="ck"><input type="checkbox" data-l="${k}"${S.layers[k] ? " checked" : ""}> ${t}</label>`).join("");
}
$("layers").onchange = (e) => { S.layers[e.target.dataset.l] = e.target.checked; applyMode(); };
renderLayers();
$("explode").oninput = (e) => { S.explode = +e.target.value; $("explode-v").textContent = f(S.explode * 100, 0) + "%"; if (S.data) layout(); };
$("vscale").oninput = (e) => { S.vscale = +e.target.value; $("vscale-v").textContent = `${S.vscale}×`; if (S.data) build(); };
$("render").onclick = (e) => {
  const b = e.target.closest("button"); if (!b) return;
  S.wire = b.dataset.r === "wire";
  for (const x of $("render").children) x.classList.toggle("on", x === b);
  if (S.data) applyWire();
};
$("move").onchange = (e) => { S.move = e.target.checked; };
$("reset-place").onclick = () => S.data && edit({ reset_placement: true }, false, "Placement reset");
$("frame").onclick = () => S.data && frameAll();
$("design").onchange = () => load();
$("workload").onchange = () => load();

function status(t, err) { const s = $("status"); s.textContent = t || ""; s.style.color = err ? "var(--bad)" : ""; }
function setBusy(b, t) { S.busy = b; $("loading").style.display = b ? "flex" : "none"; if (t) $("loading").textContent = t; }

async function loadDesigns() {
  const r = await fetch("/api/designs");
  const j = await r.json();
  const list = Array.isArray(j) ? j : j.designs || [];
  S.designs = list;
  const prev = $("design").value;
  $("design").innerHTML = list.map((d) => `<option value="${esc(d.design_id)}">${esc(d.name)} · v${esc(d.version)}</option>`).join("");
  const want = new URLSearchParams(location.search).get("design");
  if (prev && list.some((d) => d.design_id === prev)) $("design").value = prev;
  else if (want && list.some((d) => d.design_id === want)) $("design").value = want;
  else {
    // prefer a design with real hardware detail
    const good = list.find((d) => /hbm|llm|accelerator|echo/i.test(d.name)) || list[0];
    if (good) $("design").value = good.design_id;
  }
  return list;
}

async function load(keepView) {
  const id = $("design").value;
  if (!id) { setBusy(true, "No designs yet. Ask an agent to create one with chiplab, or use chip_design_from_template."); return; }
  setBusy(true, "Simulating package…");
  try {
    const wl = $("workload").value || "llm8_decode";
    const r = await fetch(`/api/studio/scene?design=${encodeURIComponent(id)}&workload=${encodeURIComponent(wl)}`);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    show(j, keepView);
    history.replaceState(null, "", `?design=${encodeURIComponent(id)}`);
  } catch (e) { setBusy(true, `Could not load this design: ${e.message}`); return; }
  setBusy(false);
}

function show(j, keepView) {
  S.data = j;
  if (!$("workload").options.length) $("workload").innerHTML = j.workloads.map((w) => `<option value="${w}">${w.replace(/_/g, " ")}</option>`).join("");
  $("workload").value = j.workload;
  const all = j.thermal.grid.die_c.flat();
  S.tmin = j.thermal.ambient_c;
  S.tmax = Math.max(...all, j.thermal.ambient_c + 1);
  if (S.selected && !j.components.some((c) => c.id === S.selected.id) && S.selected.type !== "pcb_part") S.selected = null;
  build();
  inspector(); memoryStudio(); summary();
  if (!keepView) frameAll();
  status(`${j.components.length} chiplets · ${f(j.power.package_w)} W · ${f(j.thermal.hotspot.temp_c)} °C peak`);
}

// AI agent activity: reload when an agent touches the open design.
function connect() {
  const es = new EventSource(`/api/stream?after=${S.lastEventId}`);
  es.addEventListener("activity", (m) => {
    const e = JSON.parse(m.data);
    S.lastEventId = e.id;
    const div = document.createElement("div");
    div.className = "ev";
    div.innerHTML = `<b>${esc(e.tool)}</b> <span>${esc(e.client || "agent")} · ${new Date(e.ts * 1000).toLocaleTimeString()}</span>${e.summary ? `<div>${esc(e.summary)}</div>` : ""}`;
    $("feed").prepend(div);
    while ($("feed").children.length > 60) $("feed").lastChild.remove();
    const cur = S.data?.design;
    if (cur && e.design_id && (e.design_id === cur.id || e.design_id.startsWith(cur.name)) && /create|update|set|edit|design/.test(e.tool) && !S.busy) {
      loadDesigns().then(() => load(true));
    }
  });
  es.onerror = () => { es.close(); setTimeout(connect, 3000); };
}

// ---------------------------------------------------------------- render loop
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);
  for (const a of animators) a(dt);
  controls.update();
  renderer.render(scene3, camera);
});

resize();
loadDesigns().then(() => load()).catch((e) => setBusy(true, e.message));
connect();
