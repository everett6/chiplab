// ChipLab GPU Studio: interactive 3D package view driven by ChipLab's design + simulators.
// Every color, animation speed and highlight comes from the scene JSON
// (/api/studio/scene): analytical power, the package thermal/airflow solve and the
// discrete-event simulation's per-resource utilization. Nothing is decorative.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildDieLayout, STACK } from "/layout3d.js";
import { renderChipletFloorplanSvg, renderRooflineSvg, renderTopologySvg, renderStackupCrossSectionSvg } from "./eda_diagrams.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 }));
const dark = true;                                  // the workstation theme (eda.css) is always dark

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
  ["cells", "Die layout (cells + metal stack)"],
  ["interposer", "Interposer & routes"], ["substrate", "Package substrate"], ["pcb", "PCB & board parts"],
];
const DEFAULT_LAYERS = {
  assembly: { cooling: true, ihs: true, dies: true, cells: true, interposer: true, substrate: true, pcb: true },
  die: { cooling: false, ihs: false, dies: true, cells: true, interposer: true, substrate: false, pcb: false },
  memory: { cooling: false, ihs: false, dies: true, cells: false, interposer: true, substrate: true, pcb: false },
  thermal: { cooling: false, ihs: false, dies: true, cells: false, interposer: true, substrate: true, pcb: true },
  power: { cooling: false, ihs: false, dies: true, cells: false, interposer: true, substrate: true, pcb: true },
  interconnect: { cooling: false, ihs: false, dies: true, cells: true, interposer: true, substrate: true, pcb: false },
  airflow: { cooling: true, ihs: true, dies: true, cells: false, interposer: true, substrate: true, pcb: true },
};

const S = {
  data: null, mode: "assembly", layers: { ...DEFAULT_LAYERS.assembly }, explode: 0, vscale: 6,
  lyr: {}, wire: false, selected: null, move: false, playing: false, frame: -1, designs: [],
  lastEventId: 0, busy: false,
  edaOpen: false, edaTab: "chiplet", camAnim: null,
  memPresets: [], memDraft: null, customMemResult: null, memComparisons: [],
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
renderer.toneMappingExposure = 1.15;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
vp.prepend(renderer.domElement);
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--viewport").trim() || "#dfe4ea");

// Procedural CAD studio environment map for realistic metallic reflections (gold, copper, solder, silicon)
(() => {
  const ec = document.createElement("canvas");
  ec.width = 256; ec.height = 128;
  const g = ec.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 128);
  grad.addColorStop(0.0, "#eef4ff");
  grad.addColorStop(0.45, "#9bb0cc");
  grad.addColorStop(0.55, "#232a36");
  grad.addColorStop(1.0, "#10141c");
  g.fillStyle = grad; g.fillRect(0, 0, 256, 128);
  g.fillStyle = "rgba(255,255,255,0.85)";
  g.fillRect(28, 14, 60, 22);
  g.fillRect(156, 18, 72, 18);
  const envTex = new THREE.CanvasTexture(ec);
  envTex.mapping = THREE.EquirectangularReflectionMapping;
  envTex.colorSpace = THREE.SRGBColorSpace;
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene3.environment = pmrem.fromEquirectangular(envTex).texture;
  envTex.dispose(); pmrem.dispose();
})();
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 5000);
camera.position.set(140, 160, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.screenSpacePanning = true;
controls.addEventListener("start", () => { S.camAnim = null; });
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

function buildPcbTextures(pcb, pkg) {
  const W = 1024, H = Math.max(384, Math.round(1024 * pcb.d / Math.max(pcb.w, 1)));
  const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  const bcv = document.createElement("canvas"); bcv.width = W; bcv.height = H;
  const bg = bcv.getContext("2d");

  const toX = (x) => (x + pcb.w / 2) / pcb.w * W;
  const toY = (y) => (-y + pcb.d / 2) / pcb.d * H;

  // 1. Matte LPI soldermask base + subtle FR-4 weave
  g.fillStyle = "#084232"; g.fillRect(0, 0, W, H);
  bg.fillStyle = "#505050"; bg.fillRect(0, 0, W, H);
  g.strokeStyle = "rgba(255,255,255,0.028)"; g.lineWidth = 1;
  for (let x = 0; x < W; x += 8) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  for (let y = 0; y < H; y += 8) { g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }

  // 2. CUDA PDN current-density / copper-plane thermal tint (if computed by backend)
  const pdn = pcb.cad?.cuda_pdn_field;
  if (pdn && Array.isArray(pdn.grid_u8) && pdn.nx && pdn.ny) {
    const cw = W / pdn.nx, ch = H / pdn.ny;
    for (let j = 0; j < pdn.ny; j++) {
      for (let i = 0; i < pdn.nx; i++) {
        const v = pdn.grid_u8[j * pdn.nx + i] / 255;
        if (v > 0.08) {
          g.fillStyle = `rgba(38, 165, 118, ${(v * 0.22).toFixed(3)})`;
          g.fillRect(i * cw, (pdn.ny - 1 - j) * ch, cw + 0.5, ch + 0.5);
        }
      }
    }
  }

  // 3. Exposed copper ground pour hatches & soldermask-covered trace reliefs
  g.strokeStyle = "#17684f"; g.lineWidth = 1.2;
  for (let x = 18; x < W - 18; x += 14) {
    g.beginPath(); g.moveTo(x, 18); g.lineTo(x + 12, H - 18); g.stroke();
  }
  for (const tr of pcb.traces || []) {
    g.strokeStyle = tr.kind === "power" ? "#1d7a5b" : "#146148";
    bg.strokeStyle = "#b8b8b8";
    const lw = Math.max(1.5, tr.width / pcb.w * W);
    g.lineWidth = lw; bg.lineWidth = lw;
    g.lineJoin = bg.lineJoin = "round";
    g.beginPath(); bg.beginPath();
    tr.points.forEach(([px0, py0], idx) => {
      const sx = toX(px0), sy = toY(py0);
      if (idx === 0) { g.moveTo(sx, sy); bg.moveTo(sx, sy); }
      else { g.lineTo(sx, sy); bg.lineTo(sx, sy); }
    });
    g.stroke(); bg.stroke();
  }

  // 4. White silkscreen: board perimeter, component courtyards, reference designators
  g.strokeStyle = "rgba(235,242,248,0.72)"; g.lineWidth = 1.6;
  g.strokeRect(10, 10, W - 20, H - 20);
  // Package U1 courtyard & Pin-1 triangle
  const ux0 = toX(pcb.package_x - pkg.substrate.w / 2 - 2), ux1 = toX(pcb.package_x + pkg.substrate.w / 2 + 2);
  const uy0 = toY(pkg.substrate.d / 2 + 2), uy1 = toY(-pkg.substrate.d / 2 - 2);
  g.strokeRect(ux0, Math.min(uy0, uy1), ux1 - ux0, Math.abs(uy1 - uy0));
  g.fillStyle = "rgba(235,242,248,0.85)";
  g.beginPath();
  g.moveTo(ux0 - 4, Math.min(uy0, uy1));
  g.lineTo(ux0 + 6, Math.min(uy0, uy1));
  g.lineTo(ux0 - 4, Math.min(uy0, uy1) + 10);
  g.closePath(); g.fill();

  for (const p of pcb.parts || []) {
    if (p.kind === "edge_connector") continue;
    const x0 = toX(p.x - p.w / 2 - 0.8), x1 = toX(p.x + p.w / 2 + 0.8);
    const y0 = toY(p.y + p.d / 2 + 0.8), y1 = toY(p.y - p.d / 2 - 0.8);
    g.strokeStyle = "rgba(235,242,248,0.55)"; g.lineWidth = 1;
    g.strokeRect(x0, Math.min(y0, y1), x1 - x0, Math.abs(y1 - y0));
  }

  g.fillStyle = "rgba(240,245,250,0.88)";
  g.font = "bold 12px monospace";
  for (const s of pcb.cad?.silkscreen || []) {
    g.fillText(s.text, toX(s.x), toY(s.y));
  }
  if (pdn) {
    g.fillStyle = "rgba(180,235,215,0.78)";
    g.font = "11px monospace";
    g.fillText(`CUDA PDN (${pdn.backend}) | Peak J=${pdn.peak_current_density} A/mm2 | ${pcb.cad?.layers || 8}L FR-4`, 18, H - 16);
  }

  const map = new THREE.CanvasTexture(cv);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  const bumpMap = new THREE.CanvasTexture(bcv);
  bumpMap.anisotropy = 4;
  return { map, bumpMap };
}

function buildPcbAssembly(pcb, pkg, grp) {
  const { map, bumpMap } = buildPcbTextures(pcb, pkg);
  const edgeMat = new THREE.MeshStandardMaterial({ color: 0x1b2b1e, roughness: 0.72, metalness: 0.08 });
  const topMat = new THREE.MeshStandardMaterial({ map, bumpMap, bumpScale: 0.06, roughness: 0.52, metalness: 0.12 });
  const board = new THREE.Mesh(
    new THREE.BoxGeometry(pcb.w, pcb.t, pcb.d),
    [edgeMat, edgeMat, topMat, edgeMat, edgeMat, edgeMat],
  );
  board.castShadow = board.receiveShadow = true;
  board.userData.pick = { type: "pcb", id: "PCB" };
  pickables.push(board);
  board.position.set(0, pcb.t / 2, 0);
  grp.add(board);

  // Internal copper plane edge bands (FR-4 multi-layer stackup visualization)
  const cuEdgeMat = new THREE.MeshStandardMaterial({ color: 0xc98242, metalness: 0.85, roughness: 0.3 });
  for (const frac of [0.25, 0.5, 0.75]) {
    const band = new THREE.Mesh(new THREE.BoxGeometry(pcb.w + 0.08, 0.05, pcb.d + 0.08), cuEdgeMat);
    band.position.set(0, pcb.t * frac, 0);
    grp.add(band);
  }

  // Copper traces with slight physical height
  const cuMat = MAT.copper();
  for (const tr of pcb.traces || []) {
    for (let i = 1; i < tr.points.length; i++) {
      const [ax, ay] = tr.points[i - 1], [bx, by] = tr.points[i];
      const len = Math.hypot(bx - ax, by - ay);
      if (len < 0.05) continue;
      const m = box(len, 0.07, tr.width, cuMat);
      m.position.set((ax + bx) / 2, pcb.t + 0.035, -(ay + by) / 2);
      m.rotation.y = Math.atan2(by - ay, bx - ax);
      grp.add(m);
    }
  }

  // Plated mounting holes (ENIG gold annular ring + dark drill bore)
  const goldMat = MAT.gold();
  const boreMat = new THREE.MeshStandardMaterial({ color: 0x0b0d12, roughness: 0.9 });
  for (const mh of pcb.cad?.mounting_holes || []) {
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(mh.pad_d / 2, mh.pad_d / 2, pcb.t + 0.08, 24), goldMat);
    ring.position.set(mh.x, pcb.t / 2, -mh.y);
    const bore = new THREE.Mesh(new THREE.CylinderGeometry(mh.drill_d / 2, mh.drill_d / 2, pcb.t + 0.14, 20), boreMat);
    bore.position.set(mh.x, pcb.t / 2, -mh.y);
    grp.add(ring, bore);
  }

  // ENIG Fiducials
  for (const fid of pcb.cad?.fiducials || []) {
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(fid.d / 2, fid.d / 2, 0.06, 16), goldMat);
    disc.position.set(fid.x, pcb.t + 0.03, -fid.y);
    grp.add(disc);
  }

  // Instanced Via Stitching Fence
  const vias = pcb.cad?.vias || [];
  if (vias.length) {
    const viaMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.5, 0.5, pcb.t + 0.1, 8), goldMat, vias.length);
    const mtx = new THREE.Matrix4();
    vias.forEach((v, i) => {
      mtx.makeScale(v.d, 1, v.d).setPosition(v.x, pcb.t / 2, -v.y);
      viaMesh.setMatrixAt(i, mtx);
    });
    viaMesh.instanceMatrix.needsUpdate = true;
    grp.add(viaMesh);
  }

  // Instanced BGA Solder Balls under the package substrate
  const balls = pcb.cad?.bga_balls || [];
  if (balls.length) {
    const solderMat = new THREE.MeshStandardMaterial({ color: 0xd8dde5, metalness: 0.94, roughness: 0.18 });
    const ballMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.24, 8, 6), solderMat, balls.length);
    const mtx = new THREE.Matrix4();
    balls.forEach((b, i) => {
      mtx.makeTranslation(b.x, pcb.t + 0.14, -b.y);
      ballMesh.setMatrixAt(i, mtx);
    });
    ballMesh.instanceMatrix.needsUpdate = true;
    grp.add(ballMesh);
  }

  // Individual PCIe Gold Fingers
  const fingers = pcb.cad?.gold_fingers || [];
  if (fingers.length) {
    const fMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 0.12, 1), goldMat, fingers.length);
    const mtx = new THREE.Matrix4();
    fingers.forEach((fg, i) => {
      mtx.makeScale(fg.w, 1, fg.d).setPosition(fg.x, pcb.t + 0.05, -fg.y);
      fMesh.setMatrixAt(i, mtx);
    });
    fMesh.instanceMatrix.needsUpdate = true;
    grp.add(fMesh);
  }

  // Detailed SMD & Power Components
  const ceramicMat = new THREE.MeshStandardMaterial({ color: 0x9c744f, roughness: 0.48, metalness: 0.08 });
  const silverCapMat = new THREE.MeshStandardMaterial({ color: 0xd4dae2, metalness: 0.9, roughness: 0.22 });
  const ferriteMat = new THREE.MeshStandardMaterial({ color: 0x3f4550, metalness: 0.35, roughness: 0.55 });
  const drmosMat = new THREE.MeshStandardMaterial({ color: 0x181b22, roughness: 0.45, metalness: 0.1 });

  for (const p of pcb.parts || []) {
    const pick = { type: "pcb_part", id: p.id, part: p };
    if (p.kind === "capacitor") {
      // SMD MLCC / Polymer cap with ceramic body + silver terminations
      const body = box(p.w * 0.7, p.h * 0.85, p.d * 0.85, ceramicMat, pick);
      body.position.set(p.x, pcb.t + p.h * 0.425, -p.y);
      const capL = box(p.w * 0.22, p.h * 0.9, p.d * 0.9, silverCapMat);
      capL.position.set(p.x - p.w * 0.36, pcb.t + p.h * 0.45, -p.y);
      const capR = box(p.w * 0.22, p.h * 0.9, p.d * 0.9, silverCapMat);
      capR.position.set(p.x + p.w * 0.36, pcb.t + p.h * 0.45, -p.y);
      grp.add(body, capL, capR);
    } else if (p.kind === "inductor") {
      const core = box(p.w, p.h * 0.88, p.d, ferriteMat, pick);
      core.position.set(p.x, pcb.t + p.h * 0.44, -p.y);
      const top = box(p.w * 0.82, p.h * 0.14, p.d * 0.82, silverCapMat);
      top.position.set(p.x, pcb.t + p.h * 0.92, -p.y);
      grp.add(core, top);
    } else if (p.kind === "power_stage") {
      const pkgBody = box(p.w, p.h, p.d, drmosMat, pick);
      pkgBody.position.set(p.x, pcb.t + p.h / 2, -p.y);
      const pad = box(p.w + 0.8, 0.12, p.d + 0.6, silverCapMat);
      pad.position.set(p.x, pcb.t + 0.06, -p.y);
      grp.add(pkgBody, pad);
    } else if (p.kind === "power_connector") {
      const shroud = box(p.w, p.h, p.d, drmosMat, pick);
      shroud.position.set(p.x, pcb.t + p.h / 2, -p.y);
      for (let px1 = -1; px1 <= 1; px1++) {
        for (let pz1 = -1; pz1 <= 1; pz1++) {
          const pin = box(1.1, p.h * 0.7, 1.1, goldMat);
          pin.position.set(p.x + px1 * 3.2, pcb.t + p.h * 0.55, -p.y + pz1 * 2.6);
          grp.add(pin);
        }
      }
      grp.add(shroud);
    } else if (p.kind !== "edge_connector") {
      const m = box(p.w, p.h, p.d, MAT.part(), pick);
      m.position.set(p.x, pcb.t + p.h / 2, -p.y);
      grp.add(m);
    }
  }
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
  // PCB (EDA CAD multi-layer board, soldermask/silkscreen PBR, vias, PTH, BGA, gold fingers)
  buildPcbAssembly(pcb, pkg, groups.pcb);
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
    if (c.kind !== "hbm_stack") {
      const lay = buildDieLayout(c, D.components, V);
      g.add(lay);
      H.comp[c.id].lay = lay;
      for (const [k, objs] of Object.entries(lay.userData.layers)) for (const o of objs) o.visible = S.lyr[k] !== false;
    }
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
    if (h.lay) h.lay.position.y = top + 0.02;
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
  setTimeout(statusBar, 0);
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
  // die layout sheets replace the flat floorplan face where the cell/metal view applies
  for (const h of Object.values(H.comp)) {
    if (!h.lay) continue;
    const on = !!S.layers.cells && ["assembly", "die", "interconnect", "airflow"].includes(mode);
    h.lay.visible = on;
    h.face.material.visible = !on;   // stays pickable underneath
  }
  applyWire();
  layerPalette(); applyLayoutLayers();
  legend();
  hud();
  $("timeline").style.display = mode === "thermal" ? "flex" : "none";
  if (mode === "thermal") drawTimeline();
}

// Layout-layer palette (die mode): per-mask-layer visibility, like a layer window in a layout editor.
function applyLayoutLayers() {
  for (const h of Object.values(H.comp)) {
    const L = h.lay?.userData.layers;
    if (L) for (const [k, objs] of Object.entries(L)) for (const o of objs) o.visible = S.lyr[k] !== false;
  }
}
function layerPalette() {
  const show = S.mode === "die" && !!S.layers.cells;
  $("lyrbox").hidden = !show;
  if (!show) return;
  $("lyrpal").innerHTML = STACK.slice().reverse().map((l) =>
    `<label class="ck"><input type="checkbox" data-k="${l.key}"${S.lyr[l.key] === false ? "" : " checked"}><i style="width:14px;height:9px;border-radius:1px;border:1px solid rgba(255,255,255,.28);background:${l.color}"></i> ${esc(l.name)}</label>`).join("");
}
$("lyrpal").onchange = (e) => { S.lyr[e.target.dataset.k] = e.target.checked; applyLayoutLayers(); };

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
  } else if (m === "die" && S.layers.cells) {
    const h = S.selected && H.comp[S.selected.id], st = h?.lay?.userData.stats;
    html = `<b>Die layout · layer stack</b><div class="lpal">${STACK.slice().reverse().map((l) => `<i style="background:${l.color}"></i><span>${esc(l.name)}</span><span></span>`).join("")}</div>
      <div class="note">Drawn from ChipLab's floorplan for each die (clusters, SRAM share, PHY edges, channels × banks, CIM share). Representative, not a GDS: each cell row stands for many.${st ? ` ${f(st.cell_rows, 0)} rows · ${f(st.signal_wires, 0)} wires · ${f(st.vias, 0)} vias · ${f(st.clock_buffers, 0)} clock buffers · ${f(st.tap_cells, 0)} taps · ${f(st.arrays, 0)} arrays shown.` : ""} Signal-wire density follows simulated cluster activity.</div>`;
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
  const selChip = S.selected && D.components.find((c) => c.id === S.selected.id);
  $("hud").innerHTML = `<b>${esc(D.design.name)}</b> v${esc(D.design.version)} · ${esc(D.design.package_type)} · ${esc(D.package.interposer.kind.replace(/_/g, " "))}<br>
    ${D.components.length} chiplets · ${f(cap)} GB · ${f(bw / 1000, 2)} TB/s · ${f(D.power.package_w)} W · peak ${f(D.thermal.hotspot.temp_c)} °C${sim ? ` · ${f(sim.throughput_tokens_per_sec)} tok/s (${esc(D.workload)})` : ""}
    <div class="hud-actions">
      <button class="btn" id="btn-eda">${S.edaOpen ? "Hide EDA Diagrams" : "EDA Diagrams"}</button>
      ${selChip ? `<button class="btn primary" id="btn-zoom-chiplet">Zoom Chiplet Up Close</button>` : ""}
    </div>`;
  $("btn-eda").onclick = () => toggleEdaDrawer();
  if (selChip) $("btn-zoom-chiplet").onclick = () => focusChipletUpClose(selChip.id);
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
renderer.domElement.addEventListener("dblclick", () => {
  if (!S.selected) return;
  if (H.comp[S.selected.id]) focusChipletUpClose(S.selected.id);
  else focusOn(S.selected.id);
});

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
  setTimeout(statusBar, 0);
  inspector();
  memoryStudio();
  if (S.data) hud();
  if (S.edaOpen) renderEdaDrawer();
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
    let html = `<h2>${esc(c.id)} <span class="badge">${esc(c.kind.replace(/_/g, " "))}</span></h2>
      <div class="row" style="margin:0 0 8px"><button class="btn primary" id="ins-zoom-chiplet">Zoom Chiplet Up Close</button></div>`;
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
    $("ins-zoom-chiplet").onclick = () => focusChipletUpClose(c.id);
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

// ---------------------------------------------------------------- Custom Memory & Architecture Explorer
const TECH_CLASSES = ["dram", "sram", "reram", "fefet", "mram", "pcm", "optical", "custom"];
const FALLBACK_MEM_PRESETS = [
  { name: "HBM3e", tech_class: "dram", node_nm: 14, stacks: 8, dies_per_stack: 12, cap_per_stack_gb: 36, io_gbps_per_pin: 9.6, bus_width_bits: 1024, energy_pj_per_bit: 3.5, read_latency_ns: 45, density_gb_per_mm2: 0.33, cim_capable: false, cim_tops_per_w: 0 },
  { name: "HBM4", tech_class: "dram", node_nm: 10, stacks: 8, dies_per_stack: 16, cap_per_stack_gb: 48, io_gbps_per_pin: 8.0, bus_width_bits: 2048, energy_pj_per_bit: 2.8, read_latency_ns: 40, density_gb_per_mm2: 0.44, cim_capable: false, cim_tops_per_w: 0 },
  { name: "3D-ReRAM", tech_class: "reram", node_nm: 22, stacks: 8, dies_per_stack: 8, cap_per_stack_gb: 64, io_gbps_per_pin: 6.4, bus_width_bits: 1024, energy_pj_per_bit: 1.4, read_latency_ns: 18, density_gb_per_mm2: 0.72, cim_capable: true, cim_tops_per_w: 18 },
  { name: "FeFET-CIM", tech_class: "fefet", node_nm: 22, stacks: 8, dies_per_stack: 8, cap_per_stack_gb: 48, io_gbps_per_pin: 6.4, bus_width_bits: 1024, energy_pj_per_bit: 0.9, read_latency_ns: 12, density_gb_per_mm2: 0.58, cim_capable: true, cim_tops_per_w: 28 },
  { name: "STT-MRAM", tech_class: "mram", node_nm: 16, stacks: 8, dies_per_stack: 8, cap_per_stack_gb: 32, io_gbps_per_pin: 8.0, bus_width_bits: 1024, energy_pj_per_bit: 1.8, read_latency_ns: 10, density_gb_per_mm2: 0.40, cim_capable: true, cim_tops_per_w: 12 },
  { name: "3D-SRAM", tech_class: "sram", node_nm: 3, stacks: 8, dies_per_stack: 4, cap_per_stack_gb: 4, io_gbps_per_pin: 16.0, bus_width_bits: 2048, energy_pj_per_bit: 0.45, read_latency_ns: 2, density_gb_per_mm2: 0.08, cim_capable: true, cim_tops_per_w: 22 },
  { name: "Optical-DRAM", tech_class: "optical", node_nm: 10, stacks: 8, dies_per_stack: 12, cap_per_stack_gb: 64, io_gbps_per_pin: 32.0, bus_width_bits: 1024, energy_pj_per_bit: 1.2, read_latency_ns: 28, density_gb_per_mm2: 0.48, cim_capable: false, cim_tops_per_w: 0 },
];

function normalizePreset(p) {
  return {
    name: p.name ?? "Custom-Mem",
    tech_class: (p.tech_class ?? "dram").toLowerCase(),
    node_nm: Number(p.node_nm ?? 14),
    stacks: Number(p.stacks ?? 8),
    dies_per_stack: Number(p.dies_per_stack ?? 12),
    cap_per_stack_gb: Number(p.cap_per_stack_gb ?? p.capacity_gb_per_stack ?? 36),
    io_gbps_per_pin: Number(p.io_gbps_per_pin ?? 9.6),
    bus_width_bits: Number(p.bus_width_bits ?? 1024),
    energy_pj_per_bit: Number(p.energy_pj_per_bit ?? 3.5),
    read_latency_ns: Number(p.read_latency_ns ?? 45),
    density_gb_per_mm2: Number(p.density_gb_per_mm2 ?? 0.33),
    cim_capable: Boolean(p.cim_capable),
    cim_tops_per_w: Number(p.cim_tops_per_w ?? 0),
  };
}

async function loadMemPresets() {
  try {
    const r = await fetch("/api/lab/custom_memory/presets");
    if (r.ok) {
      const j = await r.json();
      const list = Array.isArray(j) ? j : j.presets || [];
      if (list.length) {
        S.memPresets = list.map(normalizePreset);
        if (!S.memDraft) S.memDraft = { ...S.memPresets[0] };
        if (S.data) memoryStudio();
        return;
      }
    }
  } catch (_) { /* fallback below */ }
  if (!S.memPresets.length) {
    S.memPresets = FALLBACK_MEM_PRESETS.map(normalizePreset);
    if (!S.memDraft) S.memDraft = { ...S.memPresets[0] };
    if (S.data) memoryStudio();
  }
}

function readMemDraftFromDom(el) {
  const g = (k) => el.querySelector(`[data-cm="${k}"]`);
  return {
    name: g("name")?.value?.trim() || "Custom-Mem",
    tech_class: g("tech_class")?.value || "dram",
    node_nm: Number(g("node_nm")?.value || 14),
    stacks: Number(g("stacks")?.value || 8),
    dies_per_stack: Number(g("dies_per_stack")?.value || 12),
    cap_per_stack_gb: Number(g("cap_per_stack_gb")?.value || 36),
    io_gbps_per_pin: Number(g("io_gbps_per_pin")?.value || 9.6),
    bus_width_bits: Number(g("bus_width_bits")?.value || 1024),
    energy_pj_per_bit: Number(g("energy_pj_per_bit")?.value || 3.5),
    read_latency_ns: Number(g("read_latency_ns")?.value || 45),
    density_gb_per_mm2: Number(g("density_gb_per_mm2")?.value || 0.33),
    cim_capable: Boolean(g("cim_capable")?.checked),
    cim_tops_per_w: Number(g("cim_tops_per_w")?.value || 0),
  };
}

function renderCustomMemDiff() {
  const diffEl = $("cm-diff");
  if (!diffEl) return;
  const res = S.customMemResult;
  if (!res) { diffEl.innerHTML = ""; return; }
  const b = res.before || res.baseline || {};
  const a = res.after || res.metrics || res.custom || res.result || res;
  const rows = [
    ["TB/s", f(b.bandwidth_tbps ?? b.tb_s, 2), f(a.bandwidth_tbps ?? a.tb_s, 2)],
    ["TOPS", f(b.tops ?? b.effective_tops ?? b.peak_tops, 0), f(a.tops ?? a.effective_tops ?? a.peak_tops, 0)],
    ["tok/s", f(b.tok_s ?? b.throughput_tokens_per_sec, 1), f(a.tok_s ?? a.throughput_tokens_per_sec, 1)],
    ["Total Power W", f(b.total_power_w ?? b.power_w, 1), f(a.total_power_w ?? a.power_w, 1)],
    ["Peak Temp °C", f(b.peak_temp_c ?? b.temp_c, 1), f(a.peak_temp_c ?? a.temp_c, 1)],
    ["Bottleneck", esc(b.bottleneck || "–"), esc(a.bottleneck || "–")],
  ];
  diffEl.innerHTML = `<div class="cmp-grid">
    <span class="hdr">Metric</span><span class="hdr" style="text-align:right">Before</span><span class="hdr" style="text-align:right">After</span>
    ${rows.map(([k, bv, av]) => `<span>${esc(k)}</span><span class="val" style="color:var(--muted)">${bv}</span><span class="val" style="color:var(--accent)">${av}</span>`).join("")}
  </div>`;
}

function renderComparisonsTable() {
  const wrap = $("cm-table");
  if (!wrap) return;
  if (!S.memComparisons.length) {
    wrap.innerHTML = `<div class="empty" style="margin-top:6px">No saved runs yet. Click "Test Custom Memory" to compare architectures.</div>`;
    return;
  }
  wrap.innerHTML = `<div class="row" style="justify-content:space-between;margin-top:10px">
      <h3 style="margin:0;border:0;padding:0">Saved Architecture Comparisons</h3>
      <button class="btn" id="cm-clear" style="padding:2px 7px;font-size:10.5px">Clear</button>
    </div>
    <div class="cmp-table-wrap"><table class="cmp-table">
      <thead><tr><th>Name</th><th>TB/s</th><th>TOPS</th><th>tok/s</th><th>W</th><th>°C</th><th>Bound</th></tr></thead>
      <tbody>${S.memComparisons.map((r) => `<tr>
        <td>${esc(r.name)}</td><td>${f(r.tbps, 2)}</td><td>${f(r.tops, 0)}</td><td>${f(r.toks, 0)}</td>
        <td>${f(r.power_w, 0)}</td><td>${f(r.temp_c, 1)}</td><td>${esc(r.bottleneck)}</td>
      </tr>`).join("")}</tbody>
    </table></div>`;
  $("cm-clear").onclick = () => { S.memComparisons = []; renderComparisonsTable(); };
}

function memoryStudio() {
  const el = $("mps");
  if (!S.memPresets.length) S.memPresets = FALLBACK_MEM_PRESETS.map(normalizePreset);
  if (!S.memDraft) S.memDraft = { ...S.memPresets[0] };
  const d = S.memDraft;
  const cmNum = (label, key, val, step) => `<label for="cm-${key}">${esc(label)}</label><input id="cm-${key}" data-cm="${key}" type="number" step="${step}" value="${Number.isFinite(val) ? val : 0}">`;

  el.innerHTML = `<h2>Custom Memory &amp; Architecture Explorer</h2>
    <div class="form" id="cm-form">
      <label for="cm-preset">Preset</label>
      <select id="cm-preset">${S.memPresets.map((p, i) => `<option value="${i}"${p.name === d.name ? " selected" : ""}>${esc(p.name)}</option>`).join("")}</select>
      <label for="cm-name">Name</label>
      <input id="cm-name" data-cm="name" type="text" value="${esc(d.name)}">
      <label for="cm-tech_class">Tech Class</label>
      <select id="cm-tech_class" data-cm="tech_class">${TECH_CLASSES.map((t) => `<option value="${t}"${t === d.tech_class ? " selected" : ""}>${t}</option>`).join("")}</select>
      ${cmNum("Node (nm)", "node_nm", d.node_nm, 1)}
      ${cmNum("Stacks", "stacks", d.stacks, 1)}
      ${cmNum("Dies/Stack", "dies_per_stack", d.dies_per_stack, 1)}
      ${cmNum("Cap/Stack (GB)", "cap_per_stack_gb", d.cap_per_stack_gb, 0.5)}
      ${cmNum("I/O Gbps/Pin", "io_gbps_per_pin", d.io_gbps_per_pin, 0.1)}
      ${cmNum("Bus Width (bits)", "bus_width_bits", d.bus_width_bits, 64)}
      ${cmNum("Energy (pJ/bit)", "energy_pj_per_bit", d.energy_pj_per_bit, 0.05)}
      ${cmNum("Read Lat (ns)", "read_latency_ns", d.read_latency_ns, 0.5)}
      ${cmNum("Density (GB/mm²)", "density_gb_per_mm2", d.density_gb_per_mm2, 0.01)}
      <label for="cm-cim_capable">CIM Capable</label>
      <input id="cm-cim_capable" data-cm="cim_capable" type="checkbox"${d.cim_capable ? " checked" : ""}>
      ${cmNum("CIM TOPS/W", "cim_tops_per_w", d.cim_tops_per_w, 0.5)}
    </div>
    <div class="row"><button class="btn primary" id="cm-test">Test Custom Memory</button></div>
    <div class="msg" id="mps-msg"></div>
    <div id="cm-diff"></div>
    <div id="cm-table"></div>`;

  $("cm-form").oninput = () => { S.memDraft = readMemDraftFromDom(el); };
  $("cm-preset").onchange = (e) => {
    const p = S.memPresets[Number(e.target.value)];
    if (!p) return;
    S.memDraft = { ...p };
    memoryStudio();
  };
  $("cm-test").onclick = async () => {
    const draft = readMemDraftFromDom(el);
    S.memDraft = draft;
    const msg = $("mps-msg");
    msg.className = "msg";
    msg.textContent = "Evaluating custom memory…";
    try {
      const r = await fetch("/api/lab/custom_memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ design_id: S.data?.design?.id, workload: $("workload").value, ...draft, memory: draft }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      S.customMemResult = j;
      const a = j.after || j.metrics || j.custom || j.result || j;
      S.memComparisons.push({
        name: draft.name,
        tbps: a.bandwidth_tbps ?? a.tb_s,
        tops: a.tops ?? a.effective_tops ?? a.peak_tops,
        toks: a.tok_s ?? a.throughput_tokens_per_sec,
        power_w: a.total_power_w ?? a.power_w,
        temp_c: a.peak_temp_c ?? a.temp_c,
        bottleneck: a.bottleneck || "–",
      });
      msg.className = "msg ok";
      msg.textContent = `Evaluated ${draft.name} (${draft.tech_class})`;
      renderCustomMemDiff();
      renderComparisonsTable();
      if (S.edaOpen && S.edaTab === "roofline") renderEdaDrawer();
    } catch (e) {
      msg.className = "msg err";
      msg.textContent = e.message;
    }
  };
  renderCustomMemDiff();
  renderComparisonsTable();
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

function focusChipletUpClose(cid) {
  const h = H.comp[cid];
  if (!h || !S.data) return;
  const b = new THREE.Box3().setFromObject(h.group);
  const c = b.getCenter(new THREE.Vector3());
  const r = b.getSize(new THREE.Vector3()).length() / 2;
  const endTarget = c.clone();
  const endPos = c.clone().add(new THREE.Vector3(0.18, 0.92, 0.38).normalize().multiplyScalar(Math.max(r * 1.85, 6)));
  S.camAnim = {
    t0: performance.now(),
    dur: 450,
    p0: camera.position.clone(),
    t0v: controls.target.clone(),
    p1: endPos,
    t1v: endTarget,
  };
  openEdaDrawer("chiplet");
}

// ---------------------------------------------------------------- EDA Diagrams Drawer
function mountSvg(el, out) {
  if (typeof out === "string") el.innerHTML = out;
  else if (out instanceof Node) el.replaceChildren(out);
  else el.innerHTML = `<div class="empty">Diagram unavailable.</div>`;
}

function openEdaDrawer(tab) {
  S.edaOpen = true;
  if (tab) S.edaTab = tab;
  $("edaDrawer").hidden = false;
  for (const b of $("edaTabs").children) b.classList.toggle("on", b.dataset.tab === S.edaTab);
  renderEdaDrawer();
  if (S.data) hud();
}

function closeEdaDrawer() {
  S.edaOpen = false;
  $("edaDrawer").hidden = true;
  if (S.data) hud();
}

function toggleEdaDrawer() {
  if (S.edaOpen) closeEdaDrawer();
  else openEdaDrawer(S.edaTab);
}

function renderEdaDrawer() {
  if (!S.edaOpen || !S.data) return;
  const D = S.data, body = $("edaBody"), sub = $("edaSubtitle");
  const selComp = (S.selected && D.components.find((c) => c.id === S.selected.id)) || D.components.find((c) => c.kind === "compute_die") || D.components[0];
  try {
    if (S.edaTab === "chiplet") {
      sub.textContent = selComp ? `${selComp.id} · ${selComp.kind.replace(/_/g, " ")}` : "";
      if (!selComp) { body.innerHTML = `<div class="empty">No chiplet available.</div>`; return; }
      mountSvg(body, renderChipletFloorplanSvg(selComp, D));
    } else if (S.edaTab === "roofline") {
      sub.textContent = S.customMemResult ? `${D.workload} · custom memory overlay` : D.workload;
      mountSvg(body, renderRooflineSvg(D, S.customMemResult));
    } else if (S.edaTab === "topology") {
      sub.textContent = `${D.package.topology} · ${D.links.length} links`;
      mountSvg(body, renderTopologySvg(D));
    } else if (S.edaTab === "stackup") {
      sub.textContent = selComp ? `${selComp.id} cross-section` : D.design.package_type;
      mountSvg(body, renderStackupCrossSectionSvg(D, selComp));
    }
  } catch (e) {
    body.innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
  }
}

$("edaTabs").onclick = (e) => {
  const b = e.target.closest("button[data-tab]");
  if (!b) return;
  openEdaDrawer(b.dataset.tab);
};
$("edaClose").onclick = () => closeEdaDrawer();
(() => {
  const handle = $("edaResize"), drawer = $("edaDrawer");
  let dragY = null, startH = 290;
  handle.addEventListener("pointerdown", (ev) => {
    dragY = ev.clientY;
    startH = drawer.getBoundingClientRect().height;
    handle.setPointerCapture(ev.pointerId);
  });
  handle.addEventListener("pointermove", (ev) => {
    if (dragY === null) return;
    const h = Math.max(150, Math.min(vp.clientHeight * 0.78, startH + (dragY - ev.clientY)));
    drawer.style.height = `${Math.round(h)}px`;
  });
  const end = () => { dragY = null; };
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
})();

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
  const chipletDesigns = list.filter((d) => d.has_chiplets !== false);
  const pool = chipletDesigns.length ? chipletDesigns : list;
  if (prev && pool.some((d) => d.design_id === prev)) $("design").value = prev;
  else if (want && pool.some((d) => d.design_id === want)) $("design").value = want;
  else {
    // prefer a design with real hardware detail
    const good = pool.find((d) => /hbm|llm|accelerator|echo/i.test(d.name)) || pool[0];
    if (good) $("design").value = good.design_id;
  }
  return list;
}

async function load(keepView) {
  let id = $("design").value;
  const pool = S.designs.filter((d) => d.has_chiplets !== false);
  if (!id && pool.length) { id = pool[0].design_id; $("design").value = id; }
  if (!id) { setBusy(true, "No designs yet. Ask an agent to create one with chiplab, or use chip_design_from_template."); return; }
  setBusy(true, "Simulating package…");
  try {
    const wl = $("workload").value || "llm8_decode";
    const r = await fetch(`/api/studio/scene?design=${encodeURIComponent(id)}&workload=${encodeURIComponent(wl)}`);
    const j = await r.json();
    if (!r.ok) {
      if (pool.length > 1) {
        const next = pool.find((d) => d.design_id !== id);
        if (next) { $("design").value = next.design_id; return load(keepView); }
      }
      throw new Error(j.error || r.statusText);
    }
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
  inspector(); memoryStudio(); summary(); statusBar();
  if (S.edaOpen) renderEdaDrawer();
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
let fpsN = 0, fpsT = performance.now();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const NICE = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
function scaleBar() {                        // length of a round number of mm at the orbit target, like a layout editor ruler
  const dist = camera.position.distanceTo(controls.target), h = renderer.domElement.clientHeight || 1;
  const pxPerMm = h / (2 * dist * Math.tan((camera.fov * Math.PI) / 360));
  const L = NICE.find((n) => n * pxPerMm >= 70) || NICE[NICE.length - 1];
  $("sb-bar").style.width = `${Math.round(L * pxPerMm)}px`; $("sb-len").textContent = `${L} mm`;
  $("scalebar").hidden = false;
}
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);
  for (const a of animators) a(dt);
  if (S.camAnim) {
    const a = S.camAnim;
    const u = Math.min(1, (performance.now() - a.t0) / a.dur);
    const k = 1 - Math.pow(1 - u, 3);
    camera.position.lerpVectors(a.p0, a.p1, k);
    controls.target.lerpVectors(a.t0v, a.t1v, k);
    if (u >= 1) S.camAnim = null;
  }
  controls.update();
  renderer.render(scene3, camera);
  if (++fpsN >= 30) { const n = performance.now(); $("sb-fps").textContent = `${Math.round(fpsN * 1000 / (n - fpsT))} fps`; fpsN = 0; fpsT = n; scaleBar(); }
});
renderer.domElement.addEventListener("pointermove", (ev) => {   // live cursor position on the board plane
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), camera);
  const p = new THREE.Vector3();
  if (ray.ray.intersectPlane(groundPlane, p)) { $("cx").textContent = p.x.toFixed(2); $("cz").textContent = p.z.toFixed(2); }
});
function statusBar() {
  if (!S.data) return;
  const D = S.data, sel = S.selected;
  $("sb-led").className = "led ok"; $("sb-conn").textContent = "design loaded";
  $("sb-design").textContent = `${D.design.name} v${D.design.version}`;
  $("sb-mode").textContent = S.mode;
  $("sb-dies").textContent = String(D.components.length);
  $("sb-sel").textContent = sel ? String(sel.id) : "none";
}

resize();
loadMemPresets();
loadDesigns().then(() => load()).catch((e) => setBusy(true, e.message));
connect();
