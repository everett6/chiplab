// ChipLab GPU Studio: die layout view.
// Approach borrowed from open-source GDS viewers (Tiny Tapeout's gds_viewer / GDS2glTF, GDS3D):
// each mask layer is its own 3D sheet at its real place in the stack (diffusion and poly at
// the bottom, local interconnect, then met1..met5), colored with the familiar KLayout/sky130
// palette. Here there is no GDS: geometry is drawn from ChipLab's floorplan for the die
// (cluster count, SRAM fraction, PHY edges facing memory, channels x banks, CIM fraction),
// so it is a representative layout, not the taped-out one. Each "cell row" stands for many.
import * as THREE from "three";

// Layer stack, bottom to top: z in stack units (scaled by vertical exaggeration).
export const STACK = [
  { key: "diff", name: "Diffusion / poly (cells)", color: "#c8687a", z: 0.0 },
  { key: "li1", name: "Local interconnect (li1)", color: "#e6c35c", z: 0.02 },
  { key: "met1", name: "met1 power rails", color: "#4256c8", z: 0.06, h: 0.025 },
  { key: "met2", name: "met2 signal routing", color: "#8fb3ff", z: 0.11, h: 0.02 },
  { key: "met3", name: "met3 signal routing", color: "#57d1c9", z: 0.16, h: 0.02 },
  { key: "met4", name: "met4 power straps", color: "#3b2b8f", z: 0.22, h: 0.04 },
  { key: "met5", name: "met5 power grid", color: "#6d4fd8", z: 0.32, h: 0.05 },
  { key: "pad", name: "Bumps / pads", color: "#c9894a", z: 0.40 },
];
const C = Object.fromEntries(STACK.map((l) => [l.key, l]));

function rng(seed) {
  let h = 2166136261;
  for (const ch of String(seed)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return () => {
    h += 0x6d2b79f5;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Regions in normalized die coordinates: u across x (0 = west), v across z (0 = north).
export function regions(c, comps) {
  const it = c.internal || {};
  const out = [];
  if (it.kind === "compute") {
    const sides = new Set();
    for (const o of comps) {
      if (o === c || !["hbm_stack", "memory_chiplet", "dram_chiplet", "pim_chiplet"].includes(o.kind)) continue;
      const dx = o.x - c.x, dy = o.y - c.y;
      if (Math.abs(dx) > Math.abs(dy)) sides.add(dx < 0 ? "W" : "E"); else sides.add(dy > 0 ? "N" : "S");
    }
    const p = 0.045, m = 0.012;
    const u0 = sides.has("W") ? p : m, u1 = 1 - (sides.has("E") ? p : m);
    const v0 = sides.has("N") ? p : m, v1 = 1 - (sides.has("S") ? p : m);
    if (sides.has("W")) out.push({ kind: "phy", u0: m / 2, v0, u1: p - 0.006, v1 });
    if (sides.has("E")) out.push({ kind: "phy", u0: 1 - p + 0.006, v0, u1: 1 - m / 2, v1 });
    if (sides.has("N")) out.push({ kind: "phy", u0, v0: m / 2, u1, v1: p - 0.006 });
    if (sides.has("S")) out.push({ kind: "phy", u0, v0: 1 - p + 0.006, u1, v1: 1 - m / 2 });
    const sf = Math.min(0.6, Math.max(0.05, it.sram_fraction ?? 0.2));
    const vm = (v0 + v1) / 2, sh = (v1 - v0) * sf;
    // SRAM band as a row of macros
    const nm = Math.max(2, Math.round((u1 - u0) / Math.max(sh, 0.04) * 1.6));
    const mw = (u1 - u0) / nm;
    for (let i = 0; i < nm; i++) out.push({ kind: "sram", u0: u0 + i * mw + 0.004, v0: vm - sh / 2 + 0.004, u1: u0 + (i + 1) * mw - 0.004, v1: vm + sh / 2 - 0.004 });
    // compute clusters (standard-cell logic), above and below the SRAM band
    const n = Math.max(1, it.clusters || 8);
    const aspect = ((u1 - u0) * c.w) / Math.max(1e-6, (v1 - v0 - sh) * c.d);
    const cols = Math.max(2, Math.round(Math.sqrt(n * aspect))), rows = Math.ceil(n / cols);
    const rt = Math.ceil(rows / 2), rb = rows - rt;
    const cw = (u1 - u0) / cols;
    const ht = (vm - sh / 2 - v0) / Math.max(rt, 1), hb = (v1 - (vm + sh / 2)) / Math.max(rb, 1);
    let k = 0;
    const g = 0.006;
    for (let r = 0; r < rows; r++) {
      const top = r < rt;
      const va = top ? v0 + r * ht : vm + sh / 2 + (r - rt) * hb, hh = top ? ht : hb;
      for (let q = 0; q < cols && k < n; q++, k++) {
        out.push({ kind: "logic", u0: u0 + q * cw + g, v0: va + g, u1: u0 + (q + 1) * cw - g, v1: va + hh - g, util: it.utilization });
      }
    }
  } else {
    const ch = Math.max(1, it.channels || 1), nb = Math.max(1, it.banks_per_channel || 1);
    const cim = Math.min(0.6, it.cim_fraction || 0);
    const m = 0.02;
    if (cim) out.push({ kind: "logic", u0: m, v0: m, u1: 1 - m, v1: cim - 0.008, cim: true });
    const vb = cim || m;
    // peripheral strip (row decoders, I/O) along the south edge
    out.push({ kind: "periph", u0: m, v0: 1 - 0.07, u1: 1 - m, v1: 1 - m });
    const cw = (1 - 2 * m) / ch, bh = (1 - 0.07 - vb - 0.006) / nb;
    for (let a = 0; a < ch; a++) for (let b = 0; b < nb; b++) {
      out.push({ kind: "bank", u0: m + a * cw + 0.003, v0: vb + b * bh + 0.003, u1: m + (a + 1) * cw - 0.003, v1: vb + (b + 1) * bh - 0.003, ch: a, bank: b });
    }
  }
  return out;
}

// Base sheet: diffusion/poly/li1 drawn into a canvas, like the bottom layers of a GDS render.
function baseCanvas(c, regs, rand) {
  const W = c.w * c.d > 200 ? 4096 : 2048;
  const H = Math.max(256, Math.min(4096, Math.round(W * c.d / c.w)));
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  g.fillStyle = "#14171f"; g.fillRect(0, 0, W, H);
  // seal ring
  g.strokeStyle = "#5a5f6e"; g.lineWidth = Math.max(4, W / 400);
  g.strokeRect(g.lineWidth, g.lineWidth, W - 2 * g.lineWidth, H - 2 * g.lineWidth);
  const rowH = Math.max(5, Math.round(W / 520));
  for (const r of regs) {
    const x0 = r.u0 * W, y0 = r.v0 * H, x1 = r.u1 * W, y1 = r.v1 * H;
    if (r.kind === "logic") {
      // standard-cell rows: alternating n-well, cells of random width, li1 contacts
      for (let y = y0, row = 0; y + rowH <= y1; y += rowH, row++) {
        g.fillStyle = row % 2 ? "#251c33" : "#1c1824"; g.fillRect(x0, y, x1 - x0, rowH);
        for (let x = x0; x < x1;) {
          const w = rowH * (0.5 + rand() * 2.6);
          if (rand() < 0.08) { x += w; continue; }                      // filler / whitespace
          g.fillStyle = rand() < 0.15 ? "#a24d63" : "#c8687a";
          g.fillRect(x + 0.5, y + rowH * 0.18, Math.min(w, x1 - x) - 1, rowH * 0.64);
          g.fillStyle = C.li1.color;
          const nc = 1 + Math.floor(rand() * 3);
          for (let k = 0; k < nc; k++) g.fillRect(x + (k + 0.5) * w / (nc + 0.5), y + rowH * 0.3, Math.max(1, rowH * 0.18), rowH * 0.4);
          x += w;
        }
      }
    } else if (r.kind === "sram" || r.kind === "bank") {
      // bitcell array: dense regular grid, split into subarrays by sense-amp / wordline-driver strips
      g.fillStyle = "#2a2140"; g.fillRect(x0, y0, x1 - x0, y1 - y0);
      const pitch = Math.max(2, rowH * 0.45);
      const subR = Math.max(2, Math.round((y1 - y0) / (rowH * 10))), subC = Math.max(1, Math.round((x1 - x0) / (rowH * 18)));
      g.fillStyle = "#9c5470";
      for (let y = y0 + 2; y < y1 - 2; y += pitch) g.fillRect(x0 + 2, y, x1 - x0 - 4, pitch * 0.45);
      g.fillStyle = "rgba(230,195,92,.55)";
      for (let x = x0 + 2; x < x1 - 2; x += pitch * 2) g.fillRect(x, y0 + 2, Math.max(1, pitch * 0.3), y1 - y0 - 4);
      g.fillStyle = "#d9a24a";                                          // sense amps
      for (let s = 1; s < subR; s++) g.fillRect(x0, y0 + s * (y1 - y0) / subR - rowH * 0.4, x1 - x0, rowH * 0.8);
      g.fillStyle = "#4a8f6a";                                          // wordline drivers
      for (let s = 1; s < subC; s++) g.fillRect(x0 + s * (x1 - x0) / subC - rowH * 0.3, y0, rowH * 0.6, y1 - y0);
      g.fillStyle = "#c8687a"; g.fillRect(x0, y1 - rowH * 1.4, x1 - x0, rowH * 1.4);  // column decoder
    } else if (r.kind === "phy" || r.kind === "periph") {
      g.fillStyle = "#3a2f22"; g.fillRect(x0, y0, x1 - x0, y1 - y0);
      g.fillStyle = "#8a6a3a";
      const s = rowH * 3;
      for (let y = y0 + s / 2; y < y1; y += s) for (let x = x0 + s / 2; x < x1; x += s) g.fillRect(x - s * 0.3, y - s * 0.3, s * 0.6, s * 0.6);
    }
  }
  return cv;
}

function inst(geom, color, n, opts = {}) {
  const mat = new THREE.MeshStandardMaterial({ color, metalness: opts.metal ?? 0.65, roughness: opts.rough ?? 0.35, transparent: !!opts.opacity, opacity: opts.opacity ?? 1 });
  const m = new THREE.InstancedMesh(geom, mat, Math.max(1, n));
  m.count = 0;
  m.userData.layer = opts.layer;
  return m;
}
const _m = new THREE.Matrix4();
function put(mesh, x, y, z, sx, sy, sz) {
  if (mesh.count >= mesh.instanceMatrix.count) return;
  _m.makeScale(sx, sy, sz).setPosition(x, y, z);
  mesh.setMatrixAt(mesh.count++, _m);
}

// Build the layer stack for one die. Returns a Group whose y = 0 is the die's top surface.
export function buildDieLayout(c, comps, V) {
  const rand = rng(c.id + ":" + c.w + "x" + c.d);
  const regs = regions(c, comps);
  const w = c.w, d = c.d, zs = 0.35 * V / 6 * Math.max(1, Math.min(w, d) / 12);   // stack height scale
  const X = (u) => (u - 0.5) * w, Z = (v) => (v - 0.5) * d;
  const g = new THREE.Group();
  g.userData.layers = {};
  const add = (key, obj) => { obj.userData.layer = key; g.add(obj); (g.userData.layers[key] ||= []).push(obj); };

  // base sheet (diffusion, poly, li1)
  const tex = new THREE.CanvasTexture(baseCanvas(c, regs, rand));
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const base = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.995, d * 0.995), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.55, metalness: 0.15 }));
  base.rotation.x = -Math.PI / 2;
  base.position.y = 0.005;
  base.receiveShadow = true;
  add("diff", base);

  const unit = new THREE.BoxGeometry(1, 1, 1);
  const rowPitch = Math.max(d / 140, 0.08);                 // representative cell-row pitch (mm)
  const logic = regs.filter((r) => r.kind === "logic"), arrays = regs.filter((r) => r.kind === "sram" || r.kind === "bank");
  const rails = [];
  for (const r of logic) {
    const n = Math.floor(((r.v1 - r.v0) * d) / rowPitch) + 1;
    for (let i = 0; i < n; i++) rails.push([X(r.u0), X(r.u1), Z(r.v0) + i * rowPitch]);
  }
  // met1: horizontal VDD/VSS rails along every cell row
  const met1 = inst(unit, C.met1.color, rails.length, { layer: "met1" });
  for (const [a, b, z] of rails) put(met1, (a + b) / 2, C.met1.z * zs, z, b - a, C.met1.h * zs, rowPitch * 0.16);
  add("met1", met1);

  // met2 (vertical) / met3 (horizontal) signal routing inside logic regions, density ~ cluster activity
  const nSig = Math.min(9000, Math.round(logic.length * 160));
  const met2 = inst(unit, C.met2.color, nSig, { layer: "met2", opacity: 0.85 });
  const met3 = inst(unit, C.met3.color, nSig, { layer: "met3", opacity: 0.85 });
  for (const r of logic) {
    const k = Math.round(160 * (r.util == null ? 0.6 : 0.3 + 0.7 * Math.min(1, r.util)));
    const rw = (r.u1 - r.u0) * w, rd = (r.v1 - r.v0) * d;
    for (let i = 0; i < k; i++) {
      const x = X(r.u0) + rand() * rw, z = Z(r.v0) + rand() * rd;
      const L2 = rd * (0.05 + rand() * 0.35), L3 = rw * (0.05 + rand() * 0.35);
      put(met2, x, C.met2.z * zs, Math.min(z, Z(r.v1) - L2 / 2), rowPitch * 0.12, C.met2.h * zs, L2);
      put(met3, Math.min(X(r.u0) + rand() * rw, X(r.u1) - L3 / 2), C.met3.z * zs, Z(r.v0) + rand() * rd, L3, C.met3.h * zs, rowPitch * 0.12);
    }
  }
  add("met2", met2); add("met3", met3);

  // met4: vertical power straps across the whole die (and over SRAM / banks: macro rings)
  const strapPitch = Math.max(w / 28, rowPitch * 8);
  const nStrap = Math.floor(w / strapPitch);
  const met4 = inst(unit, C.met4.color, nStrap * 2 + arrays.length * 4, { layer: "met4", metal: 0.55 });
  for (let i = 0; i < nStrap; i++) {
    const x = -w / 2 + (i + 0.5) * strapPitch;
    put(met4, x - strapPitch * 0.08, C.met4.z * zs, 0, strapPitch * 0.1, C.met4.h * zs, d * 0.97);
    put(met4, x + strapPitch * 0.08, C.met4.z * zs, 0, strapPitch * 0.1, C.met4.h * zs, d * 0.97);
  }
  for (const r of arrays) {
    const a = X(r.u0), b = X(r.u1), p = Z(r.v0), q = Z(r.v1), t = rowPitch * 0.4;
    put(met4, (a + b) / 2, C.met4.z * zs, p, b - a, C.met4.h * zs, t);
    put(met4, (a + b) / 2, C.met4.z * zs, q, b - a, C.met4.h * zs, t);
    put(met4, a, C.met4.z * zs, (p + q) / 2, t, C.met4.h * zs, q - p);
    put(met4, b, C.met4.z * zs, (p + q) / 2, t, C.met4.h * zs, q - p);
  }
  add("met4", met4);

  // met5: horizontal top-level power grid (thick, wide)
  const gridPitch = Math.max(d / 10, strapPitch * 2.5);
  const nG = Math.max(1, Math.floor(d / gridPitch));
  const met5 = inst(unit, C.met5.color, nG, { layer: "met5", metal: 0.6, opacity: 0.7 });
  for (let i = 0; i < nG; i++) put(met5, 0, C.met5.z * zs, -d / 2 + (i + 0.5) * gridPitch, w * 0.97, C.met5.h * zs, gridPitch * 0.1);
  add("met5", met5);

  // bumps / pads on PHY and periphery regions
  const pads = regs.filter((r) => r.kind === "phy" || r.kind === "periph");
  const pitch = Math.max(Math.min(w, d) / 45, 0.25);
  let nPad = 0;
  for (const r of pads) nPad += Math.ceil(((r.u1 - r.u0) * w) / pitch) * Math.ceil(((r.v1 - r.v0) * d) / pitch);
  nPad = Math.min(nPad, 6000);
  const pad = inst(new THREE.CylinderGeometry(0.5, 0.5, 1, 10), C.pad.color, nPad, { layer: "pad", metal: 0.9, rough: 0.25 });
  for (const r of pads) {
    for (let x = X(r.u0) + pitch / 2; x < X(r.u1); x += pitch) for (let z = Z(r.v0) + pitch / 2; z < Z(r.v1); z += pitch) {
      put(pad, x, C.pad.z * zs, z, pitch * 0.55, 0.06 * zs, pitch * 0.55);
    }
  }
  if (pad.count) add("pad", pad);

  for (const o of g.children) if (o.isInstancedMesh) { o.instanceMatrix.needsUpdate = true; o.computeBoundingSphere(); o.castShadow = false; }
  g.userData.stats = {
    regions: regs.length, logic_regions: logic.length, arrays: arrays.length,
    cell_rows: rails.length, signal_wires: met2.count + met3.count, straps: met4.count, pads: pad.count,
  };
  return g;
}
