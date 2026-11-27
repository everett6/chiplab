/**
 * ChipLab GPU Studio — Professional EDA SVG Diagrams Module
 *
 * Pure, dependency-free ES module generating Cadence/Synopsys-inspired
 * dark-mode SVG diagrams for:
 *   1. Die Floorplan & Microarchitecture (`renderChipletFloorplanSvg`)
 *   2. Log-Log Roofline Performance Plot (`renderRooflineSvg`)
 *   3. Chiplet-to-Chiplet & Memory Interconnect Topology (`renderTopologySvg`)
 *   4. 2D Package & Silicon Cross-Section Stackup (`renderStackupCrossSectionSvg`)
 */

const PALETTE = {
  bg: "#0d131c",
  panel: "#111926",
  panelAlt: "#162032",
  border: "#233248",
  grid: "#1b283b",
  gridMajor: "#273852",
  text: "#e2e8f0",
  muted: "#94a3b8",
  dim: "#64748b",
  cyan: "#38bdf8",
  emerald: "#34d399",
  amber: "#f59e0b",
  rose: "#f43f5e",
  violet: "#a78bfa",
  copper: "#d97706",
  silicon: "#334155",
};

const FONT_MONO = "'JetBrains Mono', 'Fira Code', 'SFMono-Regular', Consolas, monospace";
const FONT_SANS = "'Inter', system-ui, -apple-system, sans-serif";

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function fmt(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function lerpHex(hexA, hexB, t) {
  const k = clamp(t, 0, 1);
  const pa = parseInt(hexA.slice(1), 16);
  const pb = parseInt(hexB.slice(1), 16);
  const r = Math.round(((pa >> 16) & 255) + (((pb >> 16) & 255) - ((pa >> 16) & 255)) * k);
  const g = Math.round(((pa >> 8) & 255) + (((pb >> 8) & 255) - ((pa >> 8) & 255)) * k);
  const b = Math.round((pa & 255) + ((pb & 255) - (pa & 255)) * k);
  return `#${((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1)}`;
}

function rampColor(t, stops) {
  const k = clamp(t, 0, 1);
  for (let i = 1; i < stops.length; i++) {
    const [t0, c0] = stops[i - 1];
    const [t1, c1] = stops[i];
    if (k <= t1) {
      return lerpHex(c0, c1, (k - t0) / Math.max(1e-9, t1 - t0));
    }
  }
  return stops[stops.length - 1][1];
}

const UTIL_STOPS = [
  [0.0, "#172334"],
  [0.35, "#0284c7"],
  [0.65, "#34d399"],
  [0.85, "#f59e0b"],
  [1.0, "#f43f5e"],
];

const TEMP_STOPS = [
  [0.0, "#38bdf8"],
  [0.45, "#34d399"],
  [0.75, "#f59e0b"],
  [1.0, "#f43f5e"],
];

const LOAD_STOPS = [
  [0.0, "#38bdf8"],
  [0.5, "#34d399"],
  [0.8, "#f59e0b"],
  [1.0, "#f43f5e"],
];

function svgDocument(width, height, body, extraDefs = "", extraClass = "") {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="100%" class="eda-svg ${esc(extraClass)}" role="img" style="background:${PALETTE.bg};font-family:${FONT_SANS};user-select:none;">
  <defs>
    <pattern id="eda-grid-sm" width="16" height="16" patternUnits="userSpaceOnUse">
      <path d="M 16 0 L 0 0 0 16" fill="none" stroke="${PALETTE.grid}" stroke-width="0.6"/>
    </pattern>
    <pattern id="eda-grid-lg" width="64" height="64" patternUnits="userSpaceOnUse">
      <rect width="64" height="64" fill="url(#eda-grid-sm)"/>
      <path d="M 64 0 L 0 0 0 64" fill="none" stroke="${PALETTE.gridMajor}" stroke-width="0.9"/>
    </pattern>
    <pattern id="eda-hatch" width="6" height="6" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <line x1="0" y1="0" x2="0" y2="6" stroke="${PALETTE.dim}" stroke-width="1" stroke-opacity="0.35"/>
    </pattern>
    ${extraDefs}
  </defs>
  <style>
    .eda-mono { font-family: ${FONT_MONO}; }
    .eda-node { transition: filter 0.15s ease, opacity 0.15s ease; cursor: pointer; }
    .eda-node:hover { filter: brightness(1.18); }
    .eda-link { transition: stroke-opacity 0.15s ease, stroke-width 0.15s ease; }
    .eda-link:hover { stroke-opacity: 1 !important; }
  </style>
  ${body}
</svg>`;
}

function headerBar(width, title, badges) {
  const badgeHtml = badges
    .filter((b) => b && b.value !== undefined && b.value !== null && b.value !== "")
    .map((b, idx) => {
      const color = b.color || PALETTE.cyan;
      return `<tspan fill="${PALETTE.dim}">  ${idx > 0 ? "│  " : ""}</tspan><tspan fill="${PALETTE.muted}">${esc(b.label)} </tspan><tspan fill="${color}" font-weight="600">${esc(b.value)}</tspan>`;
    })
    .join("");

  return `<g class="eda-header">
    <rect x="0" y="0" width="${width}" height="40" fill="${PALETTE.panel}" />
    <line x1="0" y1="40" x2="${width}" y2="40" stroke="${PALETTE.border}" stroke-width="1.2" />
    <rect x="12" y="11" width="4" height="18" rx="1.5" fill="${PALETTE.cyan}" />
    <text x="24" y="25" fill="${PALETTE.text}" font-size="13" font-weight="700" letter-spacing="0.4">${esc(title)}</text>
    <text x="${width - 14}" y="25" text-anchor="end" class="eda-mono" font-size="11">${badgeHtml}</text>
  </g>`;
}

function inferPhyEdges(comp, sceneData) {
  const edges = new Set();
  const internal = comp?.internal || {};
  if (Array.isArray(internal.phy_edges)) {
    for (const e of internal.phy_edges) edges.add(String(e).toUpperCase());
  }
  const comps = Array.isArray(sceneData?.components) ? sceneData.components : [];
  const isMem = (k) => ["hbm_stack", "memory_chiplet", "dram_chiplet"].includes(k);
  for (const other of comps) {
    if (!other || other.id === comp?.id) continue;
    if (!isMem(other.kind) && comps.length > 2) continue;
    const dx = num(other.x) - num(comp?.x);
    const dy = num(other.y) - num(comp?.y);
    if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) continue;
    if (Math.abs(dx) >= Math.abs(dy)) edges.add(dx < 0 ? "W" : "E");
    else edges.add(dy < 0 ? "N" : "S");
  }
  const links = Array.isArray(sceneData?.links) ? sceneData.links : [];
  for (const lk of links) {
    if (lk.from === comp?.id || lk.to === comp?.id) {
      const otherId = lk.from === comp?.id ? lk.to : lk.from;
      const other = comps.find((c) => c.id === otherId);
      if (other) {
        const dx = num(other.x) - num(comp?.x);
        const dy = num(other.y) - num(comp?.y);
        if (Math.abs(dx) >= Math.abs(dy)) edges.add(dx < 0 ? "W" : "E");
        else edges.add(dy < 0 ? "N" : "S");
      }
    }
  }
  if (edges.size === 0) {
    edges.add("W");
    edges.add("E");
  }
  return edges;
}

function niceTickStep(spanMm) {
  if (spanMm <= 5) return 1;
  if (spanMm <= 12) return 2;
  if (spanMm <= 30) return 5;
  return 10;
}

/**
 * 1. Render Cadence/Synopsys-style SVG Die Floorplan & Architecture Diagram.
 *
 * @param {object} comp - Selected component from `sceneData.components`
 * @param {object} sceneData - Full studio scene payload
 * @param {object} [opts] - Optional rendering options ({ colorBy: "utilization"|"power"|"temp", showClockTree: true })
 * @returns {string} SVG markup
 */
export function renderChipletFloorplanSvg(comp, sceneData = {}, opts = {}) {
  const W = 860;
  const H = 620;
  if (!comp) {
    return svgDocument(
      W,
      H,
      `${headerBar(W, "DIE FLOORPLAN EXPLORER", [{ label: "STATUS", value: "NO DIE SELECTED", color: PALETTE.amber }])}
       <text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${PALETTE.muted}" font-size="14">Select a chiplet to inspect its physical floorplan.</text>`
    );
  }

  const dieW = Math.max(1, num(comp.w, 20));
  const dieD = Math.max(1, num(comp.d, 16));
  const area = num(comp.area_mm2, dieW * dieD);
  const power = num(comp.power_w, 0);
  const temp = comp.temp_c;
  const util = comp.utilization;
  const internal = comp.internal || {};
  const isCompute =
    internal.kind === "compute" ||
    ["compute_die", "pim_chiplet", "io_die"].includes(comp.kind);

  const colorBy = opts.colorBy || "utilization";
  const showClock = opts.showClockTree !== false;
  const maxTemp = num(sceneData?.max_temp_c, 85);
  const ambient = num(sceneData?.thermal?.ambient_c, 25);

  // Compute viewport for die centered with space for mm rulers & legend
  const padL = 78;
  const padR = 44;
  const padT = 82;
  const padB = 64;
  const availW = W - padL - padR;
  const availH = H - padT - padB;
  const scale = Math.min(availW / dieW, availH / dieD);
  const pxW = dieW * scale;
  const pxH = dieD * scale;
  const x0 = padL + (availW - pxW) / 2;
  const y0 = padT + (availH - pxH) / 2;

  const parts = [];
  parts.push(`<rect x="0" y="40" width="${W}" height="${H - 40}" fill="url(#eda-grid-lg)" />`);

  // Header
  parts.push(
    headerBar(W, `DIE FLOORPLAN · ${comp.id || comp.chiplet_id || "DIE"}`, [
      { label: "DIM", value: `${fmt(dieW, 2)}×${fmt(dieD, 2)} mm`, color: PALETTE.text },
      { label: "AREA", value: `${fmt(area, 1)} mm²`, color: PALETTE.cyan },
      { label: "PWR", value: `${fmt(power, 1)} W`, color: PALETTE.amber },
      {
        label: "TEMP",
        value: temp != null ? `${fmt(temp, 1)} °C` : "—",
        color: temp != null && temp > maxTemp ? PALETTE.rose : PALETTE.emerald,
      },
      {
        label: "UTIL",
        value: util != null ? `${fmt(util * 100, 0)}%` : "—",
        color: PALETTE.cyan,
      },
    ])
  );

  // MM Ruler Ticks (X top & Y left)
  const stepX = niceTickStep(dieW);
  const stepY = niceTickStep(dieD);
  parts.push(`<g class="eda-rulers" stroke="${PALETTE.dim}" stroke-width="1" fill="${PALETTE.muted}" font-size="10">`);
  // X-axis ruler bar above die
  const rxY = y0 - 14;
  parts.push(`<line x1="${x0}" y1="${rxY}" x2="${x0 + pxW}" y2="${rxY}" stroke="${PALETTE.muted}" />`);
  for (let mm = 0; mm <= dieW + 1e-6; mm += stepX) {
    const tx = x0 + mm * scale;
    parts.push(`<line x1="${tx}" y1="${rxY - 5}" x2="${tx}" y2="${rxY + 3}" />`);
    parts.push(`<text x="${tx}" y="${rxY - 8}" text-anchor="middle" class="eda-mono">${fmt(mm, 0)}</text>`);
  }
  if (Math.abs(dieW % stepX) > 0.25) {
    parts.push(`<line x1="${x0 + pxW}" y1="${rxY - 5}" x2="${x0 + pxW}" y2="${rxY + 3}" />`);
    parts.push(`<text x="${x0 + pxW}" y="${rxY - 8}" text-anchor="middle" class="eda-mono">${fmt(dieW, 1)}mm</text>`);
  }
  // Y-axis ruler bar left of die
  const ryX = x0 - 14;
  parts.push(`<line x1="${ryX}" y1="${y0}" x2="${ryX}" y2="${y0 + pxH}" stroke="${PALETTE.muted}" />`);
  for (let mm = 0; mm <= dieD + 1e-6; mm += stepY) {
    const ty = y0 + mm * scale;
    parts.push(`<line x1="${ryX - 5}" y1="${ty}" x2="${ryX + 3}" y2="${ty}" />`);
    parts.push(`<text x="${ryX - 8}" y="${ty + 3.5}" text-anchor="end" class="eda-mono">${fmt(mm, 0)}</text>`);
  }
  if (Math.abs(dieD % stepY) > 0.25) {
    parts.push(`<line x1="${ryX - 5}" y1="${y0 + pxH}" x2="${ryX + 3}" y2="${y0 + pxH}" />`);
    parts.push(`<text x="${ryX - 8}" y="${y0 + pxH + 3.5}" text-anchor="end" class="eda-mono">${fmt(dieD, 1)}</text>`);
  }
  parts.push(`</g>`);

  // Die substrate & seal ring
  parts.push(
    `<rect x="${x0 - 3}" y="${y0 - 3}" width="${pxW + 6}" height="${pxH + 6}" rx="3" fill="#090e16" stroke="${PALETTE.cyan}" stroke-width="1.5" stroke-opacity="0.7" />`,
    `<rect x="${x0}" y="${y0}" width="${pxW}" height="${pxH}" fill="#121c2b" stroke="${PALETTE.border}" stroke-width="1" />`
  );

  // Pin-1 Fiducial (top-left chamfer + crosshair)
  parts.push(
    `<g class="eda-fiducial">
      <polygon points="${x0},${y0} ${x0 + 18},${y0} ${x0},${y0 + 18}" fill="${PALETTE.amber}" fill-opacity="0.85" />
      <circle cx="${x0 + 9}" cy="${y0 + 9}" r="3" fill="${PALETTE.bg}" stroke="${PALETTE.amber}" stroke-width="1" />
      <text x="${x0 + 22}" y="${y0 + 12}" fill="${PALETTE.amber}" font-size="9" font-weight="700" class="eda-mono">PIN-1</text>
    </g>`
  );

  if (isCompute) {
    const phyEdges = inferPhyEdges(comp, sceneData);
    const phyThick = clamp(Math.min(pxW, pxH) * 0.065, 16, 28);
    const ix0 = x0 + (phyEdges.has("W") ? phyThick : 8);
    const ix1 = x0 + pxW - (phyEdges.has("E") ? phyThick : 8);
    const iy0 = y0 + (phyEdges.has("N") ? phyThick : 8);
    const iy1 = y0 + pxH - (phyEdges.has("S") ? phyThick : 8);
    const innerW = Math.max(20, ix1 - ix0);
    const innerH = Math.max(20, iy1 - iy0);

    // PHY I/O Rings on connected edges
    const drawPhy = (x, y, w, h, label, vertical = false) => {
      parts.push(
        `<g class="eda-node">
          <title>${esc(label)} PHY I/O Ring</title>
          <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="2" fill="#3a2812" stroke="${PALETTE.amber}" stroke-width="1" />
          <text x="${x + w / 2}" y="${y + h / 2 + 3.5}" text-anchor="middle" fill="#fcd34d" font-size="9" font-weight="700" class="eda-mono"${vertical ? ` transform="rotate(-90 ${x + w / 2} ${y + h / 2})"` : ""}>${esc(label)}</text>
        </g>`
      );
    };
    if (phyEdges.has("W")) drawPhy(x0 + 3, iy0, phyThick - 6, innerH, "UCIe / HBM PHY", true);
    if (phyEdges.has("E")) drawPhy(ix1 + 3, iy0, phyThick - 6, innerH, "UCIe / HBM PHY", true);
    if (phyEdges.has("N")) drawPhy(ix0, y0 + 3, innerW, phyThick - 6, "UCIe / PCIe PHY", false);
    if (phyEdges.has("S")) drawPhy(ix0, iy1 + 3, innerW, phyThick - 6, "UCIe / PCIe PHY", false);

    // Central L2 / Shared SRAM Bank Strip
    const sramFrac = clamp(num(internal.sram_fraction, 0.14), 0.08, 0.36);
    const sramH = innerH * sramFrac;
    const ym = iy0 + innerH / 2;
    const sramY = ym - sramH / 2;
    const sramBanks = 8;
    const bankW = innerW / sramBanks;
    parts.push(`<g class="eda-node"><title>Shared L2 / Scratchpad SRAM (${fmt(comp.sram_mb, 0)} MB)</title>`);
    parts.push(`<rect x="${ix0}" y="${sramY}" width="${innerW}" height="${sramH}" rx="2" fill="#172a4a" stroke="${PALETTE.cyan}" stroke-width="1.2" />`);
    for (let b = 1; b < sramBanks; b++) {
      parts.push(`<line x1="${ix0 + b * bankW}" y1="${sramY}" x2="${ix0 + b * bankW}" y2="${sramY + sramH}" stroke="${PALETTE.cyan}" stroke-opacity="0.35" stroke-width="0.9" />`);
    }
    parts.push(`<text x="${ix0 + innerW / 2}" y="${ym + 4}" text-anchor="middle" fill="#bae6fd" font-size="11" font-weight="700" class="eda-mono">L2 / SHARED SRAM · ${fmt(comp.sram_mb, 0)} MB</text>`);
    parts.push(`</g>`);

    // Compute Clusters / Systolic Array Tiles (top & bottom halves)
    const nClusters = Math.max(4, Math.round(num(internal.clusters, 16)));
    const availClusterH = Math.max(20, innerH - sramH - 12);
    const aspect = innerW / availClusterH;
    const cols = Math.max(2, Math.round(Math.sqrt(nClusters * aspect)));
    const rows = Math.ceil(nClusters / cols);
    const rowsTop = Math.ceil(rows / 2);
    const rowsBot = rows - rowsTop;
    const cellW = innerW / cols;
    const topH = (sramY - 6 - iy0) / Math.max(1, rowsTop);
    const botY0 = sramY + sramH + 6;
    const botH = (iy1 - botY0) / Math.max(1, rowsBot);

    const baseUtil = clamp(num(util, 0.5), 0, 1);
    const baseTempNorm = clamp((num(temp, 55) - ambient) / Math.max(10, maxTemp - ambient), 0, 1);
    const basePdNorm = clamp(num(comp.power_density_w_mm2, 0.5) / 1.5, 0, 1);

    const drawClusterTile = (idx, cx, cy, cw, ch) => {
      // Subtle deterministic spatial gradient toward die center
      const nx = Math.abs((cx + cw / 2 - (ix0 + innerW / 2)) / (innerW / 2));
      const ny = Math.abs((cy + ch / 2 - ym) / (innerH / 2));
      const centerWeight = 1 - 0.18 * Math.hypot(nx, ny);
      let metric = baseUtil;
      if (colorBy === "temp") metric = clamp(baseTempNorm * centerWeight, 0, 1);
      else if (colorBy === "power") metric = clamp(basePdNorm * centerWeight, 0, 1);
      else metric = clamp(baseUtil * (0.9 + 0.1 * centerWeight), 0, 1);

      const fill = colorBy === "temp" ? rampColor(metric, TEMP_STOPS) : rampColor(metric, UTIL_STOPS);
      const pad = 2.5;
      const rx = cx + pad;
      const ry = cy + pad;
      const rw = Math.max(6, cw - pad * 2);
      const rh = Math.max(6, ch - pad * 2);

      parts.push(`<g class="eda-node">`);
      parts.push(`<title>Cluster C${idx} · ${esc(comp.architecture || "systolic")} (${esc(comp.precision || "FP8")})</title>`);
      parts.push(`<rect x="${rx}" y="${ry}" width="${rw}" height="${rh}" rx="2" fill="${fill}" fill-opacity="0.72" stroke="${PALETTE.emerald}" stroke-opacity="0.45" stroke-width="0.9" />`);
      // Systolic array sub-grid lines
      if (rw > 18 && rh > 16) {
        parts.push(`<line x1="${rx + rw / 2}" y1="${ry + 2}" x2="${rx + rw / 2}" y2="${ry + rh - 2}" stroke="#ffffff" stroke-opacity="0.14" stroke-width="0.7" />`);
        parts.push(`<line x1="${rx + 2}" y1="${ry + rh / 2}" x2="${rx + rw - 2}" y2="${ry + rh / 2}" stroke="#ffffff" stroke-opacity="0.14" stroke-width="0.7" />`);
      }
      if (rw > 28 && rh > 20) {
        parts.push(`<text x="${rx + rw / 2}" y="${ry + rh / 2 + 3.5}" text-anchor="middle" fill="#f8fafc" font-size="9" font-weight="600" class="eda-mono">C${idx}</text>`);
      }
      parts.push(`</g>`);
    };

    let idx = 0;
    for (let r = 0; r < rowsTop; r++) {
      for (let q = 0; q < cols && idx < nClusters; q++, idx++) {
        drawClusterTile(idx, ix0 + q * cellW, iy0 + r * topH, cellW, topH);
      }
    }
    for (let r = 0; r < rowsBot; r++) {
      for (let q = 0; q < cols && idx < nClusters; q++, idx++) {
        drawClusterTile(idx, ix0 + q * cellW, botY0 + r * botH, cellW, botH);
      }
    }

    // Clock Spine H-Tree Overlay
    if (showClock) {
      const cx = ix0 + innerW / 2;
      const cy = ym;
      const hSpan = innerW * 0.25;
      const vSpan = innerH * 0.25;
      parts.push(`<g class="eda-clock-tree" stroke="${PALETTE.cyan}" stroke-opacity="0.55" fill="none" stroke-dasharray="4 3" stroke-width="1.2">`);
      parts.push(`<line x1="${cx - hSpan}" y1="${cy}" x2="${cx + hSpan}" y2="${cy}" />`);
      parts.push(`<line x1="${cx - hSpan}" y1="${cy - vSpan}" x2="${cx - hSpan}" y2="${cy + vSpan}" />`);
      parts.push(`<line x1="${cx + hSpan}" y1="${cy - vSpan}" x2="${cx + hSpan}" y2="${cy + vSpan}" />`);
      parts.push(`<circle cx="${cx}" cy="${cy}" r="5" fill="${PALETTE.bg}" stroke="${PALETTE.cyan}" stroke-width="1.6" stroke-dasharray="none" />`);
      parts.push(`<text x="${cx + 9}" y="${cy - 8}" fill="${PALETTE.cyan}" stroke="none" font-size="9" font-weight="700" class="eda-mono">PLL / H-TREE (${fmt(comp.clock_ghz, 2)} GHz)</text>`);
      parts.push(`</g>`);
    }
  } else {
    // Memory / HBM Stack Floorplan
    const channels = Math.max(1, Math.round(num(internal.channels || comp.channels, 8)));
    const banks = Math.max(1, Math.round(num(internal.banks_per_channel || comp.banks_per_channel, 16)));
    const cimFrac = clamp(num(internal.cim_fraction, comp.kind === "dram_chiplet" ? 0.22 : 0), 0, 0.35);

    const pad = 8;
    const ix0 = x0 + pad;
    const iy0 = y0 + pad;
    const innerW = pxW - pad * 2;
    const innerH = pxH - pad * 2;
    const baseFooterH = 26;
    const cimH = cimFrac > 0 ? innerH * cimFrac : 0;
    const arrayY0 = iy0 + (cimH > 0 ? cimH + 6 : 0);
    const arrayH = Math.max(40, innerH - cimH - baseFooterH - (cimH > 0 ? 12 : 6));

    // Optional Compute-In-Memory (CIM) region
    if (cimH > 0) {
      const cu = internal.compute_util != null ? rampColor(internal.compute_util, UTIL_STOPS) : "#4c1d95";
      parts.push(`<g class="eda-node"><title>Compute-In-Memory (CIM) Logic Region</title>`);
      parts.push(`<rect x="${ix0}" y="${iy0}" width="${innerW}" height="${cimH}" rx="2" fill="${cu}" fill-opacity="0.75" stroke="${PALETTE.violet}" stroke-width="1.2" />`);
      parts.push(`<text x="${ix0 + innerW / 2}" y="${iy0 + cimH / 2 + 4}" text-anchor="middle" fill="#ede9fe" font-size="11" font-weight="700" class="eda-mono">COMPUTE-IN-MEMORY (CIM) REGION</text>`);
      parts.push(`</g>`);
    }

    // Channel Columns & Bank Grid + Sense-Amp Strips
    const chW = innerW / channels;
    const bankH = arrayH / banks;
    const chUtils = Array.isArray(internal.channel_util) ? internal.channel_util : [];
    const bankAccess = Array.isArray(internal.bank_access) ? internal.bank_access : [];
    const bankUtil = Array.isArray(internal.bank_util) ? internal.bank_util : [];

    for (let ch = 0; ch < channels; ch++) {
      const cx = ix0 + ch * chW;
      const cu = num(chUtils[ch], num(util, 0));
      // Channel header label
      parts.push(`<text x="${cx + chW / 2}" y="${arrayY0 + 10}" text-anchor="middle" fill="${PALETTE.cyan}" font-size="8.5" font-weight="700" class="eda-mono">CH${ch}</text>`);
      const gridY0 = arrayY0 + 14;
      const gridBankH = (arrayH - 14) / banks;

      for (let b = 0; b < banks; b++) {
        const by = gridY0 + b * gridBankH;
        const bu = num(bankAccess[ch]?.[b], num(bankUtil[ch]?.[b], cu));
        const fill = rampColor(clamp(bu, 0, 1), UTIL_STOPS);
        parts.push(`<g class="eda-node"><title>CH${ch} · Bank ${b} · Load ${fmt(bu * 100, 1)}%</title>`);
        parts.push(`<rect x="${cx + 1.5}" y="${by + 1}" width="${Math.max(4, chW - 3)}" height="${Math.max(3, gridBankH - 2)}" rx="1" fill="${fill}" fill-opacity="0.82" stroke="${PALETTE.border}" stroke-width="0.6" />`);
        // Sense-Amp strip across bank
        if (gridBankH >= 8) {
          parts.push(`<line x1="${cx + 2}" y1="${by + gridBankH / 2}" x2="${cx + chW - 2}" y2="${by + gridBankH / 2}" stroke="${PALETTE.amber}" stroke-opacity="0.6" stroke-width="1" />`);
        }
        parts.push(`</g>`);
      }
    }

    // Base Logic Die & TSV Array Indicator at bottom of memory stack
    const footY = iy0 + innerH - baseFooterH;
    const tsvCount = comp.tsv?.count_estimate;
    const tsvPitch = comp.tsv?.pitch_um;
    const ubPitch = comp.microbumps?.pitch_um;
    const tsvLabel = tsvCount
      ? `BASE LOGIC DIE · ${fmt(tsvCount, 0)} TSVs @ ${fmt(tsvPitch, 0)} µm · µBUMP ${fmt(ubPitch, 0)} µm`
      : `BASE LOGIC DIE · ${comp.stack_layers ? `${comp.stack_layers}-Hi STACK` : "DRAM ARRAY"} · ${fmt(comp.capacity_gb, 1)} GB`;
    parts.push(`<g class="eda-node"><title>Base Logic Die &amp; TSV Array</title>`);
    parts.push(`<rect x="${ix0}" y="${footY}" width="${innerW}" height="${baseFooterH}" rx="2" fill="url(#eda-hatch)" stroke="${PALETTE.cyan}" stroke-width="1" />`);
    parts.push(`<rect x="${ix0}" y="${footY}" width="${innerW}" height="${baseFooterH}" rx="2" fill="${PALETTE.panelAlt}" fill-opacity="0.72" />`);
    parts.push(`<text x="${ix0 + innerW / 2}" y="${footY + 16.5}" text-anchor="middle" fill="${PALETTE.cyan}" font-size="10" font-weight="700" class="eda-mono">${esc(tsvLabel)}</text>`);
    parts.push(`</g>`);
  }

  // Footer Legend Bar
  const footY = H - 24;
  parts.push(`<g font-size="10" fill="${PALETTE.muted}" class="eda-mono">`);
  parts.push(`<rect x="14" y="${footY - 9}" width="10" height="10" rx="1.5" fill="${rampColor(0.2, UTIL_STOPS)}" /><text x="28" y="${footY}">Low</text>`);
  parts.push(`<rect x="64" y="${footY - 9}" width="10" height="10" rx="1.5" fill="${rampColor(0.65, UTIL_STOPS)}" /><text x="78" y="${footY}">Nominal</text>`);
  parts.push(`<rect x="136" y="${footY - 9}" width="10" height="10" rx="1.5" fill="${rampColor(0.95, UTIL_STOPS)}" /><text x="150" y="${footY}">Hot / Saturated</text>`);
  if (isCompute) {
    parts.push(`<rect x="255" y="${footY - 9}" width="10" height="10" rx="1.5" fill="#3a2812" stroke="${PALETTE.amber}" /><text x="269" y="${footY}">PHY Ring</text>`);
    parts.push(`<rect x="332" y="${footY - 9}" width="10" height="10" rx="1.5" fill="#172a4a" stroke="${PALETTE.cyan}" /><text x="346" y="${footY}">L2 SRAM</text>`);
  } else {
    parts.push(`<line x1="255" y1="${footY - 4}" x2="269" y2="${footY - 4}" stroke="${PALETTE.amber}" stroke-width="2" /><text x="274" y="${footY}">Sense-Amp Strip</text>`);
  }
  parts.push(`<text x="${W - 14}" y="${footY}" text-anchor="end" fill="${PALETTE.dim}">COLOR BY: ${esc(colorBy.toUpperCase())}</text>`);
  parts.push(`</g>`);

  return svgDocument(W, H, parts.join("\n"));
}

/**
 * 2. Render Log-Log Roofline Performance Plot.
 *
 * @param {object} sceneData - Full studio scene payload
 * @param {object} [customEval] - Optional custom memory/config evaluation to compare
 * @returns {string} SVG markup
 */
export function renderRooflineSvg(sceneData = {}, customEval = null) {
  const W = 860;
  const H = 520;
  const comps = Array.isArray(sceneData?.components) ? sceneData.components : [];

  const peakTops = Math.max(
    1,
    comps.reduce((acc, c) => acc + num(c.peak_tops, 0), 0) || num(sceneData?.peak_tops, 1000)
  );
  const bwTbps = Math.max(
    0.05,
    comps.reduce((acc, c) => acc + num(c.bandwidth_gbps, 0), 0) / 1000 ||
      num(sceneData?.bandwidth_tbps, 3.2)
  );

  // Current operating point derived from simulated workload & utilization
  const computeComps = comps.filter((c) => num(c.peak_tops, 0) > 0);
  const avgComputeUtil =
    computeComps.length > 0
      ? computeComps.reduce((a, c) => a + num(c.utilization, 0.5), 0) / computeComps.length
      : 0.45;
  const memComps = comps.filter((c) => num(c.bandwidth_gbps, 0) > 0);
  const avgMemUtil =
    memComps.length > 0
      ? memComps.reduce((a, c) => a + num(c.utilization, 0.5), 0) / memComps.length
      : 0.5;

  const ridgeAi = peakTops / bwTbps; // FLOP / Byte
  const wlType = String(sceneData?.simulation?.workload?.type || sceneData?.workload || "").toLowerCase();
  let opAi = ridgeAi * (avgComputeUtil / Math.max(0.08, avgMemUtil));
  if (wlType.includes("decode")) opAi = Math.min(opAi, ridgeAi * 0.45);
  else if (wlType.includes("prefill") || wlType.includes("gemm")) opAi = Math.max(opAi, ridgeAi * 1.6);
  opAi = clamp(opAi, 0.25, 4096);

  const ceilingAtOp = Math.min(peakTops, bwTbps * opAi);
  const opPerf = clamp(ceilingAtOp * clamp(avgComputeUtil * 1.15, 0.15, 0.98), 0.1, ceilingAtOp);

  // Optional custom evaluation comparison
  let custBwTbps = null;
  let custPeakTops = null;
  let custAi = null;
  let custPerf = null;
  let custLabel = "Custom Memory";
  if (customEval && typeof customEval === "object") {
    custBwTbps =
      num(customEval.bandwidth_tbps, 0) ||
      num(customEval.aggregate_memory_bandwidth_tbps, 0) ||
      (num(customEval.bandwidth_gbps, 0) > 0 ? num(customEval.bandwidth_gbps) / 1000 : bwTbps);
    custPeakTops =
      num(customEval.peak_tops, 0) ||
      num(customEval.peak_compute_tops, 0) ||
      peakTops;
    custAi = num(customEval.arithmetic_intensity || customEval.ai, opAi);
    const custCeil = Math.min(custPeakTops, custBwTbps * custAi);
    const rawCustPerf =
      num(customEval.achieved_tflops, 0) ||
      num(customEval.effective_tops, 0) ||
      num(customEval.tflops, 0) ||
      Math.min(custCeil, opPerf * (custBwTbps / bwTbps));
    custPerf = clamp(rawCustPerf, 0.1, custCeil);
    custLabel = customEval.label || customEval.name || "Custom Config";
  }

  // Log-log axis bounds
  const xMinExp = -1; // 0.1 FLOP/B
  const xMaxExp = 4; // 10,000 FLOP/B
  const maxY = Math.max(peakTops, custPeakTops || 0) * 2.2;
  const yMinExp = 0; // 1 TFLOP/s
  const yMaxExp = Math.max(2, Math.ceil(Math.log10(maxY)));

  const padL = 78;
  const padR = 40;
  const padT = 68;
  const padB = 64;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const toX = (ai) => {
    const lx = clamp(Math.log10(Math.max(1e-3, ai)), xMinExp, xMaxExp);
    return padL + ((lx - xMinExp) / (xMaxExp - xMinExp)) * plotW;
  };
  const toY = (perf) => {
    const ly = clamp(Math.log10(Math.max(1e-3, perf)), yMinExp, yMaxExp);
    return padT + plotH - ((ly - yMinExp) / (yMaxExp - yMinExp)) * plotH;
  };

  const parts = [];
  parts.push(`<rect x="0" y="40" width="${W}" height="${H - 40}" fill="${PALETTE.bg}" />`);
  parts.push(
    headerBar(W, "ROOFLINE PERFORMANCE MODEL", [
      { label: "PEAK", value: `${fmt(peakTops, 0)} TOPS`, color: PALETTE.emerald },
      { label: "MEM BW", value: `${fmt(bwTbps, 2)} TB/s`, color: PALETTE.cyan },
      { label: "RIDGE", value: `${fmt(ridgeAi, 1)} FLOP/B`, color: PALETTE.amber },
      { label: "OP", value: `${fmt(opPerf, 1)} TOPS`, color: PALETTE.rose },
    ])
  );

  // Log-grid lines
  parts.push(`<g class="eda-grid">`);
  for (let e = xMinExp; e <= xMaxExp; e++) {
    const gx = toX(10 ** e);
    parts.push(`<line x1="${gx}" y1="${padT}" x2="${gx}" y2="${padT + plotH}" stroke="${PALETTE.gridMajor}" stroke-width="1" />`);
    parts.push(`<text x="${gx}" y="${padT + plotH + 18}" text-anchor="middle" fill="${PALETTE.muted}" font-size="10.5" class="eda-mono">${10 ** e}</text>`);
    if (e < xMaxExp) {
      for (const m of [2, 5]) {
        const mx = toX(m * 10 ** e);
        parts.push(`<line x1="${mx}" y1="${padT}" x2="${mx}" y2="${padT + plotH}" stroke="${PALETTE.grid}" stroke-width="0.6" />`);
      }
    }
  }
  for (let e = yMinExp; e <= yMaxExp; e++) {
    const gy = toY(10 ** e);
    parts.push(`<line x1="${padL}" y1="${gy}" x2="${padL + plotW}" y2="${gy}" stroke="${PALETTE.gridMajor}" stroke-width="1" />`);
    parts.push(`<text x="${padL - 10}" y="${gy + 4}" text-anchor="end" fill="${PALETTE.muted}" font-size="10.5" class="eda-mono">${10 ** e}</text>`);
    if (e < yMaxExp) {
      for (const m of [2, 5]) {
        const my = toY(m * 10 ** e);
        parts.push(`<line x1="${padL}" y1="${my}" x2="${padL + plotW}" y2="${my}" stroke="${PALETTE.grid}" stroke-width="0.6" />`);
      }
    }
  }
  parts.push(`<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="none" stroke="${PALETTE.border}" stroke-width="1.2" />`);
  parts.push(`</g>`);

  // Axis Titles
  parts.push(`<text x="${padL + plotW / 2}" y="${H - 16}" text-anchor="middle" fill="${PALETTE.text}" font-size="12" font-weight="600">Arithmetic Intensity (FLOP / Byte)</text>`);
  parts.push(`<text x="22" y="${padT + plotH / 2}" text-anchor="middle" fill="${PALETTE.text}" font-size="12" font-weight="600" transform="rotate(-90 22 ${padT + plotH / 2})">Performance (TFLOP/s · TOPS)</text>`);

  // Memory-bound & Compute-bound shaded regions
  const xMinVal = 10 ** xMinExp;
  const xMaxVal = 10 ** xMaxExp;
  const yMinVal = 10 ** yMinExp;
  const ridgeX = toX(ridgeAi);
  const ridgeY = toY(peakTops);
  const startAi = Math.max(xMinVal, yMinVal / bwTbps);
  const startX = toX(startAi);
  const startY = toY(bwTbps * startAi);
  const endX = toX(xMaxVal);

  // Shaded area under roofline
  parts.push(`<polygon points="${startX},${padT + plotH} ${startX},${startY} ${ridgeX},${ridgeY} ${endX},${ridgeY} ${endX},${padT + plotH}" fill="${PALETTE.cyan}" fill-opacity="0.06" />`);

  // Custom evaluation roofline (if provided)
  if (custBwTbps !== null) {
    const cRidgeAi = custPeakTops / custBwTbps;
    const cStartAi = Math.max(xMinVal, yMinVal / custBwTbps);
    const cStartX = toX(cStartAi);
    const cStartY = toY(custBwTbps * cStartAi);
    const cRidgeX = toX(cRidgeAi);
    const cRidgeY = toY(custPeakTops);
    parts.push(`<polyline points="${cStartX},${cStartY} ${cRidgeX},${cRidgeY} ${endX},${cRidgeY}" fill="none" stroke="${PALETTE.violet}" stroke-width="2" stroke-dasharray="6 4" />`);
  }

  // Baseline Roofline Curve
  parts.push(`<polyline points="${startX},${startY} ${ridgeX},${ridgeY} ${endX},${ridgeY}" fill="none" stroke="${PALETTE.cyan}" stroke-width="2.8" />`);
  // Slope & Ceiling Labels
  const slopeMidAi = Math.sqrt(startAi * ridgeAi);
  const slopeAngle = Math.atan2(ridgeY - startY, ridgeX - startX) * (180 / Math.PI);
  parts.push(`<text x="${toX(slopeMidAi)}" y="${toY(bwTbps * slopeMidAi) - 10}" text-anchor="middle" fill="${PALETTE.cyan}" font-size="11" font-weight="700" class="eda-mono" transform="rotate(${slopeAngle.toFixed(1)} ${toX(slopeMidAi)} ${toY(bwTbps * slopeMidAi) - 10})">BW SLOPE: ${fmt(bwTbps, 2)} TB/s</text>`);
  parts.push(`<text x="${(ridgeX + endX) / 2}" y="${ridgeY - 9}" text-anchor="middle" fill="${PALETTE.emerald}" font-size="11" font-weight="700" class="eda-mono">COMPUTE PEAK: ${fmt(peakTops, 0)} TOPS</text>`);

  // Ridge Point Marker
  parts.push(`<g class="eda-node"><title>Ridge Point: ${fmt(ridgeAi, 2)} FLOP/Byte @ ${fmt(peakTops, 0)} TOPS</title>`);
  parts.push(`<line x1="${ridgeX}" y1="${ridgeY}" x2="${ridgeX}" y2="${padT + plotH}" stroke="${PALETTE.amber}" stroke-dasharray="3 3" stroke-opacity="0.7" />`);
  parts.push(`<circle cx="${ridgeX}" cy="${ridgeY}" r="5.5" fill="${PALETTE.bg}" stroke="${PALETTE.amber}" stroke-width="2.2" />`);
  parts.push(`<text x="${ridgeX + 8}" y="${ridgeY + 15}" fill="${PALETTE.amber}" font-size="10" font-weight="700" class="eda-mono">RIDGE (${fmt(ridgeAi, 1)})</text>`);
  parts.push(`</g>`);

  // Current Operating Point
  const opX = toX(opAi);
  const opY = toY(opPerf);
  const regime = opAi < ridgeAi ? "Memory-Bound" : "Compute-Bound";
  parts.push(`<g class="eda-node"><title>Current Workload (${regime}): ${fmt(opAi, 2)} FLOP/B, ${fmt(opPerf, 1)} TOPS</title>`);
  parts.push(`<circle cx="${opX}" cy="${opY}" r="9" fill="${PALETTE.rose}" fill-opacity="0.22" />`);
  parts.push(`<circle cx="${opX}" cy="${opY}" r="5" fill="${PALETTE.rose}" stroke="#ffffff" stroke-width="1.5" />`);
  parts.push(`<text x="${opX + 10}" y="${opY - 6}" fill="${PALETTE.text}" font-size="10.5" font-weight="700" class="eda-mono">CURRENT (${fmt(opPerf, 1)} TOPS · ${regime})</text>`);
  parts.push(`</g>`);

  // Custom Operating Point Comparison
  if (custPerf !== null && custAi !== null) {
    const cx = toX(custAi);
    const cy = toY(custPerf);
    const deltaPct = ((custPerf - opPerf) / Math.max(1e-6, opPerf)) * 100;
    const sign = deltaPct >= 0 ? "+" : "";
    parts.push(`<g class="eda-node"><title>${esc(custLabel)}: ${fmt(custAi, 2)} FLOP/B, ${fmt(custPerf, 1)} TOPS (${sign}${fmt(deltaPct, 1)}%)</title>`);
    parts.push(`<line x1="${opX}" y1="${opY}" x2="${cx}" y2="${cy}" stroke="${PALETTE.violet}" stroke-width="1.4" stroke-dasharray="3 2" />`);
    parts.push(`<circle cx="${cx}" cy="${cy}" r="9" fill="${PALETTE.violet}" fill-opacity="0.25" />`);
    parts.push(`<circle cx="${cx}" cy="${cy}" r="5" fill="${PALETTE.violet}" stroke="#ffffff" stroke-width="1.5" />`);
    parts.push(`<text x="${cx + 10}" y="${cy + 14}" fill="#ddd6fe" font-size="10.5" font-weight="700" class="eda-mono">${esc(custLabel)} (${fmt(custPerf, 1)} TOPS · ${sign}${fmt(deltaPct, 1)}%)</text>`);
    parts.push(`</g>`);
  }

  return svgDocument(W, H, parts.join("\n"));
}

/**
 * 3. Render Interactive Chiplet-to-Chiplet & Memory Interconnect Topology Diagram.
 *
 * @param {object} sceneData - Full studio scene payload
 * @returns {string} SVG markup
 */
export function renderTopologySvg(sceneData = {}) {
  const W = 860;
  const H = 560;
  const comps = Array.isArray(sceneData?.components) ? sceneData.components : [];
  const links = Array.isArray(sceneData?.links) ? sceneData.links : [];
  const pkg = sceneData?.package || {};

  if (!comps.length) {
    return svgDocument(
      W,
      H,
      `${headerBar(W, "INTERCONNECT TOPOLOGY", [])}
       <text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${PALETTE.muted}" font-size="14">No components in scene.</text>`
    );
  }

  // Compute bounding box in mm
  const ip = pkg.interposer || {};
  const minX = ip.w ? num(ip.x, 0) - ip.w / 2 : Math.min(...comps.map((c) => num(c.x) - num(c.w, 10) / 2)) - 4;
  const maxX = ip.w ? num(ip.x, 0) + ip.w / 2 : Math.max(...comps.map((c) => num(c.x) + num(c.w, 10) / 2)) + 4;
  const minY = ip.d ? num(ip.y, 0) - ip.d / 2 : Math.min(...comps.map((c) => num(c.y) - num(c.d, 10) / 2)) - 4;
  const maxY = ip.d ? num(ip.y, 0) + ip.d / 2 : Math.max(...comps.map((c) => num(c.y) + num(c.d, 10) / 2)) + 4;
  const spanX = Math.max(10, maxX - minX);
  const spanY = Math.max(10, maxY - minY);

  const padL = 44;
  const padR = 44;
  const padT = 66;
  const padB = 46;
  const scale = Math.min((W - padL - padR) / spanX, (H - padT - padB) / spanY);
  const ox = padL + ((W - padL - padR) - spanX * scale) / 2 - minX * scale;
  const oy = padT + ((H - padT - padB) - spanY * scale) / 2 - minY * scale;

  const px = (mmX) => ox + mmX * scale;
  const py = (mmY) => oy + mmY * scale;

  const parts = [];
  parts.push(`<rect x="0" y="40" width="${W}" height="${H - 40}" fill="url(#eda-grid-lg)" />`);
  parts.push(
    headerBar(W, `INTERCONNECT TOPOLOGY · ${String(pkg.topology || "MESH").toUpperCase()}`, [
      { label: "CHIPLETS", value: String(comps.length), color: PALETTE.cyan },
      { label: "LINKS", value: String(links.length), color: PALETTE.emerald },
      { label: "INTERPOSER", value: String(ip.kind || "silicon").replace(/_/g, " "), color: PALETTE.amber },
    ])
  );

  // Interposer boundary box
  parts.push(
    `<rect x="${px(minX)}" y="${py(minY)}" width="${spanX * scale}" height="${spanY * scale}" rx="6" fill="#0f172a" fill-opacity="0.6" stroke="${PALETTE.border}" stroke-width="1.5" stroke-dasharray="6 4" />`
  );

  // Optional central switch node if any link targets "switch"
  const hasSwitch = links.some((l) => l.to === "switch" || l.from === "switch");
  const switchPt = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };

  // Draw links underneath component boxes
  const byId = new Map(comps.map((c) => [c.id, c]));
  const maxUtil = Math.max(0.01, ...links.map((l) => num(l.util, 0)));

  for (const lk of links) {
    const a = lk.from === "switch" ? switchPt : byId.get(lk.from);
    const b = lk.to === "switch" ? switchPt : byId.get(lk.to);
    if (!a || !b) continue;

    const x1 = px(num(a.x));
    const y1 = py(num(a.y));
    const x2 = px(num(b.x));
    const y2 = py(num(b.y));
    const u = clamp(num(lk.util, 0), 0, 1);
    const color = rampColor(u / Math.max(1, maxUtil), LOAD_STOPS);
    const bw = lk.bandwidth_gbps != null ? `${fmt(lk.bandwidth_gbps, 0)} GB/s` : "";
    const epb = lk.energy_pj_per_bit != null ? `${fmt(lk.energy_pj_per_bit, 2)} pJ/b` : "";
    const lat = lk.latency_ns != null ? `${fmt(lk.latency_ns, 1)} ns` : "";
    const metricsText = [bw, epb, lat].filter(Boolean).join(" · ") || `${fmt(u * 100, 0)}% load`;

    // Slight orthogonal/curved offset so parallel links don't collide
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.max(1, Math.hypot(dx, dy));
    const nx = (-dy / len) * 10;
    const ny = (dx / len) * 10;

    parts.push(`<g class="eda-link" data-link-from="${esc(lk.from)}" data-link-to="${esc(lk.to)}">`);
    parts.push(`<title>${esc(lk.from)} ↔ ${esc(lk.to)} · ${esc(metricsText)} (${fmt(u * 100, 1)}% util)</title>`);
    parts.push(`<path d="M ${x1} ${y1} Q ${mx + nx} ${my + ny} ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="${2 + u * 2.5}" stroke-opacity="0.82" />`);
    // Midpoint pill label
    const lx = mx + nx * 0.5;
    const ly = my + ny * 0.5;
    const pillW = Math.max(66, metricsText.length * 5.8 + 10);
    parts.push(`<rect x="${lx - pillW / 2}" y="${ly - 9}" width="${pillW}" height="17" rx="4" fill="${PALETTE.bg}" fill-opacity="0.92" stroke="${color}" stroke-opacity="0.65" stroke-width="0.9" />`);
    parts.push(`<text x="${lx}" y="${ly + 3}" text-anchor="middle" fill="${PALETTE.text}" font-size="8.5" class="eda-mono">${esc(metricsText)}</text>`);
    parts.push(`</g>`);
  }

  if (hasSwitch) {
    const sx = px(switchPt.x);
    const sy = py(switchPt.y);
    parts.push(`<g class="eda-node" data-comp-id="switch"><title>Interconnect Switch / NoC Hub</title>`);
    parts.push(`<circle cx="${sx}" cy="${sy}" r="18" fill="${PALETTE.panelAlt}" stroke="${PALETTE.amber}" stroke-width="2" />`);
    parts.push(`<text x="${sx}" y="${sy + 3.5}" text-anchor="middle" fill="${PALETTE.amber}" font-size="9" font-weight="700" class="eda-mono">SW</text>`);
    parts.push(`</g>`);
  }

  // Draw Component Boxes
  for (const c of comps) {
    const cw = Math.max(24, num(c.w, 10) * scale);
    const ch = Math.max(20, num(c.d, 10) * scale);
    const cx = px(num(c.x)) - cw / 2;
    const cy = py(num(c.y)) - ch / 2;
    const isMem = ["hbm_stack", "memory_chiplet", "dram_chiplet"].includes(c.kind);
    const accent = isMem ? PALETTE.cyan : c.kind === "io_die" ? PALETTE.amber : PALETTE.emerald;
    const fill = isMem ? "#132338" : c.kind === "io_die" ? "#2c2112" : "#112925";

    parts.push(`<g class="eda-node" data-comp-id="${esc(c.id)}">`);
    parts.push(`<title>${esc(c.id)} (${esc(c.kind)}) · ${fmt(c.power_w, 1)} W · ${c.temp_c != null ? `${fmt(c.temp_c, 1)} °C` : ""}</title>`);
    parts.push(`<rect x="${cx}" y="${cy}" width="${cw}" height="${ch}" rx="4" fill="${fill}" stroke="${accent}" stroke-width="1.6" />`);
    parts.push(`<text x="${cx + cw / 2}" y="${cy + ch / 2 - 2}" text-anchor="middle" fill="${PALETTE.text}" font-size="10.5" font-weight="700" class="eda-mono">${esc(c.id)}</text>`);
    const sub = isMem
      ? `${fmt(c.capacity_gb, 0)} GB · ${fmt(c.bandwidth_gbps, 0)} GB/s`
      : `${fmt(c.peak_tops, 0)} TOPS · ${fmt(c.power_w, 0)} W`;
    if (cw > 52 && ch > 32) {
      parts.push(`<text x="${cx + cw / 2}" y="${cy + ch / 2 + 12}" text-anchor="middle" fill="${accent}" font-size="8.5" class="eda-mono">${esc(sub)}</text>`);
    }
    parts.push(`</g>`);
  }

  // Footer Load Legend
  const footY = H - 16;
  parts.push(`<g font-size="10" fill="${PALETTE.muted}" class="eda-mono">`);
  parts.push(`<line x1="16" y1="${footY - 3}" x2="36" y2="${footY - 3}" stroke="${PALETTE.cyan}" stroke-width="2.5" /><text x="42" y="${footY}">Low Load</text>`);
  parts.push(`<line x1="115" y1="${footY - 3}" x2="135" y2="${footY - 3}" stroke="${PALETTE.emerald}" stroke-width="2.5" /><text x="141" y="${footY}">Nominal</text>`);
  parts.push(`<line x1="205" y1="${footY - 3}" x2="225" y2="${footY - 3}" stroke="${PALETTE.amber}" stroke-width="2.5" /><text x="231" y="${footY}">High</text>`);
  parts.push(`<line x1="275" y1="${footY - 3}" x2="295" y2="${footY - 3}" stroke="${PALETTE.rose}" stroke-width="2.5" /><text x="301" y="${footY}">Congested</text>`);
  parts.push(`<text x="${W - 16}" y="${footY}" text-anchor="end" fill="${PALETTE.dim}">HOVER NODES &amp; LINKS FOR DETAILS</text>`);
  parts.push(`</g>`);

  return svgDocument(W, H, parts.join("\n"));
}

/**
 * 4. Render 2D Package & Silicon Cross-Section Stackup Diagram.
 *
 * @param {object} sceneData - Full studio scene payload
 * @returns {string} SVG markup
 */
export function renderStackupCrossSectionSvg(sceneData = {}) {
  const W = 860;
  const H = 600;
  const pkg = sceneData?.package || {};
  const params = sceneData?.params || {};
  const pcb = sceneData?.pcb || {};
  const comps = Array.isArray(sceneData?.components) ? sceneData.components : [];
  const cooling = params.cooling || {};

  const finH = num(pkg.heatsink?.fin_h, num(cooling.fin_height_mm, 24));
  const hsBaseT = num(pkg.heatsink?.base_t, num(cooling.base_thickness_mm, 4));
  const ihsT = num(pkg.ihs?.t, num(cooling.ihs_thickness_mm, 2.0));
  const timUm = 75;
  const dieUm = num(params.compute_die_thickness_um, 775);
  const hbmLayers = Math.max(1, comps.find((c) => c.kind === "hbm_stack")?.stack_layers || 8);
  const dramUm = num(params.die_thickness_um, 50);
  const baseUm = num(params.base_die_thickness_um, 60);
  const bondUm = num(params.bond_layer_um, 20);
  const ubPitchUm = num(params.microbump_pitch_um, 55);
  const tsvPitchUm = num(params.tsv_pitch_um, 40);
  const ipT = num(pkg.interposer?.t, 0.1);
  const ipKind = String(pkg.interposer?.kind || "silicon_interposer").replace(/_/g, " ");
  const subT = num(pkg.substrate?.t, 1.2);
  const pcbT = num(pcb.t, 1.6);

  const parts = [];
  parts.push(`<rect x="0" y="40" width="${W}" height="${H - 40}" fill="url(#eda-grid-lg)" />`);
  parts.push(
    headerBar(W, "2D PACKAGE & SILICON CROSS-SECTION STACKUP", [
      { label: "PKG", value: String(sceneData?.design?.package_type || "2.5D CoWoS"), color: PALETTE.cyan },
      { label: "INTERPOSER", value: ipKind.toUpperCase(), color: PALETTE.emerald },
      { label: "TSV PITCH", value: `${fmt(tsvPitchUm, 0)} µm`, color: PALETTE.amber },
    ])
  );

  // Schematic layer heights (not strictly to linear scale so µm layers remain readable)
  const cx = 330;
  const maxLayerW = 420;
  let y = 62;

  const addCallout = (yTop, yBot, thickLabel, matLabel, color = PALETTE.cyan) => {
    const lx = 572;
    const mid = (yTop + yBot) / 2;
    parts.push(`<g class="eda-callout">`);
    parts.push(`<line x1="${lx}" y1="${yTop + 1}" x2="${lx + 10}" y2="${yTop + 1}" stroke="${color}" stroke-width="1" />`);
    parts.push(`<line x1="${lx}" y1="${yBot - 1}" x2="${lx + 10}" y2="${yBot - 1}" stroke="${color}" stroke-width="1" />`);
    parts.push(`<line x1="${lx + 5}" y1="${yTop + 1}" x2="${lx + 5}" y2="${yBot - 1}" stroke="${color}" stroke-width="1" />`);
    parts.push(`<text x="${lx + 16}" y="${mid - 1}" fill="${color}" font-size="10.5" font-weight="700" class="eda-mono">${esc(thickLabel)}</text>`);
    parts.push(`<text x="${lx + 16}" y="${mid + 12}" fill="${PALETTE.muted}" font-size="10">${esc(matLabel)}</text>`);
    parts.push(`</g>`);
  };

  // 1. Cooler / Heatsink Fins + Base
  const hsFinPx = 54;
  const hsBasePx = 18;
  const hsW = maxLayerW;
  const finCount = 15;
  const finStep = hsW / finCount;
  for (let i = 0; i < finCount; i++) {
    parts.push(`<rect x="${cx - hsW / 2 + i * finStep + 4}" y="${y}" width="${finStep - 8}" height="${hsFinPx}" rx="1.5" fill="#334155" stroke="#64748b" stroke-width="0.9" />`);
  }
  parts.push(`<rect x="${cx - hsW / 2}" y="${y + hsFinPx}" width="${hsW}" height="${hsBasePx}" rx="2" fill="#475569" stroke="#94a3b8" stroke-width="1" />`);
  parts.push(`<text x="${cx}" y="${y + hsFinPx + 13}" text-anchor="middle" fill="${PALETTE.text}" font-size="10" font-weight="700" class="eda-mono">HEATSINK (${fmt(pkg.heatsink?.fins || cooling.fin_count || 28, 0)} FINS)</text>`);
  addCallout(y, y + hsFinPx + hsBasePx, `${fmt(finH, 1)} mm fins + ${fmt(hsBaseT, 1)} mm base`, "Al / Cu Vapor-Chamber Heatsink", "#94a3b8");
  y += hsFinPx + hsBasePx + 2;

  // 2. TIM1 / TIM2 + IHS
  const timTopPx = 8;
  parts.push(`<rect x="${cx - 170}" y="${y}" width="340" height="${timTopPx}" fill="${PALETTE.cyan}" fill-opacity="0.35" stroke="${PALETTE.cyan}" stroke-width="0.8" />`);
  y += timTopPx + 1;

  const ihsPx = 22;
  const ihsW = 350;
  parts.push(`<rect x="${cx - ihsW / 2}" y="${y}" width="${ihsW}" height="${ihsPx}" rx="2" fill="#78350f" stroke="${PALETTE.amber}" stroke-width="1.1" />`);
  parts.push(`<text x="${cx}" y="${y + 15}" text-anchor="middle" fill="#fde68a" font-size="10" font-weight="700" class="eda-mono">INTEGRATED HEAT SPREADER (IHS)</text>`);
  addCallout(y - timTopPx - 1, y + ihsPx, `${fmt(ihsT, 2)} mm IHS + ${timUm} µm TIM`, "Ni-Plated Cu Lid · Indium/Polymer TIM", PALETTE.amber);
  y += ihsPx + 2;

  // TIM1 between IHS and Dies
  const tim1Px = 8;
  parts.push(`<rect x="${cx - 150}" y="${y}" width="300" height="${tim1Px}" fill="${PALETTE.cyan}" fill-opacity="0.45" stroke="${PALETTE.cyan}" stroke-width="0.8" />`);
  y += tim1Px + 2;

  // 3. Active Dies (Compute Die + HBM Stack side-by-side) + Microbumps
  const dieZoneH = 74;
  const compDieW = 148;
  const hbmW = 112;
  const compX = cx - compDieW - 14;
  const hbmX = cx + 18;

  // Compute die
  parts.push(`<g class="eda-node"><title>Active Compute Die (${dieUm} µm)</title>`);
  parts.push(`<rect x="${compX}" y="${y + 10}" width="${compDieW}" height="${dieZoneH - 10}" rx="2" fill="#064e3b" stroke="${PALETTE.emerald}" stroke-width="1.3" />`);
  parts.push(`<text x="${compX + compDieW / 2}" y="${y + 40}" text-anchor="middle" fill="#a7f3d0" font-size="10.5" font-weight="700" class="eda-mono">COMPUTE DIE</text>`);
  parts.push(`<text x="${compX + compDieW / 2}" y="${y + 55}" text-anchor="middle" fill="${PALETTE.emerald}" font-size="9" class="eda-mono">${fmt(dieUm, 0)} µm Si</text>`);
  parts.push(`</g>`);

  // HBM Stack (Base die + stacked DRAM dies)
  const visLayers = Math.min(8, hbmLayers);
  const basePx = 12;
  const dramPx = (dieZoneH - basePx - 4) / visLayers;
  parts.push(`<g class="eda-node"><title>HBM Stack (${hbmLayers}-Hi DRAM + Base Logic Die)</title>`);
  for (let i = 0; i < visLayers; i++) {
    const ly = y + i * dramPx;
    parts.push(`<rect x="${hbmX}" y="${ly}" width="${hbmW}" height="${Math.max(4, dramPx - 1.5)}" rx="1" fill="#1e3a8a" stroke="${PALETTE.cyan}" stroke-width="0.8" />`);
  }
  parts.push(`<rect x="${hbmX}" y="${y + dieZoneH - basePx}" width="${hbmW}" height="${basePx}" rx="1.5" fill="#172554" stroke="${PALETTE.cyan}" stroke-width="1.1" />`);
  parts.push(`<text x="${hbmX + hbmW / 2}" y="${y + dieZoneH / 2 - 4}" text-anchor="middle" fill="#e0f2fe" font-size="9.5" font-weight="700" class="eda-mono">${hbmLayers}-Hi HBM</text>`);
  parts.push(`<text x="${hbmX + hbmW / 2}" y="${y + dieZoneH - 2.5}" text-anchor="middle" fill="${PALETTE.cyan}" font-size="7.5" class="eda-mono">BASE DIE (${baseUm}µm)</text>`);
  parts.push(`</g>`);
  addCallout(y, y + dieZoneH, `${fmt(dieUm, 0)} µm Si · ${hbmLayers}×${dramUm} µm + ${bondUm} µm`, "Active Logic & HBM3E DRAM Stack", PALETTE.emerald);
  y += dieZoneH + 1;

  // Microbumps
  const ubPx = 12;
  for (let bx = compX + 8; bx < compX + compDieW - 4; bx += 14) {
    parts.push(`<circle cx="${bx}" cy="${y + ubPx / 2}" r="3.2" fill="${PALETTE.amber}" stroke="#fde68a" stroke-width="0.7" />`);
  }
  for (let bx = hbmX + 8; bx < hbmX + hbmW - 4; bx += 14) {
    parts.push(`<circle cx="${bx}" cy="${y + ubPx / 2}" r="3.2" fill="${PALETTE.amber}" stroke="#fde68a" stroke-width="0.7" />`);
  }
  addCallout(y, y + ubPx, `Pitch ${fmt(ubPitchUm, 0)} µm`, "SnAg Microbumps + Underfill", PALETTE.amber);
  y += ubPx + 1;

  // 4. Silicon Interposer + TSVs
  const ipPx = 28;
  const ipW = 340;
  parts.push(`<rect x="${cx - ipW / 2}" y="${y}" width="${ipW}" height="${ipPx}" rx="2" fill="#1e293b" stroke="${PALETTE.cyan}" stroke-width="1.2" />`);
  for (let tx = cx - ipW / 2 + 16; tx < cx + ipW / 2 - 12; tx += 18) {
    parts.push(`<rect x="${tx}" y="${y + 2}" width="3" height="${ipPx - 4}" fill="${PALETTE.copper}" fill-opacity="0.85" />`);
  }
  parts.push(`<rect x="${cx - 115}" y="${y + 6}" width="230" height="16" rx="3" fill="${PALETTE.bg}" fill-opacity="0.85" />`);
  parts.push(`<text x="${cx}" y="${y + 17.5}" text-anchor="middle" fill="${PALETTE.cyan}" font-size="9.5" font-weight="700" class="eda-mono">INTERPOSER + Cu TSVs (${fmt(ipT * 1000, 0)} µm)</text>`);
  addCallout(y, y + ipPx, `${fmt(ipT * 1000, 0)} µm · TSV ${fmt(tsvPitchUm, 0)} µm`, `${ipKind} + Cu TSVs`, PALETTE.cyan);
  y += ipPx + 2;

  // C4 Bumps
  const c4Px = 12;
  for (let bx = cx - ipW / 2 + 16; bx < cx + ipW / 2 - 12; bx += 22) {
    parts.push(`<circle cx="${bx}" cy="${y + c4Px / 2}" r="4" fill="#94a3b8" stroke="#cbd5e1" stroke-width="0.7" />`);
  }
  y += c4Px + 1;

  // 5. Package Substrate + BGA Balls
  const subPx = 36;
  const subW = 380;
  parts.push(`<rect x="${cx - subW / 2}" y="${y}" width="${subW}" height="${subPx}" rx="2" fill="#14532d" stroke="${PALETTE.emerald}" stroke-width="1.1" />`);
  for (let k = 1; k <= 3; k++) {
    parts.push(`<line x1="${cx - subW / 2 + 6}" y1="${y + (subPx * k) / 4}" x2="${cx + subW / 2 - 6}" y2="${y + (subPx * k) / 4}" stroke="${PALETTE.copper}" stroke-opacity="0.65" stroke-width="1" />`);
  }
  parts.push(`<text x="${cx}" y="${y + 22}" text-anchor="middle" fill="#dcfce7" font-size="10" font-weight="700" class="eda-mono">ABF PACKAGE SUBSTRATE (${fmt(subT, 2)} mm)</text>`);
  addCallout(y, y + subPx, `${fmt(subT, 2)} mm + BGA`, "ABF Multi-Layer Organic + SAC305 BGA", PALETTE.emerald);
  y += subPx + 2;

  const bgaPx = 16;
  for (let bx = cx - subW / 2 + 18; bx < cx + subW / 2 - 14; bx += 24) {
    parts.push(`<circle cx="${bx}" cy="${y + bgaPx / 2}" r="5.5" fill="#cbd5e1" stroke="#64748b" stroke-width="1" />`);
  }
  y += bgaPx + 2;

  // 6. Multi-Layer PCB
  const pcbPx = 44;
  const pcbW = 440;
  parts.push(`<rect x="${cx - pcbW / 2}" y="${y}" width="${pcbW}" height="${pcbPx}" rx="2" fill="#052e16" stroke="#22c55e" stroke-width="1.2" />`);
  for (let k = 1; k <= 5; k++) {
    parts.push(`<line x1="${cx - pcbW / 2 + 8}" y1="${y + (pcbPx * k) / 6}" x2="${cx + pcbW / 2 - 8}" y2="${y + (pcbPx * k) / 6}" stroke="${PALETTE.amber}" stroke-opacity="0.45" stroke-width="1" />`);
  }
  parts.push(`<text x="${cx}" y="${y + 26}" text-anchor="middle" fill="#bbf7d0" font-size="10.5" font-weight="700" class="eda-mono">MULTI-LAYER SYSTEM PCB (${fmt(pcbT, 2)} mm)</text>`);
  addCallout(y, y + pcbPx, `${fmt(pcbT, 2)} mm`, "FR-4 High-Tg 12+ Layer PCB", "#22c55e");

  return svgDocument(W, H, parts.join("\n"));
}
