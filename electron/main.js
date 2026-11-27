"use strict";

const { app, BrowserWindow, Menu, clipboard, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const os = require("os");
const path = require("path");

const HOST = "127.0.0.1";
const PORT = Number(process.env.CHIPLAB_PORT || 8770);
const MCP_PORT = Number(process.env.CHIPLAB_MCP_PORT || 8000);
const BASE_URL = `http://${HOST}:${PORT}`;
const STARTUP_TIMEOUT_MS = 25000;
const POLL_INTERVAL_MS = 250;

// Linux Wayland/NVIDIA + SwiftShader WebGL2 compatibility.
// NOTE: --ozone-platform=x11 is required on GNOME/Wayland with NVIDIA: the
// default Wayland Ozone backend kills the GPU process (exit code 15) and the
// window never appears. Forcing XWayland is the reliable path.
if (process.platform === "linux") {
  app.commandLine.appendSwitch("no-sandbox");
  app.commandLine.appendSwitch("disable-gpu-sandbox");
  app.commandLine.appendSwitch("enable-unsafe-swiftshader");
  app.commandLine.appendSwitch("disable-dev-shm-usage");
  app.commandLine.appendSwitch("ozone-platform", "x11");
  app.commandLine.appendSwitch("disable-features", "WaylandWindowDecorations");
  app.commandLine.appendSwitch("class", "ChipLab");
}

let mainWindow = null;
let backendProc = null;
let mcpProc = null;
let tunnelProc = null;
let mcpTunnelUrl = null;
let mcpTunnelStatus = "starting";
let resolvedPython = null;
let isQuitting = false;
let ownsBackend = false;
const backendLogs = [];

// Single-instance: clicking the desktop icon twice must focus the existing
// window instead of spawning a second (blank) app that fights over the port.
const gotInstanceLock = app.requestSingleInstanceLock();
if (!gotInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    } else {
      createWindow();
    }
  });
}

function logPath() {
  const dir = path.join(os.homedir(), ".local", "state", "chiplab");
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch (_) {
    return null;
  }
  return path.join(dir, "electron.log");
}

function logLine(line) {
  appendLog(line);
  const target = logPath();
  if (!target) return;
  try {
    fs.appendFileSync(target, `[${new Date().toISOString()}] ${line}\n`);
  } catch (_) {
    // Logging must never break startup
  }
}

function appendLog(chunk) {
  const lines = String(chunk).split(/\r?\n/).filter(Boolean);
  for (const line of lines) {
    backendLogs.push(line);
    if (backendLogs.length > 80) backendLogs.shift();
  }
}

function getRepoRoot() {
  return path.resolve(__dirname, "..");
}

function getPythonSrcDir() {
  const devSrc = path.join(getRepoRoot(), "src");
  if (fs.existsSync(devSrc)) return devSrc;
  if (process.resourcesPath) {
    const resSrc = path.join(process.resourcesPath, "src");
    if (fs.existsSync(resSrc)) return resSrc;
  }
  return devSrc;
}

function discoverPython() {
  const candidates = [
    process.env.CHIPLAB_PYTHON,
    path.join(getRepoRoot(), ".venv", "bin", "python"),
    path.join(getRepoRoot(), ".venv", "bin", "python3"),
    "/home/everett/miniconda3/bin/python3",
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch (_) {
      // Ignore permission/stat errors and continue fallback chain
    }
  }
  return "python3";
}

function ensureOllamaRunning() {
  if (process.platform !== "linux" && process.platform !== "darwin") return;
  const ollamaBin = fs.existsSync("/usr/local/bin/ollama")
    ? "/usr/local/bin/ollama"
    : (fs.existsSync(path.join(os.homedir(), ".local", "bin", "ollama"))
      ? path.join(os.homedir(), ".local", "bin", "ollama")
      : null);
  if (!ollamaBin) return;
  const probe = http.get("http://127.0.0.1:11434/api/tags", { timeout: 800 }, (res) => {
    res.resume();
  });
  probe.on("error", () => {
    try {
      spawn(ollamaBin, ["serve"], { detached: true, stdio: "ignore" }).unref();
    } catch (_) {
      // Ollama not started
    }
  });
}

async function startBackend() {
  ensureOllamaRunning();

  // Reuse an already-healthy backend instead of failing with EADDRINUSE.
  if (await checkStatusOnce()) {
    logLine(`Reusing backend already listening on ${BASE_URL}`);
    ownsBackend = false;
    backendProc = null;
    return;
  }

  resolvedPython = discoverPython();
  const srcDir = getPythonSrcDir();
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH
    ? `${srcDir}${path.delimiter}${env.PYTHONPATH}`
    : srcDir;
  env.PYTHONUNBUFFERED = "1";

  const repoRoot = getRepoRoot();
  const cwd = fs.existsSync(path.join(repoRoot, "src", "chiplab"))
    ? repoRoot
    : app.getPath("userData");

  if (!env.CHIPLAB_DB_PATH && cwd !== repoRoot) {
    env.CHIPLAB_DB_PATH = path.join(app.getPath("userData"), "chiplab_experiments.db");
  }

  const args = [
    "-m",
    "chiplab.app.server",
    "--host",
    HOST,
    "--port",
    String(PORT),
  ];

  logLine(`Starting backend: ${resolvedPython} ${args.join(" ")} (cwd=${cwd})`);

  backendProc = spawn(resolvedPython, args, {
    cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  ownsBackend = true;

  backendProc.stdout.on("data", logLine);
  backendProc.stderr.on("data", logLine);

  backendProc.on("error", (err) => {
    logLine(`Failed to spawn Python (${resolvedPython}): ${err.message}`);
  });

  backendProc.on("exit", (code, signal) => {
    logLine(`Backend exited (code=${code}, signal=${signal})`);
    backendProc = null;
    if (!isQuitting && ownsBackend) {
      // The backend died underneath us (crash, OOM, or an external kill).
      // Leaving the window on a dead page looks exactly like "the app won't
      // open", so say so plainly and try to come back on its own.
      logLine("Backend died while app was running — scheduling restart");
      diagnosticPage(
        "ChipLab backend stopped",
        `The Python backend exited (code=${code}, signal=${signal}).\n\n` +
          `It will restart automatically in a few seconds.\n\nRecent logs:\n${backendLogs.slice(-20).join("\n") || "(none)"}`
      );
      scheduleBackendRestart();
    }
  });
}

let restartTimer = null;
let restartAttempts = 0;

function scheduleBackendRestart() {
  if (restartTimer || isQuitting) return;
  restartAttempts += 1;
  const delay = Math.min(1000 * 2 ** Math.min(restartAttempts, 4), 15000);
  restartTimer = setTimeout(async () => {
    restartTimer = null;
    if (isQuitting) return;
    logLine(`Restarting backend (attempt ${restartAttempts})`);
    await startBackend();
    const ok = await waitForBackend(20000);
    if (ok) {
      restartAttempts = 0;
      logLine("Backend restarted successfully");
      loadRoute("/");
    } else {
      scheduleBackendRestart();
    }
  }, delay);
  restartTimer.unref?.();
}

function stopBackend() {
  if (!backendProc || !ownsBackend) return;
  const proc = backendProc;
  backendProc = null;
  try {
    proc.kill("SIGTERM");
  } catch (_) {
    return;
  }
  setTimeout(() => {
    try {
      if (!proc.killed) proc.kill("SIGKILL");
    } catch (_) {
      // Already exited
    }
  }, 2000).unref();
}

function discoverCloudflared() {
  const candidates = [
    process.env.CHIPLAB_CLOUDFLARED,
    path.join(os.homedir(), ".local", "bin", "cloudflared"),
    "/usr/local/bin/cloudflared",
    "/usr/bin/cloudflared",
  ].filter(Boolean);
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c;
    } catch (_) {
      // ignore
    }
  }
  return null;
}

function writeTunnelStateFile(url) {
  const dir = path.join(os.homedir(), ".local", "state", "chiplab");
  try {
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, "mcp_tunnel_url.txt");
    if (url) {
      fs.writeFileSync(file, url + "\n");
    } else if (fs.existsSync(file)) {
      fs.unlinkSync(file);
    }
  } catch (_) {
    // non-fatal
  }
}

function stopMcpTunnel() {
  mcpTunnelUrl = null;
  mcpTunnelStatus = "stopped";
  writeTunnelStateFile(null);
  for (const proc of [tunnelProc, mcpProc]) {
    if (!proc) continue;
    try {
      proc.kill("SIGTERM");
    } catch (_) {
      // ignore
    }
    setTimeout(() => {
      try {
        if (!proc.killed) proc.kill("SIGKILL");
      } catch (_) {
        // ignore
      }
    }, 1500).unref();
  }
  tunnelProc = null;
  mcpProc = null;
  if (app.isReady()) buildMenu();
}

function startMcpServerForHost(tunnelHost) {
  if (isQuitting || mcpProc) return;
  const py = resolvedPython || discoverPython();
  const srcDir = getPythonSrcDir();
  const repoRoot = getRepoRoot();
  const cwd = fs.existsSync(path.join(repoRoot, "src", "chiplab"))
    ? repoRoot
    : app.getPath("userData");

  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH
    ? `${srcDir}${path.delimiter}${env.PYTHONPATH}`
    : srcDir;
  env.PYTHONUNBUFFERED = "1";
  if (!env.CHIPLAB_DB_PATH) {
    env.CHIPLAB_DB_PATH = cwd === repoRoot
      ? path.join(repoRoot, "chiplab_experiments.db")
      : path.join(app.getPath("userData"), "chiplab_experiments.db");
  }

  const args = [
    "-m",
    "chiplab.mcp_server",
    "--transport",
    "streamable-http",
    "--host",
    HOST,
    "--port",
    String(MCP_PORT),
    "--allowed-host",
    tunnelHost,
  ];

  logLine(`Starting MCP HTTP server: ${py} ${args.join(" ")}`);
  mcpProc = spawn(py, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"] });
  mcpProc.stdout.on("data", (d) => logLine(`[mcp] ${String(d).trim()}`));
  mcpProc.stderr.on("data", (d) => logLine(`[mcp] ${String(d).trim()}`));
  mcpProc.on("error", (err) => logLine(`MCP server spawn error: ${err.message}`));
  mcpProc.on("exit", (code, signal) => {
    logLine(`MCP server exited (code=${code}, signal=${signal})`);
    mcpProc = null;
  });
}

function startMcpTunnel() {
  if (process.env.CHIPLAB_DISABLE_TUNNEL === "1" || isQuitting) {
    mcpTunnelStatus = "disabled";
    return;
  }
  const cfBin = discoverCloudflared();
  if (!cfBin) {
    mcpTunnelStatus = "cloudflared not found";
    logLine("MCP tunnel skipped: cloudflared binary not found in ~/.local/bin or PATH");
    if (app.isReady()) buildMenu();
    return;
  }
  stopMcpTunnel();
  mcpTunnelStatus = "connecting";
  if (app.isReady()) buildMenu();

  logLine(`Starting cloudflared tunnel -> http://${HOST}:${MCP_PORT}`);
  tunnelProc = spawn(
    cfBin,
    ["tunnel", "--url", `http://${HOST}:${MCP_PORT}`, "--no-autoupdate"],
    { stdio: ["ignore", "pipe", "pipe"] }
  );

  const handleTunnelOutput = (chunk) => {
    const text = String(chunk);
    for (const line of text.split(/\r?\n/).filter(Boolean)) {
      if (line.includes("ERR ") || line.includes("failed") || line.includes("error")) {
        logLine(`[cloudflared] ${line}`);
      }
    }
    const match = text.match(/https:\/\/([a-zA-Z0-9.-]+\.trycloudflare\.com)/);
    if (match && !mcpTunnelUrl) {
      const host = match[1];
      mcpTunnelUrl = `https://${host}/mcp`;
      mcpTunnelStatus = "online";
      writeTunnelStateFile(mcpTunnelUrl);
      logLine(`MCP tunnel online: ${mcpTunnelUrl}`);
      startMcpServerForHost(host);
      if (app.isReady()) buildMenu();
    }
  };

  tunnelProc.stdout.on("data", handleTunnelOutput);
  tunnelProc.stderr.on("data", handleTunnelOutput);
  tunnelProc.on("error", (err) => {
    mcpTunnelStatus = `error: ${err.message}`;
    logLine(`cloudflared error: ${err.message}`);
    if (app.isReady()) buildMenu();
  });
  tunnelProc.on("exit", (code, signal) => {
    logLine(`cloudflared exited (code=${code}, signal=${signal})`);
    tunnelProc = null;
    if (!isQuitting) {
      mcpTunnelUrl = null;
      mcpTunnelStatus = "offline";
      writeTunnelStateFile(null);
      if (app.isReady()) buildMenu();
    }
  });
}

function checkStatusOnce() {
  return new Promise((resolve) => {
    const req = http.get(`${BASE_URL}/api/status`, { timeout: 1500 }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(timeoutMs = STARTUP_TIMEOUT_MS) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await checkStatusOnce()) {
      return true;
    }
    if (!backendProc) {
      return false;
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  return false;
}

function normalizeRoute(route) {
  if (typeof route !== "string" || !route.startsWith("/")) {
    return "/agent";
  }
  return route;
}

function loadRoute(route) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const safeRoute = normalizeRoute(route);
  mainWindow.webContents.session.clearCache().catch(() => {}).finally(() => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.loadURL(`${BASE_URL}${safeRoute}`, { extraHeaders: "pragma: no-cache\nCache-Control: no-cache\n" });
    }
  });
}

function buildMenu() {
  const template = [
    {
      label: "File",
      submenu: [
        { role: "quit", label: "Quit ChipLab" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" },
      ],
    },
    {
      label: "View",
      submenu: [
        {
          label: "Live Studio",
          accelerator: "Ctrl+1",
          click: () => loadRoute("/"),
        },
        {
          label: "Built-in AI Agent",
          accelerator: "Ctrl+2",
          click: () => loadRoute("/agent"),
        },
        {
          label: "ASIC Flow",
          accelerator: "Ctrl+3",
          click: () => loadRoute("/asic"),
        },
        { type: "separator" },
        {
          label: "Reload",
          accelerator: "Ctrl+R",
          click: () => {
            if (mainWindow && !mainWindow.isDestroyed()) {
              const current = mainWindow.webContents.getURL();
              if (!current || current.startsWith("data:")) {
                loadRoute("/");
              } else {
                mainWindow.webContents.reload();
              }
            }
          },
        },
        {
          label: "DevTools",
          accelerator: "Ctrl+Shift+I",
          click: () => {
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.toggleDevTools();
            }
          },
        },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "MCP Tunnel",
      submenu: [
        {
          label: mcpTunnelUrl
            ? `Copy ChatGPT MCP URL (${mcpTunnelUrl})`
            : `ChatGPT MCP Tunnel (${mcpTunnelStatus})`,
          enabled: Boolean(mcpTunnelUrl),
          click: () => {
            if (mcpTunnelUrl) clipboard.writeText(mcpTunnelUrl);
          },
        },
        {
          label: "Show ChatGPT MCP URL…",
          click: () => {
            dialog.showMessageBox(mainWindow || undefined, {
              type: "info",
              title: "ChatGPT MCP Tunnel",
              message: mcpTunnelUrl ? "MCP Tunnel is Online" : `MCP Tunnel Status: ${mcpTunnelStatus}`,
              detail: mcpTunnelUrl
                ? `Paste this URL into ChatGPT (Settings → Apps & Connectors → Developer mode):\n\n${mcpTunnelUrl}`
                : "Tunnel URL is not available yet. Check ~/.local/state/chiplab/electron.log.",
              buttons: mcpTunnelUrl ? ["Copy URL", "OK"] : ["OK"],
            }).then(({ response }) => {
              if (mcpTunnelUrl && response === 0) clipboard.writeText(mcpTunnelUrl);
            });
          },
        },
        { type: "separator" },
        {
          label: "Restart MCP Tunnel",
          click: () => startMcpTunnel(),
        },
        {
          label: "Stop MCP Tunnel",
          enabled: Boolean(tunnelProc || mcpProc),
          click: () => stopMcpTunnel(),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function diagnosticPage(title, detail) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>ChipLab</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:#0b0f17; color:#e6edf3; font:14px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .card { max-width:820px; padding:32px 36px; border:1px solid #2a3441; border-radius:12px; background:#151b23; }
  h1 { margin:0 0 6px; font-size:19px; color:#f2555f; }
  p  { margin:0 0 18px; color:#9aa7b4; }
  pre { margin:0; padding:16px; background:#0d1117; border:1px solid #2a3441; border-radius:8px;
        font:12px/1.55 ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace; white-space:pre-wrap;
        color:#9aa7b4; max-height:340px; overflow:auto; }
  .hint { margin-top:16px; font-size:12.5px; color:#6e7b88; }
  code { color:#6ea0ff; }
</style></head><body><div class="card">
  <h1>${esc(title)}</h1>
  <p>ChipLab's Python backend did not answer. The desktop shell is running, so this is a backend problem, not an app-launch problem.</p>
  <pre>${esc(detail)}</pre>
  <div class="hint">Log: <code>~/.local/state/chiplab/electron.log</code> &nbsp;·&nbsp; Press <code>Ctrl+R</code> to retry.</div>
</div></body></html>`;
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(html));
}

function loadingPage() {
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>ChipLab · Starting…</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:#0b0f17; color:#e6edf3; font:14px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .card { text-align:center; padding:32px 40px; border:1px solid #2a3441; border-radius:12px; background:#151b23; }
  h1 { margin:0 0 8px; font-size:18px; color:#6ea0ff; }
  p  { margin:0; color:#9aa7b4; font-size:13px; }
</style></head><body><div class="card">
  <h1>◆ ChipLab Studio</h1>
  <p>Starting Python multiphysics &amp; EDA backend…</p>
</div></body></html>`;
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(html));
}

function createWindow(opts = {}) {
  const deferLoad = Boolean(opts.deferLoad);
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1024,
    minHeight: 680,
    title: "ChipLab Desktop",
    backgroundColor: "#0b0f17",
    show: false,
    autoHideMenuBar: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  // Show only once the first paint is ready — avoids the "window opens blank
  // and looks like it never started" effect.
  mainWindow.once("ready-to-show", () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.show();
  });
  setTimeout(() => {
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) {
      mainWindow.show();
    }
  }, 4000).unref();

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http://") || url.startsWith("https://")) {
      shell.openExternal(url);
    }
    return { action: "deny" };
  });

  mainWindow.webContents.on("did-fail-load", (_e, code, desc, url) => {
    logLine(`did-fail-load ${code} ${desc} ${url}`);
    diagnosticPage(
      "ChipLab could not load its interface",
      `URL: ${url}\nError ${code}: ${desc}\n\nRecent backend output:\n${backendLogs.slice(-20).join("\n") || "(none)"}`
    );
  });

  mainWindow.webContents.on("render-process-gone", (_e, details) => {
    logLine(`render-process-gone ${JSON.stringify(details)}`);
    diagnosticPage(
      "ChipLab's renderer crashed",
      `Reason: ${details.reason} (exitCode=${details.exitCode})\n\nRecent backend output:\n${backendLogs.slice(-20).join("\n") || "(none)"}`
    );
  });

  mainWindow.webContents.on("console-message", (details) => {
    const { level, message, lineNumber, sourceId, stackTrace } = details ?? {};
    if (level >= 2) {
      const frames = (stackTrace ?? []).map((f) => `    at ${f.functionName || "?"} (${f.url}:${f.lineNumber}:${f.columnNumber})`).join("\n");
      logLine(`renderer: ${message}  [${sourceId}:${lineNumber}]${frames ? `\n${frames}` : ""}`);
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  if (deferLoad) {
    loadingPage();
  } else {
    loadRoute("/");
  }
}

ipcMain.handle("chiplab:get-info", () => ({
  appVersion: app.getVersion(),
  electronVersion: process.versions.electron,
  nodeVersion: process.versions.node,
  platform: process.platform,
  arch: process.arch,
  backendUrl: BASE_URL,
  pythonPath: resolvedPython,
  mcpTunnelUrl,
  mcpTunnelStatus,
}));

ipcMain.handle("chiplab:open-route", (_event, route) => {
  loadRoute(route);
  return { ok: true, url: `${BASE_URL}${normalizeRoute(route)}` };
});

app.whenReady().then(async () => {
  buildMenu();
  await startBackend();
  startMcpTunnel();

  // Always open a window immediately with a loading screen; only navigate to
  // BASE_URL once the backend is listening so Chromium never hits ERR_CONNECTION_REFUSED.
  createWindow({ deferLoad: true });

  const ready = await waitForBackend();
  if (ready) {
    loadRoute("/");
  } else {
    const tail = backendLogs.slice(-20).join("\n") || "(no output captured)";
    logLine("Backend did not become ready within timeout");
    diagnosticPage(
      "ChipLab backend failed to start",
      `Could not reach ${BASE_URL}/api/status within ${STARTUP_TIMEOUT_MS / 1000}s.\n\n` +
        `Python: ${resolvedPython}\n\nRecent logs:\n${tail}`
    );
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("before-quit", () => {
  isQuitting = true;
  stopMcpTunnel();
  stopBackend();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin" || isQuitting) {
    stopMcpTunnel();
    stopBackend();
    app.quit();
  }
});
