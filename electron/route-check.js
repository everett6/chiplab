// Headless-ish route smoke test: loads each ChipLab route in an offscreen
// Electron window and reports any console error / failed load. Run with:
//   electron --no-sandbox route-check.js
"use strict";

const { app, BrowserWindow } = require("electron");

const BASE = process.env.CHIPLAB_BASE || "http://127.0.0.1:8770";
const ROUTES = ["/", "/asic", "/agent", "/live", "/studio", "/memory"];

if (process.platform === "linux") {
  app.commandLine.appendSwitch("no-sandbox");
  app.commandLine.appendSwitch("disable-gpu-sandbox");
  app.commandLine.appendSwitch("enable-unsafe-swiftshader");
  app.commandLine.appendSwitch("disable-dev-shm-usage");
  app.commandLine.appendSwitch("ozone-platform", "x11");
}

const problems = [];

function checkRoute(win, route) {
  return new Promise((resolve) => {
    const errors = [];
    const onConsole = (d) => {
      if ((d && d.level ? d.level : 0) >= 2) errors.push(d.message);
    };
    win.webContents.on("console-message", onConsole);

    const done = (failed) => {
      win.webContents.off("console-message", onConsole);
      if (errors.length || failed) problems.push({ route, errors, failed });
      const tag = errors.length || failed ? "FAIL" : " OK ";
      console.log(
        tag + " " + route +
        (errors.length ? "\n      " + errors.join("\n      ") : "") +
        (failed ? "\n      load failed: " + failed : "")
      );
      resolve();
    };

    win.webContents.once("did-finish-load", () => setTimeout(() => done(null), 2500));
    win.webContents.once("did-fail-load", (_e, code, desc) => done(code + " " + desc));
    win.loadURL(BASE + route);
  });
}

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    show: false,
    width: 1280,
    height: 860,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: false },
  });

  for (const r of ROUTES) {
    await checkRoute(win, r);
  }

  console.log(problems.length ? "\n" + problems.length + " route(s) with problems" : "\nAll routes clean");
  app.exit(problems.length ? 1 : 0);
});