"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("chiplabDesktop", {
  platform: process.platform,
  arch: process.arch,
  versions: {
    node: process.versions.node,
    chrome: process.versions.chrome,
    electron: process.versions.electron,
  },
  openRoute: (route) => ipcRenderer.invoke("chiplab:open-route", route),
  getInfo: () => ipcRenderer.invoke("chiplab:get-info"),
});
