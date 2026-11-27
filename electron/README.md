# ChipLab Desktop (Linux)

Desktop wrapper for **ChipLab** built with Electron. It automatically launches the local Python backend (`python -m chiplab.app.server --host 127.0.0.1 --port 8770`) and opens the ChipLab AI Agent & Custom Circuit Lab workspace.

## Prerequisites (Linux)

- **Node.js** 18+ and **npm**
- **Python** 3.10+ with ChipLab dependencies installed (`.venv`, conda, or system Python)
- Standard Electron runtime libraries on Debian/Ubuntu:
  ```bash
  sudo apt-get install -y libgtk-3-0 libnss3 libasound2t64 libgbm1 libxss1
  ```
  *(On older Ubuntu/Debian releases use `libasound2` instead of `libasound2t64`; AppImage also uses `libfuse2`.)*

## Quick Start

```bash
cd electron
npm install
npm start
```

### Python Discovery Order

`main.js` resolves the Python interpreter in this order:
1. `CHIPLAB_PYTHON` environment variable
2. `../.venv/bin/python` (or `python3`)
3. `/home/everett/miniconda3/bin/python3`
4. `python3` on `PATH`

Override it explicitly if needed:
```bash
CHIPLAB_PYTHON=/path/to/python npm start
```

## Packaging (AppImage & .deb)

Build an unpacked directory for testing:
```bash
npm run pack
```

Build Linux `AppImage` and `.deb` installers into `electron/dist/`:
```bash
npm run dist
```

## OpenRouter AI Agent Configuration

The Built-in AI Agent tab (`/agent`) supports OpenRouter free models (default: `nvidia/nemotron-3-ultra-550b-a55b:free`).

You can provide your OpenRouter API key in either of two ways:
- **Environment variable** before launching:
  ```bash
  export OPENROUTER_API_KEY="sk-or-v1-..."
  npm start
  ```
- **In-app UI**: Paste your key into the OpenRouter API Key field in the `/agent` workspace (saved in browser `localStorage` only and sent in the chat request body when provided).
