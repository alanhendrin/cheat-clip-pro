# Cheat Clip PRO — MCP Server

Lets an AI assistant (Claude Code, Claude Desktop, or any MCP client) drive Cheat Clip PRO:
analyze videos, pick clips, render them, and manage assets.

## Setup

1. Install the SDK into the project venv:
   ```bash
   venv/Scripts/python.exe -m pip install -r mcp_server/requirements.txt
   ```
2. Put your Gemini key in `backend/.env` (`GEMINI_API_KEY=...`) and restart the backend.
3. Start the app (`npm run dev` or `start.bat`) — the MCP server talks to the backend at
   `http://127.0.0.1:8000` (override with `CHEAT_CLIP_BACKEND_URL`).
4. Open Claude Code in the project root. `.mcp.json` registers the server automatically
   (approve it on first use).

## Tools

| Tool | What it does |
|---|---|
| `backend_status` | Backend health, Gemini key presence, available encoders |
| `analyze_video` | YouTube / Google Drive / local file → numbered viral clip list (saved as an analysis) |
| `list_analyses` / `get_analysis` | Browse saved analyses (stored in `mcp_server/analyses/`) |
| `render_clips` | Render chosen clip numbers with layout, captions, fonts, watermark, BGM, SFX |
| `render_status` | Keep waiting on a long render batch |
| `upload_asset` | Upload BGM, hook SFX, watermark image, or font |
| `list_fonts` | Fonts usable for captions / titles |
| `list_exports` | Rendered MP4 / ZIP files in `backend/exports` |

Example prompt: *"Analyze https://youtu.be/xxxx for funny moments, 30s clips, then render the
top 3 with Anton font, fire_red captions and @myhandle watermark."*
