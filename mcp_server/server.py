"""
Cheat Clip PRO — MCP server.

Exposes the local Cheat Clip PRO backend (http://127.0.0.1:8000) as MCP tools so an
AI assistant can analyze videos, render clips and manage assets on the user's behalf.

The backend must already be running (`npm run dev` or `start.bat`). The Gemini API key
is read by the backend itself from backend/.env (GEMINI_API_KEY).
"""
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

import httpx
from mcp.server.mcpserver import MCPServer

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ROOT_DIR / "backend" / "exports"
ANALYSES_DIR = Path(__file__).resolve().parent / "analyses"
ANALYSES_DIR.mkdir(exist_ok=True)

BACKEND_URL = os.environ.get("CHEAT_CLIP_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

mcp = MCPServer(
    name="cheat-clip-pro",
    instructions=(
        "Controls the local Cheat Clip PRO auto-clipper. Typical flow: analyze_video -> "
        "pick clips from the returned list -> render_clips with those clip numbers -> "
        "files land in backend/exports. Clip numbers are 1-based and refer to the "
        "analysis' clip list. If a call fails with a connection error, the backend is not "
        "running: ask the user to start it with `npm run dev` or start.bat."
    ),
)


# Watermark center anchor (x%, y%) per preset; bottom presets sit above the subtitles.
WATERMARK_POSITIONS = {
    "top_right": (78.0, 6.0),
    "top_left": (22.0, 6.0),
    "top_center": (50.0, 6.0),
    "center": (50.0, 50.0),
    "bottom_right": (78.0, 94.0),
    "bottom_left": (22.0, 94.0),
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _client(timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BACKEND_URL, timeout=httpx.Timeout(timeout, read=None))


async def _iter_sse(response: httpx.Response):
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            yield json.loads(line[5:].strip())


def _analysis_path(analysis_id: str) -> Path:
    safe = "".join(ch for ch in analysis_id if ch.isalnum() or ch in "_-")
    return ANALYSES_DIR / f"{safe}.json"


def _load_analysis(analysis_id: str) -> dict:
    path = _analysis_path(analysis_id)
    if not path.exists():
        raise ValueError(f"Analysis '{analysis_id}' not found. Use list_analyses to see saved ones.")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _clip_summary(index: int, clip: dict) -> dict:
    return {
        "clip": index,
        "title": clip.get("title_suggestion") or clip.get("title"),
        "range": f"{_fmt_time(clip['start_time'])}-{_fmt_time(clip['end_time'])}",
        "start_time": clip["start_time"],
        "end_time": clip["end_time"],
        "virality_score": clip.get("virality_score"),
        "key_quotes": clip.get("key_quotes", []),
        "caption": clip.get("caption_suggestion", ""),
        "hashtags": clip.get("hashtag_suggestion", ""),
    }


def _batch_summary(batch: dict) -> dict:
    clips = []
    for c in batch.get("clips", []):
        entry = {
            "title": c.get("title"),
            "status": c.get("status"),
            "progress_percent": c.get("progress_percent"),
        }
        if c.get("output_path"):
            entry["output_path"] = c["output_path"]
        if c.get("error_message") or c.get("error"):
            entry["error"] = c.get("error_message") or c.get("error")
        clips.append(entry)
    summary = {
        "batch_id": batch.get("batch_id"),
        "overall_status": batch.get("overall_status"),
        "clips": clips,
    }
    if batch.get("zip_url"):
        summary["zip_download_url"] = BACKEND_URL + batch["zip_url"]
    if batch.get("warning_message"):
        summary["warning"] = batch["warning_message"]
    return summary


async def _wait_for_batch(batch_id: str, wait_seconds: float) -> dict:
    """Follows the render progress stream until the batch finishes or wait_seconds elapse."""
    deadline = time.monotonic() + wait_seconds
    last: dict = {"batch_id": batch_id, "overall_status": "running", "clips": []}
    async with _client() as client:
        async with client.stream("GET", f"/api/render-progress/{batch_id}") as resp:
            if resp.status_code == 404:
                raise ValueError(f"Render batch '{batch_id}' not found (the backend may have restarted).")
            resp.raise_for_status()
            async for batch in _iter_sse(resp):
                last = batch
                if batch.get("overall_status") in ("completed", "error") or time.monotonic() > deadline:
                    break
    return _batch_summary(last)


# ── Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
async def backend_status() -> dict:
    """Check whether the Cheat Clip PRO backend is running and how it is configured
    (Gemini key in .env, proxy, hardware encoders)."""
    async with _client(10) as client:
        health = (await client.get("/api/health")).json()
        hw = (await client.get("/api/hardware-accel")).json()
    return {
        "backend_url": BACKEND_URL,
        "status": health.get("status"),
        "gemini_key_in_env": health.get("gemini_env_configured"),
        "proxy_configured": health.get("proxy_configured"),
        "default_encoder": hw.get("active_default"),
        "available_hardware_accel": [o["id"] for o in hw.get("options", []) if o.get("available")],
    }


@mcp.tool()
async def analyze_video(
    source: str,
    clip_duration: Literal["15s", "30s", "60s", "auto"] = "30s",
    focus: Optional[str] = None,
    clip_count: Optional[int] = None,
    range_start_seconds: Optional[float] = None,
    range_end_seconds: Optional[float] = None,
    model: str = "gemini-2.5-flash",
    use_mock: bool = False,
) -> dict:
    """Analyze a video and find viral clip candidates with Gemini.

    source: a YouTube URL, a Google Drive share link, or an absolute path to a local
        video file on this computer.
    clip_duration: target length of each clip.
    focus: optional theme to search for, e.g. "funny moments" or "tips about investing".
    clip_count: optional number of clips to find (default: AI decides).
    range_start_seconds / range_end_seconds: optionally only search part of the video.
    use_mock: return sample clips without calling Gemini (for testing).

    Returns an analysis_id plus the numbered clip list. Pass the analysis_id and clip
    numbers to render_clips. Long videos can take several minutes (transcription).
    """
    body = {
        "url": source,
        "duration": clip_duration,
        "model": model,
        "custom_prompt": focus,
        "target_clip_count": clip_count if clip_count else "auto",
        "range_start": range_start_seconds,
        "range_end": range_end_seconds,
    }
    if use_mock:
        body["api_key"] = "mock"

    result = None
    last_stage = ""
    async with _client() as client:
        async with client.stream("POST", "/api/analyze", json=body) as resp:
            resp.raise_for_status()
            async for event in _iter_sse(resp):
                if event.get("error"):
                    raise RuntimeError(f"Analysis failed at '{last_stage}': {event['error']}")
                last_stage = event.get("stage") or last_stage
                if event.get("done"):
                    result = event["result"]
    if not result:
        raise RuntimeError(f"Analysis stream ended without a result (last stage: '{last_stage}').")

    analysis_id = f"{result['video_id']}_{int(time.time())}"
    record = {
        "analysis_id": analysis_id,
        "source": source,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
    }
    _analysis_path(analysis_id).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    return {
        "analysis_id": analysis_id,
        "title": result.get("title"),
        "channel": result.get("channel"),
        "duration": _fmt_time(result.get("duration", 0)),
        "model": result.get("model"),
        "summary": result.get("summary"),
        "clips": [_clip_summary(i, c) for i, c in enumerate(result.get("clips", []), start=1)],
    }


@mcp.tool()
def list_analyses(limit: int = 20) -> list[dict]:
    """List previously saved analyses (newest first) so their clips can be rendered again."""
    records = []
    for path in sorted(ANALYSES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append({
            "analysis_id": rec["analysis_id"],
            "title": rec["result"].get("title"),
            "source": rec["source"],
            "clips": len(rec["result"].get("clips", [])),
            "created_at": rec.get("created_at"),
        })
    return records


@mcp.tool()
def get_analysis(analysis_id: str, include_transcript: bool = False) -> dict:
    """Show the full details of a saved analysis: summary and numbered clip list.
    Set include_transcript to also return the timestamped transcript (can be long)."""
    rec = _load_analysis(analysis_id)
    result = rec["result"]
    data = {
        "analysis_id": analysis_id,
        "source": rec["source"],
        "title": result.get("title"),
        "duration": _fmt_time(result.get("duration", 0)),
        "summary": result.get("summary"),
        "clips": [_clip_summary(i, c) for i, c in enumerate(result.get("clips", []), start=1)],
    }
    if include_transcript:
        data["transcript"] = [
            {"t": _fmt_time(line["start"]), "text": line["text"]} for line in result.get("transcript") or []
        ]
    return data


@mcp.tool()
async def render_clips(
    analysis_id: str,
    clips: list[int],
    aspect_ratio: Literal["9:16", "1:1", "4:3", "16:9", "16:9_landscape"] = "9:16",
    background_style: Literal["black", "blurred"] = "black",
    face_tracking: bool = True,
    layout: Literal["none", "split_top_cam", "pip_corner"] = "none",
    subtitles: bool = True,
    caption_style: Literal[
        "viral_pop", "beast_punch", "cyber_violet", "electric_cyan", "fire_red", "golden_aura"
    ] = "viral_pop",
    caption_font: str = "Outfit",
    title_font: str = "Outfit",
    font_size: Literal["small", "medium", "big"] = "medium",
    text_case: Literal["uppercase", "lowercase", "capitalize", "original"] = "uppercase",
    show_title: bool = True,
    title_duration: Literal["entire", "5s", "10s"] = "entire",
    custom_titles: Optional[dict[str, str]] = None,
    watermark_image_path: Optional[str] = None,
    watermark_text: Optional[str] = None,
    watermark_position: Literal["top_right", "top_left", "top_center", "center", "bottom_right", "bottom_left"] = "top_right",
    watermark_size_percent: float = 20.0,
    bgm_path: Optional[str] = None,
    bgm_volume_percent: float = 25.0,
    hook_sfx_path: Optional[str] = None,
    original_audio_percent: float = 100.0,
    merge_into_one_video: bool = False,
    hardware_accel: Literal["auto", "nvenc", "amf", "qsv", "cpu"] = "auto",
    wait_seconds: int = 900,
) -> dict:
    """Render clips from a saved analysis into finished MP4 videos (saved in backend/exports).

    clips: 1-based clip numbers from analyze_video / get_analysis, e.g. [1, 3].
    layout: "split_top_cam" = streamer facecam on top + gameplay below, "pip_corner" =
        picture-in-picture facecam, "none" = normal crop.
    caption_font / title_font: any name from list_fonts.
    show_title: burn the clip title into the video. custom_titles overrides titles per
        clip number, e.g. {"1": "Judul baru"}.
    watermark_image_path / bgm_path / hook_sfx_path: file_path values from upload_asset
        (or any absolute local path). watermark_size_percent is relative to video width.
    merge_into_one_video: combine all selected clips into a single highlight video.
    wait_seconds: how long to wait for the render. If it is still running afterwards,
        call render_status with the returned batch_id.
    """
    rec = _load_analysis(analysis_id)
    result = rec["result"]
    all_clips = result.get("clips", [])
    if not clips:
        raise ValueError("Pick at least one clip number.")
    bad = [n for n in clips if n < 1 or n > len(all_clips)]
    if bad:
        raise ValueError(f"Clip numbers {bad} are out of range (this analysis has {len(all_clips)} clips).")

    selected = []
    for n in clips:
        clip = dict(all_clips[n - 1])
        if custom_titles and str(n) in custom_titles:
            clip["custom_title"] = custom_titles[str(n)]
        selected.append(clip)

    if result.get("source_type") == "youtube" or not result.get("video_url"):
        video_url = rec["source"]
    else:
        video_url = result["video_url"]

    settings = {
        "aspect_ratio": aspect_ratio,
        "background_style": background_style,
        "enable_face_tracking": face_tracking,
        "streamer_preset": layout,
        "title_position": "auto" if show_title else "none",
        "title_duration": title_duration,
        "subtitles_enabled": subtitles,
        "caption_style": caption_style,
        "caption_font": caption_font,
        "title_font": title_font,
        "font_size": font_size,
        "title_font_size": font_size,
        "text_case": text_case,
        "title_text_case": text_case,
        "original_audio_volume": original_audio_percent,
        "hardware_accel": hardware_accel,
        "render_mode": "merged" if merge_into_one_video else "separate",
    }
    if watermark_image_path or watermark_text:
        # The backend anchors the watermark by its center, so keep it away from the edges.
        wm_x, wm_y = WATERMARK_POSITIONS[watermark_position]
        settings.update({
            "watermark_enabled": True,
            "watermark_type": "image" if watermark_image_path else "text",
            "watermark_file_path": watermark_image_path,
            "watermark_text": watermark_text,
            "watermark_size": watermark_size_percent,
            "watermark_x": wm_x,
            "watermark_y": wm_y,
        })
    if bgm_path:
        settings.update({"bgm_enabled": True, "bgm_file_path": bgm_path, "bgm_volume": bgm_volume_percent})
    if hook_sfx_path:
        settings.update({"hook_sfx_enabled": True, "hook_sfx_file_path": hook_sfx_path})

    body = {
        "video_url": video_url,
        "video_id": result["video_id"],
        "clips": selected,
        "settings": settings,
        "transcript": result.get("transcript"),
    }
    async with _client() as client:
        resp = await client.post("/api/render-batch", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"Render request rejected: {resp.text}")
        batch_id = resp.json()["batch_id"]

    return await _wait_for_batch(batch_id, wait_seconds)


@mcp.tool()
async def render_status(batch_id: str, wait_seconds: int = 300) -> dict:
    """Check (and wait up to wait_seconds for) a render batch started by render_clips."""
    return await _wait_for_batch(batch_id, wait_seconds)


@mcp.tool()
async def upload_asset(file_path: str, kind: Literal["bgm", "sfx", "watermark", "font"]) -> dict:
    """Upload a local file for use in renders.

    kind: "bgm" (background music), "sfx" (hook sound effect at the start),
        "watermark" (logo image), or "font" (.ttf/.otf caption/title font).
    Returns the file_path to pass to render_clips (or the font name for fonts).
    """
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    endpoint = {"bgm": "/api/upload-bgm", "sfx": "/api/upload-sfx",
                "watermark": "/api/upload-watermark", "font": "/api/upload-font"}[kind]
    async with _client(120) as client:
        with path.open("rb") as fh:
            resp = await client.post(endpoint, files={"file": (path.name, fh)})
    if resp.status_code >= 400:
        raise RuntimeError(f"Upload failed: {resp.text}")
    data = resp.json()
    return {k: v for k, v in data.items() if k in ("file_path", "font_name", "saved_name", "filename", "url")}


@mcp.tool()
async def list_fonts() -> list[str]:
    """List font names available for caption_font / title_font."""
    async with _client(10) as client:
        data = (await client.get("/api/fonts")).json()
    return [f["name"] for f in data.get("fonts", [])]


@mcp.tool()
def list_exports(limit: int = 20) -> list[dict]:
    """List rendered videos in backend/exports (newest first) with their full paths."""
    if not EXPORTS_DIR.exists():
        return []
    files = sorted(
        (p for p in EXPORTS_DIR.iterdir() if p.suffix.lower() in (".mp4", ".zip")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:limit]
    return [{
        "path": str(p),
        "size_mb": round(p.stat().st_size / 1_048_576, 1),
        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
    } for p in files]


if __name__ == "__main__":
    mcp.run()
