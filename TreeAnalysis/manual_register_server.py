"""Flask server for TIFF + SWC manual registration.

Whole-cell ``fov1.tif`` stack as the base image with SWC overlay; align one FOV
at a time (MIP or Z-slice viewing).

Start:
    python manual_register_server.py config/GP04.json --session pre-droplet
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from export_registration_scene import (
    _bounds,
    _decimate_segments,
    _resolve_cell_tiff,
    _segments_from_nodes,
    export_scene,
)
from endpoint_register import trace_endpoints
from correct_swc_z import slice_to_depth_um
from io_pvscan_xml import parse_pvscan
from run_tree_analysis import (
    _exists,
    _load_combined_swc,
    _resolve,
    _resolve_swc_list,
    _z_schedule_from_cfg,
    load_config,
)
from tiff_slices import (
    load_or_build_mip_png_bytes,
    resolve_frame_path,
    slice_to_png_bytes,
    stack_info,
    stack_page_to_png_bytes,
)

try:
    from flask import Flask, Response, jsonify, request, send_from_directory
except ImportError as exc:
    raise SystemExit("Flask required: pip install flask") from exc

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

_STATE: Dict[str, Any] = {
    "scene": None,
    "scene_loading": False,
    "scene_error": None,
    "fov_meta": {},
    "fov_meta_loaded": False,
    "fov_xml_paths": {},
    "cell_tiff": None,
    "cell_n_pages": 0,
    "mip_png": None,
    "mip_loading": False,
    "mip_ready": False,
    "slice_png_cache": {},
    "config_path": None,
    "session": None,
}


def _slim_scene(scene: dict) -> dict:
    """Drop fields the browser does not need (smaller / faster JSON)."""
    cell = dict(scene["cell"])
    cell.pop("segments", None)
    fovs_out = {}
    for fov_id, fov in scene["fovs"].items():
        fovs_out[fov_id] = {
            "fov": fov.get("fov"),
            "color": fov["color"],
            "segments": fov["segments"],
            "initial_offset_um": fov.get("initial_offset_um"),
            "centroid_stage_um": fov.get("centroid_stage_um"),
            "endpoints_stage_um": fov.get("endpoints_stage_um"),
        }
    return {
        "version": scene.get("version"),
        "animal_id": scene.get("animal_id"),
        "session": scene.get("session"),
        "coordinate_frames": scene.get("coordinate_frames"),
        "cell": cell,
        "fovs": fovs_out,
    }


def _load_fov_meta() -> None:
    if _STATE["fov_meta_loaded"]:
        return
    scene = _STATE["scene"]
    if scene is None:
        return
    for fov_id, xml in _STATE.get("fov_xml_paths", {}).items():
        if Path(xml).is_file():
            _STATE["fov_meta"][fov_id] = parse_pvscan(xml)
    _STATE["fov_meta_loaded"] = True


def _warm_mip() -> None:
    path = _STATE["cell_tiff"]
    if path is None or not path.is_file():
        return
    _STATE["mip_loading"] = True
    t0 = time.perf_counter()
    try:
        _STATE["mip_png"] = load_or_build_mip_png_bytes(path)
        _STATE["mip_ready"] = True
        elapsed = time.perf_counter() - t0
        cache_hit = elapsed < 2.0
        print(f"MIP {'cached' if cache_hit else 'built'} in {elapsed:.1f}s")
    except Exception as exc:
        print(f"MIP build failed: {exc}")
    finally:
        _STATE["mip_loading"] = False


def _nearest_slice_index(z_um: float, z_um_per_page: list[float]) -> float:
    if not z_um_per_page:
        return z_um
    i = int(np.argmin([abs(z - z_um) for z in z_um_per_page]))
    return float(i + 1)


def _enrich_scene_cell(scene: dict, cfg: dict, base: Path) -> Optional[Path]:
    """Add TIFF/SWC display fields when scene was loaded from an old cache JSON."""
    wc = cfg.get("whole_cell", {})
    xml_path = _resolve(base, wc.get("pvscan_xml", ""))
    if not _exists(xml_path):
        print("WARNING: whole_cell pvscan_xml not found — check E: drive / config paths", flush=True)
        return None

    meta = parse_pvscan(xml_path)
    mpp_x, mpp_y, _ = meta.microns_per_pixel
    schedule = _z_schedule_from_cfg(wc["z_schedule"])
    cell = scene["cell"]

    cell["mpp"] = [round(mpp_x, 6), round(mpp_y, 6)]
    cell["size_px"] = [int(meta.pixels[0]), int(meta.pixels[1])]

    xml_p = Path(xml_path)
    cell_tiff = _resolve_cell_tiff(wc, xml_p, base)
    if cell_tiff:
        cell["tiff_stack"] = cell_tiff

    n_pages = 0
    z_um_per_page: list[float] = []
    cell_path: Optional[Path] = Path(cell_tiff) if cell_tiff else None
    if cell_path and cell_path.is_file():
        n_pages, _, _ = stack_info(cell_path)
        for p in range(n_pages):
            z_val = slice_to_depth_um(float(p + 1), schedule)
            z_um_per_page.append(round(float(np.asarray(z_val).ravel()[0]), 3))
    else:
        print("WARNING: whole-cell TIFF not found — grey SWC only (no image background)", flush=True)

    cell["n_pages"] = n_pages
    cell["z_um_per_page"] = z_um_per_page

    if not cell.get("segments_pixel"):
        swc_files = _resolve_swc_list(base, wc.get("swc", ""))
        if swc_files:
            print("Building cell SWC overlay from whole-cell trace…", flush=True)
            raw = _load_combined_swc(swc_files)
            cell["segments_pixel"] = _segments_from_nodes(raw, x="x", y="y", z="z")
        elif cell.get("segments"):
            print("Converting cached cell segments (µm) → pixel overlay…", flush=True)
            px_segs: list = []
            for a, b in cell["segments"]:
                px_segs.append(
                    [
                        [a[0] / mpp_x, a[1] / mpp_y, _nearest_slice_index(a[2], z_um_per_page)],
                        [b[0] / mpp_x, b[1] / mpp_y, _nearest_slice_index(b[2], z_um_per_page)],
                    ]
                )
            cell["segments_pixel"] = px_segs

    if cell.get("segments_pixel"):
        cell["segments_pixel"] = _decimate_segments(cell["segments_pixel"], max_segments=12000, stride=2)
        print(f"cell SWC overlay: {len(cell['segments_pixel'])} segments for display", flush=True)

    cell_bounds_um = _bounds(scene["cell"].get("segments", []))
    if cell_bounds_um == [0, 0, 0, 0, 0, 0] and cell.get("segments_pixel"):
        mpp_x, mpp_y = cell["mpp"]
        px_segs = cell["segments_pixel"]
        um_segs = [
            [[a[0] * mpp_x, a[1] * mpp_y, a[2]], [b[0] * mpp_x, b[1] * mpp_y, b[2]]]
            for a, b in px_segs[:500]
        ]
        cell_bounds_um = _bounds(um_segs)

    for fov in scene.get("fovs", {}).values():
        if not fov.get("initial_offset_um") and fov.get("segments") and cell_bounds_um != [0, 0, 0, 0, 0, 0]:
            fb = _bounds(fov["segments"])
            fov_cx = (fb[0] + fb[1]) / 2
            fov_cy = (fb[2] + fb[3]) / 2
            fov_cz = (fb[4] + fb[5]) / 2
            cell_cx = (cell_bounds_um[0] + cell_bounds_um[1]) / 2
            cell_cy = (cell_bounds_um[2] + cell_bounds_um[3]) / 2
            cell_cz = (cell_bounds_um[4] + cell_bounds_um[5]) / 2
            fov["initial_offset_um"] = [
                round(cell_cx - fov_cx, 2),
                round(cell_cy - fov_cy, 2),
                round(cell_cz - fov_cz, 2),
            ]

    for fov in scene.get("fovs", {}).values():
        if not fov.get("endpoints_stage_um") and fov.get("segments"):
            try:
                ep_a, ep_b = trace_endpoints(fov["segments"])
                fov["endpoints_stage_um"] = [
                    [round(float(c), 2) for c in ep_a],
                    [round(float(c), 2) for c in ep_b],
                ]
            except ValueError:
                pass

    return cell_path if cell_path and cell_path.is_file() else None


def _build_state(config_path: Path, session_name: str, scene_cache: Optional[Path] = None) -> None:
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    base = config_path.parent.parent
    if scene_cache and scene_cache.is_file():
        print(f"Loading cached scene: {scene_cache.name}", flush=True)
        scene = json.loads(scene_cache.read_text(encoding="utf-8"))
        print(f"scene cache loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    else:
        print("Building scene from SWC files (1–3 min if E: drive is slow)…", flush=True)
        scene = export_scene(cfg, base, session_name)
        print(f"scene built in {time.perf_counter() - t0:.1f}s", flush=True)

    _enrich_scene_cell(scene, cfg, base)

    if scene["cell"].get("segments"):
        cell_segs = scene["cell"]["segments"]
        scene["cell"]["segments"] = _decimate_segments(cell_segs, max_segments=12000, stride=2)
        scene["cell"]["n_segments_full"] = scene["cell"].get("n_segments", len(cell_segs))
        scene["cell"]["n_segments"] = len(scene["cell"]["segments"])

    # Keep XML paths server-side only (lazy PVScan parse for FOV TIFF API).
    fov_xml_paths: Dict[str, str] = {}
    for fov_id, fov in scene["fovs"].items():
        xml = fov.get("pvscan_xml")
        if xml:
            fov_xml_paths[fov_id] = xml
            fov.pop("pvscan_xml", None)

    cell_tiff = scene["cell"].get("tiff_stack")
    cell_path = Path(cell_tiff) if cell_tiff else None
    n_pages = int(scene["cell"].get("n_pages") or 0)
    if cell_path and cell_path.is_file():
        print(f"cell TIFF: {cell_path.name} ({n_pages} pages)", flush=True)
    elif not cell_path or not cell_path.is_file():
        cell_path = None
        print("WARNING: whole-cell TIFF not available — SWC overlay only", flush=True)

    _STATE["scene"] = _slim_scene(scene)
    _STATE["fov_meta"] = {}
    _STATE["fov_meta_loaded"] = False
    _STATE["fov_xml_paths"] = fov_xml_paths
    _STATE["cell_tiff"] = cell_path
    _STATE["cell_n_pages"] = n_pages
    _STATE["mip_png"] = None
    _STATE["mip_ready"] = False
    _STATE["mip_loading"] = False
    _STATE["slice_png_cache"] = {}
    _STATE["config_path"] = str(config_path)
    _STATE["session"] = session_name


def _build_state_async(config_path: Path, session_name: str, scene_cache: Optional[Path] = None) -> None:
    _STATE["scene_loading"] = True
    _STATE["scene_error"] = None
    try:
        _build_state(config_path, session_name, scene_cache=scene_cache)
        scene = _STATE["scene"]
        print(f"animal  : {scene['animal_id']}", flush=True)
        print(f"session : {session_name}", flush=True)
        print(f"fovs    : {list(scene['fovs'].keys())}", flush=True)
        threading.Thread(target=_warm_mip, daemon=True).start()
    except Exception as exc:
        _STATE["scene_error"] = str(exc)
        print(f"SCENE BUILD FAILED: {exc}", flush=True)
    finally:
        _STATE["scene_loading"] = False


@app.get("/")
def index() -> Response:
    return send_from_directory(STATIC_DIR, "manual_register_tiff.html")


@app.get("/endpoints")
def index_endpoints() -> Response:
    return send_from_directory(STATIC_DIR, "manual_register_endpoints.html")


@app.get("/api/status")
def api_status():
    return jsonify(
        {
            "mip_ready": _STATE["mip_ready"],
            "mip_loading": _STATE["mip_loading"],
            "scene_ready": _STATE["scene"] is not None,
            "scene_loading": _STATE["scene_loading"],
            "scene_error": _STATE["scene_error"],
            "n_pages": _STATE["cell_n_pages"],
        }
    )


@app.get("/api/scene")
def api_scene():
    if _STATE["scene"] is None:
        if _STATE["scene_error"]:
            return jsonify({"error": _STATE["scene_error"], "loading": False}), 500
        return jsonify(
            {
                "loading": True,
                "message": "Loading scene (reading SWC traces — can take 1–3 minutes)…",
            }
        )
    return jsonify(_STATE["scene"])


@app.get("/api/cell/mip.png")
def api_cell_mip():
    if _STATE["cell_tiff"] is None:
        return jsonify({"error": "whole-cell TIFF not configured"}), 404
    if not _STATE["mip_ready"]:
        if not _STATE["mip_loading"]:
            threading.Thread(target=_warm_mip, daemon=True).start()
        return jsonify({"error": "MIP still building — use Z slice mode or retry shortly"}), 503
    return Response(_STATE["mip_png"], mimetype="image/png")


@app.get("/api/cell/slice/<int:page>.png")
def api_cell_slice(page: int):
    path = _STATE["cell_tiff"]
    if path is None or not path.is_file():
        return jsonify({"error": "whole-cell TIFF not configured"}), 404
    if page < 0 or page >= _STATE["cell_n_pages"]:
        return jsonify({"error": f"page {page} out of range (0..{_STATE['cell_n_pages'] - 1})"}), 404
    cache = _STATE["slice_png_cache"]
    if page not in cache:
        try:
            cache[page] = stack_page_to_png_bytes(path, page)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    return Response(cache[page], mimetype="image/png")


@app.get("/api/fov/<fov_id>/frames")
def api_fov_frames(fov_id: str):
    scene = _STATE["scene"]
    if scene is None or fov_id not in scene["fovs"]:
        return jsonify({"error": f"unknown fov {fov_id!r}"}), 404
    _load_fov_meta()
    fov = scene["fovs"][fov_id]
    meta = _STATE["fov_meta"].get(fov_id)
    frames = []
    for fr in fov.get("frames", []):
        idx = fr["index"]
        path = resolve_frame_path(meta, idx) if meta else None
        frames.append(
            {
                "index": idx,
                "z_um": fr["z_um"],
                "file": fr.get("file"),
                "available": path is not None,
            }
        )
    return jsonify({"fov_id": fov_id, "frames": frames})


@app.get("/api/fov/<fov_id>/slice/<int:frame_index>.png")
def api_fov_slice(fov_id: str, frame_index: int):
    _load_fov_meta()
    meta = _STATE["fov_meta"].get(fov_id)
    if meta is None:
        return jsonify({"error": f"no TIFF metadata for {fov_id!r}"}), 404
    path = resolve_frame_path(meta, frame_index)
    if path is None:
        return jsonify({"error": f"frame {frame_index} file not found on disk"}), 404
    try:
        png = slice_to_png_bytes(path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return Response(png, mimetype="image/png")


@app.post("/api/export")
def api_export():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "expected JSON body"}), 400
    out = Path(request.args.get("out", "manual_registration.json"))
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return jsonify({"saved": str(out.resolve())})


def main(argv: Optional[list[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="TIFF + SWC manual registration server")
    parser.add_argument("config", help="config/<ANIMAL>.json")
    parser.add_argument("--session", required=True, help="session name (e.g. pre-droplet)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--no-browser", action="store_true", help="do not open browser tab")
    args = parser.parse_args(argv[1:] if argv else None)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    animal = cfg.get("animal_id", "unknown")
    tree_dir = Path(__file__).resolve().parent
    scene_cache = tree_dir / f"registration_scene_{animal}_{args.session}.json"

    url = f"http://{args.host}:{args.port}/endpoints"
    print(f"Server URL: {url}", flush=True)
    print("Opening browser… (if it does not open, paste the URL above manually)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    threading.Thread(
        target=_build_state_async,
        args=(cfg_path, args.session, scene_cache if scene_cache.is_file() else None),
        daemon=True,
    ).start()

    print("Tip: Z slice mode loads instantly; MIP builds in background.", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
