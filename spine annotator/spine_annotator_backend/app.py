from __future__ import annotations

from datetime import datetime
import copy
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from . import baseline_adapter, crop_service, io_service, models, session_store

app = FastAPI(title="Dendritic Spine Annotator Backend", version="0.1.0")
# .../learning_project_spines/code final/spine annotator/spine_annotator_backend/app.py
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
LAST_SESSION_DIR = WORKSPACE_ROOT / "results" / "session_state" / "last_session"
LAST_SESSION_FILE = LAST_SESSION_DIR / "session.json"
LAST_SESSION_BACKUP_FILE = LAST_SESSION_DIR / "session_prev.json"
CLIENT_ACTIVITY_LOG = LAST_SESSION_DIR / "matching_activity.log"
AUTO_MATCH_THRESHOLD = 0.85
DEFAULT_MAX_MATCH_Z_GAP = float(baseline_adapter.MAX_MATCH_Z_GAP)


def _session_z_gap(session: session_store.LoadedSession) -> float:
    v = float(getattr(session, "max_match_z_gap", DEFAULT_MAX_MATCH_Z_GAP))
    return max(0.0, v)


def _load_session_internal(payload: models.LoadSessionRequest) -> models.SessionStats:
    t1_tiff = io_service.validate_existing_path(payload.t1_tiff_path, "t1_tiff_path")
    t2_tiff = io_service.validate_existing_path(payload.t2_tiff_path, "t2_tiff_path")
    t1_csv = io_service.validate_existing_path(payload.t1_csv_path, "t1_csv_path")
    t2_csv = io_service.validate_existing_path(payload.t2_csv_path, "t2_csv_path")

    t1_stack = baseline_adapter.load_stack(t1_tiff)
    t2_stack = baseline_adapter.load_stack(t2_tiff)
    t1_df = baseline_adapter.load_spines(t1_csv)
    t2_df = baseline_adapter.load_spines(t2_csv)

    session = session_store.LoadedSession(
        t1_tiff_path=t1_tiff,
        t2_tiff_path=t2_tiff,
        t1_csv_path=t1_csv,
        t2_csv_path=t2_csv,
        t1_stack=t1_stack,
        t2_stack=t2_stack,
        t1_df=t1_df,
        t2_df=t2_df,
        t1_lookup=baseline_adapter.to_lookup(t1_df),
        t2_lookup=baseline_adapter.to_lookup(t2_df),
    )
    session_store.set_active_session(session)
    _ensure_default_dendrite_links(session)
    _refresh_algo_matches(session, use_anchors=True)

    return models.SessionStats(
        t1_spine_count=int(len(t1_df)),
        t2_spine_count=int(len(t2_df)),
        t1_dendrite_ids=sorted([d for d, _ in baseline_adapter.dendrite_groups(t1_df)]),
        t2_dendrite_ids=sorted([d for d, _ in baseline_adapter.dendrite_groups(t2_df)]),
        t1_z_range=[0, int(max(t1_stack.shape[0] - 1, 0))],
        t2_z_range=[0, int(max(t2_stack.shape[0] - 1, 0))],
    )


def _save_last_session() -> None:
    session = session_store.require_active_session()
    LAST_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if LAST_SESSION_FILE.exists():
        LAST_SESSION_BACKUP_FILE.write_text(LAST_SESSION_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {
        "selected_files": {
            "t1_tiff_path": str(session.t1_tiff_path),
            "t2_tiff_path": str(session.t2_tiff_path),
            "t1_csv_path": str(session.t1_csv_path),
            "t2_csv_path": str(session.t2_csv_path),
        },
        "dendrite_links": session.dendrite_links,
        "review_decisions": session.review_decisions,
        "algo_matches": session.algo_matches,
        "confirmed_matches": session.confirmed_matches,
        "matched_t1_ids": sorted(list(session.matched_t1_ids)),
        "matched_t2_ids": sorted(list(session.matched_t2_ids)),
        "rejected_pairs": sorted(list(session.rejected_pairs)),
        "lost_t1_ids": sorted(list(session.lost_t1_ids)),
        "new_t2_ids": sorted(list(session.new_t2_ids)),
        "removed_t1_ids": sorted(list(session.removed_t1_ids)),
        "removed_t2_ids": sorted(list(session.removed_t2_ids)),
        "ignored_t1_ids": sorted(list(session.ignored_t1_ids)),
        "ignored_t2_ids": sorted(list(session.ignored_t2_ids)),
        "manual_t2_click_spines": session.manual_t2_click_spines,
        "manual_t1_click_spines": session.manual_t1_click_spines,
        "manual_click_matches": session.manual_click_matches,
        "viewer_state": session.viewer_state,
        "max_match_z_gap": float(session.max_match_z_gap),
        "action_history": session.action_history[-200:],
    }
    LAST_SESSION_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _restore_previous_saved_session() -> tuple[models.SelectFilesResponse, models.SessionStats]:
    if not LAST_SESSION_BACKUP_FILE.exists():
        raise FileNotFoundError("No previous saved session snapshot found.")
    raw = json.loads(LAST_SESSION_BACKUP_FILE.read_text(encoding="utf-8"))
    selected = models.SelectFilesResponse(**raw["selected_files"])
    stats = _load_session_internal(models.LoadSessionRequest(**selected.model_dump()))
    session = session_store.require_active_session()
    session.dendrite_links = list(raw.get("dendrite_links", []))
    session.review_decisions = dict(raw.get("review_decisions", {}))
    session.algo_matches = dict(raw.get("algo_matches", {}))
    session.confirmed_matches = list(raw.get("confirmed_matches", []))
    session.matched_t1_ids = set(str(x) for x in raw.get("matched_t1_ids", []))
    session.matched_t2_ids = set(str(x) for x in raw.get("matched_t2_ids", []))
    session.rejected_pairs = set(str(x) for x in raw.get("rejected_pairs", []))
    session.lost_t1_ids = set(str(x) for x in raw.get("lost_t1_ids", []))
    session.new_t2_ids = set(str(x) for x in raw.get("new_t2_ids", []))
    session.removed_t1_ids = set(str(x) for x in raw.get("removed_t1_ids", []))
    session.removed_t2_ids = set(str(x) for x in raw.get("removed_t2_ids", []))
    session.ignored_t1_ids = set(str(x) for x in raw.get("ignored_t1_ids", []))
    session.ignored_t2_ids = set(str(x) for x in raw.get("ignored_t2_ids", []))
    session.manual_t2_click_spines = list(raw.get("manual_t2_click_spines", []))
    session.manual_t1_click_spines = list(raw.get("manual_t1_click_spines", []))
    session.manual_click_matches = list(raw.get("manual_click_matches", []))
    _apply_manual_click_spines_to_lookup(session)
    session.viewer_state = dict(raw.get("viewer_state", {"modal_slice_t1": 10, "modal_slice_t2": 10}))
    session.max_match_z_gap = float(raw.get("max_match_z_gap", DEFAULT_MAX_MATCH_Z_GAP))
    session.action_history = list(raw.get("action_history", []))
    _refresh_algo_matches(session, use_anchors=True)
    _save_last_session()
    return selected, stats


def _has_saved_session() -> bool:
    return LAST_SESSION_FILE.exists()


def _restore_last_session() -> tuple[models.SelectFilesResponse, models.SessionStats]:
    if not LAST_SESSION_FILE.exists():
        raise FileNotFoundError("No saved session found. Start with 'Choose Files and Load'.")
    raw = json.loads(LAST_SESSION_FILE.read_text(encoding="utf-8"))
    selected = models.SelectFilesResponse(**raw["selected_files"])
    stats = _load_session_internal(models.LoadSessionRequest(**selected.model_dump()))
    session = session_store.require_active_session()
    session.dendrite_links = list(raw.get("dendrite_links", []))
    session.review_decisions = dict(raw.get("review_decisions", {}))
    session.algo_matches = dict(raw.get("algo_matches", {}))
    session.confirmed_matches = list(raw.get("confirmed_matches", []))
    session.matched_t1_ids = set(str(x) for x in raw.get("matched_t1_ids", []))
    session.matched_t2_ids = set(str(x) for x in raw.get("matched_t2_ids", []))
    session.rejected_pairs = set(str(x) for x in raw.get("rejected_pairs", []))
    session.lost_t1_ids = set(str(x) for x in raw.get("lost_t1_ids", []))
    session.new_t2_ids = set(str(x) for x in raw.get("new_t2_ids", []))
    session.removed_t1_ids = set(str(x) for x in raw.get("removed_t1_ids", []))
    session.removed_t2_ids = set(str(x) for x in raw.get("removed_t2_ids", []))
    session.ignored_t1_ids = set(str(x) for x in raw.get("ignored_t1_ids", []))
    session.ignored_t2_ids = set(str(x) for x in raw.get("ignored_t2_ids", []))
    session.manual_t2_click_spines = list(raw.get("manual_t2_click_spines", []))
    session.manual_t1_click_spines = list(raw.get("manual_t1_click_spines", []))
    session.manual_click_matches = list(raw.get("manual_click_matches", []))
    _apply_manual_click_spines_to_lookup(session)
    session.viewer_state = dict(raw.get("viewer_state", {"modal_slice_t1": 10, "modal_slice_t2": 10}))
    session.max_match_z_gap = float(raw.get("max_match_z_gap", DEFAULT_MAX_MATCH_Z_GAP))
    # Backward compatibility if old session format lacked explicit sets.
    if not session.matched_t1_ids or not session.matched_t2_ids:
        for row in session.confirmed_matches:
            t1 = str(row.get("t1_spine_id", ""))
            t2 = str(row.get("t2_spine_id", ""))
            if t1:
                session.matched_t1_ids.add(t1)
            if t2:
                session.matched_t2_ids.add(t2)
        for row in session.review_decisions.values():
            if str(row.get("action", "")) == "match":
                t2 = str(row.get("t2_spine_id", ""))
                t1 = str(row.get("t1_spine_id", ""))
                if t1:
                    session.matched_t1_ids.add(t1)
                if t2:
                    session.matched_t2_ids.add(t2)
        for t2_id, m in session.algo_matches.items():
            t1_id = str(m.get("t1_spine_id", ""))
            if t1_id:
                session.matched_t1_ids.add(t1_id)
            session.matched_t2_ids.add(str(t2_id))
    _refresh_algo_matches(session, use_anchors=True)
    return selected, stats


def _open_new_session() -> None:
    if LAST_SESSION_DIR.exists():
        try:
            shutil.rmtree(LAST_SESSION_DIR)
        except OSError as exc:
            raise OSError(
                f"Cannot remove saved session folder {LAST_SESSION_DIR}: {exc}. "
                "Close this page's export preview, stop other tools using that folder, "
                "then try Open New Session again."
            ) from exc
    session_store.clear_active_session()


def _apply_manual_click_spines_to_lookup(session: session_store.LoadedSession) -> None:
    for row in session.manual_t2_click_spines:
        mid = str(row.get("manual_id", "")).strip()
        if not mid:
            continue
        session.t2_lookup[mid] = {
            "spine_id": mid,
            "dendrite_id": "manual_added",
            "x": float(row.get("x", np.nan)),
            "y": float(row.get("y", np.nan)),
            "z": float(row.get("z", np.nan)),
            "features": {},
        }
    for row in session.manual_t1_click_spines:
        mid = str(row.get("manual_id", "")).strip()
        if not mid:
            continue
        session.t1_lookup[mid] = {
            "spine_id": mid,
            "dendrite_id": "manual_added",
            "x": float(row.get("x", np.nan)),
            "y": float(row.get("y", np.nan)),
            "z": float(row.get("z", np.nan)),
            "features": {},
        }


def _ensure_default_dendrite_links(session: session_store.LoadedSession) -> None:
    if session.dendrite_links:
        return
    t1_ids = sorted([d for d, _ in baseline_adapter.dendrite_groups(session.t1_df)])
    t2_ids = sorted([d for d, _ in baseline_adapter.dendrite_groups(session.t2_df)])
    shared = sorted(list(set(t1_ids).intersection(set(t2_ids))))
    if shared:
        for did in shared:
            session.dendrite_links.append(
                {"link_id": f"auto_{did}", "t1_dendrite_ids": [did], "t2_dendrite_ids": [did], "notes": "auto-bootstrap"}
            )
        return
    if t1_ids and t2_ids:
        session.dendrite_links.append(
            {"link_id": "auto_all", "t1_dendrite_ids": t1_ids, "t2_dendrite_ids": t2_ids, "notes": "auto-bootstrap"}
        )


def _extract_manual_anchors(session: session_store.LoadedSession) -> List[dict]:
    out: List[dict] = []
    # Anchors come from explicitly confirmed manual matches and must persist.
    for row in session.confirmed_matches:
        t2_id = str(row.get("t2_spine_id", ""))
        t1_id = str(row.get("t1_spine_id", ""))
        if not t2_id or not t1_id:
            continue
        if t2_id not in session.t2_lookup or t1_id not in session.t1_lookup:
            continue
        out.append({"t2_spine_id": t2_id, "t1_spine_id": t1_id})
    # Backward compatibility for sessions that predate confirmed_matches.
    if not out:
        for row in session.review_decisions.values():
            if str(row.get("action", "")) != "match":
                continue
            t2_id = str(row.get("t2_spine_id", ""))
            t1_id = str(row.get("t1_spine_id", ""))
            if not t2_id or not t1_id:
                continue
            if t2_id not in session.t2_lookup or t1_id not in session.t1_lookup:
                continue
            out.append({"t2_spine_id": t2_id, "t1_spine_id": t1_id})
    return out


def _rebuild_decision_state(session: session_store.LoadedSession) -> None:
    matched_t1: set[str] = set()
    matched_t2: set[str] = set()
    rejected_pairs: set[str] = set()
    lost_t1: set[str] = set(session.lost_t1_ids)
    new_t2: set[str] = set(session.new_t2_ids)
    removed_t1: set[str] = set(session.removed_t1_ids)
    removed_t2: set[str] = set(session.removed_t2_ids)
    ignored_t1: set[str] = set(session.ignored_t1_ids)
    ignored_t2: set[str] = set(session.ignored_t2_ids)
    confirmed: list[dict[str, str]] = []
    for row in session.review_decisions.values():
        action = str(row.get("action", ""))
        t2_id = str(row.get("t2_spine_id", ""))
        t1_id = str(row.get("t1_spine_id", "")) if row.get("t1_spine_id") else ""
        if action == "match" and t1_id and t2_id:
            matched_t1.add(t1_id)
            matched_t2.add(t2_id)
            confirmed.append({"t1_spine_id": t1_id, "t2_spine_id": t2_id})
        elif action == "manual_match" and t2_id:
            matched_t2.add(t2_id)
            if t1_id and t1_id in session.t1_lookup:
                matched_t1.add(t1_id)
        elif action == "no_match" and t1_id and t2_id:
            rejected_pairs.add(f"{t1_id}|{t2_id}")
        elif action == "lost" and t1_id:
            lost_t1.add(t1_id)
        elif action == "new" and t2_id:
            new_t2.add(t2_id)
        elif action == "remove_t1" and t1_id:
            removed_t1.add(t1_id)
        elif action == "remove_t2" and t2_id:
            removed_t2.add(t2_id)
        elif action == "ignore_t1" and t1_id:
            ignored_t1.add(t1_id)
        elif action == "ignore_t2" and t2_id:
            ignored_t2.add(t2_id)
    for t2_id, m in session.algo_matches.items():
        t2s = str(t2_id)
        t1s = str(m.get("t1_spine_id", "")) if m.get("t1_spine_id") else ""
        matched_t2.add(t2s)
        if t1s:
            matched_t1.add(t1s)
    session.confirmed_matches = confirmed
    session.rejected_pairs = rejected_pairs
    session.lost_t1_ids = lost_t1
    session.new_t2_ids = new_t2
    session.removed_t1_ids = removed_t1
    session.removed_t2_ids = removed_t2
    session.ignored_t1_ids = ignored_t1
    session.ignored_t2_ids = ignored_t2
    session.matched_t1_ids = matched_t1
    session.matched_t2_ids = matched_t2


def _capture_undo_snapshot(session: session_store.LoadedSession, action_label: str) -> None:
    snap = {
        "action_label": action_label,
        "review_decisions": copy.deepcopy(session.review_decisions),
        "algo_matches": copy.deepcopy(session.algo_matches),
        "confirmed_matches": copy.deepcopy(session.confirmed_matches),
        "matched_t1_ids": sorted(list(session.matched_t1_ids)),
        "matched_t2_ids": sorted(list(session.matched_t2_ids)),
        "rejected_pairs": sorted(list(session.rejected_pairs)),
        "lost_t1_ids": sorted(list(session.lost_t1_ids)),
        "new_t2_ids": sorted(list(session.new_t2_ids)),
        "removed_t1_ids": sorted(list(session.removed_t1_ids)),
        "removed_t2_ids": sorted(list(session.removed_t2_ids)),
        "ignored_t1_ids": sorted(list(session.ignored_t1_ids)),
        "ignored_t2_ids": sorted(list(session.ignored_t2_ids)),
        "manual_t2_click_spines": copy.deepcopy(session.manual_t2_click_spines),
        "manual_t1_click_spines": copy.deepcopy(session.manual_t1_click_spines),
        "manual_click_matches": copy.deepcopy(session.manual_click_matches),
        "max_match_z_gap": float(session.max_match_z_gap),
    }
    session.action_history.append(snap)
    if len(session.action_history) > 200:
        session.action_history = session.action_history[-200:]


def _restore_undo_snapshot(session: session_store.LoadedSession, snap: dict) -> None:
    session.review_decisions = copy.deepcopy(snap.get("review_decisions", {}))
    session.algo_matches = copy.deepcopy(snap.get("algo_matches", {}))
    session.confirmed_matches = copy.deepcopy(snap.get("confirmed_matches", []))
    session.matched_t1_ids = set(str(x) for x in snap.get("matched_t1_ids", []))
    session.matched_t2_ids = set(str(x) for x in snap.get("matched_t2_ids", []))
    session.rejected_pairs = set(str(x) for x in snap.get("rejected_pairs", []))
    session.lost_t1_ids = set(str(x) for x in snap.get("lost_t1_ids", []))
    session.new_t2_ids = set(str(x) for x in snap.get("new_t2_ids", []))
    session.removed_t1_ids = set(str(x) for x in snap.get("removed_t1_ids", []))
    session.removed_t2_ids = set(str(x) for x in snap.get("removed_t2_ids", []))
    session.ignored_t1_ids = set(str(x) for x in snap.get("ignored_t1_ids", []))
    session.ignored_t2_ids = set(str(x) for x in snap.get("ignored_t2_ids", []))
    session.manual_t2_click_spines = copy.deepcopy(snap.get("manual_t2_click_spines", []))
    session.manual_t1_click_spines = copy.deepcopy(snap.get("manual_t1_click_spines", []))
    session.manual_click_matches = copy.deepcopy(snap.get("manual_click_matches", []))
    if "max_match_z_gap" in snap:
        session.max_match_z_gap = float(snap.get("max_match_z_gap", DEFAULT_MAX_MATCH_Z_GAP))


def _refresh_algo_matches(session: session_store.LoadedSession, use_anchors: bool, nearby_xy: float = 140.0) -> None:
    zmax = _session_z_gap(session)
    preserved_t1 = set(session.matched_t1_ids)
    preserved_t2 = set(session.matched_t2_ids)
    existing_algo = dict(session.algo_matches)
    _ensure_default_dendrite_links(session)
    t1_df = session.t1_df.copy()
    t2_df = session.t2_df.copy()
    if "dendrite_id" not in t1_df.columns or "dendrite_id" not in t2_df.columns:
        session.algo_matches = existing_algo
        session.matched_t1_ids.update(preserved_t1)
        session.matched_t2_ids.update(preserved_t2)
        return
    t1_df["dendrite_id"] = t1_df["dendrite_id"].astype(str)
    t2_df["dendrite_id"] = t2_df["dendrite_id"].astype(str)

    manual_anchors = _extract_manual_anchors(session)
    manual_t2 = {str(a["t2_spine_id"]) for a in manual_anchors}.union(session.removed_t2_ids).union(session.new_t2_ids).union(session.ignored_t2_ids)
    manual_t1 = {str(a["t1_spine_id"]) for a in manual_anchors}.union(session.removed_t1_ids).union(session.lost_t1_ids).union(session.ignored_t1_ids)
    # Keep previously accepted algo matches; add to them, don't drop them.

    allowed_t1_by_t2 = baseline_adapter.linked_dendrite_map(session.dendrite_links)

    candidate_rows: List[dict] = []
    for _, t2 in t2_df.iterrows():
        t2_id = str(t2["id"])
        if t2_id in manual_t2:
            continue
        allowed_d = allowed_t1_by_t2.get(str(t2["dendrite_id"]), set())
        if not allowed_d:
            continue
        t1_sub = t1_df[t1_df["dendrite_id"].isin(allowed_d)]
        if manual_t1:
            t1_sub = t1_sub[~t1_sub["id"].astype(str).isin(manual_t1)]
        for _, t1 in t1_sub.iterrows():
            t1_id = str(t1["id"])
            pair_key = f"{t1_id}|{t2_id}"
            if pair_key in session.rejected_pairs:
                continue
            dz_gap = float(abs(float(t1["z"]) - float(t2["z"])))
            if dz_gap > zmax:
                continue
            candidate_rows.append(
                {
                    "t1_spine_id": t1_id,
                    "t2_spine_id": t2_id,
                    "distance_xy": float(np.hypot(float(t1["x"]) - float(t2["x"]), float(t1["y"]) - float(t2["y"]))),
                    "distance_z": dz_gap,
                }
            )
    if not candidate_rows:
        session.algo_matches = existing_algo
        session.matched_t1_ids.update(preserved_t1)
        session.matched_t2_ids.update(preserved_t2)
        return

    anchors = manual_anchors if use_anchors else []
    scored = baseline_adapter.score_candidates_hybrid(
        pd.DataFrame(candidate_rows),
        t1_df,
        t2_df,
        anchors=anchors,
        nearby_xy=nearby_xy,
        gating_z=zmax,
    ).sort_values("final_score", ascending=False)

    taken_t2 = set(manual_t2).union(set(str(k) for k in existing_algo.keys()))
    taken_t1 = set(manual_t1).union(set(str(v.get("t1_spine_id", "")) for v in existing_algo.values()))
    auto_new: dict[str, dict] = {}
    for _, r in scored.iterrows():
        t2_id = str(r["t2_spine_id"])
        t1_id = str(r["t1_spine_id"])
        s = float(r.get("final_score", 0.0))
        if s < AUTO_MATCH_THRESHOLD:
            continue
        if t2_id in taken_t2 or t1_id in taken_t1:
            continue
        auto_new[t2_id] = {"t1_spine_id": t1_id, "final_score": s}
        taken_t2.add(t2_id)
        taken_t1.add(t1_id)
    merged_algo = dict(existing_algo)
    merged_algo.update(auto_new)
    session.algo_matches = merged_algo
    # Sync promoted/known algo matches into exclusion sets immediately.
    for t2_id, m in merged_algo.items():
        session.matched_t2_ids.add(str(t2_id))
        t1_id = m.get("t1_spine_id")
        if t1_id:
            session.matched_t1_ids.add(str(t1_id))
    session.matched_t1_ids.update(preserved_t1)
    session.matched_t2_ids.update(preserved_t2)
    _rebuild_decision_state(session)


def _build_review_queue(
    *,
    offset: int,
    limit: int,
    use_local_registration: bool,
    nearby_xy: float,
    allow_cross_dendrite: bool = False,
) -> models.ReviewQueueResponse:
    session = session_store.require_active_session()
    zmax = _session_z_gap(session)
    if not session.dendrite_links:
        return models.ReviewQueueResponse(offset=offset, limit=limit, total_candidates=0, items=[])

    not_in_t1_t2_ids = _collect_not_in_t1_t2_ids(session)
    matched_t2_ids = set(session.matched_t2_ids).union(session.removed_t2_ids).union(session.new_t2_ids).union(session.ignored_t2_ids).union(not_in_t1_t2_ids)
    matched_t1_ids = set(session.matched_t1_ids).union(session.removed_t1_ids).union(session.lost_t1_ids).union(session.ignored_t1_ids)

    allowed_t1_by_t2 = baseline_adapter.linked_dendrite_map(session.dendrite_links)

    t1_df = session.t1_df.copy()
    t2_df = session.t2_df.copy()
    if "dendrite_id" not in t1_df.columns or "dendrite_id" not in t2_df.columns:
        return models.ReviewQueueResponse(offset=offset, limit=limit, total_candidates=0, items=[])
    t1_df["dendrite_id"] = t1_df["dendrite_id"].astype(str)
    t2_df["dendrite_id"] = t2_df["dendrite_id"].astype(str)
    t1_df, t2_df = baseline_adapter.exclude_matched_spines(t1_df, t2_df, matched_t1_ids, matched_t2_ids)

    anchors: List[dict] = _extract_manual_anchors(session) if use_local_registration else []

    candidate_rows: List[dict] = []
    for _, t2 in t2_df.iterrows():
        t2_id = str(t2["id"])
        if t2_id in matched_t2_ids:
            continue
        t2_d = str(t2["dendrite_id"])
        allowed_t1_d = allowed_t1_by_t2.get(t2_d, set())
        if not allow_cross_dendrite:
            if not allowed_t1_d:
                continue
            t1_sub = t1_df[t1_df["dendrite_id"].isin(allowed_t1_d)]
        else:
            t1_sub = t1_df
        if matched_t1_ids:
            t1_sub = t1_sub[~t1_sub["id"].astype(str).isin(matched_t1_ids)]
        if t1_sub.empty:
            continue

        for _, t1 in t1_sub.iterrows():
            t1_id = str(t1["id"])
            pair_key = f"{t1_id}|{t2_id}"
            if pair_key in session.rejected_pairs:
                continue
            dz_gap = float(abs(float(t1["z"]) - float(t2["z"])))
            if dz_gap > zmax:
                continue
            candidate_rows.append(
                {
                    "t1_spine_id": t1_id,
                    "t2_spine_id": t2_id,
                    "distance_xy": float(np.hypot(float(t1["x"]) - float(t2["x"]), float(t1["y"]) - float(t2["y"]))),
                    "distance_z": dz_gap,
                }
            )

    if not candidate_rows:
        return models.ReviewQueueResponse(offset=offset, limit=limit, total_candidates=0, items=[])

    cand = pd.DataFrame(candidate_rows)
    scored = baseline_adapter.score_candidates_hybrid(
        cand,
        t1_df,
        t2_df,
        anchors=anchors if use_local_registration else None,
        nearby_xy=nearby_xy,
        gating_z=zmax,
    )
    scored = scored.sort_values(["t2_spine_id", "final_score"], ascending=[True, False]).copy()

    items: List[models.ReviewQueueItem] = []
    for t2_id, grp_all in scored.groupby("t2_spine_id"):
        grp = grp_all.head(2).copy()
        first = grp.iloc[0]
        second = grp.iloc[1] if len(grp) > 1 else None
        margin = float(first["final_score"] - (float(second["final_score"]) if second is not None else 0.0))
        t2_row = t2_df[t2_df["id"].astype(str) == str(t2_id)].iloc[0]
        t1_row = t1_df[t1_df["id"].astype(str) == str(first["t1_spine_id"])].iloc[0]
        nearby = []
        for _, c in grp_all.head(5).iterrows():
            t1_row_near = t1_df[t1_df["id"].astype(str) == str(c["t1_spine_id"])].iloc[0]
            nearby.append(
                {
                    "t1_spine_id": str(c["t1_spine_id"]),
                    "distance_xy": float(c["distance_xy"]),
                    "distance_z": float(c["distance_z"]),
                    "final_score": float(c.get("final_score", np.nan)),
                    "x": float(t1_row_near["x"]),
                    "y": float(t1_row_near["y"]),
                    "z": float(t1_row_near["z"]),
                }
            )
        t2_curr = t2_df[t2_df["id"].astype(str) == str(t2_id)].iloc[0]
        same_d = t2_df[t2_df["dendrite_id"].astype(str) == str(t2_curr["dendrite_id"])].copy()
        same_d["distance_xy"] = np.hypot(
            same_d["x"].astype(float) - float(t2_curr["x"]),
            same_d["y"].astype(float) - float(t2_curr["y"]),
        )
        nearby_t2 = []
        for _, c2 in same_d.sort_values("distance_xy", ascending=True).head(5).iterrows():
            nearby_t2.append(
                {
                    "t2_spine_id": str(c2["id"]),
                    "distance_xy": float(c2["distance_xy"]),
                    "x": float(c2["x"]),
                    "y": float(c2["y"]),
                    "z": float(c2["z"]),
                }
            )
        items.append(
            models.ReviewQueueItem(
                t2_spine_id=str(t2_id),
                t2_dendrite_id=str(t2_row["dendrite_id"]),
                suggested_t1_spine_id=str(first["t1_spine_id"]),
                suggested_t1_dendrite_id=str(t1_row["dendrite_id"]),
                distance_xy=float(first["distance_xy"]),
                distance_z=float(first["distance_z"]),
                score_feature_weighted=float(first.get("score_feature_weighted", np.nan)),
                score_toolb_model=float(first.get("score_toolb_model", np.nan))
                if np.isfinite(float(first.get("score_toolb_model", np.nan)))
                else None,
                stability_score=float(first.get("stability_score", np.nan)),
                final_score=float(first.get("final_score", np.nan)),
                margin=margin,
                local_shift_xyz={"x": 0.0, "y": 0.0, "z": 0.0},
                registration_applied=bool(use_local_registration and len(anchors) > 0),
                nearby_t1_candidates=nearby,
                nearby_t2_candidates=nearby_t2,
            )
        )

    items.sort(key=lambda r: (r.final_score if r.final_score is not None else -1), reverse=True)
    total = len(items)
    start = max(0, min(offset, total))
    eff_limit = min(max(0, limit), 100)
    if eff_limit == 0:
        return models.ReviewQueueResponse(offset=start, limit=limit, total_candidates=total, items=[])
    end = min(total, start + eff_limit)
    return models.ReviewQueueResponse(offset=start, limit=limit, total_candidates=total, items=items[start:end])


def _build_algo_review_queue(*, offset: int, limit: int) -> models.ReviewQueueResponse:
    session = session_store.require_active_session()
    zmax = _session_z_gap(session)
    if not session.algo_matches:
        return models.ReviewQueueResponse(offset=offset, limit=limit, total_candidates=0, items=[])
    allowed_t1_by_t2 = baseline_adapter.linked_dendrite_map(session.dendrite_links)
    reviewed_t2 = set(str(k) for k in session.review_decisions.keys())
    excluded_t2 = set(session.removed_t2_ids).union(session.new_t2_ids).union(session.ignored_t2_ids).union(_collect_not_in_t1_t2_ids(session)).union(reviewed_t2)
    excluded_t1 = set(session.removed_t1_ids).union(session.lost_t1_ids).union(session.ignored_t1_ids)
    for row in session.review_decisions.values():
        if str(row.get("action", "")) == "match":
            t1_id = str(row.get("t1_spine_id", "")) if row.get("t1_spine_id") else ""
            if t1_id:
                excluded_t1.add(t1_id)
    pairs: list[tuple[float, str, str]] = []
    for t2_id, m in session.algo_matches.items():
        t2s = str(t2_id)
        t1s = str(m.get("t1_spine_id", ""))
        if not t1s:
            continue
        if t2s in excluded_t2 or t1s in excluded_t1:
            continue
        t2_row_chk = session.t2_lookup.get(t2s)
        t1_row_chk = session.t1_lookup.get(t1s)
        if t2_row_chk and t1_row_chk:
            if float(abs(float(t1_row_chk["z"]) - float(t2_row_chk["z"]))) > zmax:
                continue
        pairs.append((float(m.get("final_score", 0.0)), t2s, t1s))
    pairs.sort(key=lambda x: x[0])  # lower confidence first
    total = len(pairs)
    if total == 0:
        return models.ReviewQueueResponse(offset=offset, limit=limit, total_candidates=0, items=[])
    start = max(0, offset)
    stop = min(total, start + max(1, min(limit, 50)))
    page = pairs[start:stop]
    items: list[models.ReviewQueueItem] = []
    t1_df = session.t1_df.copy()
    t2_df = session.t2_df.copy()
    if "dendrite_id" in t1_df.columns:
        t1_df["dendrite_id"] = t1_df["dendrite_id"].astype(str)
    if "dendrite_id" in t2_df.columns:
        t2_df["dendrite_id"] = t2_df["dendrite_id"].astype(str)
    for score, t2_id, t1_id in page:
        t2 = session.t2_lookup.get(t2_id)
        t1 = session.t1_lookup.get(t1_id)
        if not t2 or not t1:
            continue
        dxy = float(np.hypot(float(t1["x"]) - float(t2["x"]), float(t1["y"]) - float(t2["y"])))
        dz = float(abs(float(t1["z"]) - float(t2["z"])))
        if dz > zmax:
            continue
        t2_d = str(t2.get("dendrite_id", ""))
        allowed_d = allowed_t1_by_t2.get(t2_d, set())
        cand = t1_df.copy()
        if allowed_d and "dendrite_id" in cand.columns:
            cand = cand[cand["dendrite_id"].isin(allowed_d)]
        if cand.empty:
            cand = t1_df.copy()
        cand = cand.assign(
            _dxy=np.hypot(cand["x"].astype(float) - float(t2["x"]), cand["y"].astype(float) - float(t2["y"])),
            _dz=np.abs(cand["z"].astype(float) - float(t2["z"])),
        )
        cand = cand[cand["_dz"] <= zmax].sort_values(["_dxy", "_dz"], ascending=[True, True])
        nearby_t1: list[dict] = []
        for _, r in cand.head(8).iterrows():
            nearby_t1.append(
                {
                    "t1_spine_id": str(r["id"]),
                    "distance_xy": float(r["_dxy"]),
                    "distance_z": float(r["_dz"]),
                    "final_score": max(0.0, 1.0 - float(r["_dxy"]) / 120.0 - 0.1 * float(r["_dz"])),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "z": float(r["z"]),
                }
            )
        if all(str(c.get("t1_spine_id", "")) != t1_id for c in nearby_t1):
            nearby_t1.insert(
                0,
                {
                    "t1_spine_id": t1_id,
                    "distance_xy": dxy,
                    "distance_z": dz,
                    "final_score": float(score),
                    "x": float(t1["x"]),
                    "y": float(t1["y"]),
                    "z": float(t1["z"]),
                },
            )
        items.append(
            models.ReviewQueueItem(
                t2_spine_id=t2_id,
                t2_dendrite_id=t2_d,
                suggested_t1_spine_id=t1_id,
                suggested_t1_dendrite_id=str(t1.get("dendrite_id", "")),
                distance_xy=dxy,
                distance_z=dz,
                final_score=float(score),
                margin=None,
                nearby_t1_candidates=nearby_t1,
                nearby_t2_candidates=[
                    {
                        "t2_spine_id": t2_id,
                        "distance_xy": 0.0,
                        "distance_z": 0.0,
                        "x": float(t2["x"]),
                        "y": float(t2["y"]),
                        "z": float(t2["z"]),
                    }
                ],
            )
        )
    return models.ReviewQueueResponse(offset=start, limit=max(1, min(limit, 50)), total_candidates=total, items=items)


def _build_top5_next() -> models.Top5NextResponse:
    session = session_store.require_active_session()
    zmax = _session_z_gap(session)
    if not session.dendrite_links:
        return models.Top5NextResponse(has_item=False, candidates=[])
    t1_df = session.t1_df.copy()
    t2_df = session.t2_df.copy()
    if "dendrite_id" not in t1_df.columns or "dendrite_id" not in t2_df.columns:
        return models.Top5NextResponse(has_item=False, candidates=[])
    t1_df["dendrite_id"] = t1_df["dendrite_id"].astype(str)
    t2_df["dendrite_id"] = t2_df["dendrite_id"].astype(str)

    # Strict exclusion pool: never resurface these.
    excluded_t1 = set(session.matched_t1_ids).union(session.removed_t1_ids).union(session.lost_t1_ids).union(session.ignored_t1_ids)
    excluded_t2 = set(session.matched_t2_ids).union(session.removed_t2_ids).union(session.new_t2_ids).union(session.ignored_t2_ids).union(_collect_not_in_t1_t2_ids(session))
    reviewed_t2 = set(str(k) for k in session.review_decisions.keys())
    excluded_t2.update(reviewed_t2)
    for row in session.review_decisions.values():
        if str(row.get("action", "")) in {"lost", "remove_t1"}:
            t1 = str(row.get("t1_spine_id", "")) if row.get("t1_spine_id") else ""
            if t1:
                excluded_t1.add(t1)

    allowed_t1_by_t2 = baseline_adapter.linked_dendrite_map(session.dendrite_links)
    allowed_t2_by_t1: dict[str, set[str]] = {}
    for t2_d, t1_set in allowed_t1_by_t2.items():
        for t1_d in t1_set:
            allowed_t2_by_t1.setdefault(str(t1_d), set()).add(str(t2_d))

    remaining_t1 = t1_df[~t1_df["id"].astype(str).isin(excluded_t1)].copy()
    if remaining_t1.empty:
        return models.Top5NextResponse(has_item=False, candidates=[])

    # Deterministic order for manual loop.
    remaining_t1 = remaining_t1.sort_values("id", ascending=True)
    for _, t1 in remaining_t1.iterrows():
        t1_id = str(t1["id"])
        allowed_t2_d = allowed_t2_by_t1.get(str(t1["dendrite_id"]), set())
        if not allowed_t2_d:
            continue
        t2_sub = t2_df[t2_df["dendrite_id"].isin(allowed_t2_d)].copy()
        if t2_sub.empty:
            continue
        t2_sub = t2_sub[~t2_sub["id"].astype(str).isin(excluded_t2)]
        if t2_sub.empty:
            continue

        candidate_rows: list[dict] = []
        for _, t2 in t2_sub.iterrows():
            t2_id = str(t2["id"])
            pair_key = f"{t1_id}|{t2_id}"
            if pair_key in session.rejected_pairs:
                continue
            dz_gap = float(abs(float(t1["z"]) - float(t2["z"])))
            if dz_gap > zmax:
                continue
            candidate_rows.append(
                {
                    "t1_spine_id": t1_id,
                    "t2_spine_id": t2_id,
                    "distance_xy": float(np.hypot(float(t1["x"]) - float(t2["x"]), float(t1["y"]) - float(t2["y"]))),
                    "distance_z": dz_gap,
                }
            )
        if not candidate_rows:
            continue
        scored = baseline_adapter.score_candidates_hybrid(
            pd.DataFrame(candidate_rows),
            t1_df,
            t2_df,
            anchors=None,
            nearby_xy=140.0,
            gating_z=zmax,
        ).sort_values("final_score", ascending=False)
        top = scored.head(5)
        candidates: list[models.Top5Candidate] = []
        for _, r in top.iterrows():
            t2_id = str(r["t2_spine_id"])
            t2_row = session.t2_lookup.get(t2_id, {})
            candidates.append(
                models.Top5Candidate(
                    t2_spine_id=t2_id,
                    t2_dendrite_id=str(t2_row.get("dendrite_id", "")) if t2_row else None,
                    distance_xy=float(r.get("distance_xy", np.nan)),
                    distance_z=float(r.get("distance_z", np.nan)),
                    final_score=float(r.get("final_score", np.nan)),
                    x=float(t2_row.get("x", np.nan)) if t2_row else None,
                    y=float(t2_row.get("y", np.nan)) if t2_row else None,
                    z=float(t2_row.get("z", np.nan)) if t2_row else None,
                )
            )
        return models.Top5NextResponse(
            has_item=True,
            t1_spine_id=t1_id,
            t1_dendrite_id=str(t1["dendrite_id"]),
            t1_xyz={"x": float(t1["x"]), "y": float(t1["y"]), "z": float(t1["z"])},
            candidates=candidates,
        )
    return models.Top5NextResponse(has_item=False, candidates=[])


def _safe_export_folder_name(name: str) -> str:
    sanitized = "".join(c if (c.isalnum() or c in {"-", "_", " "}) else "_" for c in str(name))
    sanitized = "_".join(sanitized.strip().split())
    return sanitized[:120] if sanitized else "spine_annotator_export"


def _collect_non_matched_dendrite_spines(
    session: session_store.LoadedSession,
) -> tuple[set[str], set[str], list[dict[str, str]]]:
    """
    Spines that belong to dendrites not covered by any T1<->T2 dendrite link.
    These are excluded from final biological calculations and exported separately.
    """
    if "dendrite_id" not in session.t1_df.columns or "dendrite_id" not in session.t2_df.columns:
        return set(), set(), []

    allowed_t1_by_t2 = baseline_adapter.linked_dendrite_map(session.dendrite_links)
    matched_t2_dendrites = {str(d) for d in allowed_t1_by_t2.keys()}
    matched_t1_dendrites = {str(d) for vals in allowed_t1_by_t2.values() for d in vals}

    excluded_t1_ids: set[str] = set()
    excluded_t2_ids: set[str] = set()
    rows: list[dict[str, str]] = []

    t1_work = session.t1_df.copy()
    t2_work = session.t2_df.copy()
    t1_work["dendrite_id"] = t1_work["dendrite_id"].astype(str)
    t2_work["dendrite_id"] = t2_work["dendrite_id"].astype(str)

    t1_excluded = t1_work[~t1_work["dendrite_id"].isin(matched_t1_dendrites)]
    for _, r in t1_excluded.iterrows():
        sid = str(r["id"])
        did = str(r["dendrite_id"])
        excluded_t1_ids.add(sid)
        rows.append(
            {
                "timepoint": "t1",
                "spine_id": sid,
                "dendrite_id": did,
                "reason": "dendrite_not_linked_between_timepoints",
            }
        )

    t2_excluded = t2_work[~t2_work["dendrite_id"].isin(matched_t2_dendrites)]
    for _, r in t2_excluded.iterrows():
        sid = str(r["id"])
        did = str(r["dendrite_id"])
        excluded_t2_ids.add(sid)
        rows.append(
            {
                "timepoint": "t2",
                "spine_id": sid,
                "dendrite_id": did,
                "reason": "dendrite_not_linked_between_timepoints",
            }
        )

    return excluded_t1_ids, excluded_t2_ids, rows


def _collect_not_in_t1_t2_ids(session: session_store.LoadedSession) -> set[str]:
    out: set[str] = set()
    for row in session.review_decisions.values():
        if str(row.get("action", "")) == "not_in_t1":
            out.add(str(row.get("t2_spine_id", "")))
    return out


def _strict_export_unclassified_counts(
    session: session_store.LoadedSession,
    *,
    excluded_t1_nonmatch: set[str],
    excluded_t2_nonmatch: set[str],
    matched_t1: set[str],
    matched_t2: set[str],
    new_t2: set[str],
    lost_t1: set[str],
    removed_t1: set[str],
    removed_t2: set[str],
    ignored_t1: set[str],
    ignored_t2: set[str],
) -> tuple[set[str], set[str]]:
    """
    Strict gate for export integrity:
    every eligible spine must be classified into one of:
      matched / new / lost / removed_* / ignored_*
    """
    eligible_t1 = set(str(k) for k in session.t1_lookup.keys()) - set(excluded_t1_nonmatch)
    eligible_t2 = set(str(k) for k in session.t2_lookup.keys()) - set(excluded_t2_nonmatch)
    classified_t1 = set(matched_t1).union(set(lost_t1), set(removed_t1), set(ignored_t1))
    classified_t2 = set(matched_t2).union(set(new_t2), set(removed_t2), set(ignored_t2))
    unclassified_t1 = eligible_t1 - classified_t1
    unclassified_t2 = eligible_t2 - classified_t2
    return unclassified_t1, unclassified_t2


def _compute_inferred_new_t2_candidates(
    session: session_store.LoadedSession,
    *,
    excluded_t2_nonmatch: set[str],
    excluded_t1_nonmatch: set[str],
) -> list[str]:
    matched_t2 = set(session.matched_t2_ids) - excluded_t2_nonmatch
    matched_t1 = set(session.matched_t1_ids) - excluded_t1_nonmatch
    removed_t2 = set(session.removed_t2_ids) - excluded_t2_nonmatch
    removed_t1 = set(session.removed_t1_ids) - excluded_t1_nonmatch
    explicit_new_t2 = set(session.new_t2_ids) - excluded_t2_nonmatch
    all_t2 = {str(k) for k in session.t2_lookup.keys()} - excluded_t2_nonmatch
    all_t1 = {str(k) for k in session.t1_lookup.keys()} - excluded_t1_nonmatch
    _ = all_t1 - matched_t1 - removed_t1  # Keeps logic parallel with finalize.
    return sorted(list(explicit_new_t2.union(all_t2 - matched_t2 - removed_t2)))


def _build_new_validation_queue(session: session_store.LoadedSession) -> list[str]:
    excluded_t1_nonmatch, excluded_t2_nonmatch, _rows = _collect_non_matched_dendrite_spines(session)
    inferred_new = _compute_inferred_new_t2_candidates(
        session,
        excluded_t2_nonmatch=excluded_t2_nonmatch,
        excluded_t1_nonmatch=excluded_t1_nonmatch,
    )
    reviewed = {
        str(row.get("t2_spine_id", ""))
        for row in session.review_decisions.values()
        if str(row.get("action", "")) in {"new", "remove_t2", "not_in_t1", "ignore_t2", "manual_match"}
    }
    return [sid for sid in inferred_new if sid not in reviewed]


def _ensure_t2_is_new_validation_candidate(session: session_store.LoadedSession, t2_spine_id: str) -> None:
    t2 = str(t2_spine_id)
    pending = set(_build_new_validation_queue(session))
    if t2 not in pending:
        raise ValueError(
            f"T2 spine '{t2}' is not currently classified as pending T2-new validation."
        )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Browsers request this automatically; no icon bundled."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def simple_home() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Spine Annotator - Dendrite Match</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; max-width: 1000px; padding-left: 160px; }
    button { font-size: 15px; padding: 10px 14px; margin-right: 8px; margin-bottom: 8px; }
    select { min-width: 220px; min-height: 140px; padding: 6px; }
    .row { margin: 10px 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
    .canvas-wrap { display: flex; gap: 16px; margin-top: 12px; }
    canvas { border: 1px solid #ddd; image-rendering: pixelated; }
    pre { background: #f5f5f5; padding: 12px; border-radius: 6px; overflow: auto; }
    .queue-grid { display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 12px; margin-top: 10px; }
    .mode-tabs { display:flex; gap:8px; margin-top:10px; margin-bottom:8px; }
    .mode-tab { border:1px solid #bbb; background:#f8f8f8; border-radius:8px; padding:8px 12px; cursor:pointer; }
    .mode-tab.active { background:#e8f2ff; border-color:#4f7fd8; }
    .qcard { border: 1px solid #ddd; border-radius: 8px; padding: 8px; background: #fff; }
    .qrow { display: flex; gap: 8px; margin-top: 6px; }
    .qimg { width: 180px; height: 180px; }
    .qactions button { font-size: 12px; padding: 4px 8px; margin-right: 5px; }
    .picked { background: #d7f5dc; }
    .left-runner {
      position: fixed;
      left: 10px;
      top: 220px;
      z-index: 50;
      background: #ffffff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .right-stats {
      position: fixed;
      right: 10px;
      top: 140px;
      z-index: 40;
      background: #ffffff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 220px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      font-size: 14px;
      line-height: 1.6;
    }
    .modal-viewport {
      width: 520px;
      height: 520px;
      border: 1px solid #ddd;
      overflow: hidden;
      position: relative;
      cursor: grab;
      background: #111;
    }
    .modal-viewport.grabbing { cursor: grabbing; }
    .slice-overlay {
      position: absolute;
      right: 8px;
      top: 8px;
      z-index: 5;
      color: #fff;
      background: rgba(0, 0, 0, 0.55);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 12px;
      line-height: 1.3;
      text-shadow: 0 1px 2px rgba(0,0,0,0.8);
      pointer-events: none;
      user-select: none;
    }
    .modal-canvas {
      width: 520px;
      height: 520px;
      transform-origin: 0 0;
      will-change: transform;
      transition: transform 0.2s ease-out;
      image-rendering: pixelated;
      display: block;
    }
  </style>
</head>
<body>
  <div class="left-runner">
    <button onclick="runLocalRegistration()">Run Local Re-Registration</button>
  </div>
  <div class="right-stats">
    <div><b>Progress</b></div>
    <div>manual matched = <span id="statManualMatched">0</span></div>
    <div>algo_matched = <span id="statAlgoMatched">0</span></div>
    <div>to review = <span id="statToReview">0</span></div>
    <div>t1 lost = <span id="statT1Lost">0</span></div>
    <div>t2 new = <span id="statT2New">0</span></div>
    <div style="margin-top:10px; font-size:13px;">
      <label for="maxMatchZGapInput">Max |Δz| (T1↔T2)</label><br/>
      <input id="maxMatchZGapInput" type="number" min="0" max="100" step="0.5" value="7" style="width:56px; margin-top:4px;" />
      <button onclick="applyMaxMatchZGap()" style="font-size:12px; padding:3px 8px; margin-top:4px;">Apply</button>
    </div>
    <div style="margin-top:6px;">
      <button onclick="undoLastChoice()">Undo Last Choice</button>
    </div>
  </div>
  <h2>Spine Annotator (Phase 2 - Dendrite Match)</h2>
  <p>Only dendrite linking in this phase: full FOV preview for both timepoints, selected dendrite spines highlighted in red, then create T1↔T2 links (many-to-many allowed).</p>
  <button onclick="useLastSaved()">Use Last Saved Dataset</button>
  <button onclick="undoLastSession()">Undo Last Session</button>
  <button onclick="openNewSession()">Open New Session</button>
  <button onclick="selectAndLoad()">Choose Files and Load</button>
  <button onclick="refreshDendrites()">Refresh Dendrite IDs</button>
  <button onclick="previewSelectedDendrites()">Refresh FOV Preview</button>
  <button onclick="createLink()">Create Link</button>
  <button onclick="showLinks()">Show Links</button>
  <button onclick="clearLinks()">Clear Links</button>
  <button onclick="saveNow()">Save Now</button>
  <button onclick="exportResults()">Export Results</button>
  <button onclick="loadTop5Next()">Load Top 5 Matcher</button>
  <button onclick="loadAlgoReviewQueue()">Review Algo Matches (2-window)</button>
  <button onclick="openDetachedCompareViewer()">Open Detached Large Viewer</button>
  <button onclick="rerunAlgoOnly()">Re-run algo (saved matches)</button>
  <button onclick="finalizeMatches()">Finalize New/Lost Inference</button>
  <button onclick="loadNewValidationNext()">Validate New T2 (Final Pass)</button>
  <div class="mode-tabs">
    <button id="tabLargeFov" class="mode-tab active" onclick="setMatchMode('large')">Large FOV Match (Manual)</button>
    <button id="tabTop5" class="mode-tab" onclick="setMatchMode('top5')">Top 5 Fast Match</button>
  </div>
  <div class="grid">
    <div>
      <div><b>T1 Dendrite IDs</b> (multi-select)</div>
      <select id="t1Dendrites" multiple></select>
    </div>
    <div>
      <div><b>T2 Dendrite IDs</b> (multi-select)</div>
      <select id="t2Dendrites" multiple></select>
    </div>
  </div>
  <div class="row">
    <input id="linkNotes" placeholder="optional notes" style="width: 360px; padding: 8px;" />
  </div>
  <div id="largeModePanel">
    <div class="canvas-wrap">
      <div>
        <div><b>T1 Full FOV</b></div>
        <canvas id="t1Canvas" width="420" height="320"></canvas>
      </div>
      <div>
        <div><b>T2 Full FOV</b></div>
        <canvas id="t2Canvas" width="420" height="320"></canvas>
      </div>
    </div>
    <div id="queue" class="queue-grid"></div>
  </div>
  <div id="top5Panel" style="display:none; margin-top:12px; border:1px solid #ddd; border-radius:10px; padding:10px;">
    <div style="font-weight:700; margin-bottom:8px;">Top 5 Manual Matching (T1 vs T2 candidates)</div>
    <div style="display:grid; grid-template-columns: 1fr 2fr; gap:12px;">
      <div style="border:1px solid #eee; border-radius:8px; padding:8px;">
        <div><b>T1 Current</b></div>
        <div id="top5T1Meta" style="font-size:12px; margin:4px 0;"></div>
        <canvas id="top5T1Canvas" width="260" height="260" style="border:1px solid #ddd;"></canvas>
        <div style="margin-top:8px;">
          <button style="background:#b22222;color:#fff;" onclick="markTop5T1Lost()">T1 LOST (biological)</button>
          <button onclick="removeTop5T1()">Remove T1 (artifact)</button>
        </div>
      </div>
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <b>T2 Top 5 Candidates</b>
          <button onclick="openDetachedTop5Viewer()">Open #1 in Detached Viewer</button>
        </div>
        <div id="top5Candidates" style="display:grid; grid-template-columns: repeat(5, minmax(180px, 1fr)); gap:8px; margin-top:6px;"></div>
      </div>
    </div>
  </div>
  <div id="newValidationPanel" style="display:none; margin-top:12px; border:1px solid #ddd; border-radius:10px; padding:10px;">
    <div style="font-weight:700; margin-bottom:8px;">Final Pass: Validate Candidate New T2 Spines</div>
    <div id="newValidationMeta" style="font-size:12px; margin-bottom:8px;"></div>
    <div style="font-size:12px; color:#444;">
      <b>New-T2 validation</b> (only candidate newly formed T2 spines; no auto T1 matches). T1 shows the same region as the T2 candidate for context. Use click mode <b>Add manual T1 point</b> then <b>Save Manual Point</b>, or pick an existing T1 in the list after it appears. Then choose one of the four actions in the large viewer.
    </div>
  </div>
  <div id="compareModal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.45); z-index:2000;">
    <div style="background:#fff; margin:3% auto; padding:12px; width:92%; max-width:1300px; border-radius:10px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <b>Large Compare Viewer</b>
        <div>
          <button onclick="closeCompareModal()">Close</button>
          <button onclick="closeCompareModal()">Dismiss</button>
        </div>
      </div>
      <div class="row">
        <span id="modalFinalPassT2Wrap">
        <label>T2 spine</label>
        <select id="modalT2Pick" style="min-width:220px; min-height:30px;"></select>
        </span>
        <label>T1 spine</label>
        <select id="modalT1Pick" style="min-width:220px; min-height:30px;"></select>
        <span id="modalRowSuggestWrap">
        <label><input id="modalSuggestByClick" type="checkbox" /> Suggest by click</label>
        </span>
        <label>Click mode</label>
        <select id="modalClickMode" style="min-width:180px; min-height:30px;">
          <option value="suggest">Suggest alternative</option>
          <option value="mark_t1_lost">Mark clicked T1 as lost</option>
          <option value="mark_t2_new">Mark clicked T2 as new</option>
          <option value="add_t2_manual">Add manual T2 point (x,y,z only)</option>
          <option value="add_t1_manual">Add manual T1 point (x,y,z only)</option>
        </select>
        <button id="modalSaveManualDraftBtnMain" onclick="saveManualT2Draft()">Save Manual Point</button>
        <button id="modalClearManualDraftBtnMain" onclick="clearManualT2Draft()">Clear Draft</button>
        <button id="modalMatchLastManualToSelectedBtnMain" onclick="matchLastManualT2ToSelectedT1()">Match Last Manual T2 -> Selected T1</button>
        <button id="modalMatchLastManualToManualBtnMain" onclick="matchLastManualT2ToLastManualT1()">Match Last Manual T2 -> Last Manual T1</button>
        <button type="button" id="modalUndoLastMarkBtnMain" onclick="modalUndoLastMark()">Undo Last Mark</button>
        <button type="button" id="modalFinalPassMatchBtnMain" style="display:none;background:#2b6cb0;color:#fff;" onclick="modalSetDecision('match')">4) Match / approve</button>
        <button id="modalApproveBtn" onclick="modalSetDecision('match')">Approve / Match</button>
        <button id="modalMarkNewBtnMain" onclick="modalSetDecision('new')">Mark as T2 New</button>
        <button id="modalRejectBtn" onclick="modalSetDecision('no_match')">Reject / No Match</button>
        <button id="modalRemoveT1BtnMain" onclick="modalSetDecision('remove_t1')">Remove T1 (artifact)</button>
        <button id="modalRemoveT2BtnMain" onclick="modalSetDecision('remove_t2')">Remove T2 (artifact)</button>
        <button id="modalNotInT1BtnMain" onclick="modalSetDecision('not_in_t1')">Not in T1 focus</button>
        <button id="modalIgnoreT1BtnMain" onclick="modalSetDecision('ignore_t1')">Ignore T1 (T2 out of focus)</button>
        <button id="modalIgnoreT2BtnMain" onclick="modalSetDecision('ignore_t2')">Ignore T2 (T1 out of focus)</button>
        <button class="undo-btn" onclick="modalUndoDecision()">Undo</button>
        <button id="modalUnmatchBtnMain" onclick="modalUnmatchAlgo()">Unmatch</button>
      </div>
      <div class="row" style="font-size:12px;">
        <span id="modalTagT2" style="margin-right:16px;"></span>
        <span id="modalTagT1"></span>
      </div>
      <div class="row">
        <label>T2 Z <span id="modalSliceT2Label">10/20</span></label>
        <input id="modalSliceT2" type="range" min="0" max="20" value="10" />
        <label>T1 Z <span id="modalSliceT1Label">10/20</span></label>
        <input id="modalSliceT1" type="range" min="0" max="20" value="10" />
        <label>Brightness</label>
        <input id="modalBrightness" type="range" min="-50" max="50" value="0" />
        <label>Contrast</label>
        <input id="modalContrast" type="range" min="50" max="200" value="100" />
      </div>
      <div style="display:flex; gap:16px;">
        <div style="min-width:90px; border:1px solid #ddd; border-radius:8px; padding:8px; height:fit-content;">
          <div style="font-size:12px; font-weight:700; margin-bottom:6px;">Zoom</div>
          <button style="width:100%; margin:2px 0;" onclick="setModalZoomBoth(0.25)">25%</button>
          <button style="width:100%; margin:2px 0;" onclick="setModalZoomBoth(0.5)">50%</button>
          <button style="width:100%; margin:2px 0;" onclick="setModalZoomBoth(0.75)">75%</button>
          <button style="width:100%; margin:2px 0;" onclick="setModalZoomBoth(1.0)">100%</button>
          <button style="width:100%; margin:2px 0;" onclick="setModalZoomBoth(2.0)">200%</button>
        </div>
        <div>
          <div>T2</div>
          <div id="modalViewportT2" class="modal-viewport">
            <canvas id="modalT2" class="modal-canvas" width="520" height="520"></canvas>
            <div id="modalSliceOverlayT2" class="slice-overlay">Slice: 10/20</div>
          </div>
        </div>
        <div>
          <div>T1</div>
          <div id="modalViewportT1" class="modal-viewport">
            <canvas id="modalT1" class="modal-canvas" width="520" height="520"></canvas>
            <div id="modalSliceOverlayT1" class="slice-overlay">Slice: 10/20</div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    let sessionLoaded = false;
    let previewDebounce = null;
    let queueOffset = 0;
    let currentQueue = [];
    let pendingDecisions = {};
    let reviewMode = 'manual';
    let top5Current = null;
    let top5LargeContext = { active: false };
    let newValidationCurrent = null;
    let newValidationLargeContext = { active: false };
    /** Max Z index (0-based) for large-viewer slice sliders; from /review/counters after load. */
    let sessionStackZ = { t1: 20, t2: 20 };
    let previewBrightness = 0.0;
    let previewContrast = 1.0;
    let previewZoom = 1.0;
    let previewProjection = 'mip';
    let previewSliceIndex = 8;
    let matchMode = 'large';
    let modalState = { open: false, idx: null };
    let modalView = { t1: { scale: 1.0, tx: 0, ty: 0 }, t2: { scale: 1.0, tx: 0, ty: 0 } };
    let modalDrag = { active: false, pane: null, startX: 0, startY: 0, startTx: 0, startTy: 0, pendingDx: 0, pendingDy: 0, moved: false };
    let modalCache = { key: '', p1: null, p2: null, pickedT1: null, pickedT2: null };
    let modalAlignZOnFetch = false;
    let modalLastMark = null;
    let modalManualT2Draft = null;
    let modalManualT1Draft = null;
    let lastSavedManualT2Id = null;
    let lastSavedManualT1Id = null;
    let manualT2Cache = [];
    let manualT1Cache = [];
    let modalRaf = null;
    let detachedCompareWindow = null;
    let detachedCompareReady = false;
    const syncModalMovement = false; // Ready for future sync mode.
    const FOCUS_ZOOM_SCALE = 6.0;

    function setOut(data) {
      const text = (typeof data === 'string') ? data : JSON.stringify(data, null, 2);
      fetch('/session/client-log', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: text })
      }).catch(() => {});
    }

    async function refreshManualT2Cache() {
      try {
        const resp = await fetch('/review/manual-t2-clicks');
        const data = await resp.json();
        if (!resp.ok) return;
        manualT2Cache = Array.isArray(data.items) ? data.items : [];
      } catch (e) {
        // ignore
      }
    }

    async function refreshManualT1Cache() {
      try {
        const resp = await fetch('/review/manual-t1-clicks');
        const data = await resp.json();
        if (!resp.ok) return;
        manualT1Cache = Array.isArray(data.items) ? data.items : [];
      } catch (e) {
        // ignore
      }
    }

    function ensureManualT2OptionsOnSelect(sel) {
      if (!sel) return;
      const doc = sel.ownerDocument || document;
      (manualT2Cache || []).forEach((m) => {
        const mid = String(m.manual_id || '');
        if (!mid) return;
        const exists = Array.from(sel.options).some((o) => String(o.value) === mid);
        if (exists) return;
        const o = doc.createElement('option');
        o.value = mid;
        o.textContent = `${mid} (manual x=${Number(m.x ?? NaN).toFixed(1)} y=${Number(m.y ?? NaN).toFixed(1)} z=${Number(m.z ?? NaN).toFixed(1)})`;
        sel.appendChild(o);
      });
    }

    function ensureManualT1OptionsOnSelect(sel) {
      if (!sel) return;
      const doc = sel.ownerDocument || document;
      (manualT1Cache || []).forEach((m) => {
        const mid = String(m.manual_id || '');
        if (!mid) return;
        const exists = Array.from(sel.options).some((o) => String(o.value) === mid);
        if (exists) return;
        const o = doc.createElement('option');
        o.value = mid;
        o.textContent = `${mid} (manual x=${Number(m.x ?? NaN).toFixed(1)} y=${Number(m.y ?? NaN).toFixed(1)} z=${Number(m.z ?? NaN).toFixed(1)})`;
        sel.appendChild(o);
      });
    }

    function setMatchMode(mode) {
      matchMode = (mode === 'top5') ? 'top5' : 'large';
      const large = document.getElementById('largeModePanel');
      const top5 = document.getElementById('top5Panel');
      const nv = document.getElementById('newValidationPanel');
      const tabL = document.getElementById('tabLargeFov');
      const tabT = document.getElementById('tabTop5');
      if (large) large.style.display = (matchMode === 'large') ? '' : 'none';
      if (top5) top5.style.display = (matchMode === 'top5') ? '' : 'none';
      if (nv) nv.style.display = 'none';
      if (tabL) tabL.classList.toggle('active', matchMode === 'large');
      if (tabT) tabT.classList.toggle('active', matchMode === 'top5');
      if (matchMode !== 'large') {
        newValidationLargeContext.active = false;
      }
    }

    function setElVisible(el, visible) {
      if (!el) return;
      el.style.display = visible ? '' : 'none';
    }

    function applyFinalPassButtonLabels() {
      const g = (id) => modalEl(id) || document.getElementById(id);
      if (newValidationLargeContext.active) {
        const pairs = [
          ['modalMarkNewBtn', '1) Mark as T2 new (newly formed)'],
          ['modalMarkNewBtnMain', '1) Mark as T2 new (newly formed)'],
          ['modalRemoveT2Btn', '2) Ignore T2 — artifact'],
          ['modalRemoveT2BtnMain', '2) Ignore T2 — artifact'],
          ['modalNotInT1Btn', '3) Not relevant (not in T1 focus)'],
          ['modalNotInT1BtnMain', '3) Not relevant (not in T1 focus)'],
          ['modalFinalPassMatchBtn', '4) Match / approve'],
          ['modalFinalPassMatchBtnMain', '4) Match / approve'],
        ];
        pairs.forEach(([id, t]) => { const el = g(id); if (el) el.textContent = t; });
      } else {
        const pairs = [
          ['modalMarkNewBtn', 'Mark as T2 New'],
          ['modalMarkNewBtnMain', 'Mark as T2 New'],
          ['modalRemoveT2Btn', 'Remove T2 (artifact)'],
          ['modalRemoveT2BtnMain', 'Remove T2 (artifact)'],
          ['modalNotInT1Btn', 'Not in T1 focus'],
          ['modalNotInT1BtnMain', 'Not in T1 focus'],
        ];
        pairs.forEach(([id, t]) => { const el = g(id); if (el) el.textContent = t; });
      }
    }

    function applyLargeViewerModeUI() {
      const finalPass = !!newValidationLargeContext.active;
      const idsFinalOnlyShow = [
        'modalFinalPassMatchBtn',
        'modalFinalPassMatchBtnMain',
        'modalMarkNewBtn',
        'modalMarkNewBtnMain',
        'modalRemoveT2Btn',
        'modalRemoveT2BtnMain',
        'modalNotInT1Btn',
        'modalNotInT1BtnMain',
      ];
      const idsFinalOnlyHide = [
        'modalApproveBtn',
        'modalApproveBtnMain',
        'modalRejectBtn',
        'modalRejectBtnMain',
        'modalRemoveT1Btn',
        'modalRemoveT1BtnMain',
        'modalIgnoreT1Btn',
        'modalIgnoreT1BtnMain',
        'modalIgnoreT2Btn',
        'modalIgnoreT2BtnMain',
        'modalUnmatchBtn',
        'modalUnmatchBtnMain',
      ];
      const idsHiddenWhenFinalPass = [
        'modalFinalPassT2Wrap',
        'modalMatchLastManualToSelectedBtn',
        'modalMatchLastManualToSelectedBtnMain',
        'modalMatchLastManualToManualBtn',
        'modalMatchLastManualToManualBtnMain',
        'modalUndoLastMarkBtn',
        'modalUndoLastMarkBtnMain',
      ];
      idsFinalOnlyShow.forEach((id) => setElVisible(modalEl(id) || document.getElementById(id), finalPass));
      idsFinalOnlyHide.forEach((id) => setElVisible(modalEl(id) || document.getElementById(id), !finalPass));
      idsHiddenWhenFinalPass.forEach((id) => setElVisible(modalEl(id) || document.getElementById(id), !finalPass));
      const clickMode = modalEl('modalClickMode');
      if (clickMode) {
        Array.from(clickMode.options || []).forEach((opt) => {
          if (!finalPass) {
            opt.disabled = false;
            return;
          }
          const v = String(opt.value);
          opt.disabled = v !== 'add_t1_manual' && v !== 'suggest';
        });
        if (finalPass) {
          const cur = String(clickMode.value || '');
          if (cur !== 'add_t1_manual' && cur !== 'suggest') clickMode.value = 'suggest';
        }
      }
      const sug = modalEl('modalSuggestByClick');
      if (sug && !finalPass) sug.checked = false;
      applyFinalPassButtonLabels();
    }

    function isDetachedCompareOpen() {
      return !!detachedCompareWindow && !detachedCompareWindow.closed && detachedCompareReady;
    }

    function modalDocument() {
      return isDetachedCompareOpen() ? detachedCompareWindow.document : document;
    }

    function modalEl(id) {
      return modalDocument().getElementById(id);
    }

    function largeViewerActive() {
      return modalState.open || isDetachedCompareOpen();
    }

    function detachedViewerMarkup() {
      return `
<!doctype html>
<html>
<head>
  <title>Detached Large Compare Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 12px; background: #fff; color: #222; }
    .row { margin: 10px 0; }
    select { min-width: 220px; min-height: 30px; padding: 4px; }
    button { margin-right: 5px; }
    .viewer-header { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .viewer-status { font-size:12px; color:#2f6f3e; margin-top:4px; }
    .modal-viewport {
      width: 520px;
      height: 520px;
      border: 1px solid #ddd;
      overflow: hidden;
      position: relative;
      cursor: grab;
      background: #111;
    }
    .modal-viewport.grabbing { cursor: grabbing; }
    .slice-overlay {
      position: absolute;
      right: 8px;
      top: 8px;
      z-index: 5;
      color: #fff;
      background: rgba(0, 0, 0, 0.55);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 12px;
      line-height: 1.3;
      text-shadow: 0 1px 2px rgba(0,0,0,0.8);
      pointer-events: none;
      user-select: none;
    }
    .modal-canvas {
      width: 520px;
      height: 520px;
      transform-origin: 0 0;
      will-change: transform;
      transition: transform 0.2s ease-out;
      image-rendering: pixelated;
      display: block;
    }
  </style>
</head>
<body>
  <div class="viewer-header">
    <div>
      <b>Large Compare Viewer</b>
      <div id="modalViewerStatus" class="viewer-status">Connected to main window.</div>
    </div>
    <div>
      <button id="modalCloseBtn">Close</button>
      <button id="modalDismissBtn">Dismiss</button>
    </div>
  </div>
  <div class="row">
    <span id="modalFinalPassT2Wrap">
    <label>T2 spine</label>
    <select id="modalT2Pick"></select>
    </span>
    <label>T1 spine</label>
    <select id="modalT1Pick"></select>
    <span id="modalRowSuggestWrap">
    <label><input id="modalSuggestByClick" type="checkbox" /> Suggest by click</label>
    </span>
    <label>Click mode</label>
    <select id="modalClickMode" style="min-width:180px;">
      <option value="suggest">Suggest alternative</option>
      <option value="mark_t1_lost">Mark clicked T1 as lost</option>
      <option value="mark_t2_new">Mark clicked T2 as new</option>
      <option value="add_t2_manual">Add manual T2 point (x,y,z only)</option>
      <option value="add_t1_manual">Add manual T1 point (x,y,z only)</option>
    </select>
    <button id="modalSaveManualDraftBtn">Save Manual Point</button>
    <button id="modalClearManualDraftBtn">Clear Draft</button>
    <button id="modalMatchLastManualToSelectedBtn">Match Last Manual T2 -> Selected T1</button>
    <button id="modalMatchLastManualToManualBtn">Match Last Manual T2 -> Last Manual T1</button>
    <button id="modalUndoLastMarkBtn">Undo Last Mark</button>
    <button type="button" id="modalFinalPassMatchBtn" style="display:none;background:#2b6cb0;color:#fff;">4) Match / approve</button>
    <button id="modalApproveBtn">Approve / Match</button>
    <button id="modalMarkNewBtn">Mark as T2 New</button>
    <button id="modalRejectBtn">Reject / No Match</button>
    <button id="modalRemoveT1Btn">Remove T1 (artifact)</button>
    <button id="modalRemoveT2Btn">Remove T2 (artifact)</button>
    <button id="modalNotInT1Btn">Not in T1 focus</button>
    <button id="modalIgnoreT1Btn">Ignore T1 (T2 out of focus)</button>
    <button id="modalIgnoreT2Btn">Ignore T2 (T1 out of focus)</button>
    <button id="modalUndoDecisionBtn">Undo</button>
    <button id="modalUnmatchBtn">Unmatch</button>
  </div>
  <div class="row" style="font-size:12px;">
    <span id="modalTagT2" style="margin-right:16px;"></span>
    <span id="modalTagT1"></span>
  </div>
  <div class="row">
    <label>T2 Z <span id="modalSliceT2Label">10/20</span></label>
    <input id="modalSliceT2" type="range" min="0" max="20" value="10" />
    <label>T1 Z <span id="modalSliceT1Label">10/20</span></label>
    <input id="modalSliceT1" type="range" min="0" max="20" value="10" />
    <label>Brightness</label>
    <input id="modalBrightness" type="range" min="-50" max="50" value="0" />
    <label>Contrast</label>
    <input id="modalContrast" type="range" min="50" max="200" value="100" />
  </div>
  <div style="display:flex; gap:16px;">
    <div style="min-width:90px; border:1px solid #ddd; border-radius:8px; padding:8px; height:fit-content;">
      <div style="font-size:12px; font-weight:700; margin-bottom:6px;">Zoom</div>
      <button id="modalZoom25" style="width:100%; margin:2px 0;">25%</button>
      <button id="modalZoom50" style="width:100%; margin:2px 0;">50%</button>
      <button id="modalZoom75" style="width:100%; margin:2px 0;">75%</button>
      <button id="modalZoom100" style="width:100%; margin:2px 0;">100%</button>
      <button id="modalZoom200" style="width:100%; margin:2px 0;">200%</button>
    </div>
    <div>
      <div>T2</div>
      <div id="modalViewportT2" class="modal-viewport">
        <canvas id="modalT2" class="modal-canvas" width="520" height="520"></canvas>
        <div id="modalSliceOverlayT2" class="slice-overlay">Slice: 10/20</div>
      </div>
    </div>
    <div>
      <div>T1</div>
      <div id="modalViewportT1" class="modal-viewport">
        <canvas id="modalT1" class="modal-canvas" width="520" height="520"></canvas>
        <div id="modalSliceOverlayT1" class="slice-overlay">Slice: 10/20</div>
      </div>
    </div>
  </div>
</body>
</html>`;
    }

    function setDetachedViewerMessage(message) {
      if (!isDetachedCompareOpen()) return;
      const status = detachedCompareWindow.document.getElementById('modalViewerStatus');
      if (status) status.textContent = message;
    }

    function ensureDetachedCompareWindow() {
      if (isDetachedCompareOpen()) {
        detachedCompareWindow.focus();
        return detachedCompareWindow;
      }
      const w = window.open('', 'spineLargeCompareViewer', 'width=1280,height=760,resizable=yes,scrollbars=yes');
      if (!w) {
        setOut('Popup was blocked. Allow popups for this local annotator page, then try again.');
        return null;
      }
      detachedCompareWindow = w;
      detachedCompareReady = false;
      w.document.open();
      w.document.write(detachedViewerMarkup());
      w.document.close();
      detachedCompareReady = true;
      wireModalControls(w.document);
      syncModalSliceSliderMaxFromSession();
      w.addEventListener('beforeunload', () => {
        if (detachedCompareWindow === w) {
          detachedCompareReady = false;
          detachedCompareWindow = null;
          modalState.open = false;
          modalState.idx = null;
          top5LargeContext.active = false;
        }
      });
      w.focus();
      return w;
    }

    function firstAvailableQueueIndex() {
      for (let i = 0; i < currentQueue.length; i++) {
        const card = document.getElementById(`qcard-${i}`);
        if (!card || !card.classList.contains('picked')) return i;
      }
      return null;
    }

    function t2IdAtQueueIndex(idx) {
      const it = currentQueue[idx];
      if (!it) return null;
      const sel = document.getElementById(`q-t2-pick-${idx}`);
      return String(sel?.value || it.t2_spine_id || '');
    }

    function nextUnresolvedQueueIndex(currentIdx = null) {
      if (!Array.isArray(currentQueue) || currentQueue.length === 0) return null;
      const start = Number.isInteger(currentIdx) ? currentIdx : -1;
      for (let step = 1; step < currentQueue.length; step++) {
        const j = (start + step) % currentQueue.length;
        if (Number.isInteger(currentIdx) && j === currentIdx) continue;
        const t2 = t2IdAtQueueIndex(j);
        if (!t2) continue;
        if (!pendingDecisions[String(t2)]) return j;
      }
      return null;
    }

    async function openDetachedCompareViewer(idx = null) {
      if (!ensureDetachedCompareWindow()) return;
      if (matchMode === 'top5' && top5Current?.has_item) {
        await openDetachedTop5Viewer();
        return;
      }
      const target = Number.isInteger(idx) ? idx : (modalState.idx ?? firstAvailableQueueIndex() ?? 0);
      if (!currentQueue[target]) {
        setDetachedViewerMessage('Connected. No review item is loaded yet.');
        return;
      }
      await openCompareModal(target, { detached: true });
    }

    async function openDetachedTop5Viewer(t2Id = null) {
      if (!ensureDetachedCompareWindow()) return;
      if (!top5Current?.has_item || !top5Current?.t1_spine_id) {
        setDetachedViewerMessage('Connected. No Top 5 item is loaded yet.');
        return;
      }
      const pick = String(t2Id || top5Current.candidates?.[0]?.t2_spine_id || '');
      if (!pick) {
        setDetachedViewerMessage('Connected. This Top 5 item has no T2 candidates.');
        return;
      }
      await openLargeFromTop5(pick, true);
    }

    async function syncLargeViewerToQueueCard(idx, forceFetch = true) {
      if (!largeViewerActive() || !currentQueue[idx]) return;
      modalState.open = true;
      modalState.idx = idx;
      hydrateModalSelectors(idx);
      syncModalSliceSliderMaxFromSession();
      modalAlignZOnFetch = true;
      await renderCompareModal(forceFetch);
    }

    function syncModalSelectionsToQueue() {
      if (modalState.idx === null) return;
      const i = modalState.idx;
      const t2 = modalEl('modalT2Pick')?.value;
      const t1 = modalEl('modalT1Pick')?.value;
      const q2 = document.getElementById(`q-t2-pick-${i}`);
      const q1 = document.getElementById(`q-t1-pick-${i}`);
      if (q2 && t2) q2.value = t2;
      if (q1 && t1) q1.value = t1;
    }

    function updateLargeViewerStatus() {
      const status = modalEl('modalViewerStatus');
      if (!status || modalState.idx === null) return;
      const itemNumber = modalState.idx + 1;
      const pickedT2 = modalEl('modalT2Pick')?.value || '?';
      const pickedT1 = modalEl('modalT1Pick')?.value || '?';
      status.textContent = `Connected to main window. Showing item #${itemNumber}: T2 ${pickedT2} vs T1 ${pickedT1}.`;
    }

    function syncMaxMatchZGapInput(value) {
      const el = document.getElementById('maxMatchZGapInput');
      if (el && Number.isFinite(Number(value))) el.value = String(Number(value));
    }

    async function loadMatchSettings() {
      if (!sessionLoaded) return;
      try {
        const resp = await fetch('/session/match-settings');
        if (!resp.ok) return;
        const data = await resp.json();
        syncMaxMatchZGapInput(data.max_match_z_gap ?? 7);
      } catch (e) {
        // ignore
      }
    }

    async function applyMaxMatchZGap() {
      if (!sessionLoaded) {
        setOut('Please load a session first.');
        return;
      }
      const raw = Number(document.getElementById('maxMatchZGapInput')?.value);
      if (!Number.isFinite(raw) || raw < 0) {
        setOut('Max |Δz| must be a non-negative number.');
        return;
      }
      const resp = await fetch('/session/match-settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ max_match_z_gap: raw }),
      });
      const data = await resp.json();
      setOut(data);
      if (!resp.ok) return;
      syncMaxMatchZGapInput(data.max_match_z_gap);
      await refreshCounters();
    }

    async function refreshCounters() {
      if (!sessionLoaded) {
        document.getElementById('statManualMatched').textContent = '0';
        document.getElementById('statAlgoMatched').textContent = '0';
        document.getElementById('statToReview').textContent = '0';
        document.getElementById('statT1Lost').textContent = '0';
        document.getElementById('statT2New').textContent = '0';
        syncMaxMatchZGapInput(7);
        return;
      }
      try {
        const resp = await fetch('/review/counters');
        const data = await resp.json();
        document.getElementById('statManualMatched').textContent = String(data.manual_matched ?? 0);
        document.getElementById('statAlgoMatched').textContent = String(data.algo_matched ?? 0);
        document.getElementById('statToReview').textContent = String(data.to_review ?? 0);
        document.getElementById('statT1Lost').textContent = String(data.t1_lost ?? 0);
        document.getElementById('statT2New').textContent = String(data.t2_new ?? 0);
        if (Number.isFinite(Number(data.max_match_z_gap))) {
          syncMaxMatchZGapInput(data.max_match_z_gap);
        }
      } catch (e) {
        // keep UI usable even if counters fail
      }
    }

    function renderQueue(items) {
      const root = document.getElementById('queue');
      if (!items || !items.length) {
        root.innerHTML = '<i>No review items.</i>';
        currentQueue = [];
        pendingDecisions = {};
        setDetachedViewerMessage('Connected. No review items are currently loaded.');
        return;
      }
      currentQueue = items;
      pendingDecisions = {};
      root.innerHTML = items.map((it, idx) => `
        <div id="qcard-${idx}" class="qcard">
          <div><b>#${idx + 1} T2 ${it.t2_spine_id}</b> → T1 ${it.suggested_t1_spine_id ?? 'none'}</div>
          <div style="font-size:12px;">dxy=${Number(it.distance_xy ?? 0).toFixed(2)} dz=${Number(it.distance_z ?? 0).toFixed(2)} score=${Number(it.final_score ?? 0).toFixed(3)} margin=${Number(it.margin ?? 0).toFixed(3)}</div>
          <div style="margin-top:6px;">
            <label style="font-size:12px;">Choose nearby T2:</label>
            <select id="q-t2-pick-${idx}" style="min-width:180px; min-height:30px;"></select>
          </div>
          <div style="margin-top:6px;">
            <label style="font-size:12px;">Choose nearby T1:</label>
            <select id="q-t1-pick-${idx}" style="min-width:180px; min-height:30px;"></select>
          </div>
          <div class="qrow">
            <div>
              <div style="font-size:11px;">T2</div>
              <canvas id="q-t2-${idx}" class="qimg" width="180" height="180"></canvas>
              <div id="q-t2-tag-${idx}" style="font-size:11px; color:#b22222; margin-top:2px;"></div>
            </div>
            <div>
              <div style="font-size:11px;">T1</div>
              <canvas id="q-t1-${idx}" class="qimg" width="180" height="180"></canvas>
              <div id="q-t1-tag-${idx}" style="font-size:11px; color:#b22222; margin-top:2px;"></div>
            </div>
          </div>
          <div class="qactions" style="margin-top:6px;">
            <button onclick="setDecision(${idx}, 'match')">${reviewMode === 'algo' ? 'Approve' : 'Match'}</button>
            <button onclick="setDecision(${idx}, 'no_match')">${reviewMode === 'algo' ? 'Reject' : 'No Match'}</button>
            <button onclick="setDecision(${idx}, 'remove_t1')">Remove T1 (artifact)</button>
            <button onclick="setDecision(${idx}, 'remove_t2')">Remove T2 (artifact)</button>
            <button class="undo-btn" onclick="undoDecision(${idx})">Undo</button>
            <button onclick="unmatchAlgoPair(${idx})">Unmatch</button>
            <button onclick="openCompareModal(${idx})">Open Large View</button>
            <span id="q-picked-${idx}" style="font-size:12px;color:#2f6f3e;"></span>
          </div>
        </div>
      `).join('');
      for (let i = 0; i < items.length; i++) {
        const selT2 = document.getElementById(`q-t2-pick-${i}`);
        const optsT2 = items[i].nearby_t2_candidates || [];
        if (!optsT2.length && items[i].t2_spine_id) {
          const o = document.createElement('option');
          o.value = String(items[i].t2_spine_id);
          o.textContent = `${items[i].t2_spine_id}`;
          selT2.appendChild(o);
        } else {
          optsT2.forEach((cand) => {
            const o = document.createElement('option');
            o.value = String(cand.t2_spine_id);
            const dxy = Number(cand.distance_xy ?? 0).toFixed(1);
            o.textContent = `${cand.t2_spine_id} (dxy ${dxy})`;
            selT2.appendChild(o);
          });
        }
        if (items[i].t2_spine_id) {
          selT2.value = String(items[i].t2_spine_id);
        }
        ensureManualT2OptionsOnSelect(selT2);
        const sel = document.getElementById(`q-t1-pick-${i}`);
        const opts = items[i].nearby_t1_candidates || [];
        if (!opts.length && items[i].suggested_t1_spine_id) {
          const o = document.createElement('option');
          o.value = String(items[i].suggested_t1_spine_id);
          o.textContent = `${items[i].suggested_t1_spine_id}`;
          sel.appendChild(o);
        } else {
          opts.forEach((cand) => {
            const o = document.createElement('option');
            o.value = String(cand.t1_spine_id);
            const dxy = Number(cand.distance_xy ?? 0).toFixed(1);
            const score = Number(cand.final_score ?? 0).toFixed(3);
            o.textContent = `${cand.t1_spine_id} (dxy ${dxy}, s ${score})`;
            sel.appendChild(o);
          });
        }
        if (items[i].suggested_t1_spine_id) {
          sel.value = String(items[i].suggested_t1_spine_id);
        }
        ensureManualT1OptionsOnSelect(sel);
        selT2.addEventListener('change', async () => {
          await renderQueuePreviewCard(i);
          await syncLargeViewerToQueueCard(i);
        });
        sel.addEventListener('change', async () => {
          await renderQueuePreviewCard(i);
          await syncLargeViewerToQueueCard(i);
        });
        const t2c = document.getElementById(`q-t2-${i}`);
        const t1c = document.getElementById(`q-t1-${i}`);
        if (t2c) {
          t2c.addEventListener('dblclick', () => openCompareModal(i));
          t2c.addEventListener('click', (ev) => {
            if (ev && ev.shiftKey) openCompareModal(i);
          });
        }
        if (t1c) {
          t1c.addEventListener('dblclick', () => openCompareModal(i));
          t1c.addEventListener('click', (ev) => {
            if (ev && ev.shiftKey) openCompareModal(i);
          });
        }
      }
      renderQueuePreviews();
      if (isDetachedCompareOpen()) {
        syncLargeViewerToQueueCard(firstAvailableQueueIndex() ?? 0);
      }
    }

    async function selectAndLoad() {
      setOut('Opening dialogs...');
      try {
        const resp = await fetch('/select-and-load', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({use_dialog: true})
        });
        const data = await resp.json();
        sessionLoaded = resp.ok;
        setOut(data);
        if (sessionLoaded) {
          applySessionStackZFromStats(data.session);
          await refreshDendrites();
          await refreshManualT2Cache();
          await refreshManualT1Cache();
          await refreshCounters();
          await refreshStackSliceBounds();
          await hydrateViewerState();
          await loadMatchSettings();
          await loadTop5Next();
        }
      } catch (err) {
        setOut('Error: ' + String(err));
      }
    }

    async function useLastSaved() {
      setOut('Loading last saved session...');
      try {
        const resp = await fetch('/session/use-last', { method: 'POST' });
        const data = await resp.json();
        sessionLoaded = resp.ok;
        setOut(data);
        if (sessionLoaded) {
          applySessionStackZFromStats(data.session);
          await refreshDendrites();
          await refreshManualT2Cache();
          await refreshManualT1Cache();
          await previewSelectedDendrites(false);
          await refreshCounters();
          queueOffset = 0;
          await refreshStackSliceBounds();
          await hydrateViewerState();
          await loadMatchSettings();
          await loadTop5Next();
        }
      } catch (err) {
        setOut('Error: ' + String(err));
      }
    }

    async function undoLastSession() {
      setOut('Restoring previous session snapshot...');
      try {
        const resp = await fetch('/session/undo-last', { method: 'POST' });
        const data = await resp.json();
        sessionLoaded = resp.ok;
        setOut(data);
        if (sessionLoaded) {
          applySessionStackZFromStats(data.session);
          await refreshDendrites();
          await refreshManualT2Cache();
          await refreshManualT1Cache();
          await previewSelectedDendrites(false);
          await refreshCounters();
          await refreshStackSliceBounds();
          await loadMatchSettings();
          await loadTop5Next();
        }
      } catch (err) {
        setOut('Error: ' + String(err));
      }
    }

    async function openNewSession() {
      setOut('Opening a new session...');
      try {
        const resp = await fetch('/session/open-new', { method: 'POST' });
        const data = await resp.json();
        sessionLoaded = false;
        queueOffset = 0;
        document.getElementById('t1Dendrites').innerHTML = '';
        document.getElementById('t2Dendrites').innerHTML = '';
        const c1 = document.getElementById('t1Canvas').getContext('2d');
        const c2 = document.getElementById('t2Canvas').getContext('2d');
        c1.clearRect(0, 0, 420, 320);
        c2.clearRect(0, 0, 420, 320);
        renderQueue([]);
        setOut(data);
        await refreshCounters();
        top5Current = null;
        await renderTop5Panel();
      } catch (err) {
        setOut('Error: ' + String(err));
      }
    }

    async function saveNow() {
      if (!sessionLoaded) {
        setOut('No active session to save.');
        return;
      }
      const resp = await fetch('/session/save', { method: 'POST' });
      const data = await resp.json();
      setOut({
        ...data,
        saved_path: data.saved_path || null
      });
    }

    async function exportResults() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const outputName = window.prompt('Export folder name (optional):', 'spine_annotator_export') || '';
      const resp = await fetch('/results/export', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ use_dialog: true, output_name: outputName })
      });
      const data = await resp.json();
      if (!resp.ok) {
        const msg = (typeof data?.detail === 'string' && data.detail)
          ? data.detail
          : (`Export failed: ${JSON.stringify(data)}`);
        alert(msg);
      }
      setOut(data);
    }

    async function undoLastChoice() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const resp = await fetch('/review/undo-last-choice', { method: 'POST' });
      const data = await resp.json();
      setOut(data);
      await refreshCounters();
      if (reviewMode === 'algo') {
        await loadAlgoReviewQueue();
      } else {
        await loadTop5Next();
      }
    }

    function selectedValues(selectId) {
      const el = document.getElementById(selectId);
      return Array.from(el.selectedOptions).map(o => o.value);
    }

    function setSelectOptions(el, values) {
      el.innerHTML = '';
      values.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        el.appendChild(opt);
      });
    }

    function applyLinkedColors(linksData) {
      const t1Linked = new Set();
      const t2Linked = new Set();
      const links = linksData?.links || [];
      for (const link of links) {
        (link.t1_dendrite_ids || []).forEach(v => t1Linked.add(String(v)));
        (link.t2_dendrite_ids || []).forEach(v => t2Linked.add(String(v)));
      }
      for (const opt of document.getElementById('t1Dendrites').options) {
        opt.style.backgroundColor = t1Linked.has(opt.value) ? '#d9f2ff' : '';
      }
      for (const opt of document.getElementById('t2Dendrites').options) {
        opt.style.backgroundColor = t2Linked.has(opt.value) ? '#ffe8d9' : '';
      }
    }

    async function refreshLinkedColors() {
      if (!sessionLoaded) return;
      const resp = await fetch('/dendrites/links');
      const data = await resp.json();
      applyLinkedColors(data);
    }

    function scheduleAutoPreview() {
      if (!sessionLoaded) return;
      if (previewDebounce) {
        clearTimeout(previewDebounce);
      }
      previewDebounce = setTimeout(() => {
        previewSelectedDendrites(false);
      }, 180);
    }

    async function refreshDendrites() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      try {
        const resp = await fetch('/dendrites/ids');
        const data = await resp.json();
        setSelectOptions(document.getElementById('t1Dendrites'), data.t1_dendrite_ids || []);
        setSelectOptions(document.getElementById('t2Dendrites'), data.t2_dendrite_ids || []);
        await refreshLinkedColors();
        setOut(data);
      } catch (err) {
        setOut('Error: ' + String(err));
      }
    }

    async function createLink() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      try {
        const t1 = selectedValues('t1Dendrites');
        const t2 = selectedValues('t2Dendrites');
        if (!t1.length || !t2.length) {
          setOut('Select at least one dendrite on each side.');
          return;
        }
        const payload = { t1_dendrite_ids: t1, t2_dendrite_ids: t2, notes: document.getElementById('linkNotes').value || '' };
        const resp = await fetch('/dendrites/link', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        await refreshLinkedColors();
        setOut(data);
        await refreshCounters();
      } catch (err) {
        setOut('Error: ' + String(err));
      }
    }

    function drawImageOnCanvas(canvasId, image2d, minV, maxV, points, annotations) {
      const canvas = document.getElementById(canvasId);
      const h = image2d.length;
      const w = h ? image2d[0].length : 0;
      if (!h || !w) return;
      const tmp = document.createElement('canvas');
      tmp.width = w;
      tmp.height = h;
      const ctx = tmp.getContext('2d');
      const img = ctx.createImageData(w, h);
      const den = (maxV - minV) > 1e-9 ? (maxV - minV) : 1.0;
      let k = 0;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const raw = image2d[y][x];
          let n = (raw - minV) / den;
          n = ((n - 0.5) * previewContrast + 0.5) + previewBrightness;
          const v = Math.max(0, Math.min(255, Math.round(n * 255)));
          img.data[k++] = v;
          img.data[k++] = v;
          img.data[k++] = v;
          img.data[k++] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      const out = canvas.getContext('2d');
      out.clearRect(0, 0, canvas.width, canvas.height);
      const dw = canvas.width * previewZoom;
      const dh = canvas.height * previewZoom;
      const dx = (canvas.width - dw) / 2;
      const dy = (canvas.height - dh) / 2;
      out.drawImage(tmp, dx, dy, dw, dh);

      const sx = dw / w;
      const sy = dh / h;
      out.strokeStyle = '#ff2b2b';
      out.lineWidth = 2;
      (points || []).forEach((p) => {
        const cx = dx + (p.x * sx);
        const cy = dy + (p.y * sy);
        out.beginPath();
        out.arc(cx, cy, 4, 0, Math.PI * 2);
        out.stroke();
      });

      if (annotations && annotations.length) {
        out.fillStyle = '#ff2b2b';
        out.font = '11px Arial';
        for (const a of annotations) {
          const ax = dx + ((a.x || 0) * sx) + 6;
          const ay = dy + ((a.y || 0) * sy) - 6;
          out.fillText(a.label || '', ax, ay);
        }
      }
    }

    async function previewSelectedDendrites(showOutput = true) {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const t1 = selectedValues('t1Dendrites');
      const t2 = selectedValues('t2Dendrites');
      const [r1, r2] = await Promise.all([
        fetch('/fov/preview', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({timepoint: 't1', dendrite_ids: t1, projection: 'mip'})
        }),
        fetch('/fov/preview', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({timepoint: 't2', dendrite_ids: t2, projection: 'mip'})
        })
      ]);
      const [d1, d2] = await Promise.all([r1.json(), r2.json()]);
      if (!r1.ok || !r2.ok) {
        if (showOutput) {
          setOut({t1_error: d1, t2_error: d2});
        }
        return;
      }
      drawImageOnCanvas('t1Canvas', d1.image_2d, d1.intensity_min, d1.intensity_max, d1.highlighted_points, []);
      drawImageOnCanvas('t2Canvas', d2.image_2d, d2.intensity_min, d2.intensity_max, d2.highlighted_points, []);
      if (showOutput) {
        setOut({
          t1_selected: t1,
          t2_selected: t2,
          t1_highlighted_spines: (d1.highlighted_points || []).length,
          t2_highlighted_spines: (d2.highlighted_points || []).length,
          t1_shape_zyx: d1.shape_zyx,
          t2_shape_zyx: d2.shape_zyx
        });
      }
    }

    async function showLinks() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const resp = await fetch('/dendrites/links');
      const data = await resp.json();
      applyLinkedColors(data);
      setOut(data);
      await refreshCounters();
    }

    async function clearLinks() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const resp = await fetch('/dendrites/links', { method: 'DELETE' });
      const data = await resp.json();
      applyLinkedColors(data);
      setOut(data);
    }

    async function loadTop5Next() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      setMatchMode('top5');
      const resp = await fetch('/review/top5-next');
      const data = await resp.json();
      if (!resp.ok) {
        setOut(data);
        return;
      }
      top5Current = data;
      await renderTop5Panel();
      if (!data.has_item) {
        setDetachedViewerMessage('Connected. Top 5 matching is complete.');
        setOut({ done: true, message: 'Matching session completed: all T1 spines are matched/lost/removed.' });
      } else {
        if (isDetachedCompareOpen()) await openDetachedTop5Viewer();
        setOut(data);
      }
    }

    async function renderTop5Panel() {
      const meta = document.getElementById('top5T1Meta');
      const candRoot = document.getElementById('top5Candidates');
      const canvas = document.getElementById('top5T1Canvas');
      if (!meta || !candRoot || !canvas) return;
      if (!top5Current || !top5Current.has_item || !top5Current.t1_spine_id) {
        meta.textContent = 'No remaining unmatched T1 spines.';
        candRoot.innerHTML = '<i>Done.</i>';
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      const t1Id = String(top5Current.t1_spine_id);
      meta.textContent = `T1 ${t1Id} | dendrite ${top5Current.t1_dendrite_id ?? '?'} | z=${Number(top5Current.t1_xyz?.z ?? NaN).toFixed(1)}`;
      const p1 = await fetchCropPreview('t1', t1Id);
      if (p1?.image_2d) {
        drawImageOnCanvas('top5T1Canvas', p1.image_2d, p1.intensity_min, p1.intensity_max, [{x: (p1.image_2d[0].length / 2), y: (p1.image_2d.length / 2)}], [
          {x: (p1.image_2d[0].length / 2), y: (p1.image_2d.length / 2), label: `id ${t1Id}`}
        ]);
      }
      const cands = top5Current.candidates || [];
      candRoot.innerHTML = cands.map((c, idx) => `
        <div style="border:1px solid #ddd; border-radius:8px; padding:6px;">
          <div style="font-size:12px;"><b>#${idx + 1} T2 ${c.t2_spine_id}</b></div>
          <div style="font-size:11px;">score=${Number(c.final_score ?? 0).toFixed(3)} dxy=${Number(c.distance_xy ?? 0).toFixed(1)}</div>
          <canvas id="top5-t2-${idx}" width="170" height="170" style="border:1px solid #ddd; margin-top:4px;"></canvas>
          <div style="margin-top:6px;">
            <button onclick="confirmTop5Match('${String(c.t2_spine_id)}')">Match</button>
            <button onclick="removeTop5T2('${String(c.t2_spine_id)}')">Remove T2 (artifact)</button>
            <button onclick="openLargeFromTop5('${String(c.t2_spine_id)}')">Open Large View</button>
          </div>
        </div>
      `).join('');
      for (let i = 0; i < cands.length; i++) {
        const c = cands[i];
        const p2 = await fetchCropPreview('t2', String(c.t2_spine_id));
        if (!p2?.image_2d) continue;
        drawImageOnCanvas(`top5-t2-${i}`, p2.image_2d, p2.intensity_min, p2.intensity_max, [{x: (p2.image_2d[0].length / 2), y: (p2.image_2d.length / 2)}], [
          {x: (p2.image_2d[0].length / 2), y: (p2.image_2d.length / 2), label: `id ${c.t2_spine_id}`}
        ]);
      }
    }

    async function confirmTop5Match(t2Id) {
      if (!top5Current?.has_item || !top5Current?.t1_spine_id) return;
      const resp = await fetch('/review/decision', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          t2_spine_id: String(t2Id),
          t1_spine_id: String(top5Current.t1_spine_id),
          action: 'match',
          notes: 'top5 manual'
        })
      });
      const data = await resp.json();
      setOut(data);
      await refreshCounters();
      await loadTop5Next();
    }

    async function markTop5T1Lost() {
      if (!top5Current?.has_item || !top5Current?.t1_spine_id) return;
      const resp = await fetch(`/review/mark-lost/${encodeURIComponent(String(top5Current.t1_spine_id))}`, { method: 'POST' });
      const data = await resp.json();
      setOut(data);
      await refreshCounters();
      await loadTop5Next();
    }

    async function removeTop5T1() {
      if (!top5Current?.has_item || !top5Current?.t1_spine_id) return;
      const resp = await fetch('/review/decision', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          t2_spine_id: String((top5Current.candidates?.[0]?.t2_spine_id) || ''),
          t1_spine_id: String(top5Current.t1_spine_id),
          action: 'remove_t1',
          notes: 'top5 remove t1'
        })
      });
      const data = await resp.json();
      setOut(data);
      await refreshCounters();
      await loadTop5Next();
    }

    async function removeTop5T2(t2Id) {
      if (!top5Current?.has_item) return;
      const resp = await fetch('/review/decision', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          t2_spine_id: String(t2Id),
          t1_spine_id: top5Current?.t1_spine_id ? String(top5Current.t1_spine_id) : null,
          action: 'remove_t2',
          notes: 'top5 remove t2'
        })
      });
      const data = await resp.json();
      setOut(data);
      await refreshCounters();
      await loadTop5Next();
    }

    async function loadNewValidationNext() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const resp = await fetch('/review/new-validation-next');
      const data = await resp.json();
      if (!resp.ok) {
        setOut(data);
        return;
      }
      const panel = document.getElementById('newValidationPanel');
      const meta = document.getElementById('newValidationMeta');
      if (panel) panel.style.display = '';
      const large = document.getElementById('largeModePanel');
      const top5 = document.getElementById('top5Panel');
      if (large) large.style.display = 'none';
      if (top5) top5.style.display = 'none';
      if (!data.has_item || !data.item?.t2_spine_id) {
        newValidationCurrent = null;
        newValidationLargeContext.active = false;
        applyLargeViewerModeUI();
        updateReviewModeLabels();
        if (meta) meta.textContent = 'Done. No pending new-T2 validation items.';
        if (modalState.open && isDetachedCompareOpen()) {
          setDetachedViewerMessage('Connected. New-T2 validation queue is complete.');
        }
        setOut({ message: 'Final new-T2 validation queue is complete.' });
        return;
      }
      newValidationCurrent = data.item;
      if (Number.isFinite(Number(data.t1_slice_max))) sessionStackZ.t1 = Math.max(0, Math.floor(Number(data.t1_slice_max)));
      if (Number.isFinite(Number(data.t2_slice_max))) sessionStackZ.t2 = Math.max(0, Math.floor(Number(data.t2_slice_max)));
      syncModalSliceSliderMaxFromSession();
      if (meta) {
        meta.textContent = `Pending: ${Number(data.pending_count || 0)} | T2 ${data.item.t2_spine_id} | dendrite ${data.item.t2_dendrite_id ?? '?'} | z=${Number(data.item.z ?? NaN).toFixed(1)} | T1 Z 1-${sessionStackZ.t1 + 1} T2 Z 1-${sessionStackZ.t2 + 1}`;
      }
      await attachNewValidationToLargeViewer(data.item);
    }

    async function attachNewValidationToLargeViewer(item) {
      try {
        await refreshStackSliceBounds();
        const t2Id = String(item.t2_spine_id || '');
        if (!t2Id) return;
        newValidationLargeContext.active = true;
        const ax = Number(item.x ?? NaN);
        const ay = Number(item.y ?? NaN);
        const az = Number(item.z ?? NaN);
        currentQueue = [{
          t2_spine_id: t2Id,
          t2_dendrite_id: item.t2_dendrite_id || null,
          suggested_t1_spine_id: null,
          suggested_t1_dendrite_id: null,
          distance_xy: null,
          distance_z: null,
          final_score: null,
          margin: null,
          nearby_t1_candidates: [],
          final_pass_new_t2_validation: true,
          t1_region_xyz: { x: ax, y: ay, z: az },
          nearby_t2_candidates: [{
            t2_spine_id: t2Id,
            distance_xy: 0.0,
            distance_z: 0.0,
            x: ax,
            y: ay,
            z: az,
          }],
        }];
        renderQueue(currentQueue);
        await openCompareModal(0, { detached: isDetachedCompareOpen() });
        const ziT2 = Math.max(0, Math.min(sessionStackZ.t2, Math.round(az)));
        const rz1 = currentQueue[0]?.t1_region_xyz;
        const ziT1 = Math.max(0, Math.min(sessionStackZ.t1, Math.round(Number(rz1?.z ?? az))));
        if (Number.isFinite(az)) {
          eachModalSliceSlider((el, pane) => {
            el.value = String(pane === 't2' ? ziT2 : ziT1);
          });
        }
        await renderCompareModal(true);
        applyLargeViewerModeUI();
      } catch (e) {
        // keep validation panel usable even if large viewer sync fails
      }
    }

    async function submitNewValidationDecision(decision) {
      if (!newValidationCurrent?.t2_spine_id) {
        setOut('No pending new-T2 item to validate.');
        return;
      }
      const t2Id = String(newValidationCurrent.t2_spine_id);
      const resp = await fetch(`/review/new-validation/${encodeURIComponent(t2Id)}/${encodeURIComponent(String(decision))}`, {
        method: 'POST'
      });
      const data = await resp.json();
      setOut(data);
      if (!resp.ok) return;
      await refreshCounters();
      await loadNewValidationNext();
    }

    async function handleFinalPassDecision(action, t1Id) {
      if (!newValidationCurrent?.t2_spine_id) {
        setOut('No pending new-T2 item to validate.');
        return;
      }
      const t2Id = String(newValidationCurrent.t2_spine_id);
      if (action === 'match' || action === 'manual_match') {
        await matchCurrentNewValidationToExistingT1(t1Id);
        return;
      }
      if (action === 'new') {
        await submitNewValidationDecision('new');
        return;
      }
      if (action === 'remove_t2') {
        await submitNewValidationDecision('artifact');
        return;
      }
      if (action === 'not_in_t1') {
        await submitNewValidationDecision('not_in_t1');
        return;
      }
      if (action === 'ignore_t1') {
        const t1 = String(t1Id || '').trim();
        if (!t1) {
          setOut('No T1 spine selected for Ignore T1.');
          return;
        }
        const resp = await fetch('/review/decision', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            t2_spine_id: t2Id,
            t1_spine_id: t1,
            action: 'ignore_t1',
            notes: 'final pass ignore t1',
          }),
        });
        const data = await resp.json();
        setOut(data);
        if (!resp.ok) return;
        await refreshCounters();
        await loadNewValidationNext();
        return;
      }
      if (action === 'ignore_t2') {
        const t1 = String(t1Id || '').trim();
        const resp = await fetch('/review/decision', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            t2_spine_id: t2Id,
            t1_spine_id: t1 || null,
            action: 'ignore_t2',
            notes: 'final pass ignore t2',
          }),
        });
        const data = await resp.json();
        setOut(data);
        if (!resp.ok) return;
        await refreshCounters();
        await loadNewValidationNext();
        return;
      }
      setOut('This action is not available during Final Pass validation.');
    }

    async function openLargeFromTop5(t2Id, detached = false) {
      if (!top5Current?.has_item || !top5Current?.t1_spine_id) return;
      const t1Id = String(top5Current.t1_spine_id);
      const t2Pick = String(t2Id);
      const nearbyT2 = (top5Current.candidates || []).map((c) => ({
        t2_spine_id: String(c.t2_spine_id),
        distance_xy: Number(c.distance_xy ?? 0),
        x: Number(c.x ?? NaN),
        y: Number(c.y ?? NaN),
        z: Number(c.z ?? NaN),
      }));
      const nearbyT1 = [{
        t1_spine_id: t1Id,
        distance_xy: 0.0,
        distance_z: 0.0,
        final_score: 1.0,
        x: Number(top5Current.t1_xyz?.x ?? NaN),
        y: Number(top5Current.t1_xyz?.y ?? NaN),
        z: Number(top5Current.t1_xyz?.z ?? NaN),
      }];
      currentQueue = [{
        t2_spine_id: t2Pick,
        t2_dendrite_id: top5Current.candidates?.find(c => String(c.t2_spine_id) === t2Pick)?.t2_dendrite_id ?? null,
        suggested_t1_spine_id: t1Id,
        suggested_t1_dendrite_id: top5Current.t1_dendrite_id ?? null,
        distance_xy: Number(top5Current.candidates?.find(c => String(c.t2_spine_id) === t2Pick)?.distance_xy ?? 0),
        distance_z: Number(top5Current.candidates?.find(c => String(c.t2_spine_id) === t2Pick)?.distance_z ?? 0),
        final_score: Number(top5Current.candidates?.find(c => String(c.t2_spine_id) === t2Pick)?.final_score ?? 0),
        margin: null,
        nearby_t1_candidates: nearbyT1,
        nearby_t2_candidates: nearbyT2,
      }];
      renderQueue(currentQueue);
      top5LargeContext = { active: true, baseT1: t1Id, baseT2: t2Pick };
      await openCompareModal(0, { detached });
    }

    async function loadAlgoReviewQueue() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      setMatchMode('large');
      reviewMode = 'algo';
      queueOffset = 0;
      const resp = await fetch('/review/algo-queue?offset=0&limit=2');
      const data = await resp.json();
      if (!resp.ok) {
        setOut(data);
        return;
      }
      queueOffset = data.offset + data.items.length;
      renderQueue(data.items);
      updateReviewModeLabels();
      setOut({
        mode: 'algo_review_2_window',
        loaded: data.items.length,
        total_candidates: data.total_candidates,
        next_offset: queueOffset,
      });
    }

    async function runLocalRegistration() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const resp = await fetch('/review/reoptimize-local?offset=0&limit=0', { method: 'POST' });
      const data = await resp.json();
      if (!resp.ok) {
        setOut(data);
        return;
      }
      queueOffset = data.offset;
      renderQueue(data.items);
      updateReviewModeLabels();
      setOut({
        mode: 'local_reoptimization',
        total_candidates: data.total_candidates,
        note: 'Algo refreshed from saved matches; use Top 5 or Algo review for candidate cards.',
      });
      await loadTop5Next();
    }

    async function fetchCropPreview(timepoint, spineId) {
      const resp = await fetch('/crops/preview', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          timepoint: timepoint,
          spine_id: String(spineId),
          width: 120,
          height: 120,
          depth: 17,
          projection: previewProjection,
          slice_index: previewSliceIndex
        })
      });
      return await resp.json();
    }

    async function renderQueuePreviews() {
      for (let i = 0; i < currentQueue.length; i++) {
        await renderQueuePreviewCard(i);
      }
    }

    async function renderQueuePreviewCard(i) {
      const it = currentQueue[i];
      if (!it) return;
      try {
        const picker = document.getElementById(`q-t1-pick-${i}`);
        const pickerT2 = document.getElementById(`q-t2-pick-${i}`);
        const pickedT1 = picker?.value || it.suggested_t1_spine_id;
        const pickedT2 = pickerT2?.value || it.t2_spine_id;
        const [p2, p1] = await Promise.all([
          fetchCropPreview('t2', pickedT2),
          pickedT1 ? fetchCropPreview('t1', pickedT1) : Promise.resolve(null)
        ]);
        if (p2?.image_2d) {
          const c2 = p2.image_2d?.length || 0;
          const c2w = c2 ? p2.image_2d[0].length : 0;
          drawImageOnCanvas(`q-t2-${i}`, p2.image_2d, p2.intensity_min, p2.intensity_max, [{x: c2w / 2, y: c2 / 2}], [
            {x: c2w / 2, y: c2 / 2, label: `id ${pickedT2} z=${p2.meta?.center_index_source?.z ?? '?'}`}
          ]);
          const tag2 = document.getElementById(`q-t2-tag-${i}`);
          if (tag2) tag2.textContent = `T2 id ${pickedT2} | z ${p2.meta?.center_index_source?.z ?? '?'}`;
        }
        if (p1?.image_2d) {
          const c1 = p1.image_2d?.length || 0;
          const c1w = c1 ? p1.image_2d[0].length : 0;
          drawImageOnCanvas(`q-t1-${i}`, p1.image_2d, p1.intensity_min, p1.intensity_max, [{x: c1w / 2, y: c1 / 2}], [
            {x: c1w / 2, y: c1 / 2, label: `id ${pickedT1 ?? '?'} z=${p1.meta?.center_index_source?.z ?? '?'}`}
          ]);
          const tag1 = document.getElementById(`q-t1-tag-${i}`);
          if (tag1) tag1.textContent = `T1 id ${pickedT1 ?? '?'} | z ${p1.meta?.center_index_source?.z ?? '?'}`;
        }
      } catch (e) {
        // ignore single card preview failure
      }
    }

    const MANUAL_COMMIT_ACTIONS = new Set([
      'match',
      'no_match',
      'remove_t1',
      'remove_t2',
      'ignore_t1',
      'ignore_t2',
      'not_in_t1',
    ]);

    async function setDecision(idx, action) {
      const it = currentQueue[idx];
      if (!it) return;
      const picked = document.getElementById(`q-t1-pick-${idx}`);
      const picked2 = document.getElementById(`q-t2-pick-${idx}`);
      const t1Picked = picked ? picked.value : null;
      const t2Picked = picked2 ? picked2.value : null;
      const finalT2 = t2Picked ? String(t2Picked) : String(it.t2_spine_id);
      pendingDecisions[finalT2] = {
        t2_spine_id: finalT2,
        t1_spine_id: t1Picked ? String(t1Picked) : (it.suggested_t1_spine_id ? String(it.suggested_t1_spine_id) : null),
        action: action,
        notes: ''
      };
      const card = document.getElementById(`qcard-${idx}`);
      if (card) {
        card.classList.add('picked');
        if (action === 'match' || action === 'no_match') {
          card.style.opacity = '0.45';
          const btns = card.querySelectorAll('button, select');
          btns.forEach((el) => {
            if (el.classList && el.classList.contains('undo-btn')) return;
            el.disabled = true;
          });
        }
      }
      const pickedTag = document.getElementById(`q-picked-${idx}`);
      if (pickedTag) pickedTag.textContent = `picked: ${action}`;
      if (reviewMode === 'algo') {
        await commitAlgoDecisionAndAdvance(finalT2);
        return;
      }
      if (MANUAL_COMMIT_ACTIONS.has(String(action || ''))) {
        await commitManualDecisionAndAdvance(finalT2, idx);
      }
    }

    async function commitAlgoDecisionAndAdvance(t2Id) {
      const row = pendingDecisions[String(t2Id)];
      if (!row) return;
      const payload = { ...row };
      if (payload.action === 'no_match') {
        await fetch(`/review/algo-unmatch/${encodeURIComponent(String(payload.t2_spine_id))}`, { method: 'POST' });
      }
      if (
        payload.action !== 'match' &&
        payload.action !== 'remove_t1' &&
        payload.action !== 'no_match' &&
        payload.action !== 'ignore_t1' &&
        payload.action !== 'ignore_t2'
      ) {
        payload.t1_spine_id = null;
      }
      const resp = await fetch('/review/decision', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      delete pendingDecisions[String(t2Id)];
      setOut(data);
      await refreshCounters();
      await loadAlgoReviewQueue();
    }

    async function commitManualDecisionAndAdvance(t2Id, currentIdx = null) {
      const row = pendingDecisions[String(t2Id)];
      if (!row) return false;
      const payload = { ...row };
      if (
        payload.action !== 'match' &&
        payload.action !== 'remove_t1' &&
        payload.action !== 'no_match' &&
        payload.action !== 'ignore_t1' &&
        payload.action !== 'ignore_t2' &&
        payload.action !== 'manual_match'
      ) {
        payload.t1_spine_id = null;
      }
      const resp = await fetch('/review/decision', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      setOut(data);
      if (!resp.ok) return false;
      delete pendingDecisions[String(t2Id)];
      if (Number.isInteger(currentIdx)) {
        const card = document.getElementById(`qcard-${currentIdx}`);
        if (card) card.classList.add('picked');
      }
      await refreshCounters();
      const nextIdx = nextUnresolvedQueueIndex(currentIdx) ?? firstAvailableQueueIndex();
      if (nextIdx !== null) {
        await syncLargeViewerToQueueCard(nextIdx);
        return true;
      }
      await runLocalRegistration();
      const refreshedIdx = nextUnresolvedQueueIndex(modalState.idx) ?? firstAvailableQueueIndex();
      if (refreshedIdx !== null) {
        await syncLargeViewerToQueueCard(refreshedIdx);
        return true;
      }
      return true;
    }

    async function undoDecision(idx) {
      const it = currentQueue[idx];
      if (!it) return;
      const picked2 = document.getElementById(`q-t2-pick-${idx}`);
      const t2 = picked2?.value ? String(picked2.value) : String(it.t2_spine_id);
      delete pendingDecisions[t2];
      const resp = await fetch(`/review/decision/${encodeURIComponent(t2)}/undo`, { method: 'POST' });
      const data = await resp.json();
      const card = document.getElementById(`qcard-${idx}`);
      if (card) {
        card.classList.remove('picked');
        card.style.opacity = '1';
        const btns = card.querySelectorAll('button, select');
        btns.forEach((el) => { el.disabled = false; });
      }
      const pickedTag = document.getElementById(`q-picked-${idx}`);
      if (pickedTag) pickedTag.textContent = '';
      await refreshCounters();
      await renderQueuePreviewCard(idx);
      setOut(data);
    }

    async function unmatchAlgoPair(idx) {
      if (reviewMode !== 'algo') {
        setOut('Unmatch is available in "Review Algo Matches (2-window)" mode.');
        return;
      }
      const it = currentQueue[idx];
      if (!it) return;
      const t2 = String(it.t2_spine_id);
      const resp = await fetch(`/review/algo-unmatch/${encodeURIComponent(t2)}`, { method: 'POST' });
      const data = await resp.json();
      setOut(data);
      await refreshCounters();
      await loadAlgoReviewQueue();
    }

    async function rerunAlgoOnly() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      setOut({
        message: 'Re-running matching using anchors from already-saved matches only.',
      });
      await runLocalRegistration();
      await refreshCounters();
    }

    async function finalizeMatches() {
      if (!sessionLoaded) {
        setOut('Please click "Choose Files and Load" first.');
        return;
      }
      const resp = await fetch('/review/finalize', { method: 'POST' });
      const data = await resp.json();
      setOut(data);
      await refreshCounters();
      if (resp.ok && Number(data.pending_new_validation_count || 0) > 0) {
        await loadNewValidationNext();
      }
    }

    function closeCompareModal() {
      modalState.open = false;
      modalState.idx = null;
      modalDrag.active = false;
      modalDrag.pane = null;
      top5LargeContext.active = false;
      newValidationLargeContext.active = false;
      modalManualT2Draft = null;
      modalManualT1Draft = null;
      const v1 = modalEl('modalViewportT1');
      const v2 = modalEl('modalViewportT2');
      if (v1) v1.classList.remove('grabbing');
      if (v2) v2.classList.remove('grabbing');
      document.getElementById('compareModal').style.display = 'none';
      if (isDetachedCompareOpen()) {
        const w = detachedCompareWindow;
        detachedCompareReady = false;
        detachedCompareWindow = null;
        w.close();
      }
    }

    async function openCompareModal(idx, options = {}) {
      const useDetached = options.detached || isDetachedCompareOpen();
      if (useDetached && !ensureDetachedCompareWindow()) return;
      await refreshStackSliceBounds();
      await refreshManualT2Cache();
      await refreshManualT1Cache();
      modalState.open = true;
      modalState.idx = idx;
      modalView = { t1: { scale: 1.0, tx: 0, ty: 0 }, t2: { scale: 1.0, tx: 0, ty: 0 } };
      document.getElementById('compareModal').style.display = useDetached ? 'none' : 'block';
      hydrateModalSelectors(idx);
      syncModalSliceSliderMaxFromSession();
      modalAlignZOnFetch = true;
      await renderCompareModal();
      updateReviewModeLabels();
      updateLargeViewerStatus();
      applyLargeViewerModeUI();
    }

    function updateReviewModeLabels() {
      const approve = modalEl('modalApproveBtn');
      const reject = modalEl('modalRejectBtn');
      if (!approve || !reject) return;
      if (newValidationLargeContext.active) {
        approve.textContent = 'Match to selected T1';
        reject.textContent = 'Reject / No Match';
        return;
      }
      if (reviewMode === 'algo') {
        approve.textContent = 'Approve';
        reject.textContent = 'Reject';
      } else {
        approve.textContent = 'Approve / Match';
        reject.textContent = 'Reject / No Match';
      }
    }

    function eachModalSliceSlider(fn) {
      const docs = [document];
      if (isDetachedCompareOpen() && detachedCompareWindow?.document) {
        docs.push(detachedCompareWindow.document);
      }
      for (const doc of docs) {
        for (const id of ['modalSliceT1', 'modalSliceT2']) {
          const el = doc.getElementById(id);
          if (el) fn(el, id === 'modalSliceT1' ? 't1' : 't2');
        }
      }
    }

    function applySessionStackZFromStats(stats) {
      if (!stats) return;
      const z1 = stats.t1_z_range;
      const z2 = stats.t2_z_range;
      if (Array.isArray(z1) && z1.length >= 2 && Number.isFinite(Number(z1[1]))) {
        sessionStackZ.t1 = Math.max(0, Math.floor(Number(z1[1])));
      } else if (Number.isFinite(Number(stats.t1_slice_max))) {
        sessionStackZ.t1 = Math.max(0, Math.floor(Number(stats.t1_slice_max)));
      }
      if (Array.isArray(z2) && z2.length >= 2 && Number.isFinite(Number(z2[1]))) {
        sessionStackZ.t2 = Math.max(0, Math.floor(Number(z2[1])));
      } else if (Number.isFinite(Number(stats.t2_slice_max))) {
        sessionStackZ.t2 = Math.max(0, Math.floor(Number(stats.t2_slice_max)));
      }
      syncModalSliceSliderMaxFromSession();
    }

    function syncModalSliceSliderMaxFromSession() {
      const m1 = Math.max(0, Math.floor(Number(sessionStackZ.t1)));
      const m2 = Math.max(0, Math.floor(Number(sessionStackZ.t2)));
      eachModalSliceSlider((el, pane) => {
        const m = pane === 't1' ? m1 : m2;
        el.min = '0';
        el.max = String(m);
        el.step = '1';
        let c = Number(el.value || 0);
        if (c > m) el.value = String(m);
        if (c < 0) el.value = '0';
      });
    }

    async function refreshStackSliceBounds() {
      if (!sessionLoaded) return;
      try {
        const resp = await fetch('/session/stack-bounds');
        if (!resp.ok) return;
        const d = await resp.json();
        if (Number.isFinite(Number(d.t1_slice_max))) sessionStackZ.t1 = Math.max(0, Math.floor(Number(d.t1_slice_max)));
        if (Number.isFinite(Number(d.t2_slice_max))) sessionStackZ.t2 = Math.max(0, Math.floor(Number(d.t2_slice_max)));
        syncModalSliceSliderMaxFromSession();
      } catch (e) {
        // ignore
      }
    }

    async function hydrateViewerState() {
      try {
        if (!sessionLoaded) return;
        await refreshStackSliceBounds();
        const resp = await fetch('/session/viewer-state');
        const data = await resp.json();
        const s1 = modalEl('modalSliceT1');
        const s2 = modalEl('modalSliceT2');
        if (s1 && Number.isFinite(Number(data.modal_slice_t1))) s1.value = String(data.modal_slice_t1);
        if (s2 && Number.isFinite(Number(data.modal_slice_t2))) s2.value = String(data.modal_slice_t2);
        syncModalSliceSliderMaxFromSession();
      } catch (e) {
        // ignore
      }
    }

    function hydrateModalSelectors(idx) {
      const it = currentQueue[idx];
      if (!it) return;
      const t2Sel = modalEl('modalT2Pick');
      const t1Sel = modalEl('modalT1Pick');
      if (!t2Sel || !t1Sel) return;
      t2Sel.innerHTML = '';
      t1Sel.innerHTML = '';
      (it.nearby_t2_candidates || []).forEach((c) => {
        const o = modalDocument().createElement('option');
        o.value = String(c.t2_spine_id);
        const dxy = Number(c.distance_xy ?? 0).toFixed(1);
        o.textContent = `${c.t2_spine_id} (dxy ${dxy})`;
        t2Sel.appendChild(o);
      });
      if (it.final_pass_new_t2_validation) {
        const ph = modalDocument().createElement('option');
        ph.value = '';
        ph.textContent = '— No suggested T1: click T1 or pick manual T1 —';
        t1Sel.appendChild(ph);
      }
      (it.nearby_t1_candidates || []).forEach((c) => {
        const o = modalDocument().createElement('option');
        o.value = String(c.t1_spine_id);
        const dxy = Number(c.distance_xy ?? 0).toFixed(1);
        const s = Number(c.final_score ?? 0).toFixed(3);
        o.textContent = `${c.t1_spine_id} (dxy ${dxy}, s ${s})`;
        t1Sel.appendChild(o);
      });
      if (!t2Sel.options.length) {
        const o = modalDocument().createElement('option'); o.value = String(it.t2_spine_id); o.textContent = String(it.t2_spine_id); t2Sel.appendChild(o);
      }
      ensureManualT2OptionsOnSelect(t2Sel);
      if (!it.final_pass_new_t2_validation && !t1Sel.options.length && it.suggested_t1_spine_id) {
        const o = modalDocument().createElement('option'); o.value = String(it.suggested_t1_spine_id); o.textContent = String(it.suggested_t1_spine_id); t1Sel.appendChild(o);
      }
      ensureManualT1OptionsOnSelect(t1Sel);
      t2Sel.value = String(document.getElementById(`q-t2-pick-${idx}`)?.value || it.t2_spine_id);
      if (it.final_pass_new_t2_validation) {
        const qv = document.getElementById(`q-t1-pick-${idx}`)?.value;
        t1Sel.value = (qv !== undefined && qv !== null && String(qv).trim() !== '') ? String(qv) : '';
      } else {
        t1Sel.value = String(document.getElementById(`q-t1-pick-${idx}`)?.value || it.suggested_t1_spine_id || '');
      }
    }

    function candidateById(list, key, value) {
      return (list || []).find((x) => String(x[key]) === String(value)) || null;
    }

    function drawModalCanvas(canvasId, image2d, minV, maxV, zoom, brightness, contrast, label, marker, panX, panY) {
      const canvas = modalEl(canvasId);
      const h = image2d.length;
      const w = h ? image2d[0].length : 0;
      if (!canvas || !h || !w) return;
      const tmp = modalDocument().createElement('canvas');
      tmp.width = w; tmp.height = h;
      const ctx = tmp.getContext('2d');
      const img = ctx.createImageData(w, h);
      const den = (maxV - minV) > 1e-9 ? (maxV - minV) : 1.0;
      let k = 0;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let n = (image2d[y][x] - minV) / den;
          n = ((n - 0.5) * contrast + 0.5) + brightness;
          const v = Math.max(0, Math.min(255, Math.round(n * 255)));
          img.data[k++] = v; img.data[k++] = v; img.data[k++] = v; img.data[k++] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      const out = canvas.getContext('2d');
      out.clearRect(0, 0, canvas.width, canvas.height);
      out.drawImage(tmp, 0, 0, canvas.width, canvas.height);
      if (marker) {
        const sx = canvas.width / w;
        const sy = canvas.height / h;
        const mx = (Number(marker.x || 0) * sx);
        const my = (Number(marker.y || 0) * sy);
        out.strokeStyle = '#ff2b2b';
        // Keep marker stroke visually thin at high zoom.
        out.lineWidth = Math.max(0.6, 1.4 / Math.max(zoom || 1.0, 1e-6));
        out.beginPath();
        out.arc(mx, my, Math.max(2.0, 5.0 / Math.max(zoom || 1.0, 1e-6)), 0, Math.PI * 2);
        out.stroke();
        out.fillStyle = '#ff2b2b';
        out.font = '13px Arial';
        out.fillText(marker.text || '', mx + 9, my - 8);
      }
      out.fillStyle = '#ff2b2b'; out.font = '13px Arial'; out.fillText(label, 10, 20);
    }

    function clampModalPan(pane) {
      const v = modalView[pane];
      const viewport = modalEl(pane === 't1' ? 'modalViewportT1' : 'modalViewportT2');
      const canvas = modalEl(pane === 't1' ? 'modalT1' : 'modalT2');
      if (!viewport || !canvas) return;
      const vw = viewport.clientWidth;
      const vh = viewport.clientHeight;
      const cw = canvas.clientWidth * v.scale;
      const ch = canvas.clientHeight * v.scale;
      const marginX = vw * 0.8;
      const marginY = vh * 0.8;
      const minTx = vw - cw - marginX;
      const maxTx = marginX;
      const minTy = vh - ch - marginY;
      const maxTy = marginY;
      v.tx = Math.max(minTx, Math.min(maxTx, v.tx));
      v.ty = Math.max(minTy, Math.min(maxTy, v.ty));
    }

    function applyModalTransforms() {
      if (modalDrag.active && modalDrag.pane) {
        const pane = modalDrag.pane;
        modalView[pane].tx = modalDrag.startTx + modalDrag.pendingDx;
        modalView[pane].ty = modalDrag.startTy + modalDrag.pendingDy;
        maybeSyncModalPane(pane);
      }
      ['t1', 't2'].forEach((pane) => {
        clampModalPan(pane);
        const v = modalView[pane];
        const canvas = modalEl(pane === 't1' ? 'modalT1' : 'modalT2');
        if (!canvas) return;
        canvas.style.transform = `translate3d(${v.tx}px, ${v.ty}px, 0) scale(${v.scale})`;
      });
    }

    function scheduleModalTransformApply() {
      if (modalRaf) cancelAnimationFrame(modalRaf);
      modalRaf = requestAnimationFrame(() => {
        modalRaf = null;
        applyModalTransforms();
      });
    }

    function maybeSyncModalPane(sourcePane) {
      if (!syncModalMovement) return;
      const target = sourcePane === 't1' ? 't2' : 't1';
      modalView[target] = { ...modalView[sourcePane] };
    }

    function focusModalPaneAtEvent(pane, ev) {
      const viewport = modalEl(pane === 't1' ? 'modalViewportT1' : 'modalViewportT2');
      if (!viewport) return;
      const rect = viewport.getBoundingClientRect();
      const clickX = ev.clientX - rect.left;
      const clickY = ev.clientY - rect.top;
      const scale = FOCUS_ZOOM_SCALE;
      modalView[pane].scale = scale;
      modalView[pane].tx = (rect.width / 2) - (clickX * scale);
      modalView[pane].ty = (rect.height / 2) - (clickY * scale);
      maybeSyncModalPane(pane);
      scheduleModalTransformApply();
    }

    function eventToCanvasImageXY(pane, ev, image2d) {
      const viewport = modalEl(pane === 't1' ? 'modalViewportT1' : 'modalViewportT2');
      const canvas = modalEl(pane === 't1' ? 'modalT1' : 'modalT2');
      if (!viewport || !canvas || !image2d || !image2d.length) return null;
      const rect = viewport.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const py = ev.clientY - rect.top;
      const v = modalView[pane];
      const xCanvas = (px - v.tx) / Math.max(v.scale, 1e-6);
      const yCanvas = (py - v.ty) / Math.max(v.scale, 1e-6);
      const h = image2d.length;
      const w = image2d[0].length;
      const xImg = xCanvas * (w / canvas.width);
      const yImg = yCanvas * (h / canvas.height);
      return { xImg, yImg };
    }

    function clickEventToSourceXYZ(pane, ev) {
      const p = pane === 't1' ? modalCache.p1 : modalCache.p2;
      if (!p?.image_2d || !p?.meta?.source_bounds) return null;
      const pos = eventToCanvasImageXY(pane, ev, p.image_2d);
      if (!pos) return null;
      const x = Number(p.meta.source_bounds.x0 || 0) + Number(pos.xImg || 0);
      const y = Number(p.meta.source_bounds.y0 || 0) + Number(pos.yImg || 0);
      const z = Number(p.meta.center_index_source?.z || 0);
      return { x, y, z };
    }

    async function nearestSpineByClick(pane, ev) {
      const xyz = clickEventToSourceXYZ(pane, ev);
      if (!xyz) return null;
      const tp = pane === 't1' ? 't1' : 't2';
      const resp = await fetch('/spines/nearest', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ timepoint: tp, x: xyz.x, y: xyz.y, z: xyz.z, limit: 5 })
      });
      const data = await resp.json();
      if (!resp.ok || !data.items?.length) return null;
      return String(data.items[0].spine_id);
    }

    async function addManualT2ByClick(ev) {
      const xyz = clickEventToSourceXYZ('t2', ev);
      if (!xyz) return;
      modalManualT2Draft = { x: xyz.x, y: xyz.y, z: xyz.z };
      scheduleModalVisualRender();
      setOut({
        message: 'Manual T2 draft point placed. Click again to adjust, then click "Save Manual Point".',
        draft_xyz: modalManualT2Draft,
      });
    }

    async function addManualT1ByClick(ev) {
      const xyz = clickEventToSourceXYZ('t1', ev);
      if (!xyz) return;
      modalManualT1Draft = { x: xyz.x, y: xyz.y, z: xyz.z };
      scheduleModalVisualRender();
      setOut({
        message: 'Manual T1 draft point placed. Click again to adjust, then click "Save Manual Point".',
        draft_xyz: modalManualT1Draft,
      });
    }

    function clearManualT2Draft() {
      modalManualT2Draft = null;
      modalManualT1Draft = null;
      scheduleModalVisualRender();
    }

    async function saveManualT2Draft() {
      const mode = modalEl('modalClickMode')?.value || 'add_t2_manual';
      if (mode === 'add_t1_manual') {
        if (!modalManualT1Draft) {
          setOut('No manual T1 draft point to save. Use click mode and click on T1 first.');
          return;
        }
        const resp = await fetch('/review/manual-t1-click', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            x: modalManualT1Draft.x,
            y: modalManualT1Draft.y,
            z: modalManualT1Draft.z,
            notes: 'manual t1 click (x,y,z only)'
          })
        });
        const data = await resp.json();
        setOut(data);
        if (resp.ok) {
          lastSavedManualT1Id = data?.item?.manual_id || null;
          if (data?.item?.manual_id) {
            manualT1Cache = [...manualT1Cache.filter((x) => String(x.manual_id) !== String(data.item.manual_id)), data.item];
          }
          const i = modalState.idx;
          if (i !== null) {
            ensureManualT1OptionsOnSelect(document.getElementById(`q-t1-pick-${i}`));
            ensureManualT1OptionsOnSelect(modalEl('modalT1Pick'));
            if (lastSavedManualT1Id) {
              const q = document.getElementById(`q-t1-pick-${i}`);
              const m = modalEl('modalT1Pick');
              if (q) q.value = String(lastSavedManualT1Id);
              if (m) m.value = String(lastSavedManualT1Id);
            }
          }
          modalManualT1Draft = null;
          scheduleModalVisualRender();
          await renderCompareModal(true);
        }
        return;
      }
      if (!modalManualT2Draft) {
        setOut('No manual T2 draft point to save. Use click mode and click on T2 first.');
        return;
      }
      const resp = await fetch('/review/manual-t2-click', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          x: modalManualT2Draft.x,
          y: modalManualT2Draft.y,
          z: modalManualT2Draft.z,
          notes: 'manual t2 click (x,y,z only)'
        })
      });
      const data = await resp.json();
      setOut(data);
      if (resp.ok) {
        lastSavedManualT2Id = data?.item?.manual_id || null;
        if (data?.item?.manual_id) {
          manualT2Cache = [...manualT2Cache.filter((x) => String(x.manual_id) !== String(data.item.manual_id)), data.item];
        }
        const i = modalState.idx;
        if (i !== null) {
          ensureManualT2OptionsOnSelect(document.getElementById(`q-t2-pick-${i}`));
          ensureManualT2OptionsOnSelect(modalEl('modalT2Pick'));
          if (lastSavedManualT2Id) {
            const q = document.getElementById(`q-t2-pick-${i}`);
            const m = modalEl('modalT2Pick');
            if (q) q.value = String(lastSavedManualT2Id);
            if (m) m.value = String(lastSavedManualT2Id);
          }
        }
        modalManualT2Draft = null;
        scheduleModalVisualRender();
      }
    }

    async function matchLastManualT2ToSelectedT1() {
      if (!lastSavedManualT2Id) {
        setOut('No saved manual T2 point yet. Save one first.');
        return;
      }
      const t1Id = String(modalEl('modalT1Pick')?.value || '');
      if (!t1Id) {
        setOut('No selected T1 spine to match.');
        return;
      }
      const resp = await fetch('/review/manual-click-match', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          manual_t2_id: lastSavedManualT2Id,
          t1_spine_id: t1Id,
          notes: 'manual t2 -> selected t1'
        })
      });
      const data = await resp.json();
      setOut(data);
      if (!resp.ok) return;
      alert('Manual T2 -> T1 match saved.');
      // When the large viewer is on a real queue item, commit a real decision and
      // advance exactly like regular matching.
      if (
        modalState.open &&
        modalState.idx !== null &&
        currentQueue[modalState.idx] &&
        !top5LargeContext.active
      ) {
        const i = modalState.idx;
        const t2Id = String(modalEl('modalT2Pick')?.value || document.getElementById(`q-t2-pick-${i}`)?.value || currentQueue[i].t2_spine_id || '');
        const decisionResp = await fetch('/review/decision', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            t2_spine_id: t2Id,
            action: 'manual_match',
            t1_spine_id: t1Id,
            notes: 'manual t2 click matched to selected t1',
          })
        });
        const decisionData = await decisionResp.json();
        setOut(decisionData);
        if (!decisionResp.ok) return;
        await refreshCounters();
        if (reviewMode === 'algo') {
          await loadAlgoReviewQueue();
        } else {
          const card = document.getElementById(`qcard-${i}`);
          if (card) {
            card.classList.add('picked');
            card.style.opacity = '0.45';
          }
          const pickedTag = document.getElementById(`q-picked-${i}`);
          if (pickedTag) pickedTag.textContent = 'picked: manual_match';
          const nextIdx = firstAvailableQueueIndex();
          if (nextIdx !== null) {
            await syncLargeViewerToQueueCard(nextIdx);
          } else {
            // Current manual batch is consumed; pull next batch.
            await runLocalRegistration();
            const refreshedIdx = firstAvailableQueueIndex();
            if (refreshedIdx !== null) await syncLargeViewerToQueueCard(refreshedIdx);
          }
        }
      }
    }

    async function matchLastManualT2ToLastManualT1() {
      if (!lastSavedManualT2Id) {
        setOut('No saved manual T2 point yet. Save one first.');
        return;
      }
      if (!lastSavedManualT1Id) {
        setOut('No saved manual T1 point yet. Save one first.');
        return;
      }
      const resp = await fetch('/review/manual-click-match', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          manual_t2_id: lastSavedManualT2Id,
          manual_t1_id: lastSavedManualT1Id,
          notes: 'manual t2 -> manual t1'
        })
      });
      const data = await resp.json();
      setOut(data);
      if (!resp.ok) return;
      if (
        modalState.open &&
        modalState.idx !== null &&
        currentQueue[modalState.idx] &&
        !top5LargeContext.active
      ) {
        const i = modalState.idx;
        const t2Id = String(modalEl('modalT2Pick')?.value || document.getElementById(`q-t2-pick-${i}`)?.value || currentQueue[i].t2_spine_id || '');
        const decisionResp = await fetch('/review/decision', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            t2_spine_id: t2Id,
            action: 'manual_match',
            t1_spine_id: String(lastSavedManualT1Id),
            notes: 'manual t2 click matched to manual t1',
          })
        });
        const decisionData = await decisionResp.json();
        setOut(decisionData);
        if (!decisionResp.ok) return;
        await refreshCounters();
        if (reviewMode === 'algo') {
          await loadAlgoReviewQueue();
        } else {
          const card = document.getElementById(`qcard-${i}`);
          if (card) {
            card.classList.add('picked');
            card.style.opacity = '0.45';
          }
          const pickedTag = document.getElementById(`q-picked-${i}`);
          if (pickedTag) pickedTag.textContent = 'picked: manual_match';
          const nextIdx = firstAvailableQueueIndex();
          if (nextIdx !== null) {
            await syncLargeViewerToQueueCard(nextIdx);
          } else {
            // Current manual batch is consumed; pull next batch.
            await runLocalRegistration();
            const refreshedIdx = firstAvailableQueueIndex();
            if (refreshedIdx !== null) await syncLargeViewerToQueueCard(refreshedIdx);
          }
        }
      }
    }

    async function addManualT1FromNewValidation() {
      const x = Number(document.getElementById('newValidationManualT1X')?.value);
      const y = Number(document.getElementById('newValidationManualT1Y')?.value);
      const z = Number(document.getElementById('newValidationManualT1Z')?.value);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        setOut('Provide numeric x, y, z for manual T1 point.');
        return;
      }
      const resp = await fetch('/review/manual-t1-click', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ x, y, z, notes: 'manual t1 from new-validation panel' })
      });
      const data = await resp.json();
      setOut(data);
      if (resp.ok) {
        lastSavedManualT1Id = data?.item?.manual_id || null;
      }
    }

    async function matchCurrentNewValidationToExistingT1() {
      if (!newValidationCurrent?.t2_spine_id) {
        setOut('No pending T2 item in new-validation panel.');
        return;
      }
      const t1Id = String(
        arguments[0]
        || modalEl('modalT1Pick')?.value
        || document.getElementById('q-t1-pick-0')?.value
        || document.getElementById('newValidationMatchT1Id')?.value
        || ''
      ).trim();
      if (!t1Id) {
        setOut('Select an existing T1 in the dropdown, or add a manual T1 (click mode + Save Manual Point), then use 4) Match / approve.');
        return;
      }
      const resp = await fetch(`/review/new-validation/match/${encodeURIComponent(String(newValidationCurrent.t2_spine_id))}/${encodeURIComponent(t1Id)}`, {
        method: 'POST'
      });
      const data = await resp.json();
      setOut(data);
      if (!resp.ok) return;
      await refreshCounters();
      await loadNewValidationNext();
    }

    async function matchCurrentNewValidationToLastManualT1() {
      if (!newValidationCurrent?.t2_spine_id) {
        setOut('No pending T2 item in new-validation panel.');
        return;
      }
      if (!lastSavedManualT1Id) {
        setOut('No saved manual T1 point yet. Add one first.');
        return;
      }
      const resp = await fetch('/review/manual-click-match', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          t2_spine_id: String(newValidationCurrent.t2_spine_id),
          manual_t1_id: String(lastSavedManualT1Id),
          notes: 'new-validation t2 -> manual t1'
        })
      });
      const data = await resp.json();
      setOut(data);
      if (!resp.ok) return;
      const markResp = await fetch('/review/decision', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          t2_spine_id: String(newValidationCurrent.t2_spine_id),
          action: 'manual_match',
          t1_spine_id: String(lastSavedManualT1Id),
          notes: 'new-validation matched to manual t1',
        })
      });
      const markData = await markResp.json();
      setOut(markData);
      if (!markResp.ok) return;
      await refreshCounters();
      await loadNewValidationNext();
    }

    async function suggestNearestByClick(pane, ev) {
      if (!modalState.open || modalState.idx === null) return;
      const i = modalState.idx;
      const bestId = await nearestSpineByClick(pane, ev);
      if (!bestId) return;
      if (pane === 't1') {
        const modalSel = modalEl('modalT1Pick');
        if (modalSel) {
          let opt = Array.from(modalSel.options).find(o => o.value === bestId);
          if (!opt) {
            opt = modalDocument().createElement('option');
            opt.value = bestId;
            opt.textContent = `${bestId} (clicked)`;
            modalSel.appendChild(opt);
          }
          modalSel.value = bestId;
        }
        const cardSel = document.getElementById(`q-t1-pick-${i}`);
        if (cardSel) {
          let opt = Array.from(cardSel.options).find(o => o.value === bestId);
          if (!opt) {
            opt = document.createElement('option');
            opt.value = bestId;
            opt.textContent = `${bestId} (clicked)`;
            cardSel.appendChild(opt);
          }
          cardSel.value = bestId;
        }
      } else {
        const modalSel = modalEl('modalT2Pick');
        if (modalSel) {
          let opt = Array.from(modalSel.options).find(o => o.value === bestId);
          if (!opt) {
            opt = modalDocument().createElement('option');
            opt.value = bestId;
            opt.textContent = `${bestId} (clicked)`;
            modalSel.appendChild(opt);
          }
          modalSel.value = bestId;
        }
        const cardSel = document.getElementById(`q-t2-pick-${i}`);
        if (cardSel) {
          let opt = Array.from(cardSel.options).find(o => o.value === bestId);
          if (!opt) {
            opt = document.createElement('option');
            opt.value = bestId;
            opt.textContent = `${bestId} (clicked)`;
            cardSel.appendChild(opt);
          }
          cardSel.value = bestId;
        }
      }
      await renderCompareModal(true);
      await renderQueuePreviewCard(i);
    }

    async function markLostNewByClick(pane, ev) {
      if (!modalState.open || modalState.idx === null) return;
      const i = modalState.idx;
      const bestId = await nearestSpineByClick(pane, ev);
      if (!bestId) return;
      if (pane === 't1') {
        const resp = await fetch(`/review/mark-lost/${encodeURIComponent(bestId)}`, { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) modalLastMark = { type: 'lost', spineId: bestId };
        setOut(data);
      } else {
        const resp = await fetch(`/review/mark-new/${encodeURIComponent(bestId)}`, { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) modalLastMark = { type: 'new', spineId: bestId };
        setOut(data);
      }
      const card = document.getElementById(`qcard-${i}`);
      if (card) {
        card.classList.add('picked');
        card.style.opacity = '0.45';
      }
      const pickedTag = document.getElementById(`q-picked-${i}`);
      if (pickedTag) {
        pickedTag.textContent = (pane === 't1') ? 'completed: t1 lost' : 'completed: t2 new';
      }
      await refreshCounters();
      if (reviewMode === 'algo') {
        if (!isDetachedCompareOpen()) closeCompareModal();
        await loadAlgoReviewQueue();
      } else {
        await renderQueuePreviewCard(i);
        await renderCompareModal(true);
      }
    }

    function resetModalPane(pane) {
      modalView[pane].scale = 1.0;
      modalView[pane].tx = 0.0;
      modalView[pane].ty = 0.0;
      maybeSyncModalPane(pane);
      scheduleModalTransformApply();
    }

    function setModalZoomBoth(scale) {
      const s = Math.max(0.25, Math.min(8.0, Number(scale || 1.0)));
      modalView.t1.scale = s;
      modalView.t2.scale = s;
      clampModalPan('t1');
      clampModalPan('t2');
      scheduleModalTransformApply();
    }

    function renderCompareModalVisual() {
      if (!modalState.open || modalState.idx === null) return;
      const i = modalState.idx;
      const it = currentQueue[i];
      if (!it) return;
      const p2 = modalCache.p2;
      const p1 = modalCache.p1;
      const pickedT2 = modalCache.pickedT2;
      const picked = modalCache.pickedT1;
      if (!p1 || !p2) return;
      const brightness = Number(modalEl('modalBrightness')?.value || 0) / 100.0;
      const contrast = Number(modalEl('modalContrast')?.value || 100) / 100.0;
      if (p2?.image_2d) {
        drawModalCanvas(
          'modalT2',
          p2.image_2d,
          p2.intensity_min,
          p2.intensity_max,
          modalView.t2.scale,
          brightness,
          contrast,
          `T2 ${pickedT2}`,
          {
            x: p2.meta?.center_index_local?.x,
            y: p2.meta?.center_index_local?.y,
            text: `id ${pickedT2}`,
          },
          modalView.t2.tx,
          modalView.t2.ty
        );
        if (modalManualT2Draft) {
          const cv = modalEl('modalT2');
          const out = cv ? cv.getContext('2d') : null;
          const bx = Number(p2.meta?.source_bounds?.x0 ?? 0);
          const by = Number(p2.meta?.source_bounds?.y0 ?? 0);
          const lx = Number(modalManualT2Draft.x) - bx;
          const ly = Number(modalManualT2Draft.y) - by;
          const ih = p2.image_2d.length || 0;
          const iw = ih ? p2.image_2d[0].length : 0;
          if (out && iw > 0 && ih > 0) {
            const sx = cv.width / iw;
            const sy = cv.height / ih;
            const mx = lx * sx;
            const my = ly * sy;
            out.strokeStyle = '#ffd54f';
            out.lineWidth = Math.max(0.8, 1.8 / Math.max(modalView.t2.scale || 1.0, 1e-6));
            out.beginPath();
            out.arc(mx, my, Math.max(3.0, 7.0 / Math.max(modalView.t2.scale || 1.0, 1e-6)), 0, Math.PI * 2);
            out.stroke();
            out.fillStyle = '#ffd54f';
            out.font = '12px Arial';
            out.fillText('manual draft', mx + 8, my - 8);
          }
        }
      }
      if (p1?.image_2d) {
        const t1Hdr = picked ? `T1 ${picked}` : (it.final_pass_new_t2_validation ? 'T1 region (candidate xyz)' : 'T1');
        const t1Mrk = picked ? `id ${picked}` : (it.final_pass_new_t2_validation ? 'no T1 id yet' : 'id ?');
        drawModalCanvas(
          'modalT1',
          p1.image_2d,
          p1.intensity_min,
          p1.intensity_max,
          modalView.t1.scale,
          brightness,
          contrast,
          t1Hdr,
          {
            x: p1.meta?.center_index_local?.x,
            y: p1.meta?.center_index_local?.y,
            text: t1Mrk,
          },
          modalView.t1.tx,
          modalView.t1.ty
        );
        if (modalManualT1Draft) {
          const cv = modalEl('modalT1');
          const out = cv ? cv.getContext('2d') : null;
          const bx = Number(p1.meta?.source_bounds?.x0 ?? 0);
          const by = Number(p1.meta?.source_bounds?.y0 ?? 0);
          const lx = Number(modalManualT1Draft.x) - bx;
          const ly = Number(modalManualT1Draft.y) - by;
          const ih = p1.image_2d.length || 0;
          const iw = ih ? p1.image_2d[0].length : 0;
          if (out && iw > 0 && ih > 0) {
            const sx = cv.width / iw;
            const sy = cv.height / ih;
            const mx = lx * sx;
            const my = ly * sy;
            out.strokeStyle = '#6ee7ff';
            out.lineWidth = Math.max(0.8, 1.8 / Math.max(modalView.t1.scale || 1.0, 1e-6));
            out.beginPath();
            out.arc(mx, my, Math.max(3.0, 7.0 / Math.max(modalView.t1.scale || 1.0, 1e-6)), 0, Math.PI * 2);
            out.stroke();
            out.fillStyle = '#6ee7ff';
            out.font = '12px Arial';
            out.fillText('manual draft', mx + 8, my - 8);
          }
        }
      }

      const c2 = candidateById(it.nearby_t2_candidates, 't2_spine_id', pickedT2);
      const c1 = candidateById(it.nearby_t1_candidates, 't1_spine_id', picked);
      const tag2 = modalEl('modalTagT2');
      const tag1 = modalEl('modalTagT1');
      if (tag2) tag2.textContent = `T2 id ${pickedT2} | x=${Number(c2?.x ?? NaN).toFixed(1)} y=${Number(c2?.y ?? NaN).toFixed(1)} z=${Number(c2?.z ?? NaN).toFixed(1)}`;
      if (tag1) {
        if (it.final_pass_new_t2_validation && !picked) {
          const r = it.t1_region_xyz || {};
          tag1.textContent = `T1 region (same xyz as candidate T2) | x=${Number(r.x ?? NaN).toFixed(1)} y=${Number(r.y ?? NaN).toFixed(1)} z=${Number(r.z ?? NaN).toFixed(1)} — pick or click T1`;
        } else {
          tag1.textContent = `T1 id ${picked || '—'} | x=${Number(c1?.x ?? NaN).toFixed(1)} y=${Number(c1?.y ?? NaN).toFixed(1)} z=${Number(c1?.z ?? NaN).toFixed(1)}`;
        }
      }
      if (tag2 && modalManualT2Draft) {
        tag2.textContent += ` | draft manual x=${Number(modalManualT2Draft.x).toFixed(1)} y=${Number(modalManualT2Draft.y).toFixed(1)} z=${Number(modalManualT2Draft.z).toFixed(1)}`;
      }
      if (tag1 && modalManualT1Draft) {
        tag1.textContent += ` | draft manual x=${Number(modalManualT1Draft.x).toFixed(1)} y=${Number(modalManualT1Draft.y).toFixed(1)} z=${Number(modalManualT1Draft.z).toFixed(1)}`;
      }
      const s2 = modalEl('modalSliceT2');
      const s1 = modalEl('modalSliceT1');
      const maxT2 = Math.max(0, Math.floor(Number(sessionStackZ.t2)));
      const maxT1 = Math.max(0, Math.floor(Number(sessionStackZ.t1)));
      if (s2) {
        const v = Number(s2.value || 0);
        const txt = `Z ${v + 1}/${maxT2 + 1}`;
        const label2 = modalEl('modalSliceT2Label');
        if (label2) label2.textContent = txt;
        const ov2 = modalEl('modalSliceOverlayT2');
        if (ov2) ov2.textContent = txt;
      }
      if (s1) {
        const v = Number(s1.value || 0);
        const txt = `Z ${v + 1}/${maxT1 + 1}`;
        const label1 = modalEl('modalSliceT1Label');
        if (label1) label1.textContent = txt;
        const ov1 = modalEl('modalSliceOverlayT1');
        if (ov1) ov1.textContent = txt;
      }
      applyModalTransforms();
      updateLargeViewerStatus();
    }

    function scheduleModalVisualRender() {
      requestAnimationFrame(() => { renderCompareModalVisual(); });
    }

    async function renderCompareModal(forceFetch = false) {
      if (!modalState.open || modalState.idx === null) return;
      syncModalSliceSliderMaxFromSession();
      const i = modalState.idx;
      const it = currentQueue[i];
      if (!it) return;
      const pickedT2 = modalEl('modalT2Pick')?.value || document.getElementById(`q-t2-pick-${i}`)?.value || it.t2_spine_id;
      const pickedRaw = modalEl('modalT1Pick')?.value ?? document.getElementById(`q-t1-pick-${i}`)?.value ?? it.suggested_t1_spine_id;
      const picked = (pickedRaw !== undefined && pickedRaw !== null && String(pickedRaw).trim() !== '') ? String(pickedRaw).trim() : '';
      const selectionChanged = (String(modalCache.pickedT2 || '') !== String(pickedT2 || '')) || (String(modalCache.pickedT1 || '') !== String(picked || ''));
      if (selectionChanged) {
        const s1e = modalEl('modalSliceT1');
        const s2e = modalEl('modalSliceT2');
        if (picked) {
          const c1 = candidateById(it.nearby_t1_candidates, 't1_spine_id', picked);
          if (s1e && c1 && Number.isFinite(Number(c1.z))) {
            const zc = Math.round(Number(c1.z));
            s1e.value = String(Math.max(0, Math.min(sessionStackZ.t1, zc)));
          }
        }
        const c2 = candidateById(it.nearby_t2_candidates, 't2_spine_id', String(pickedT2));
        if (s2e && c2 && Number.isFinite(Number(c2.z))) {
          const zc = Math.round(Number(c2.z));
          s2e.value = String(Math.max(0, Math.min(sessionStackZ.t2, zc)));
        }
      }
      const sliceT2Raw = Number(modalEl('modalSliceT2')?.value || 10);
      const sliceT1Raw = Number(modalEl('modalSliceT1')?.value || 10);
      function clampStackSlice(raw, zmax) {
        const m = Math.max(0, Math.floor(Number(zmax)));
        if (!Number.isFinite(m)) return 0;
        return Math.max(0, Math.min(m, Math.round(Number(raw) || 0)));
      }
      const sliceT1Req = clampStackSlice(sliceT1Raw, sessionStackZ.t1);
      const sliceT2Req = clampStackSlice(sliceT2Raw, sessionStackZ.t2);
      fetch('/session/viewer-state', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ modal_slice_t1: sliceT1Req, modal_slice_t2: sliceT2Req }),
      });
      const fp = !!it.final_pass_new_t2_validation;
      const key = `${pickedT2}|${picked || '@xyz'}|${sliceT2Req}|${sliceT1Req}|fp${fp ? 1 : 0}|sg`;
      if (!forceFetch && modalCache.key === key && modalCache.p1 && modalCache.p2) {
        scheduleModalVisualRender();
        return;
      }
      let p1Body;
      if (fp && !picked) {
        const r = it.t1_region_xyz || {};
        p1Body = {
          timepoint: 't1',
          x: Number(r.x ?? NaN),
          y: Number(r.y ?? NaN),
          z: Number(r.z ?? NaN),
          width: 1024,
          height: 1024,
          depth: 21,
          projection: 'slice',
          slice_z_mode: 'stack_global',
          slice_index: sliceT1Req,
        };
      } else {
        p1Body = {
          timepoint: 't1',
          spine_id: String(picked),
          width: 1024,
          height: 1024,
          depth: 21,
          projection: 'slice',
          slice_z_mode: 'stack_global',
          slice_index: sliceT1Req,
        };
      }
      const t2PreviewBody = {
        timepoint: 't2',
        spine_id: String(pickedT2),
        width: 1024,
        height: 1024,
        depth: 21,
        projection: 'slice',
        slice_z_mode: 'stack_global',
        slice_index: sliceT2Req,
      };
      const [p2, p1] = await Promise.all([
        fetch('/crops/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(t2PreviewBody)}).then(r=>r.json()),
        fetch('/crops/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p1Body)}).then(r=>r.json()),
      ]);
      modalCache = { key, p1, p2, pickedT1: picked || null, pickedT2 };
      if (newValidationLargeContext.active) {
        modalAlignZOnFetch = false;
      } else if (selectionChanged || modalAlignZOnFetch) {
        const s1 = modalEl('modalSliceT1');
        const s2 = modalEl('modalSliceT2');
        let changed = false;
        if (s1 && Number.isFinite(Number(p1?.meta?.center_index_source?.z))) {
          const z1 = Math.max(Number(s1.min), Math.min(Number(s1.max), Number(p1.meta.center_index_source.z)));
          if (Number(s1.value) !== z1) {
            s1.value = String(z1);
            changed = true;
          }
        }
        if (s2 && Number.isFinite(Number(p2?.meta?.center_index_source?.z))) {
          const z2 = Math.max(Number(s2.min), Math.min(Number(s2.max), Number(p2.meta.center_index_source.z)));
          if (Number(s2.value) !== z2) {
            s2.value = String(z2);
            changed = true;
          }
        }
        modalAlignZOnFetch = false;
        if (changed) {
          await renderCompareModal(true);
          return;
        }
      }
      scheduleModalVisualRender();
    }

    async function modalSetDecision(action) {
      if (!modalState.open || modalState.idx === null) return;
      const i = modalState.idx;
      const row = currentQueue[i];
      if (!row) return;
      let t2 = String(modalEl('modalT2Pick')?.value || document.getElementById(`q-t2-pick-${i}`)?.value || row.t2_spine_id || '').trim();
      let t1 = String(modalEl('modalT1Pick')?.value || document.getElementById(`q-t1-pick-${i}`)?.value || row.suggested_t1_spine_id || '').trim();
      const q2 = document.getElementById(`q-t2-pick-${i}`);
      const q1 = document.getElementById(`q-t1-pick-${i}`);
      if (q2 && t2) q2.value = t2;
      if (q1 && t1) q1.value = t1;
      if ((action === 'ignore_t1' || action === 'remove_t1' || action === 'match') && !t1) {
        setOut('No T1 spine selected. Choose a T1 in the large viewer (or queue card) first.');
        return;
      }
      if (newValidationLargeContext.active) {
        await handleFinalPassDecision(action, t1);
        return;
      }
      if (top5LargeContext.active) {
        if (action === 'match') {
          top5LargeContext.active = false;
          confirmTop5Match(String(t2 || top5LargeContext.baseT2 || ''));
          if (!isDetachedCompareOpen()) closeCompareModal();
          return;
        }
        if (action === 'no_match') {
          const t1Pick = String(t1 || top5LargeContext.baseT1 || '');
          const t2Pick = String(t2 || top5LargeContext.baseT2 || '');
          fetch('/review/decision', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              t2_spine_id: t2Pick,
              t1_spine_id: t1Pick,
              action: 'no_match',
              notes: 'top5 large reject'
            })
          }).then(() => refreshCounters()).then(() => loadTop5Next());
          top5LargeContext.active = false;
          if (!isDetachedCompareOpen()) closeCompareModal();
          return;
        }
        if (action === 'remove_t1') {
          const t1Pick = String(t1 || top5LargeContext.baseT1 || '');
          fetch('/review/decision', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              t2_spine_id: String(t2 || top5LargeContext.baseT2 || ''),
              t1_spine_id: t1Pick,
              action: 'remove_t1',
              notes: 'top5 large remove t1'
            })
          }).then(() => refreshCounters()).then(() => loadTop5Next());
          top5LargeContext.active = false;
          if (!isDetachedCompareOpen()) closeCompareModal();
          return;
        }
        if (action === 'remove_t2') {
          const t2Pick = String(t2 || top5LargeContext.baseT2 || '');
          fetch('/review/decision', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              t2_spine_id: t2Pick,
              t1_spine_id: String(t1 || top5LargeContext.baseT1 || ''),
              action: 'remove_t2',
              notes: 'top5 large remove t2'
            })
          }).then(() => refreshCounters()).then(() => loadTop5Next());
          top5LargeContext.active = false;
          if (!isDetachedCompareOpen()) closeCompareModal();
          return;
        }
        if (action === 'ignore_t1') {
          const t1Pick = String(t1 || top5LargeContext.baseT1 || '');
          fetch('/review/decision', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              t2_spine_id: String(t2 || top5LargeContext.baseT2 || ''),
              t1_spine_id: t1Pick,
              action: 'ignore_t1',
              notes: 'top5 large ignore t1'
            })
          }).then(() => refreshCounters()).then(() => loadTop5Next());
          top5LargeContext.active = false;
          if (!isDetachedCompareOpen()) closeCompareModal();
          return;
        }
        if (action === 'ignore_t2') {
          const t2Pick = String(t2 || top5LargeContext.baseT2 || '');
          fetch('/review/decision', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              t2_spine_id: t2Pick,
              t1_spine_id: String(t1 || top5LargeContext.baseT1 || ''),
              action: 'ignore_t2',
              notes: 'top5 large ignore t2'
            })
          }).then(() => refreshCounters()).then(() => loadTop5Next());
          top5LargeContext.active = false;
          if (!isDetachedCompareOpen()) closeCompareModal();
          return;
        }
      }
      const didManualCommit = reviewMode !== 'algo' && MANUAL_COMMIT_ACTIONS.has(String(action || ''));
      await setDecision(i, action);
      if (isDetachedCompareOpen()) {
        if (reviewMode === 'algo') {
          setTimeout(() => {
            syncLargeViewerToQueueCard(firstAvailableQueueIndex() ?? i);
          }, 50);
        } else if (!didManualCommit) {
          setTimeout(() => {
            const nextIdx = nextUnresolvedQueueIndex(i) ?? firstAvailableQueueIndex();
            if (nextIdx !== null) {
              syncLargeViewerToQueueCard(nextIdx);
              return;
            }
            runLocalRegistration().then(() => {
              const refreshedIdx = nextUnresolvedQueueIndex(modalState.idx) ?? firstAvailableQueueIndex();
              if (refreshedIdx !== null) syncLargeViewerToQueueCard(refreshedIdx);
            });
          }, 50);
        }
      } else {
        closeCompareModal();
      }
    }

    function modalUndoDecision() {
      if (!modalState.open || modalState.idx === null) return;
      undoDecision(modalState.idx);
      if (!isDetachedCompareOpen()) closeCompareModal();
    }

    function modalUnmatchAlgo() {
      if (!modalState.open || modalState.idx === null) return;
      unmatchAlgoPair(modalState.idx);
      if (!isDetachedCompareOpen()) closeCompareModal();
    }

    async function modalUndoLastMark() {
      if (!modalLastMark) {
        setOut('No recent lost/new mark to undo.');
        return;
      }
      let resp;
      if (modalLastMark.type === 'lost') {
        resp = await fetch(`/review/unmark-lost/${encodeURIComponent(modalLastMark.spineId)}`, { method: 'POST' });
      } else {
        resp = await fetch(`/review/unmark-new/${encodeURIComponent(modalLastMark.spineId)}`, { method: 'POST' });
      }
      const data = await resp.json();
      if (resp.ok) modalLastMark = null;
      setOut(data);
      await refreshCounters();
      if (modalState.open && modalState.idx !== null) {
        await renderQueuePreviewCard(modalState.idx);
        await renderCompareModal(true);
      }
    }

    document.getElementById('t1Dendrites').addEventListener('change', scheduleAutoPreview);
    document.getElementById('t2Dendrites').addEventListener('change', scheduleAutoPreview);

    function wireModalControls(doc) {
      if (!doc || doc.__spineLargeViewerWired) return;
      doc.__spineLargeViewerWired = true;
      const byId = (id) => doc.getElementById(id);
      const bindClick = (id, handler) => {
        const el = byId(id);
        if (el) el.addEventListener('click', handler);
      };
      if (doc !== document) {
        bindClick('modalCloseBtn', closeCompareModal);
        bindClick('modalDismissBtn', closeCompareModal);
      bindClick('modalSaveManualDraftBtn', saveManualT2Draft);
      bindClick('modalClearManualDraftBtn', clearManualT2Draft);
      bindClick('modalMatchLastManualToSelectedBtn', matchLastManualT2ToSelectedT1);
      bindClick('modalMatchLastManualToManualBtn', matchLastManualT2ToLastManualT1);
        bindClick('modalUndoLastMarkBtn', modalUndoLastMark);
        bindClick('modalApproveBtn', () => modalSetDecision('match'));
        bindClick('modalFinalPassMatchBtn', () => modalSetDecision('match'));
        bindClick('modalMarkNewBtn', () => modalSetDecision('new'));
        bindClick('modalRejectBtn', () => modalSetDecision('no_match'));
        bindClick('modalRemoveT1Btn', () => modalSetDecision('remove_t1'));
        bindClick('modalRemoveT2Btn', () => modalSetDecision('remove_t2'));
        bindClick('modalNotInT1Btn', () => modalSetDecision('not_in_t1'));
        bindClick('modalIgnoreT1Btn', () => modalSetDecision('ignore_t1'));
        bindClick('modalIgnoreT2Btn', () => modalSetDecision('ignore_t2'));
        bindClick('modalUndoDecisionBtn', modalUndoDecision);
        bindClick('modalUnmatchBtn', modalUnmatchAlgo);
        bindClick('modalZoom25', () => setModalZoomBoth(0.25));
        bindClick('modalZoom50', () => setModalZoomBoth(0.5));
        bindClick('modalZoom75', () => setModalZoomBoth(0.75));
        bindClick('modalZoom100', () => setModalZoomBoth(1.0));
        bindClick('modalZoom200', () => setModalZoomBoth(2.0));
      }

      ['modalSliceT1','modalSliceT2','modalBrightness','modalContrast','modalT1Pick','modalT2Pick'].forEach((id) => {
        const el = byId(id);
        if (!el) return;
        const evt = (id === 'modalT1Pick' || id === 'modalT2Pick') ? 'change' : 'input';
        el.addEventListener(evt, async () => {
          if (id === 'modalT1Pick' || id === 'modalT2Pick') {
            modalAlignZOnFetch = true;
            syncModalSelectionsToQueue();
            if (modalState.idx !== null) await renderQueuePreviewCard(modalState.idx);
          }
          const fetchNeeded = (id === 'modalSliceT1' || id === 'modalSliceT2' || id === 'modalT1Pick' || id === 'modalT2Pick');
          if (fetchNeeded) renderCompareModal(true);
          else scheduleModalVisualRender();
        });
      });

      const modalT2Canvas = byId('modalT2');
      const modalT1Canvas = byId('modalT1');
      const modalViewportT2 = byId('modalViewportT2');
      const modalViewportT1 = byId('modalViewportT1');
      const compareModalRoot = byId('compareModal');
      const preventBrowserZoomScroll = (ev) => {
        if (!modalState.open) return;
        // Block browser-level zoom/scroll while interacting inside large viewer.
        ev.preventDefault();
      };
      [modalViewportT1, modalViewportT2, compareModalRoot].forEach((el) => {
        if (!el) return;
        el.addEventListener('wheel', preventBrowserZoomScroll, { passive: false });
        el.addEventListener('touchmove', preventBrowserZoomScroll, { passive: false });
        el.addEventListener('gesturestart', preventBrowserZoomScroll, { passive: false });
        el.addEventListener('gesturechange', preventBrowserZoomScroll, { passive: false });
      });
      if (modalT2Canvas) {
        modalT2Canvas.addEventListener('wheel', (ev) => {
          if (!modalState.open) return;
          ev.preventDefault();
          const z = byId('modalSliceT2');
          const min = Number(z.min), max = Number(z.max), cur = Number(z.value || 0);
          const next = Math.max(min, Math.min(max, cur + (ev.deltaY > 0 ? 1 : -1)));
          z.value = String(next);
          renderCompareModal(true);
        }, { passive: false });
        modalT2Canvas.addEventListener('click', (ev) => {
          if (!modalState.open || modalDrag.moved) {
            modalDrag.moved = false;
            return;
          }
          if (byId('modalSuggestByClick')?.checked && !newValidationLargeContext.active) {
            const mode = byId('modalClickMode')?.value || 'suggest';
            if (mode === 'mark_t2_new') markLostNewByClick('t2', ev);
            else if (mode === 'add_t2_manual') addManualT2ByClick(ev);
            else if (mode === 'add_t1_manual') focusModalPaneAtEvent('t2', ev);
            else if (mode === 'mark_t1_lost') focusModalPaneAtEvent('t2', ev);
            else suggestNearestByClick('t2', ev);
          } else {
            focusModalPaneAtEvent('t2', ev);
          }
        });
        modalT2Canvas.addEventListener('dblclick', (ev) => {
          ev.preventDefault();
          resetModalPane('t2');
        });
      }
      if (modalT1Canvas) {
        modalT1Canvas.addEventListener('wheel', (ev) => {
          if (!modalState.open) return;
          ev.preventDefault();
          const z = byId('modalSliceT1');
          const min = Number(z.min), max = Number(z.max), cur = Number(z.value || 0);
          const next = Math.max(min, Math.min(max, cur + (ev.deltaY > 0 ? 1 : -1)));
          z.value = String(next);
          renderCompareModal(true);
        }, { passive: false });
        modalT1Canvas.addEventListener('click', (ev) => {
          if (!modalState.open || modalDrag.moved) {
            modalDrag.moved = false;
            return;
          }
          if (newValidationLargeContext.active && byId('modalClickMode')?.value === 'add_t1_manual') {
            void addManualT1ByClick(ev);
            return;
          }
          if (byId('modalSuggestByClick')?.checked) {
            const mode = byId('modalClickMode')?.value || 'suggest';
            if (mode === 'mark_t1_lost') markLostNewByClick('t1', ev);
            else if (mode === 'add_t2_manual') focusModalPaneAtEvent('t1', ev);
            else if (mode === 'add_t1_manual') addManualT1ByClick(ev);
            else if (mode === 'mark_t2_new') focusModalPaneAtEvent('t1', ev);
            else suggestNearestByClick('t1', ev);
          } else {
            focusModalPaneAtEvent('t1', ev);
          }
        });
        modalT1Canvas.addEventListener('dblclick', (ev) => {
          ev.preventDefault();
          resetModalPane('t1');
        });
      }
      setupModalDrag(doc, modalT1Canvas, 't1');
      setupModalDrag(doc, modalT2Canvas, 't2');
      doc.addEventListener('mousemove', (ev) => {
        if (!modalDrag.active || !modalDrag.pane || !modalState.open) return;
        modalDrag.pendingDx = ev.clientX - modalDrag.startX;
        modalDrag.pendingDy = ev.clientY - modalDrag.startY;
        if (Math.abs(modalDrag.pendingDx) > 3 || Math.abs(modalDrag.pendingDy) > 3) modalDrag.moved = true;
        scheduleModalTransformApply();
      }, { passive: true });
      doc.addEventListener('mouseup', () => {
        if (modalDrag.pane) {
          const vp = modalEl(modalDrag.pane === 't1' ? 'modalViewportT1' : 'modalViewportT2');
          if (vp) vp.classList.remove('grabbing');
        }
        modalDrag.active = false;
        modalDrag.pane = null;
        setTimeout(() => { modalDrag.moved = false; }, 0);
      });
      const root = byId('compareModal');
      if (root) {
        root.addEventListener('click', (ev) => {
          if (ev.target && ev.target.id === 'compareModal') {
            closeCompareModal();
          }
        });
      }
      doc.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape' && modalState.open) {
          closeCompareModal();
        }
      });
    }

    function setupModalDrag(doc, canvas, paneKey) {
      if (!canvas) return;
      canvas.addEventListener('mousedown', (ev) => {
        if (!modalState.open) return;
        modalDrag.active = true;
        modalDrag.pane = paneKey;
        modalDrag.moved = false;
        modalDrag.startX = ev.clientX;
        modalDrag.startY = ev.clientY;
        modalDrag.startTx = modalView[paneKey].tx;
        modalDrag.startTy = modalView[paneKey].ty;
        const vp = doc.getElementById(paneKey === 't1' ? 'modalViewportT1' : 'modalViewportT2');
        if (vp) vp.classList.add('grabbing');
        ev.preventDefault();
      });
    }
    wireModalControls(document);
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/select-files", response_model=models.SelectFilesResponse)
def select_files(payload: models.SelectFilesRequest) -> models.SelectFilesResponse:
    try:
        if payload.use_dialog:
            chosen = io_service.select_files_via_dialog(
                payload.t1_tiff_path,
                payload.t2_tiff_path,
                payload.t1_csv_path,
                payload.t2_csv_path,
            )
        else:
            chosen = {
                "t1_tiff_path": str(io_service.validate_existing_path(payload.t1_tiff_path or "", "t1_tiff_path")),
                "t2_tiff_path": str(io_service.validate_existing_path(payload.t2_tiff_path or "", "t2_tiff_path")),
                "t1_csv_path": str(io_service.validate_existing_path(payload.t1_csv_path or "", "t1_csv_path")),
                "t2_csv_path": str(io_service.validate_existing_path(payload.t2_csv_path or "", "t2_csv_path")),
            }
        return models.SelectFilesResponse(**chosen)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/load-session", response_model=models.SessionStats)
def load_session(payload: models.LoadSessionRequest) -> models.SessionStats:
    try:
        stats = _load_session_internal(payload)
        _save_last_session()
        return stats
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/select-and-load", response_model=models.SelectAndLoadResponse)
def select_and_load(payload: models.SelectFilesRequest) -> models.SelectAndLoadResponse:
    try:
        chosen = select_files(payload)
        stats = _load_session_internal(models.LoadSessionRequest(**chosen.model_dump()))
        _save_last_session()
        return models.SelectAndLoadResponse(selected_files=chosen, session=stats)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/session/status", response_model=models.SessionPersistenceResponse)
def session_status() -> models.SessionPersistenceResponse:
    return models.SessionPersistenceResponse(
        ok=True,
        message="Session status checked.",
        has_saved_session=_has_saved_session(),
        saved_path=str(LAST_SESSION_FILE),
    )


@app.post("/session/client-log", response_model=models.ClientLogResponse)
def append_client_log(payload: models.ClientLogRequest) -> models.ClientLogResponse:
    """Append UI activity lines; copied into export folder only on Export Results."""
    try:
        LAST_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        text = (payload.message or "").replace("\r\n", "\n").replace("\r", "\n")
        if len(text) > 200_000:
            text = text[:200_000] + "\n... (truncated)\n"
        line = f"{datetime.now().isoformat()}\n{text}\n---\n"
        with CLIENT_ACTIVITY_LOG.open("a", encoding="utf-8") as f:
            f.write(line)
        return models.ClientLogResponse(ok=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/session/save", response_model=models.SessionPersistenceResponse)
def save_session_now() -> models.SessionPersistenceResponse:
    try:
        _save_last_session()
        return models.SessionPersistenceResponse(
            ok=True,
            message=f"Session saved to {LAST_SESSION_FILE}",
            has_saved_session=True,
            saved_path=str(LAST_SESSION_FILE),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/session/use-last", response_model=models.SessionPersistenceResponse)
def use_last_saved_session() -> models.SessionPersistenceResponse:
    try:
        selected, stats = _restore_last_session()
        session = session_store.require_active_session()
        _refresh_algo_matches(session, use_anchors=True)
        _save_last_session()
        return models.SessionPersistenceResponse(
            ok=True,
            message="Last saved session restored.",
            has_saved_session=True,
            saved_path=str(LAST_SESSION_FILE),
            selected_files=selected,
            session=stats,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/session/undo-last", response_model=models.SessionPersistenceResponse)
def undo_last_session() -> models.SessionPersistenceResponse:
    try:
        selected, stats = _restore_previous_saved_session()
        return models.SessionPersistenceResponse(
            ok=True,
            message="Previous session snapshot restored.",
            has_saved_session=True,
            saved_path=str(LAST_SESSION_FILE),
            selected_files=selected,
            session=stats,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/session/open-new", response_model=models.SessionPersistenceResponse)
def open_new_session() -> models.SessionPersistenceResponse:
    try:
        _open_new_session()
        return models.SessionPersistenceResponse(
            ok=True,
            message="Opened new session. Saved session folder removed.",
            has_saved_session=False,
            saved_path=str(LAST_SESSION_FILE),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/session/viewer-state", response_model=models.ViewerState)
def get_viewer_state() -> models.ViewerState:
    try:
        session = session_store.require_active_session()
        return models.ViewerState(
            modal_slice_t1=int(session.viewer_state.get("modal_slice_t1", 10)),
            modal_slice_t2=int(session.viewer_state.get("modal_slice_t2", 10)),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/session/stack-bounds", response_model=models.StackBoundsResponse)
def session_stack_bounds() -> models.StackBoundsResponse:
    """Z depth of loaded T1/T2 stacks (no algo refresh). Used for large-viewer slice sliders."""
    try:
        session = session_store.require_active_session()
        z1 = int(session.t1_stack.shape[0])
        z2 = int(session.t2_stack.shape[0])
        return models.StackBoundsResponse(
            t1_shape_z=z1,
            t2_shape_z=z2,
            t1_slice_max=max(z1 - 1, 0),
            t2_slice_max=max(z2 - 1, 0),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/session/viewer-state", response_model=models.ViewerState)
def set_viewer_state(payload: models.ViewerState) -> models.ViewerState:
    try:
        session = session_store.require_active_session()
        session.viewer_state = {
            "modal_slice_t1": int(payload.modal_slice_t1),
            "modal_slice_t2": int(payload.modal_slice_t2),
        }
        _save_last_session()
        return payload
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/spines/{timepoint}", response_model=List[models.SpineRecord])
def get_spines(timepoint: models.Timepoint) -> List[models.SpineRecord]:
    try:
        session = session_store.require_active_session()
        lookup = session.t1_lookup if timepoint == "t1" else session.t2_lookup
        return [models.SpineRecord(**item) for item in lookup.values()]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/spines/nearest", response_model=models.NearestSpinesResponse)
def get_nearest_spines(payload: models.NearestSpinesRequest) -> models.NearestSpinesResponse:
    try:
        session = session_store.require_active_session()
        lookup = session.t1_lookup if payload.timepoint == "t1" else session.t2_lookup
        rows = []
        qx = float(payload.x)
        qy = float(payload.y)
        qz = float(payload.z)
        limit = max(1, min(int(payload.limit), 25))
        zmax = _session_z_gap(session)
        for row in lookup.values():
            sx = float(row.get("x", 0.0))
            sy = float(row.get("y", 0.0))
            sz = float(row.get("z", 0.0))
            dxy = float(np.hypot(sx - qx, sy - qy))
            dz = float(abs(sz - qz))
            if dz > zmax:
                continue
            rows.append((dxy + 0.35 * dz, str(row.get("spine_id", "")), dxy, dz))
        rows.sort(key=lambda t: t[0])
        items = [
            models.NearestSpineItem(spine_id=spine_id, distance_xy=dxy, distance_z=dz)
            for _, spine_id, dxy, dz in rows[:limit]
            if spine_id
        ]
        return models.NearestSpinesResponse(
            timepoint=payload.timepoint,
            query={"x": qx, "y": qy, "z": qz},
            items=items,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dendrite-groups/{timepoint}", response_model=List[models.DendriteGroup])
def get_dendrites(timepoint: models.Timepoint) -> List[models.DendriteGroup]:
    try:
        session = session_store.require_active_session()
        df = session.t1_df if timepoint == "t1" else session.t2_df
        groups = baseline_adapter.dendrite_groups(df)
        return [models.DendriteGroup(dendrite_id=d_id, spine_ids=ids) for d_id, ids in groups]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dendrites/ids", response_model=models.DendriteIdsResponse)
def get_dendrite_ids() -> models.DendriteIdsResponse:
    try:
        session = session_store.require_active_session()
        t1_ids = sorted([d for d, _ in baseline_adapter.dendrite_groups(session.t1_df)])
        t2_ids = sorted([d for d, _ in baseline_adapter.dendrite_groups(session.t2_df)])
        return models.DendriteIdsResponse(t1_dendrite_ids=t1_ids, t2_dendrite_ids=t2_ids)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/dendrites/link", response_model=models.DendriteLink)
def add_dendrite_link(payload: models.DendriteLinkRequest) -> models.DendriteLink:
    try:
        session = session_store.require_active_session()
        known = get_dendrite_ids()
        t1_set = set(known.t1_dendrite_ids)
        t2_set = set(known.t2_dendrite_ids)
        t1_ids = sorted({str(v) for v in payload.t1_dendrite_ids})
        t2_ids = sorted({str(v) for v in payload.t2_dendrite_ids})
        if not t1_ids or not t2_ids:
            raise ValueError("Select at least one T1 dendrite and one T2 dendrite.")
        unknown_t1 = [v for v in t1_ids if v not in t1_set]
        unknown_t2 = [v for v in t2_ids if v not in t2_set]
        if unknown_t1 or unknown_t2:
            raise ValueError(f"Unknown dendrite ids. t1={unknown_t1}, t2={unknown_t2}")

        link_id = f"link_{len(session.dendrite_links) + 1}"
        row = {
            "link_id": link_id,
            "t1_dendrite_ids": t1_ids,
            "t2_dendrite_ids": t2_ids,
            "notes": payload.notes or "",
        }
        session.dendrite_links.append(row)
        _refresh_algo_matches(session, use_anchors=True)
        _save_last_session()
        return models.DendriteLink(**row)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dendrites/links", response_model=models.DendriteLinksResponse)
def get_dendrite_links() -> models.DendriteLinksResponse:
    try:
        session = session_store.require_active_session()
        links = [models.DendriteLink(**row) for row in session.dendrite_links]
        return models.DendriteLinksResponse(count=len(links), links=links)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/dendrites/links", response_model=models.DendriteLinksResponse)
def clear_dendrite_links() -> models.DendriteLinksResponse:
    try:
        session = session_store.require_active_session()
        session.dendrite_links = []
        session.algo_matches = {}
        _save_last_session()
        return models.DendriteLinksResponse(count=0, links=[])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/dendrites/preview", response_model=models.DendritePreviewResponse)
def dendrite_preview(payload: models.DendritePreviewRequest) -> models.DendritePreviewResponse:
    try:
        session = session_store.require_active_session()
        if payload.timepoint == "t1":
            df = session.t1_df
            stack = session.t1_stack
        else:
            df = session.t2_df
            stack = session.t2_stack
        if "dendrite_id" not in df.columns:
            raise ValueError("dendrite_id column is missing in loaded CSV.")

        previews: List[models.DendritePreviewItem] = []
        for dendrite_id in payload.dendrite_ids:
            sub = df[df["dendrite_id"].astype(str) == str(dendrite_id)]
            if sub.empty:
                continue
            xc = float(sub["x"].mean())
            yc = float(sub["y"].mean())
            zc = float(sub["z"].mean())
            crop, meta = crop_service.centered_crop(
                stack=stack,
                x=xc,
                y=yc,
                z=zc,
                width=payload.width,
                height=payload.height,
                depth=payload.depth,
            )
            if crop.size == 0:
                continue
            if payload.projection == "mid":
                plane = crop[crop.shape[0] // 2]
            else:
                plane = np.max(crop, axis=0)
            previews.append(
                models.DendritePreviewItem(
                    dendrite_id=str(dendrite_id),
                    spine_count=int(len(sub)),
                    center_xyz={"x": xc, "y": yc, "z": zc},
                    shape=[int(s) for s in crop.shape],
                    projection=payload.projection,
                    image_2d=plane.astype(float).tolist(),
                    intensity_min=float(np.min(plane)),
                    intensity_max=float(np.max(plane)),
                    meta=models.CropMeta(**meta),
                )
            )
        return models.DendritePreviewResponse(
            timepoint=payload.timepoint,
            selected_count=len(previews),
            previews=previews,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/fov/preview", response_model=models.FovPreviewResponse)
def fov_preview(payload: models.FovPreviewRequest) -> models.FovPreviewResponse:
    try:
        session = session_store.require_active_session()
        if payload.timepoint == "t1":
            df = session.t1_df
            stack = session.t1_stack
        else:
            df = session.t2_df
            stack = session.t2_stack

        if payload.projection == "mid":
            plane = stack[stack.shape[0] // 2]
        else:
            plane = np.max(stack, axis=0)

        selected = {str(v) for v in payload.dendrite_ids}
        points: List[models.FovPreviewPoint] = []
        excluded = (session.matched_t1_ids if payload.timepoint == "t1" else session.matched_t2_ids).copy()
        if payload.timepoint == "t1":
            excluded.update(session.removed_t1_ids)
            excluded.update(session.lost_t1_ids)
            excluded.update(session.ignored_t1_ids)
        else:
            excluded.update(session.removed_t2_ids)
            excluded.update(session.new_t2_ids)
            excluded.update(session.ignored_t2_ids)
        if "dendrite_id" in df.columns and selected:
            sub = df[df["dendrite_id"].astype(str).isin(selected)]
            sub = sub[~sub["id"].astype(str).isin(excluded)]
            for _, row in sub.iterrows():
                points.append(
                    models.FovPreviewPoint(
                        spine_id=str(row["id"]),
                        dendrite_id=None if pd.isna(row["dendrite_id"]) else str(row["dendrite_id"]),
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=float(row["z"]),
                    )
                )

        return models.FovPreviewResponse(
            timepoint=payload.timepoint,
            shape_zyx=[int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2])],
            projection=payload.projection,
            image_2d=plane.astype(float).tolist(),
            intensity_min=float(np.min(plane)),
            intensity_max=float(np.max(plane)),
            highlighted_points=points,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/crops/local", response_model=models.LocalCropResponse)
def local_crops(payload: models.LocalCropRequest) -> models.LocalCropResponse:
    try:
        session = session_store.require_active_session()
        out: List[models.CropResult] = []
        for target in payload.targets:
            if target.timepoint == "t1":
                stack = session.t1_stack
                lookup = session.t1_lookup
            else:
                stack = session.t2_stack
                lookup = session.t2_lookup

            x = target.x
            y = target.y
            z = target.z
            spine_id = target.spine_id
            if spine_id:
                record = lookup.get(str(spine_id))
                if record is None:
                    raise ValueError(f"Unknown spine_id '{spine_id}' for {target.timepoint}")
                x = float(record["x"])
                y = float(record["y"])
                z = float(record["z"])
            if x is None or y is None or z is None:
                raise ValueError("Each crop target must include spine_id or x,y,z coordinates.")

            crop, meta = crop_service.centered_crop(
                stack=stack,
                x=float(x),
                y=float(y),
                z=float(z),
                width=payload.width,
                height=payload.height,
                depth=payload.depth,
            )
            out.append(
                models.CropResult(
                    timepoint=target.timepoint,
                    spine_id=spine_id,
                    shape=[int(s) for s in crop.shape],
                    crop=crop.astype(float).tolist(),
                    meta=models.CropMeta(**meta),
                )
            )
        return models.LocalCropResponse(count=len(out), crops=out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/crops/preview", response_model=models.CropPreviewResponse)
def crop_preview(payload: models.CropPreviewRequest) -> models.CropPreviewResponse:
    try:
        session = session_store.require_active_session()
        if payload.timepoint == "t1":
            stack = session.t1_stack
            lookup = session.t1_lookup
        else:
            stack = session.t2_stack
            lookup = session.t2_lookup

        x = payload.x
        y = payload.y
        z = payload.z
        if payload.spine_id:
            record = lookup.get(str(payload.spine_id))
            if record is None:
                raise ValueError(f"Unknown spine_id '{payload.spine_id}' for {payload.timepoint}")
            x = float(record["x"])
            y = float(record["y"])
            z = float(record["z"])
        if x is None or y is None or z is None:
            raise ValueError("Provide spine_id or x,y,z for preview.")

        if payload.projection == "slice" and payload.slice_z_mode == "stack_global":
            z_plane = int(payload.slice_index) if payload.slice_index is not None else int(round(float(z)))
            plane, meta = crop_service.xy_plane_at_stack_z(
                stack,
                z_plane=z_plane,
                x=float(x),
                y=float(y),
                width=int(payload.width),
                height=int(payload.height),
            )
            if plane.size == 0:
                raise ValueError("Preview plane is empty.")
            zp_used = int(meta["source_bounds"]["z0"])
            return models.CropPreviewResponse(
                timepoint=payload.timepoint,
                spine_id=payload.spine_id,
                shape=[1, int(plane.shape[0]), int(plane.shape[1])],
                projection=payload.projection,
                slice_index=zp_used,
                image_2d=plane.astype(float).tolist(),
                intensity_min=float(np.min(plane)),
                intensity_max=float(np.max(plane)),
                meta=models.CropMeta(**meta),
            )

        crop, meta = crop_service.centered_crop(
            stack=stack,
            x=float(x),
            y=float(y),
            z=float(z),
            width=payload.width,
            height=payload.height,
            depth=payload.depth,
        )
        if crop.size == 0:
            raise ValueError("Preview crop is empty.")

        slice_index: int | None = None
        if payload.projection == "mid":
            slice_index = int(crop.shape[0] // 2)
            plane = crop[slice_index]
        elif payload.projection == "slice":
            if payload.slice_index is None:
                slice_index = int(crop.shape[0] // 2)
            else:
                slice_index = int(np.clip(payload.slice_index, 0, crop.shape[0] - 1))
            plane = crop[slice_index]
        else:
            plane = np.max(crop, axis=0)

        return models.CropPreviewResponse(
            timepoint=payload.timepoint,
            spine_id=payload.spine_id,
            shape=[int(s) for s in crop.shape],
            projection=payload.projection,
            slice_index=slice_index,
            image_2d=plane.astype(float).tolist(),
            intensity_min=float(np.min(plane)),
            intensity_max=float(np.max(plane)),
            meta=models.CropMeta(**meta),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/suggestions", response_model=List[models.ReviewSuggestion])
def review_suggestions(limit: int = 50) -> List[models.ReviewSuggestion]:
    try:
        session = session_store.require_active_session()
        t1 = session.t1_df[["id", "x", "y", "z"]].copy()
        t2 = session.t2_df[["id", "x", "y", "z"]].copy()
        if t1.empty or t2.empty:
            return []
        t1_xyz = t1[["x", "y", "z"]].to_numpy(dtype=np.float32)
        out: List[models.ReviewSuggestion] = []
        zmax = _session_z_gap(session)
        for _, row in t2.iterrows():
            dx = t1_xyz[:, 0] - float(row["x"])
            dy = t1_xyz[:, 1] - float(row["y"])
            dz = t1_xyz[:, 2] - float(row["z"])
            dxy = np.hypot(dx, dy)
            adz = np.abs(dz)
            valid = adz <= zmax
            if not np.any(valid):
                continue
            masked_dxy = np.where(valid, dxy, np.inf)
            idx = int(np.argmin(masked_dxy))
            out.append(
                models.ReviewSuggestion(
                    t2_spine_id=str(row["id"]),
                    suggested_t1_spine_id=str(t1.iloc[idx]["id"]),
                    distance_xy=float(dxy[idx]),
                    distance_z=float(abs(dz[idx])),
                )
            )
        out.sort(key=lambda x: (x.distance_xy if x.distance_xy is not None else 1e9))
        return out[: max(1, min(limit, len(out)))]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/decision", response_model=models.ReviewDecision)
def review_decision(payload: models.ReviewDecisionRequest) -> models.ReviewDecision:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, f"decision:{payload.action}")
        t2_id = str(payload.t2_spine_id)
        if t2_id not in session.t2_lookup:
            raise ValueError(f"Unknown t2_spine_id '{t2_id}'")
        if payload.action == "match":
            if not payload.t1_spine_id:
                raise ValueError("t1_spine_id is required for action='match'")
            if str(payload.t1_spine_id) not in session.t1_lookup:
                raise ValueError(f"Unknown t1_spine_id '{payload.t1_spine_id}'")
        if payload.action == "manual_match":
            if not payload.t1_spine_id:
                raise ValueError("t1_spine_id is required for action='manual_match'")
        if payload.action == "remove_t1":
            if not payload.t1_spine_id:
                raise ValueError("t1_spine_id is required for action='remove_t1'")
            if str(payload.t1_spine_id) not in session.t1_lookup:
                raise ValueError(f"Unknown t1_spine_id '{payload.t1_spine_id}'")
        if payload.action == "lost":
            if not payload.t1_spine_id:
                raise ValueError("t1_spine_id is required for action='lost'")
            if str(payload.t1_spine_id) not in session.t1_lookup:
                raise ValueError(f"Unknown t1_spine_id '{payload.t1_spine_id}'")
        if payload.action == "ignore_t1":
            if not payload.t1_spine_id:
                raise ValueError("t1_spine_id is required for action='ignore_t1'")
            if str(payload.t1_spine_id) not in session.t1_lookup:
                raise ValueError(f"Unknown t1_spine_id '{payload.t1_spine_id}'")
        session.review_decisions[t2_id] = {
            "t2_spine_id": t2_id,
            "action": payload.action,
            "t1_spine_id": None if not payload.t1_spine_id else str(payload.t1_spine_id),
            "notes": payload.notes or "",
        }
        _rebuild_decision_state(session)
        _save_last_session()
        return models.ReviewDecision(**session.review_decisions[t2_id])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/decision/{t2_spine_id}/undo", response_model=models.ReviewUndoResponse)
def undo_review_decision(t2_spine_id: str) -> models.ReviewUndoResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "undo-specific-decision")
        key = str(t2_spine_id)
        if key not in session.review_decisions:
            return models.ReviewUndoResponse(
                ok=False,
                message=f"No saved decision found for T2 spine '{key}'.",
                removed_decision=None,
            )
        removed = session.review_decisions.pop(key)
        _rebuild_decision_state(session)
        _save_last_session()
        return models.ReviewUndoResponse(
            ok=True,
            message=f"Undid decision for T2 spine '{key}'.",
            removed_decision=models.ReviewDecision(**removed),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/mark-lost/{t1_spine_id}", response_model=models.AlgoMatchUpdateResponse)
def mark_t1_lost(t1_spine_id: str) -> models.AlgoMatchUpdateResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "mark-t1-lost")
        t1 = str(t1_spine_id)
        if t1 not in session.t1_lookup:
            raise ValueError(f"Unknown t1_spine_id '{t1}'")
        if t1 in session.lost_t1_ids:
            return models.AlgoMatchUpdateResponse(
                ok=False,
                message=f"T1 spine '{t1}' is already marked LOST (counted once).",
                t2_spine_id="",
                t1_spine_id=t1,
            )
        session.lost_t1_ids.add(t1)
        # Drop algo matches that use this t1 so it cannot be re-proposed.
        session.algo_matches = {k: v for k, v in session.algo_matches.items() if str(v.get("t1_spine_id", "")) != t1}
        _rebuild_decision_state(session)
        _save_last_session()
        return models.AlgoMatchUpdateResponse(
            ok=True,
            message=f"Marked T1 spine '{t1}' as LOST; excluded from rematching.",
            t2_spine_id="",
            t1_spine_id=t1,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/mark-new/{t2_spine_id}", response_model=models.AlgoMatchUpdateResponse)
def mark_t2_new(t2_spine_id: str) -> models.AlgoMatchUpdateResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "mark-t2-new")
        t2 = str(t2_spine_id)
        if t2 not in session.t2_lookup:
            raise ValueError(f"Unknown t2_spine_id '{t2}'")
        if t2 in session.new_t2_ids:
            return models.AlgoMatchUpdateResponse(
                ok=False,
                message=f"T2 spine '{t2}' is already marked NEW (counted once).",
                t2_spine_id=t2,
                t1_spine_id=None,
            )
        session.new_t2_ids.add(t2)
        session.algo_matches.pop(t2, None)
        _rebuild_decision_state(session)
        _save_last_session()
        return models.AlgoMatchUpdateResponse(
            ok=True,
            message=f"Marked T2 spine '{t2}' as NEW; excluded from rematching.",
            t2_spine_id=t2,
            t1_spine_id=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/unmark-lost/{t1_spine_id}", response_model=models.AlgoMatchUpdateResponse)
def unmark_t1_lost(t1_spine_id: str) -> models.AlgoMatchUpdateResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "unmark-t1-lost")
        t1 = str(t1_spine_id)
        if t1 not in session.t1_lookup:
            raise ValueError(f"Unknown t1_spine_id '{t1}'")
        existed = t1 in session.lost_t1_ids
        session.lost_t1_ids.discard(t1)
        _rebuild_decision_state(session)
        _save_last_session()
        return models.AlgoMatchUpdateResponse(
            ok=True,
            message=(
                f"Removed LOST mark from T1 spine '{t1}'. It can be matched again."
                if existed
                else f"T1 spine '{t1}' was not marked LOST."
            ),
            t2_spine_id="",
            t1_spine_id=t1,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/unmark-new/{t2_spine_id}", response_model=models.AlgoMatchUpdateResponse)
def unmark_t2_new(t2_spine_id: str) -> models.AlgoMatchUpdateResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "unmark-t2-new")
        t2 = str(t2_spine_id)
        if t2 not in session.t2_lookup:
            raise ValueError(f"Unknown t2_spine_id '{t2}'")
        existed = t2 in session.new_t2_ids
        session.new_t2_ids.discard(t2)
        _rebuild_decision_state(session)
        _save_last_session()
        return models.AlgoMatchUpdateResponse(
            ok=True,
            message=(
                f"Removed NEW mark from T2 spine '{t2}'. It can be matched again."
                if existed
                else f"T2 spine '{t2}' was not marked NEW."
            ),
            t2_spine_id=t2,
            t1_spine_id=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/manual-t2-click", response_model=models.ManualT2ClickAddResponse)
def review_add_manual_t2_click(payload: models.ManualT2ClickAddRequest) -> models.ManualT2ClickAddResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "manual-add-t2-click")
        next_idx = len(session.manual_t2_click_spines) + 1
        manual_id = f"manual_t2_click_{next_idx:04d}"
        item = {
            "manual_id": manual_id,
            "timepoint": "t2",
            "x": float(payload.x),
            "y": float(payload.y),
            "z": float(payload.z),
            "notes": str(payload.notes or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        session.manual_t2_click_spines.append(item)
        session.t2_lookup[manual_id] = {
            "spine_id": manual_id,
            "dendrite_id": "manual_added",
            "x": float(item["x"]),
            "y": float(item["y"]),
            "z": float(item["z"]),
            "features": {},
        }
        _save_last_session()
        return models.ManualT2ClickAddResponse(
            ok=True,
            message=f"Added manual T2 click spine '{manual_id}' at x={item['x']:.1f}, y={item['y']:.1f}, z={item['z']:.1f}.",
            item=models.ManualT2ClickItem(**item),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/manual-t2-clicks", response_model=models.ManualT2ClickListResponse)
def review_list_manual_t2_clicks() -> models.ManualT2ClickListResponse:
    try:
        session = session_store.require_active_session()
        items = [models.ManualT2ClickItem(**row) for row in session.manual_t2_click_spines]
        return models.ManualT2ClickListResponse(count=len(items), items=items)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/manual-t1-click", response_model=models.ManualT1ClickAddResponse)
def review_add_manual_t1_click(payload: models.ManualT1ClickAddRequest) -> models.ManualT1ClickAddResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "manual-add-t1-click")
        next_idx = len(session.manual_t1_click_spines) + 1
        manual_id = f"manual_t1_click_{next_idx:04d}"
        item = {
            "manual_id": manual_id,
            "timepoint": "t1",
            "x": float(payload.x),
            "y": float(payload.y),
            "z": float(payload.z),
            "notes": str(payload.notes or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        session.manual_t1_click_spines.append(item)
        session.t1_lookup[manual_id] = {
            "spine_id": manual_id,
            "dendrite_id": "manual_added",
            "x": float(item["x"]),
            "y": float(item["y"]),
            "z": float(item["z"]),
            "features": {},
        }
        _save_last_session()
        return models.ManualT1ClickAddResponse(
            ok=True,
            message=f"Added manual T1 click spine '{manual_id}' at x={item['x']:.1f}, y={item['y']:.1f}, z={item['z']:.1f}.",
            item=models.ManualT1ClickItem(**item),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/manual-t1-clicks", response_model=models.ManualT1ClickListResponse)
def review_list_manual_t1_clicks() -> models.ManualT1ClickListResponse:
    try:
        session = session_store.require_active_session()
        items = [models.ManualT1ClickItem(**row) for row in session.manual_t1_click_spines]
        return models.ManualT1ClickListResponse(count=len(items), items=items)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/manual-click-match", response_model=models.ManualClickMatchResponse)
def review_manual_click_match(payload: models.ManualClickMatchRequest) -> models.ManualClickMatchResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "manual-click-match")
        t2_existing = str(payload.t2_spine_id or "").strip()
        t2_manual = str(payload.manual_t2_id or "").strip()
        t1_existing = str(payload.t1_spine_id or "").strip()
        t1_manual = str(payload.manual_t1_id or "").strip()
        if bool(t2_existing) == bool(t2_manual):
            raise ValueError("Provide exactly one of t2_spine_id or manual_t2_id.")
        if bool(t1_existing) == bool(t1_manual):
            raise ValueError("Provide exactly one of t1_spine_id or manual_t1_id.")

        if t2_existing:
            row_t2 = session.t2_lookup.get(t2_existing)
            if row_t2 is None:
                raise ValueError(f"Unknown t2_spine_id '{t2_existing}'")
            t2_kind = "existing"
            t2_id = t2_existing
            t2_xyz = (float(row_t2.get("x", np.nan)), float(row_t2.get("y", np.nan)), float(row_t2.get("z", np.nan)))
        else:
            row_t2 = next((r for r in session.manual_t2_click_spines if str(r.get("manual_id", "")) == t2_manual), None)
            if row_t2 is None:
                raise ValueError(f"Unknown manual_t2_id '{t2_manual}'")
            t2_kind = "manual"
            t2_id = t2_manual
            t2_xyz = (float(row_t2.get("x", np.nan)), float(row_t2.get("y", np.nan)), float(row_t2.get("z", np.nan)))

        if t1_existing:
            row_t1 = session.t1_lookup.get(t1_existing)
            if row_t1 is None:
                raise ValueError(f"Unknown t1_spine_id '{t1_existing}'")
            t1_kind = "existing"
            t1_id = t1_existing
            t1_xyz = (float(row_t1.get("x", np.nan)), float(row_t1.get("y", np.nan)), float(row_t1.get("z", np.nan)))
        else:
            row_t1 = next((r for r in session.manual_t1_click_spines if str(r.get("manual_id", "")) == t1_manual), None)
            if row_t1 is None:
                raise ValueError(f"Unknown manual_t1_id '{t1_manual}'")
            t1_kind = "manual"
            t1_id = t1_manual
            t1_xyz = (float(row_t1.get("x", np.nan)), float(row_t1.get("y", np.nan)), float(row_t1.get("z", np.nan)))

        match_id = f"manual_match_{len(session.manual_click_matches) + 1:04d}"
        item = {
            "match_id": match_id,
            "t2_kind": t2_kind,
            "t2_id": t2_id,
            "t2_x": t2_xyz[0],
            "t2_y": t2_xyz[1],
            "t2_z": t2_xyz[2],
            "t1_kind": t1_kind,
            "t1_id": t1_id,
            "t1_x": t1_xyz[0],
            "t1_y": t1_xyz[1],
            "t1_z": t1_xyz[2],
            "notes": str(payload.notes or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        session.manual_click_matches.append(item)
        _save_last_session()
        return models.ManualClickMatchResponse(
            ok=True,
            message=f"Saved manual match '{match_id}' ({t2_kind}:{t2_id} -> {t1_kind}:{t1_id}).",
            item=models.ManualClickMatchItem(**item),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/new-validation/match/{t2_spine_id}/{t1_spine_id}", response_model=models.ReviewDecision)
def review_new_validation_match_existing(t2_spine_id: str, t1_spine_id: str) -> models.ReviewDecision:
    session = session_store.require_active_session()
    _ensure_t2_is_new_validation_candidate(session, str(t2_spine_id))
    payload = models.ReviewDecisionRequest(
        t2_spine_id=str(t2_spine_id),
        action="match",
        t1_spine_id=str(t1_spine_id),
        notes="new-validation direct match",
    )
    return review_decision(payload)


@app.post("/review/undo-last-choice", response_model=models.UndoLastChoiceResponse)
def review_undo_last_choice() -> models.UndoLastChoiceResponse:
    try:
        session = session_store.require_active_session()
        if not session.action_history:
            return models.UndoLastChoiceResponse(ok=False, message="No previous choice to undo.", action_label=None)
        snap = session.action_history.pop()
        label = str(snap.get("action_label", "last-choice"))
        _restore_undo_snapshot(session, snap)
        _save_last_session()
        return models.UndoLastChoiceResponse(
            ok=True,
            message=f"Undid last choice: {label}",
            action_label=label,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/decisions", response_model=models.ReviewDecisionsResponse)
def review_decisions() -> models.ReviewDecisionsResponse:
    try:
        session = session_store.require_active_session()
        decisions = [
            models.ReviewDecision(**row)
            for row in sorted(session.review_decisions.values(), key=lambda x: str(x["t2_spine_id"]))
        ]
        return models.ReviewDecisionsResponse(count=len(decisions), decisions=decisions)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/finalize", response_model=models.MatchFinalizeResponse)
def review_finalize() -> models.MatchFinalizeResponse:
    try:
        session = session_store.require_active_session()
        # Rebuild global state first so finalize uses all accepted actions/matches.
        _rebuild_decision_state(session)
        excluded_t1_nonmatch, excluded_t2_nonmatch, _excluded_rows = _collect_non_matched_dendrite_spines(session)
        not_in_t1_t2 = _collect_not_in_t1_t2_ids(session) - excluded_t2_nonmatch
        matched_t2 = set(session.matched_t2_ids) - excluded_t2_nonmatch
        matched_t1 = set(session.matched_t1_ids) - excluded_t1_nonmatch
        no_match_t2 = set()
        lost_t1 = set(session.lost_t1_ids) - excluded_t1_nonmatch
        new_t2 = set(session.new_t2_ids) - excluded_t2_nonmatch
        removed_t1 = set(session.removed_t1_ids) - excluded_t1_nonmatch
        removed_t2 = set(session.removed_t2_ids) - excluded_t2_nonmatch
        ignored_t1 = set(session.ignored_t1_ids) - excluded_t1_nonmatch
        ignored_t2 = set(session.ignored_t2_ids) - excluded_t2_nonmatch
        for row in session.review_decisions.values():
            action = str(row.get("action", ""))
            t2 = str(row.get("t2_spine_id", ""))
            t1 = row.get("t1_spine_id")
            if action == "no_match" and t2 not in excluded_t2_nonmatch and t2 not in not_in_t1_t2:
                no_match_t2.add(t2)

        all_t2 = ({str(k) for k in session.t2_lookup.keys()} - excluded_t2_nonmatch) - not_in_t1_t2 - ignored_t2
        all_t1 = ({str(k) for k in session.t1_lookup.keys()} - excluded_t1_nonmatch) - ignored_t1
        inferred_new = _compute_inferred_new_t2_candidates(
            session,
            excluded_t2_nonmatch=excluded_t2_nonmatch.union(not_in_t1_t2),
            excluded_t1_nonmatch=excluded_t1_nonmatch,
        )
        inferred_lost = sorted(list(lost_t1.union(all_t1 - matched_t1 - removed_t1)))
        pending_new_validation = _build_new_validation_queue(session)
        _save_last_session()
        return models.MatchFinalizeResponse(
            matched_count=len(matched_t2),
            no_match_count=len(no_match_t2),
            inferred_new_count=len(inferred_new),
            pending_new_validation_count=len(pending_new_validation),
            inferred_lost_count=len(inferred_lost),
            removed_t1_count=len(removed_t1),
            removed_t2_count=len(removed_t2),
            ignored_t1_count=len(ignored_t1),
            ignored_t2_count=len(ignored_t2),
            inferred_new_t2_ids=inferred_new,
            inferred_lost_t1_ids=inferred_lost,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/counters", response_model=models.ReviewCountersResponse)
def review_counters() -> models.ReviewCountersResponse:
    try:
        session = session_store.require_active_session()
        _refresh_algo_matches(session, use_anchors=True, nearby_xy=140.0)
        manual_matched = 0
        for row in session.review_decisions.values():
            if str(row.get("action", "")) in {"match", "manual_match"}:
                manual_matched += 1

        queue = _build_review_queue(offset=0, limit=100000, use_local_registration=False, nearby_xy=140.0)
        to_review = int(queue.total_candidates)
        excluded_t1_nonmatch, excluded_t2_nonmatch, _rows = _collect_non_matched_dendrite_spines(session)
        matched_t2 = set(str(k) for k in session.algo_matches.keys())
        matched_t1 = set(str(v.get("t1_spine_id", "")) for v in session.algo_matches.values() if v.get("t1_spine_id"))
        for row in session.review_decisions.values():
            if str(row.get("action", "")) in {"match", "manual_match"} and row.get("t1_spine_id"):
                matched_t2.add(str(row.get("t2_spine_id", "")))
                matched_t1.add(str(row.get("t1_spine_id", "")))
        not_in_t1_t2 = _collect_not_in_t1_t2_ids(session) - excluded_t2_nonmatch
        removed_t1 = set(str(x) for x in session.removed_t1_ids) - excluded_t1_nonmatch
        removed_t2 = (set(str(x) for x in session.removed_t2_ids) - excluded_t2_nonmatch) - not_in_t1_t2
        ignored_t1 = set(str(x) for x in session.ignored_t1_ids) - excluded_t1_nonmatch
        ignored_t2 = (set(str(x) for x in session.ignored_t2_ids) - excluded_t2_nonmatch).union(not_in_t1_t2)
        new_t2 = set(str(x) for x in session.new_t2_ids) - excluded_t2_nonmatch
        lost_t1 = set(str(x) for x in session.lost_t1_ids) - excluded_t1_nonmatch
        unclassified_t1, unclassified_t2 = _strict_export_unclassified_counts(
            session,
            excluded_t1_nonmatch=excluded_t1_nonmatch,
            excluded_t2_nonmatch=excluded_t2_nonmatch,
            matched_t1=matched_t1,
            matched_t2=matched_t2,
            new_t2=new_t2,
            lost_t1=lost_t1,
            removed_t1=removed_t1,
            removed_t2=removed_t2,
            ignored_t1=ignored_t1,
            ignored_t2=ignored_t2,
        )
        to_review = max(to_review, int(len(unclassified_t1) + len(unclassified_t2)))
        algo_matched = int(len(session.algo_matches))
        t1_lost = int(len(session.lost_t1_ids))
        t2_new = int(len(session.new_t2_ids))
        z1 = max(int(session.t1_stack.shape[0]) - 1, 0)
        z2 = max(int(session.t2_stack.shape[0]) - 1, 0)
        return models.ReviewCountersResponse(
            manual_matched=manual_matched,
            algo_matched=algo_matched,
            to_review=to_review,
            t1_lost=t1_lost,
            t2_new=t2_new,
            t1_slice_max=z1,
            t2_slice_max=z2,
            max_match_z_gap=_session_z_gap(session),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/session/match-settings", response_model=models.MatchSettings)
def get_match_settings() -> models.MatchSettings:
    try:
        session = session_store.require_active_session()
        return models.MatchSettings(max_match_z_gap=_session_z_gap(session))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/session/match-settings", response_model=models.MatchSettings)
def set_match_settings(payload: models.MatchSettings) -> models.MatchSettings:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "match-settings")
        v = float(payload.max_match_z_gap)
        if not np.isfinite(v) or v < 0:
            raise ValueError("max_match_z_gap must be a non-negative finite number.")
        if v > 100.0:
            raise ValueError("max_match_z_gap must be at most 100.")
        session.max_match_z_gap = v
        _refresh_algo_matches(session, use_anchors=True)
        _save_last_session()
        return models.MatchSettings(max_match_z_gap=_session_z_gap(session))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/top5-next", response_model=models.Top5NextResponse)
def review_top5_next() -> models.Top5NextResponse:
    try:
        return _build_top5_next()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/new-validation-next", response_model=models.NewValidationNextResponse)
def review_new_validation_next() -> models.NewValidationNextResponse:
    try:
        session = session_store.require_active_session()
        pending = _build_new_validation_queue(session)
        z1_max = max(int(session.t1_stack.shape[0]) - 1, 0)
        z2_max = max(int(session.t2_stack.shape[0]) - 1, 0)
        if not pending:
            return models.NewValidationNextResponse(
                has_item=False, pending_count=0, item=None, t1_slice_max=z1_max, t2_slice_max=z2_max
            )
        t2_id = str(pending[0])
        row = session.t2_lookup.get(t2_id)
        if row is None:
            return models.NewValidationNextResponse(
                has_item=False,
                pending_count=max(0, len(pending) - 1),
                item=None,
                t1_slice_max=z1_max,
                t2_slice_max=z2_max,
            )
        return models.NewValidationNextResponse(
            has_item=True,
            pending_count=len(pending),
            item=models.NewValidationItem(
                t2_spine_id=t2_id,
                t2_dendrite_id=str(row.get("dendrite_id", "")) if row.get("dendrite_id") is not None else None,
                x=float(row.get("x", np.nan)),
                y=float(row.get("y", np.nan)),
                z=float(row.get("z", np.nan)),
            ),
            t1_slice_max=z1_max,
            t2_slice_max=z2_max,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/new-validation/{t2_spine_id}/{decision}", response_model=models.AlgoMatchUpdateResponse)
def review_new_validation_decision(t2_spine_id: str, decision: str) -> models.AlgoMatchUpdateResponse:
    try:
        session = session_store.require_active_session()
        t2 = str(t2_spine_id)
        if t2 not in session.t2_lookup:
            raise ValueError(f"Unknown t2_spine_id '{t2}'")
        _ensure_t2_is_new_validation_candidate(session, t2)
        decision_key = str(decision).strip().lower()
        if decision_key not in {"new", "artifact", "not_in_t1"}:
            raise ValueError("Decision must be one of: new, artifact, not_in_t1")
        _capture_undo_snapshot(session, f"new-validation:{decision_key}")
        if decision_key == "new":
            action = "new"
            notes = "validated new_t2 at final pass"
        elif decision_key == "artifact":
            action = "remove_t2"
            notes = "validated artifact at final pass"
        else:
            action = "not_in_t1"
            notes = "validated not-in-focus-in-t1 at final pass"
        session.review_decisions[t2] = {
            "t2_spine_id": t2,
            "action": action,
            "t1_spine_id": None,
            "notes": notes,
        }
        session.algo_matches.pop(t2, None)
        _rebuild_decision_state(session)
        _save_last_session()
        label = "T2 NEW" if action == "new" else ("artifact" if action == "remove_t2" else "not-in-T1")
        return models.AlgoMatchUpdateResponse(
            ok=True,
            message=f"Validated T2 spine '{t2}' as {label}.",
            t2_spine_id=t2,
            t1_spine_id=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/review/algo-queue", response_model=models.ReviewQueueResponse)
def review_algo_queue(offset: int = 0, limit: int = 2) -> models.ReviewQueueResponse:
    try:
        return _build_algo_review_queue(offset=offset, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/algo-unmatch/{t2_spine_id}", response_model=models.AlgoMatchUpdateResponse)
def review_algo_unmatch(t2_spine_id: str) -> models.AlgoMatchUpdateResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "algo-unmatch")
        t2 = str(t2_spine_id)
        row = session.algo_matches.pop(t2, None)
        if row is None:
            return models.AlgoMatchUpdateResponse(ok=False, message=f"No algo match for T2 '{t2}'.", t2_spine_id=t2)
        _rebuild_decision_state(session)
        _save_last_session()
        return models.AlgoMatchUpdateResponse(
            ok=True,
            message=f"Un-matched algo pair for T2 '{t2}'.",
            t2_spine_id=t2,
            t1_spine_id=str(row.get("t1_spine_id", "")) or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/algo-rematch", response_model=models.AlgoMatchUpdateResponse)
def review_algo_rematch(payload: models.AlgoMatchRemapRequest) -> models.AlgoMatchUpdateResponse:
    try:
        session = session_store.require_active_session()
        _capture_undo_snapshot(session, "algo-rematch")
        t2 = str(payload.t2_spine_id)
        t1 = str(payload.t1_spine_id)
        if t2 not in session.t2_lookup:
            raise ValueError(f"Unknown t2_spine_id '{t2}'")
        if t1 not in session.t1_lookup:
            raise ValueError(f"Unknown t1_spine_id '{t1}'")
        for other_t2, m in session.algo_matches.items():
            if str(other_t2) == t2:
                continue
            if str(m.get("t1_spine_id", "")) == t1:
                raise ValueError(f"T1 spine '{t1}' is already used by algo match with T2 '{other_t2}'")
        for row in session.review_decisions.values():
            if str(row.get("action", "")) != "match":
                continue
            if str(row.get("t1_spine_id", "")) == t1 and str(row.get("t2_spine_id", "")) != t2:
                raise ValueError(f"T1 spine '{t1}' is already manually matched.")
        prev = session.algo_matches.get(t2, {})
        session.algo_matches[t2] = {
            "t1_spine_id": t1,
            "final_score": float(prev.get("final_score", 0.0)),
        }
        _rebuild_decision_state(session)
        _save_last_session()
        return models.AlgoMatchUpdateResponse(
            ok=True,
            message=f"Re-matched T2 '{t2}' to T1 '{t1}'.",
            t2_spine_id=t2,
            t1_spine_id=t1,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review/reoptimize-local", response_model=models.ReviewQueueResponse)
def review_reoptimize_local(offset: int = 0, limit: int = 0, nearby_xy: float = 140.0) -> models.ReviewQueueResponse:
    """
    Local re-optimization:
    - restrict candidates by dendrite links
    - use accepted manual matches as nearby anchors
    - estimate local shift (t1 - t2) from anchors
    - apply shift to current t2 spine before nearest-neighbor lookup
    """
    try:
        session = session_store.require_active_session()
        _refresh_algo_matches(session, use_anchors=True, nearby_xy=nearby_xy)
        _save_last_session()
        return _build_review_queue(offset=offset, limit=limit, use_local_registration=True, nearby_xy=nearby_xy)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/results/export", response_model=models.ExportResultsResponse)
def export_results(payload: models.ExportResultsRequest | None = None) -> models.ExportResultsResponse:
    try:
        session = session_store.require_active_session()
        excluded_t1_nonmatch, excluded_t2_nonmatch, excluded_nonmatch_rows = _collect_non_matched_dendrite_spines(session)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        req = payload or models.ExportResultsRequest()
        if req.output_parent_dir:
            parent_dir = Path(req.output_parent_dir).expanduser().resolve()
        elif req.use_dialog:
            parent_dir = Path(io_service.pick_directory_via_dialog("Choose folder to save exported results")).resolve()
        else:
            parent_dir = WORKSPACE_ROOT / "results"
        parent_dir.mkdir(parents=True, exist_ok=True)
        custom_name = _safe_export_folder_name(req.output_name) if req.output_name else "spine_annotator_export"
        out_dir = parent_dir / f"{stamp}_{custom_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build matched table from manual + algo matches.
        manual_rows = []
        for row in session.review_decisions.values():
            if str(row.get("action", "")) in {"match", "manual_match"} and row.get("t1_spine_id"):
                t2_id = str(row["t2_spine_id"])
                t1_id = str(row["t1_spine_id"])
                if t2_id in excluded_t2_nonmatch or t1_id in excluded_t1_nonmatch:
                    continue
                manual_rows.append(
                    {
                        "t2_spine_id": t2_id,
                        "t1_spine_id": t1_id,
                        "source": "manual" if str(row.get("action", "")) == "match" else "manual_review",
                    }
                )
        algo_rows = [
            {"t2_spine_id": str(t2), "t1_spine_id": str(v.get("t1_spine_id", "")), "source": "algo"}
            for t2, v in session.algo_matches.items()
            if v.get("t1_spine_id")
            and str(t2) not in excluded_t2_nonmatch
            and str(v.get("t1_spine_id", "")) not in excluded_t1_nonmatch
        ]
        matched_df = pd.DataFrame(manual_rows + algo_rows, columns=["t2_spine_id", "t1_spine_id", "source"])
        if not matched_df.empty:
            matched_df = matched_df.drop_duplicates(subset=["t2_spine_id"], keep="first")
        matched_df.to_csv(out_dir / "matched.csv", index=False)

        matched_t2 = set(matched_df["t2_spine_id"].astype(str).tolist()) if not matched_df.empty else set()
        matched_t1 = set(matched_df["t1_spine_id"].astype(str).tolist()) if not matched_df.empty else set()

        # Strict classification buckets only.
        # Legacy "not_in_t1" decisions are folded into ignored_t2.
        not_in_t1_t2 = _collect_not_in_t1_t2_ids(session) - excluded_t2_nonmatch
        removed_t2 = (set(str(x) for x in session.removed_t2_ids) - excluded_t2_nonmatch) - not_in_t1_t2
        removed_t1 = set(str(x) for x in session.removed_t1_ids) - excluded_t1_nonmatch
        ignored_t2 = (set(str(x) for x in session.ignored_t2_ids) - excluded_t2_nonmatch).union(not_in_t1_t2)
        ignored_t1 = set(str(x) for x in session.ignored_t1_ids) - excluded_t1_nonmatch
        explicit_new_t2 = set(str(x) for x in session.new_t2_ids) - excluded_t2_nonmatch
        explicit_lost_t1 = set(str(x) for x in session.lost_t1_ids) - excluded_t1_nonmatch

        new_t2 = set(explicit_new_t2)
        lost_t1 = set(explicit_lost_t1)

        # Hard validation gate: block export if anything remains unclassified.
        unclassified_t1, unclassified_t2 = _strict_export_unclassified_counts(
            session,
            excluded_t1_nonmatch=excluded_t1_nonmatch,
            excluded_t2_nonmatch=excluded_t2_nonmatch,
            matched_t1=matched_t1,
            matched_t2=matched_t2,
            new_t2=new_t2,
            lost_t1=lost_t1,
            removed_t1=removed_t1,
            removed_t2=removed_t2,
            ignored_t1=ignored_t1,
            ignored_t2=ignored_t2,
        )
        unclassified_count = int(len(unclassified_t1) + len(unclassified_t2))
        if unclassified_count > 0:
            raise ValueError(
                f"Export Blocked: You have {unclassified_count} unclassified spines remaining. "
                "Every spine must be matched, marked as new/lost, or flagged as an artifact/ignored before saving."
            )

        pd.DataFrame({"t2_spine_id": sorted(list(new_t2))}).to_csv(out_dir / "new.csv", index=False)
        pd.DataFrame({"t1_spine_id": sorted(list(lost_t1))}).to_csv(out_dir / "lost.csv", index=False)
        pd.DataFrame({"t1_spine_id": sorted(list(removed_t1))}).to_csv(out_dir / "removed_t1.csv", index=False)
        pd.DataFrame({"t2_spine_id": sorted(list(removed_t2))}).to_csv(out_dir / "removed_t2.csv", index=False)
        pd.DataFrame({"t1_spine_id": sorted(list(ignored_t1))}).to_csv(out_dir / "ignored_t1.csv", index=False)
        pd.DataFrame({"t2_spine_id": sorted(list(ignored_t2))}).to_csv(out_dir / "ignored_t2.csv", index=False)

        rejected_rows = []
        for k in sorted(session.rejected_pairs):
            if "|" in k:
                t1, t2 = k.split("|", 1)
                rejected_rows.append({"t1_spine_id": t1, "t2_spine_id": t2})
        pd.DataFrame(rejected_rows, columns=["t1_spine_id", "t2_spine_id"]).to_csv(
            out_dir / "rejected_pairs.csv", index=False
        )
        pd.DataFrame(
            excluded_nonmatch_rows,
            columns=["timepoint", "spine_id", "dendrite_id", "reason"],
        ).to_csv(out_dir / "excluded_non_matched_dendrite_spines.csv", index=False)
        pd.DataFrame(
            session.manual_t2_click_spines,
            columns=["manual_id", "timepoint", "x", "y", "z", "notes", "created_at"],
        ).to_csv(out_dir / "manual_t2_click_spines.csv", index=False)
        pd.DataFrame(
            session.manual_t1_click_spines,
            columns=["manual_id", "timepoint", "x", "y", "z", "notes", "created_at"],
        ).to_csv(out_dir / "manual_t1_click_spines.csv", index=False)
        pd.DataFrame(
            session.manual_click_matches,
            columns=[
                "match_id",
                "t2_kind",
                "t2_id",
                "t2_x",
                "t2_y",
                "t2_z",
                "t1_kind",
                "t1_id",
                "t1_x",
                "t1_y",
                "t1_z",
                "notes",
                "created_at",
            ],
        ).to_csv(out_dir / "manual_click_matches.csv", index=False)

        metadata = {
            "t1_tiff_path": str(session.t1_tiff_path),
            "t2_tiff_path": str(session.t2_tiff_path),
            "t1_csv_path": str(session.t1_csv_path),
            "t2_csv_path": str(session.t2_csv_path),
            "manual_match_count": int(len(manual_rows)),
            "algo_match_count": int(len(algo_rows)),
            "new_count": int(len(new_t2)),
            "lost_count": int(len(lost_t1)),
            "excluded_non_matched_dendrite_t1_count": int(len(excluded_t1_nonmatch)),
            "excluded_non_matched_dendrite_t2_count": int(len(excluded_t2_nonmatch)),
            "not_in_t1_focus_count": 0,
            "new_validation_pending_count": 0,
            "manual_t2_click_spines_count": int(len(session.manual_t2_click_spines)),
            "manual_t1_click_spines_count": int(len(session.manual_t1_click_spines)),
            "manual_click_matches_count": int(len(session.manual_click_matches)),
            "ignored_t1_count": int(len(ignored_t1)),
            "ignored_t2_count": int(len(ignored_t2)),
            "strict_export_gate_unclassified_t1_count": int(len(unclassified_t1)),
            "strict_export_gate_unclassified_t2_count": int(len(unclassified_t2)),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        log_name = "matching_activity_log.txt"
        log_dst = out_dir / log_name
        if CLIENT_ACTIVITY_LOG.exists() and CLIENT_ACTIVITY_LOG.stat().st_size > 0:
            shutil.copyfile(CLIENT_ACTIVITY_LOG, log_dst)
            CLIENT_ACTIVITY_LOG.write_text("", encoding="utf-8")
        else:
            log_dst.write_text(
                "(No UI activity log entries were recorded since the last export.)\n", encoding="utf-8"
            )

        files = [
            "matched.csv",
            "new.csv",
            "lost.csv",
            "removed_t1.csv",
            "removed_t2.csv",
            "ignored_t1.csv",
            "ignored_t2.csv",
            "rejected_pairs.csv",
            "excluded_non_matched_dendrite_spines.csv",
            "manual_t2_click_spines.csv",
            "manual_t1_click_spines.csv",
            "manual_click_matches.csv",
            "metadata.json",
            log_name,
        ]
        return models.ExportResultsResponse(ok=True, output_dir=str(out_dir), files=files)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

