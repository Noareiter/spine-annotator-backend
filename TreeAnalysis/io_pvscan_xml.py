"""Parse Bruker PrairieView ``PVScan`` XML metadata.

This is step 1 of the TreeAnalysis pipeline. It extracts everything needed to
place a field-of-view (FOV) into microscope stage coordinates (micrometres):

- lateral calibration (micronsPerPixel) and frame size in pixels,
- objective / optical-zoom context,
- the FOV's stage XY center (``positionCurrent`` in the file-level state),
- per-frame absolute Z depth (Z Focus + Optotune ETL + piezo, summed).

The parser is read-only and intentionally has no NumPy/Pandas dependency so it
can be imported and unit-tested in isolation. Large files (one ``<Frame>`` per
z-plane and channel, tens of thousands of lines) are streamed with
``iterparse`` so memory stays flat.

CLI:
    python io_pvscan_xml.py "<path-to>.xml"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


@dataclass
class FrameMeta:
    """Per-frame state for a single z-plane of a z-series."""

    index: int
    z_um: float
    z_components: Dict[str, float] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)


@dataclass
class PVScanMeta:
    """Parsed FOV-level metadata from a ``PVScan`` XML file."""

    source_path: Path
    microns_per_pixel: Tuple[float, float, float]  # (x, y, z)
    pixels: Tuple[int, int]  # (x = pixelsPerLine, y = linesPerFrame)
    objective_mag: Optional[float]
    optical_zoom: Optional[float]
    rotation_deg: float
    stage_center_um: Tuple[float, float]  # file-level positionCurrent (X, Y)
    frames: List[FrameMeta]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def fov_size_um(self) -> Tuple[float, float]:
        """Physical width/height of the FOV in micrometres."""
        return (
            self.pixels[0] * self.microns_per_pixel[0],
            self.pixels[1] * self.microns_per_pixel[1],
        )

    @property
    def z_range_um(self) -> Tuple[float, float]:
        if not self.frames:
            return (0.0, 0.0)
        zs = [f.z_um for f in self.frames]
        return (min(zs), max(zs))

    @property
    def z_step_um(self) -> Optional[float]:
        """Median |z| spacing between consecutive frames (None if < 2 frames)."""
        if len(self.frames) < 2:
            return None
        diffs = sorted(
            abs(self.frames[i + 1].z_um - self.frames[i].z_um)
            for i in range(len(self.frames) - 1)
        )
        return diffs[len(diffs) // 2]


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_shard_values(shard: ET.Element) -> Dict[str, ET.Element]:
    """Map ``key`` -> ``PVStateValue`` element for one ``PVStateShard``."""
    out: Dict[str, ET.Element] = {}
    for pv in shard.findall("PVStateValue"):
        key = pv.get("key")
        if key:
            out[key] = pv
    return out


def _indexed_value(pv: ET.Element, index: str) -> Optional[float]:
    """Read ``<IndexedValue index=... value=.../>`` for a given axis index."""
    for iv in pv.findall("IndexedValue"):
        if iv.get("index") == index:
            return _to_float(iv.get("value"))
    return None


def _z_components(pv: ET.Element) -> Dict[str, float]:
    """Read the ZAxis ``SubindexedValue`` entries as {description: value}.

    Falls back to ``z<subindex>`` keys when a description is absent.
    """
    comps: Dict[str, float] = {}
    for sv in pv.findall("SubindexedValues"):
        if sv.get("index") != "ZAxis":
            continue
        for entry in sv.findall("SubindexedValue"):
            val = _to_float(entry.get("value"))
            if val is None:
                continue
            desc = entry.get("description") or f"z{entry.get('subindex')}"
            comps[desc] = val
    return comps


def _xy_position(pv: ET.Element) -> Tuple[Optional[float], Optional[float]]:
    x = y = None
    for sv in pv.findall("SubindexedValues"):
        axis = sv.get("index")
        entry = sv.find("SubindexedValue")
        if entry is None:
            continue
        val = _to_float(entry.get("value"))
        if axis == "XAxis":
            x = val
        elif axis == "YAxis":
            y = val
    return x, y


def parse_pvscan(xml_path: str | Path) -> PVScanMeta:
    """Parse a ``PVScan`` XML file into :class:`PVScanMeta`.

    Per-frame absolute Z is the **sum** of the ZAxis device offsets (Z Focus +
    Optotune ETL + piezo); these are additive physical offsets, so their sum is
    the true depth of the plane regardless of which device performed the step.
    """
    xml_path = Path(xml_path)

    microns_per_pixel: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    pixels: Tuple[int, int] = (0, 0)
    objective_mag: Optional[float] = None
    optical_zoom: Optional[float] = None
    rotation_deg: float = 0.0
    stage_center: Tuple[float, float] = (0.0, 0.0)
    frames: List[FrameMeta] = []

    header_done = False
    px_x: Optional[int] = None
    px_y: Optional[int] = None

    # Stream the file: the first PVStateShard is FOV-level; each <Frame> follows.
    for event, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag == "PVStateShard" and not header_done:
            vals = _read_shard_values(elem)
            if not vals:
                # The Sequence contains an empty <PVStateShard/>; skip it.
                elem.clear()
                continue

            mpp = vals.get("micronsPerPixel")
            if mpp is not None:
                microns_per_pixel = (
                    _indexed_value(mpp, "XAxis") or 1.0,
                    _indexed_value(mpp, "YAxis") or 1.0,
                    _indexed_value(mpp, "ZAxis") or 1.0,
                )
            if "pixelsPerLine" in vals:
                px_x = int(_to_float(vals["pixelsPerLine"].get("value")) or 0)
            if "linesPerFrame" in vals:
                px_y = int(_to_float(vals["linesPerFrame"].get("value")) or 0)
            if "objectiveLensMag" in vals:
                objective_mag = _to_float(vals["objectiveLensMag"].get("value"))
            if "opticalZoom" in vals:
                optical_zoom = _to_float(vals["opticalZoom"].get("value"))
            if "rotation" in vals:
                rotation_deg = _to_float(vals["rotation"].get("value")) or 0.0
            if "positionCurrent" in vals:
                x, y = _xy_position(vals["positionCurrent"])
                stage_center = (x or 0.0, y or 0.0)

            pixels = (px_x or 0, px_y or 0)
            header_done = True
            elem.clear()
            continue

        if elem.tag == "Frame":
            idx = int(elem.get("index") or len(frames) + 1)
            files = [f.get("filename") for f in elem.findall("File") if f.get("filename")]
            z_um = 0.0
            z_components: Dict[str, float] = {}
            shard = elem.find("PVStateShard")
            if shard is not None:
                vals = _read_shard_values(shard)
                pv = vals.get("positionCurrent")
                if pv is not None:
                    z_components = _z_components(pv)
                    z_um = sum(z_components.values())
            frames.append(FrameMeta(index=idx, z_um=z_um, z_components=z_components, files=files))
            elem.clear()

    if not header_done:
        raise ValueError(f"No FOV-level PVStateShard found in {xml_path}")

    return PVScanMeta(
        source_path=xml_path,
        microns_per_pixel=microns_per_pixel,
        pixels=pixels,
        objective_mag=objective_mag,
        optical_zoom=optical_zoom,
        rotation_deg=rotation_deg,
        stage_center_um=stage_center,
        frames=frames,
    )


def fov_pixel_to_stage_um(
    meta: PVScanMeta,
    px: float,
    py: float,
    frame_index: int,
    sign_x: int = 1,
    sign_y: int = 1,
) -> Tuple[float, float, float]:
    """Convert a pixel (px, py) on frame ``frame_index`` to stage micrometres.

    ``positionCurrent`` is the FOV center, so pixels are offset from the center
    pixel. ``sign_x`` / ``sign_y`` (+1 or -1) encode the per-microscope axis
    convention (default: +X to the right, +Y down-as-stored); flip if QC of an
    overlapping pair disagrees.
    """
    cx = meta.pixels[0] / 2.0
    cy = meta.pixels[1] / 2.0
    x_um = meta.stage_center_um[0] + sign_x * (px - cx) * meta.microns_per_pixel[0]
    y_um = meta.stage_center_um[1] + sign_y * (py - cy) * meta.microns_per_pixel[1]

    z_um = 0.0
    for fr in meta.frames:
        if fr.index == frame_index:
            z_um = fr.z_um
            break
    return (x_um, y_um, z_um)


def _summary(meta: PVScanMeta) -> str:
    w, h = meta.fov_size_um
    zlo, zhi = meta.z_range_um
    lines = [
        f"source           : {meta.source_path.name}",
        f"pixels           : {meta.pixels[0]} x {meta.pixels[1]}",
        f"microns/pixel    : x={meta.microns_per_pixel[0]:.5f} "
        f"y={meta.microns_per_pixel[1]:.5f} z={meta.microns_per_pixel[2]:.5f}",
        f"FOV size (um)    : {w:.1f} x {h:.1f}",
        f"objective / zoom : {meta.objective_mag}x / zoom {meta.optical_zoom}",
        f"rotation (deg)   : {meta.rotation_deg}",
        f"stage center (um): x={meta.stage_center_um[0]:.3f} y={meta.stage_center_um[1]:.3f}",
        f"frames           : {meta.n_frames}",
        f"z range (um)     : {zlo:.3f} .. {zhi:.3f}",
        f"z step (um)      : {meta.z_step_um}",
    ]
    if meta.frames:
        comps = meta.frames[0].z_components
        lines.append(f"z components     : {comps}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # console may default to cp1255
    except (AttributeError, ValueError):
        pass
    if len(argv) != 2:
        print(__doc__)
        return 2
    meta = parse_pvscan(argv[1])
    print(_summary(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
