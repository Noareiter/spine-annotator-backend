"""TIFF stack helpers for the manual registration server.

Resolves frame file paths from Bruker PVScan XML metadata and reads
individual z-slices for browser display (PNG).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from io_pvscan_xml import PVScanMeta, parse_pvscan

try:
    import tifffile
except ImportError:
    tifffile = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def resolve_frame_path(meta: PVScanMeta, frame_index: int) -> Optional[Path]:
    """Return the on-disk path for a frame (1-based ``frame_index``)."""
    xml_dir = meta.source_path.parent
    for fr in meta.frames:
        if fr.index != frame_index:
            continue
        for fn in fr.files:
            if not fn:
                continue
            for candidate in (
                xml_dir / fn,
                xml_dir / "Chan" / fn,
                meta.source_path.parent.parent / fn,
            ):
                if candidate.is_file():
                    return candidate
    return None


def list_frame_paths(meta: PVScanMeta) -> List[Tuple[int, Optional[Path], float]]:
    """``(frame_index, path, z_um)`` for every frame."""
    out: List[Tuple[int, Optional[Path], float]] = []
    for fr in meta.frames:
        path = resolve_frame_path(meta, fr.index)
        out.append((fr.index, path, fr.z_um))
    return out


def read_slice_2d(path: Path) -> np.ndarray:
    """Read a 2D image array (H, W) from a TIFF file."""
    if tifffile is not None:
        arr = tifffile.imread(str(path))
    elif Image is not None:
        arr = np.asarray(Image.open(path))
    else:
        raise ImportError("install tifffile or Pillow to read TIFF stacks")
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr)


def slice_to_png_bytes(
    path: Path,
    percentile_low: float = 1.0,
    percentile_high: float = 99.5,
) -> bytes:
    """Normalize a microscopy slice to 8-bit PNG bytes."""
    arr = read_slice_2d(path)
    lo, hi = np.percentile(arr, (percentile_low, percentile_high))
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    if Image is None:
        raise ImportError("Pillow required for PNG encoding (pip install Pillow)")
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def load_meta(xml_path: str | Path) -> PVScanMeta:
    return parse_pvscan(xml_path)


def stack_info(path: Path) -> Tuple[int, int, int]:
    """Return ``(n_pages, height, width)`` for a multi-page TIFF."""
    if tifffile is None:
        raise ImportError("tifffile required for multi-page TIFF (pip install tifffile)")
    with tifffile.TiffFile(str(path)) as tf:
        n = len(tf.pages)
        shape = tf.pages[0].shape
        h, w = (shape[0], shape[1]) if len(shape) >= 2 else (shape[-2], shape[-1])
        return n, int(h), int(w)


def read_stack_page(path: Path, page: int) -> np.ndarray:
    """Read one page (0-based) from a multi-page TIFF."""
    if tifffile is None:
        raise ImportError("tifffile required for multi-page TIFF")
    with tifffile.TiffFile(str(path)) as tf:
        if page < 0 or page >= len(tf.pages):
            raise IndexError(f"page {page} out of range (0..{len(tf.pages) - 1})")
        arr = tf.pages[page].asarray()
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr)


def build_mip(path: Path) -> np.ndarray:
    """Max-intensity projection across all pages."""
    if tifffile is None:
        raise ImportError("tifffile required for MIP")
    mip: Optional[np.ndarray] = None
    with tifffile.TiffFile(str(path)) as tf:
        for page in tf.pages:
            sl = page.asarray()
            if sl.ndim == 3:
                sl = sl[0]
            mip = sl if mip is None else np.maximum(mip, sl)
    if mip is None:
        raise ValueError(f"empty TIFF stack: {path}")
    return np.asarray(mip)


def array_to_png_bytes(
    arr: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.5,
) -> bytes:
    """Normalize a 2D array to 8-bit PNG bytes."""
    lo, hi = np.percentile(arr, (percentile_low, percentile_high))
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    if Image is None:
        raise ImportError("Pillow required for PNG encoding (pip install Pillow)")
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def stack_page_to_png_bytes(path: Path, page: int) -> bytes:
    return array_to_png_bytes(read_stack_page(path, page))


def mip_cache_path(tiff_path: Path) -> Path:
    """Sidecar cache next to the stack (rebuilt when TIFF is newer)."""
    return tiff_path.parent / f"{tiff_path.stem}_reg_mip.png"


def load_or_build_mip_png_bytes(tiff_path: Path, force: bool = False) -> bytes:
    """Return cached MIP PNG bytes, building and saving the cache if needed."""
    cache = mip_cache_path(tiff_path)
    src_mtime = tiff_path.stat().st_mtime
    if not force and cache.is_file() and cache.stat().st_mtime >= src_mtime:
        return cache.read_bytes()
    png = mip_to_png_bytes(tiff_path)
    cache.write_bytes(png)
    return png


def mip_to_png_bytes(path: Path) -> bytes:
    return array_to_png_bytes(build_mip(path))
