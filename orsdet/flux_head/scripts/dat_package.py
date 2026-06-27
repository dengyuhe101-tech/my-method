#!/usr/bin/env python3
"""Utilities for V4m single-file .dat packages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tarfile


MAGIC = b"CIANNA_V4M_DAT_V1\n"
FOOTER_STRUCT = struct.Struct("<QQ32s32s")
FOOTER_LEN = FOOTER_STRUCT.size + len(MAGIC)
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class V4mDatInfo:
    path: Path
    detector_len: int
    payload_len: int
    detector_sha256: str
    payload_sha256: str
    file_size: int

    @property
    def footer_len(self) -> int:
        return FOOTER_LEN


def _copy_and_hash(src: Path, dst_handle) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with src.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            dst_handle.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def build_payload(model_dir: Path, detector_path: Path, detector_sha256: str, detector_len: int) -> bytes:
    model_dir = model_dir.resolve()
    files = sorted(path for path in model_dir.iterdir() if path.is_file())
    if not any(path.name == "model_config.json" for path in files):
        raise FileNotFoundError(model_dir / "model_config.json")
    if not any(path.name == "fit_model.json" for path in files):
        raise FileNotFoundError(model_dir / "fit_model.json")

    manifest = {
        "format": "CIANNA_V4M_DAT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "detector_source": str(detector_path.resolve()),
        "detector_len": detector_len,
        "detector_sha256": detector_sha256,
        "model_dir_source": str(model_dir),
        "payload_files": ["model_package/" + path.name for path in files],
        "notes": "Detector bytes are stored as the .dat prefix; V4m frozen post-head files are stored in this payload.",
    }

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in files:
            _add_bytes(tar, "model_package/" + path.name, path.read_bytes())
        _add_bytes(
            tar,
            "model_package/v4m_dat_manifest.json",
            (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
        )
    return buffer.getvalue()


def pack_v4m_dat(detector_path: Path, model_dir: Path, out_dat: Path, *, force: bool = False) -> V4mDatInfo:
    detector_path = detector_path.resolve()
    model_dir = model_dir.resolve()
    out_dat = out_dat.resolve()
    if not detector_path.is_file():
        raise FileNotFoundError(detector_path)
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    if out_dat.exists() and not force:
        raise FileExistsError(out_dat)
    if detector_path == out_dat:
        raise ValueError("Refusing to overwrite detector in place: %s" % detector_path)

    out_dat.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dat.with_suffix(out_dat.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with tmp_path.open("wb") as out_handle:
        detector_len, detector_sha = _copy_and_hash(detector_path, out_handle)
        payload = build_payload(model_dir, detector_path, detector_sha, detector_len)
        payload_sha = hashlib.sha256(payload).hexdigest()
        out_handle.write(payload)
        out_handle.write(
            FOOTER_STRUCT.pack(
                detector_len,
                len(payload),
                bytes.fromhex(detector_sha),
                bytes.fromhex(payload_sha),
            )
        )
        out_handle.write(MAGIC)

    tmp_path.replace(out_dat)
    return read_v4m_dat_info(out_dat, verify=False)


def read_v4m_dat_info(path: Path, *, verify: bool = False) -> V4mDatInfo:
    path = path.resolve()
    file_size = path.stat().st_size
    if file_size < FOOTER_LEN:
        raise ValueError("File is too small to be a V4m .dat package: %s" % path)
    with path.open("rb") as handle:
        handle.seek(file_size - FOOTER_LEN)
        footer = handle.read(FOOTER_LEN)
    if not footer.endswith(MAGIC):
        raise ValueError("Missing V4m .dat trailer magic: %s" % path)
    detector_len, payload_len, detector_sha, payload_sha = FOOTER_STRUCT.unpack(footer[: FOOTER_STRUCT.size])
    expected_size = detector_len + payload_len + FOOTER_LEN
    if expected_size != file_size:
        raise ValueError("Invalid V4m .dat size: expected %d, got %d" % (expected_size, file_size))

    info = V4mDatInfo(
        path=path,
        detector_len=detector_len,
        payload_len=payload_len,
        detector_sha256=detector_sha.hex(),
        payload_sha256=payload_sha.hex(),
        file_size=file_size,
    )
    if verify:
        verify_v4m_dat(info)
    return info


def verify_v4m_dat(info: V4mDatInfo) -> None:
    detector_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    with info.path.open("rb") as handle:
        remaining = info.detector_len
        while remaining:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise ValueError("Unexpected EOF while reading detector prefix")
            detector_digest.update(chunk)
            remaining -= len(chunk)
        remaining = info.payload_len
        while remaining:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise ValueError("Unexpected EOF while reading V4m payload")
            payload_digest.update(chunk)
            remaining -= len(chunk)
    if detector_digest.hexdigest() != info.detector_sha256:
        raise ValueError("Detector sha256 mismatch in %s" % info.path)
    if payload_digest.hexdigest() != info.payload_sha256:
        raise ValueError("Payload sha256 mismatch in %s" % info.path)


def read_payload_bytes(info: V4mDatInfo) -> bytes:
    with info.path.open("rb") as handle:
        handle.seek(info.detector_len)
        payload = handle.read(info.payload_len)
    if hashlib.sha256(payload).hexdigest() != info.payload_sha256:
        raise ValueError("Payload sha256 mismatch in %s" % info.path)
    return payload


def _safe_extract_payload(payload: bytes, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError("Unsafe path in V4m payload: %s" % member.name)
        tar.extractall(out_dir)


def extract_v4m_dat(
    path: Path,
    out_dir: Path,
    *,
    extract_detector: bool = True,
    force: bool = False,
) -> tuple[Path | None, Path]:
    info = read_v4m_dat_info(path, verify=False)
    out_dir = out_dir.resolve()
    model_dir = out_dir / "model_package"
    detector_out = out_dir / "detector.dat"

    if model_dir.exists():
        if not force:
            raise FileExistsError(model_dir)
        shutil.rmtree(model_dir)
    if extract_detector and detector_out.exists():
        if not force:
            raise FileExistsError(detector_out)
        detector_out.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    if extract_detector:
        digest = hashlib.sha256()
        with info.path.open("rb") as src, detector_out.open("wb") as dst:
            remaining = info.detector_len
            while remaining:
                chunk = src.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise ValueError("Unexpected EOF while extracting detector")
                dst.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != info.detector_sha256:
            raise ValueError("Detector sha256 mismatch while extracting %s" % info.path)
    else:
        detector_out = None

    _safe_extract_payload(read_payload_bytes(info), out_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError("V4m payload did not contain model_package/")
    return detector_out, model_dir


def is_v4m_dat(path: Path) -> bool:
    try:
        read_v4m_dat_info(path, verify=False)
    except Exception:
        return False
    return True
