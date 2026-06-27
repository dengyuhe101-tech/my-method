#!/usr/bin/env python3
"""Inspect or extract a V4m single-file .dat package."""

from __future__ import annotations

import argparse
from pathlib import Path

from dat_package import extract_v4m_dat, read_v4m_dat_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dat", type=Path)
    parser.add_argument("--verify", action="store_true", help="Hash detector and payload bytes.")
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--no-detector", action="store_true", help="When extracting, only extract the V4m payload.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    info = read_v4m_dat_info(args.dat, verify=bool(args.verify))
    print("path:", info.path)
    print("file_size:", info.file_size)
    print("detector_len:", info.detector_len)
    print("payload_len:", info.payload_len)
    print("footer_len:", info.footer_len)
    print("detector_sha256:", info.detector_sha256)
    print("payload_sha256:", info.payload_sha256)

    if args.extract_dir is not None:
        detector, model_dir = extract_v4m_dat(
            args.dat,
            args.extract_dir,
            extract_detector=not bool(args.no_detector),
            force=bool(args.force),
        )
        if detector is not None:
            print("detector:", detector)
        print("model_dir:", model_dir)


if __name__ == "__main__":
    main()
