#!/usr/bin/env python3
"""Combine per-chromosome TSV/TSV.GZ shards back into one TSV or TSV.GZ file."""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path


def open_reader(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def open_writer(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wb")
    return path.open("wb")


def chrom_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"\.(chr[^./]+)\.tsv(?:\.gz)?$", path.name)
    chrom = match.group(1) if match else path.stem
    chrom = chrom.removeprefix("chr")
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if chrom.isdigit():
        return (0, int(chrom))
    if chrom in special:
        return (0, special[chrom])
    return (1, chrom)


def files_from_manifest(input_dir: Path, manifest_path: Path) -> list[Path]:
    files: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as manifest:
        header = manifest.readline().rstrip("\n").split("\t")
        if header[:2] != ["chr", "filename"]:
            raise ValueError(f"{manifest_path} does not look like a split manifest")
        for line in manifest:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            files.append(input_dir / fields[1])
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine chromosome-level TSV shards, writing the header only once."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing per-chromosome .tsv.gz files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file. Use .gz suffix to gzip-compress the combined TSV.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest from split_tsv_by_chr.py. Default: <input_dir>/manifest.tsv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    manifest_path = args.manifest or input_dir / "manifest.tsv"
    output_path = args.output or input_dir.with_name(
        f"{input_dir.name.removesuffix('.by_chr')}.recombined.tsv.gz"
    )

    if output_path.exists() and not args.force:
        raise SystemExit(f"{output_path} already exists. Re-run with --force to replace it.")

    if manifest_path.exists():
        input_files = files_from_manifest(input_dir, manifest_path)
    else:
        input_files = sorted(input_dir.glob("*.tsv.gz"), key=chrom_sort_key)

    if not input_files:
        raise SystemExit(f"No .tsv.gz files found in {input_dir}")

    header: bytes | None = None
    rows_written = 0
    with open_writer(output_path) as out_fh:
        for input_file in input_files:
            with open_reader(input_file) as in_fh:
                file_header = in_fh.readline()
                if not file_header:
                    raise ValueError(f"{input_file} is empty")
                if header is None:
                    header = file_header
                    out_fh.write(header)
                elif file_header != header:
                    raise ValueError(f"{input_file} has a different header")

                for line in in_fh:
                    out_fh.write(line)
                    rows_written += 1

    print(f"Combined {len(input_files)} files into {output_path}")
    print(f"Data rows written: {rows_written}")


if __name__ == "__main__":
    main()
