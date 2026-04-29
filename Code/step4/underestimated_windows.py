#!/usr/bin/env python3
"""
Rebuild combined underestimated-window tables for the z4 baseline and optional
Tukey-thresholded combined_k*.tsv inputs.

For each prefix (for example, combined_10kb_underest_z4 or combined_k4), this
script writes:
1) <prefix>_step1.tsv
2) <prefix>_step2_with_end.tsv
3) <prefix>_step3_with_genes.tsv
4) <prefix>_step4_with_cancer_flags.tsv
5) <prefix>_step5_cancer_genes_only.tsv
6) <prefix>_step6_unique_cancer_genes.tsv

It also writes downstream-compatible aliases:
- <prefix>_cancer_genes_only.tsv
- <prefix>_cancer_types_by_gene.tsv

Pipeline:
- Build/normalize step1 windows
- Add window end coordinate (10kb) to build step2
- Intersect windows with hg19_genes_gff.bed
- Flag genes present in OncoKB and CGC
- Keep only windows hitting cancer genes (OncoKB or CGC)
- Deduplicate to unique cancer genes
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


CANCER_ORDER = ["breast", "lung", "colorectal", "esophagus", "prostate", "skin"]
VARIANT_ORDER = ["snv", "indel"]

STEP1_CORE_FIELDS = ["chr", "start", "obs", "pred", "residual", "cancer_type", "variant"]
STEP2_CORE_FIELDS = ["chr", "start", "end", "obs", "pred", "residual", "cancer_type", "variant"]
STEP3_GENE_FIELDS = ["gene_chr", "gene_start", "gene_end", "gene", "gene_source", "gene_attributes"]
STEP4_FLAG_FIELDS = ["in_oncokb", "in_cgc"]
STEP6_FIELDS = ["gene", "cancer_types"]


def read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def require_columns(path: Path, row: Dict[str, str], columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in row]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")


def pick_existing_path(description: str, candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tested = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(f"Could not find {description}. Tried:\n{tested}")


def find_underest_file(input_dir: Path, cancer: str, variant: str) -> Path:
    candidates = [
        input_dir / f"{cancer}_{variant}_10kb_underest_z_4.0_all.tsv",
        input_dir / f"{cancer}_{variant}_10kb_underest_z>4.0_all.tsv",
        input_dir / cancer / f"{cancer}_{variant}_10kb_underest_z_4.0_all.tsv",
        input_dir / cancer / f"{cancer}_{variant}_10kb_underest_z>4.0_all.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pattern = f"**/{cancer}_{variant}_10kb_underest_z*4.0_all.tsv"
    recursive_matches = sorted(input_dir.glob(pattern))
    if recursive_matches:
        return recursive_matches[0]

    tested = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        f"Missing underestimated input for {cancer}/{variant}.\n"
        f"Tried canonical paths:\n{tested}\n"
        f"And recursive pattern: {pattern}"
    )


def infer_extra_fields(rows: List[Dict[str, str]], base_fields: Iterable[str]) -> List[str]:
    seen = set(base_fields)
    extras: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen and key not in extras:
                extras.append(key)
    return extras


def build_step1(input_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for cancer in CANCER_ORDER:
        for variant in VARIANT_ORDER:
            path = find_underest_file(input_dir, cancer, variant)

            source_rows = read_tsv_rows(path)
            if source_rows:
                require_columns(path, source_rows[0], ["chr", "start", "obs", "pred", "residual"])

            for row in source_rows:
                rows.append(
                    {
                        "chr": row["chr"],
                        "start": row["start"],
                        "obs": row["obs"],
                        "pred": row["pred"],
                        "residual": row["residual"],
                        "cancer_type": cancer,
                        "variant": variant,
                    }
                )

    return rows


def build_step1_from_combined(path: Path) -> List[Dict[str, str]]:
    source_rows = read_tsv_rows(path)
    if source_rows:
        require_columns(path, source_rows[0], STEP2_CORE_FIELDS)

    extra_fields = infer_extra_fields(source_rows, STEP2_CORE_FIELDS)
    rows: List[Dict[str, str]] = []
    for row in source_rows:
        step1_row = {
            "chr": row["chr"],
            "start": row["start"],
            "obs": row["obs"],
            "pred": row["pred"],
            "residual": row["residual"],
            "cancer_type": row["cancer_type"],
            "variant": row["variant"],
        }
        for field in extra_fields:
            step1_row[field] = row.get(field, "")
        rows.append(step1_row)

    return rows


def build_step2(step1_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    step2_rows: List[Dict[str, str]] = []
    extra_fields = infer_extra_fields(step1_rows, STEP1_CORE_FIELDS)

    for row in step1_rows:
        start = int(float(row["start"]))
        step2_row = {
            "chr": row["chr"],
            "start": row["start"],
            "end": str(start + 9999),
            "obs": row["obs"],
            "pred": row["pred"],
            "residual": row["residual"],
            "cancer_type": row["cancer_type"],
            "variant": row["variant"],
        }
        for field in extra_fields:
            step2_row[field] = row.get(field, "")
        step2_rows.append(step2_row)

    return step2_rows


def load_genes_by_chr(genes_bed_path: Path) -> Dict[str, List[Dict[str, str]]]:
    by_chr: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    genes = read_tsv_rows(genes_bed_path)
    if genes:
        require_columns(genes_bed_path, genes[0], ["chr", "start", "end", "gene", "source", "attributes"])

    for row in genes:
        by_chr[row["chr"]].append(row)

    return by_chr


def intersects(window_start: int, window_end: int, gene_start: int, gene_end: int) -> bool:
    return window_start < gene_end and window_end > gene_start


def build_step3(step2_rows: List[Dict[str, str]], genes_by_chr: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    step3_rows: List[Dict[str, str]] = []

    for window in step2_rows:
        window_chr = window["chr"]
        window_start = int(float(window["start"]))
        window_end = int(float(window["end"]))

        for gene in genes_by_chr.get(window_chr, []):
            gene_start = int(gene["start"])
            gene_end = int(gene["end"])
            if not intersects(window_start, window_end, gene_start, gene_end):
                continue

            step3_rows.append(
                {
                    **window,
                    "gene_chr": gene["chr"],
                    "gene_start": gene["start"],
                    "gene_end": gene["end"],
                    "gene": gene["gene"],
                    "gene_source": gene["source"],
                    "gene_attributes": gene["attributes"],
                }
            )

    return step3_rows


def load_gene_symbol_set(path: Path, column_name: str) -> set[str]:
    symbols = set()
    rows = read_tsv_rows(path)
    if rows:
        require_columns(path, rows[0], [column_name])
    for row in rows:
        symbol = row[column_name].strip()
        if symbol:
            symbols.add(symbol)
    return symbols


def build_step4(step3_rows: List[Dict[str, str]], oncokb_symbols: set[str], cgc_symbols: set[str]) -> List[Dict[str, str]]:
    step4_rows: List[Dict[str, str]] = []
    for row in step3_rows:
        gene = row["gene"]
        step4_rows.append(
            {
                **row,
                "in_oncokb": "Yes" if gene in oncokb_symbols else "No",
                "in_cgc": "Yes" if gene in cgc_symbols else "No",
            }
        )
    return step4_rows


def build_step5(step4_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in step4_rows if row["in_oncokb"] == "Yes" or row["in_cgc"] == "Yes"]


def build_step6(step5_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    gene_to_cancers: Dict[str, set[str]] = defaultdict(set)
    for row in step5_rows:
        gene_to_cancers[row["gene"]].add(row["cancer_type"])

    step6_rows: List[Dict[str, str]] = []
    for gene in sorted(gene_to_cancers):
        cancers = gene_to_cancers[gene]
        ordered = [ct for ct in CANCER_ORDER if ct in cancers]
        extras = sorted(ct for ct in cancers if ct not in set(CANCER_ORDER))
        ordered.extend(extras)
        step6_rows.append({"gene": gene, "cancer_types": ",".join(ordered)})

    return step6_rows


def discover_combined_k_files(input_dir: Path) -> List[Path]:
    matches = []
    for path in input_dir.glob("combined_k*.tsv"):
        stem_match = re.fullmatch(r"combined_k([0-9]+(?:\.[0-9]+)?)", path.stem)
        if stem_match:
            matches.append((float(stem_match.group(1)), path))
    matches.sort(key=lambda item: item[0])
    return [path for _, path in matches]


def build_output_paths(output_dir: Path, prefix: str) -> Dict[str, Path]:
    return {
        "step1": output_dir / f"{prefix}_step1.tsv",
        "step2": output_dir / f"{prefix}_step2_with_end.tsv",
        "step3": output_dir / f"{prefix}_step3_with_genes.tsv",
        "step4": output_dir / f"{prefix}_step4_with_cancer_flags.tsv",
        "step5": output_dir / f"{prefix}_step5_cancer_genes_only.tsv",
        "step6": output_dir / f"{prefix}_step6_unique_cancer_genes.tsv",
        "cancer_genes_only": output_dir / f"{prefix}_cancer_genes_only.tsv",
        "cancer_types_by_gene": output_dir / f"{prefix}_cancer_types_by_gene.tsv",
    }


def run_pipeline(
    prefix: str,
    step1_rows: List[Dict[str, str]],
    output_dir: Path,
    genes_by_chr: Dict[str, List[Dict[str, str]]],
    oncokb_symbols: set[str],
    cgc_symbols: set[str],
) -> Dict[str, object]:
    extra_fields = infer_extra_fields(step1_rows, STEP1_CORE_FIELDS)
    step1_fields = STEP1_CORE_FIELDS + extra_fields
    step2_fields = STEP2_CORE_FIELDS + extra_fields
    step3_fields = step2_fields + STEP3_GENE_FIELDS
    step4_fields = step3_fields + STEP4_FLAG_FIELDS

    step2_rows = build_step2(step1_rows)
    step3_rows = build_step3(step2_rows, genes_by_chr)
    step4_rows = build_step4(step3_rows, oncokb_symbols, cgc_symbols)
    step5_rows = build_step5(step4_rows)
    step6_rows = build_step6(step5_rows)

    paths = build_output_paths(output_dir, prefix)
    write_tsv_rows(paths["step1"], step1_fields, step1_rows)
    write_tsv_rows(paths["step2"], step2_fields, step2_rows)
    write_tsv_rows(paths["step3"], step3_fields, step3_rows)
    write_tsv_rows(paths["step4"], step4_fields, step4_rows)
    write_tsv_rows(paths["step5"], step4_fields, step5_rows)
    write_tsv_rows(paths["step6"], STEP6_FIELDS, step6_rows)
    write_tsv_rows(paths["cancer_genes_only"], step4_fields, step5_rows)
    write_tsv_rows(paths["cancer_types_by_gene"], STEP6_FIELDS, step6_rows)

    return {
        "prefix": prefix,
        "paths": paths,
        "counts": {
            "step1": len(step1_rows),
            "step2": len(step2_rows),
            "step3": len(step3_rows),
            "step4": len(step4_rows),
            "step5": len(step5_rows),
            "step6": len(step6_rows),
        },
    }


def print_pipeline_summary(summary: Dict[str, object]) -> None:
    prefix = str(summary["prefix"])
    paths = summary["paths"]
    counts = summary["counts"]

    if not isinstance(paths, dict) or not isinstance(counts, dict):
        raise TypeError("Invalid pipeline summary format.")

    print(f"[{prefix}] Wrote {counts['step1']} rows -> {paths['step1']}")
    print(f"[{prefix}] Wrote {counts['step2']} rows -> {paths['step2']}")
    print(f"[{prefix}] Wrote {counts['step3']} rows -> {paths['step3']}")
    print(f"[{prefix}] Wrote {counts['step4']} rows -> {paths['step4']}")
    print(f"[{prefix}] Wrote {counts['step5']} rows -> {paths['step5']}")
    print(f"[{prefix}] Wrote {counts['step6']} rows -> {paths['step6']}")
    print(f"[{prefix}] Wrote {counts['step5']} rows -> {paths['cancer_genes_only']}")
    print(f"[{prefix}] Wrote {counts['step6']} rows -> {paths['cancer_types_by_gene']}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "final_windows_genes"

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild combined_10kb_underest_z4 step tables and, if present, "
            "combined_k*.tsv step tables."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing per-cancer underestimated TSVs and annotation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory where rebuilt step files will be written.",
    )
    parser.add_argument(
        "--genes-bed",
        type=Path,
        default=None,
        help="Path to hg19_genes_gff.bed (optional; auto-discovered if omitted).",
    )
    parser.add_argument(
        "--oncokb",
        type=Path,
        default=None,
        help="Path to 20250116_oncokb_cancerGeneList.tsv (optional; auto-discovered if omitted).",
    )
    parser.add_argument(
        "--cgc",
        type=Path,
        default=None,
        help="Path to cgc_v100_17102024.tsv (optional; auto-discovered if omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    genes_bed_path = (
        args.genes_bed.resolve()
        if args.genes_bed
        else pick_existing_path(
            "hg19_genes_gff.bed",
            [input_dir / "hg19_genes_gff.bed", input_dir.parent / "hg19_genes_gff.bed"],
        )
    )
    oncokb_path = (
        args.oncokb.resolve()
        if args.oncokb
        else pick_existing_path(
            "20250116_oncokb_cancerGeneList.tsv",
            [
                input_dir / "20250116_oncokb_cancerGeneList.tsv",
                input_dir.parent / "20250116_oncokb_cancerGeneList.tsv",
            ],
        )
    )
    cgc_path = (
        args.cgc.resolve()
        if args.cgc
        else pick_existing_path(
            "cgc_v100_17102024.tsv",
            [input_dir / "cgc_v100_17102024.tsv", input_dir.parent / "cgc_v100_17102024.tsv"],
        )
    )

    genes_by_chr = load_genes_by_chr(genes_bed_path)
    oncokb_symbols = load_gene_symbol_set(oncokb_path, "Hugo Symbol")
    cgc_symbols = load_gene_symbol_set(cgc_path, "GENE_SYMBOL")

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Genes BED: {genes_bed_path}")
    print(f"OncoKB:    {oncokb_path}")
    print(f"CGC:       {cgc_path}")

    z4_step1_rows = build_step1(input_dir)
    z4_summary = run_pipeline(
        prefix="combined_10kb_underest_z4",
        step1_rows=z4_step1_rows,
        output_dir=output_dir,
        genes_by_chr=genes_by_chr,
        oncokb_symbols=oncokb_symbols,
        cgc_symbols=cgc_symbols,
    )
    print_pipeline_summary(z4_summary)

    combined_k_files = discover_combined_k_files(input_dir)
    if not combined_k_files:
        print("No combined_k*.tsv files detected in input dir.")
        return

    print(f"Detected {len(combined_k_files)} combined_k files to process.")
    for path in combined_k_files:
        print(f"Processing {path.name} ...")
        k_step1_rows = build_step1_from_combined(path)
        k_summary = run_pipeline(
            prefix=path.stem,
            step1_rows=k_step1_rows,
            output_dir=output_dir,
            genes_by_chr=genes_by_chr,
            oncokb_symbols=oncokb_symbols,
            cgc_symbols=cgc_symbols,
        )
        print_pipeline_summary(k_summary)


if __name__ == "__main__":
    main()
