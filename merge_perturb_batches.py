#!/usr/bin/env python3
"""
Merge perturbation batch outputs into ALL outputs (JSONL + Excel), with de-dup and safe cleanup.

Design goals:
- JSONL is the source of truth (dedupe on (initial_solution.edges_hash, perturbation_step)).
- Merge is idempotent and resume-friendly:
  - If you re-run merge, it can merge ALL + remaining batch files again safely.
  - Writes ALL outputs via temp files and atomic rename.
- Cleanup is safe:
  - Only deletes batch files that match the expected patterns and were present at merge time.
  - Never deletes ALL files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
from typing import Dict, Tuple, List, Optional, Set


def _iter_jsonl_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            yield rec, line


def _record_key(rec: dict) -> Optional[Tuple[str, int]]:
    init = rec.get("initial_solution") or {}
    h = init.get("edges_hash")
    if not h:
        return None
    try:
        step = int(rec.get("perturbation_step") or 0)
    except Exception:
        step = 0
    if step <= 0:
        return None
    return (str(h), step)


def _edges_to_undirected_set(edges) -> Set[Tuple[int, int]]:
    """
    Convert edge list [[u,v], ...] to normalized undirected set {(min,max), ...}.
    """
    s: Set[Tuple[int, int]] = set()
    if not edges:
        return s
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        try:
            u = int(e[0])
            v = int(e[1])
        except Exception:
            continue
        if u == v:
            continue
        s.add((u, v) if u < v else (v, u))
    return s


def _compute_broken_pairs_from_edges(original_edges, perturbed_edges) -> Tuple[int, int, float]:
    """
    Broken pairs distance for undirected CVRP:
      broken_pairs_count = |E_orig \\ E_pert|
      total_pairs_count  = |E_orig|
      ratio              = broken_pairs_count / total_pairs_count
    """
    e0 = _edges_to_undirected_set(original_edges)
    e1 = _edges_to_undirected_set(perturbed_edges)
    total = len(e0)
    broken = len(e0 - e1) if total > 0 else 0
    ratio = (broken / total) if total > 0 else 0.0
    return broken, total, ratio


def _ensure_broken_pairs_fields(rec: dict) -> dict:
    """
    If record lacks broken_pairs_* fields in perturbed_solution, compute them from JSONL edge lists.
    Works for older JSONL that didn't include these fields.
    """
    init = rec.get("initial_solution") or {}
    ps = rec.get("perturbed_solution") or {}

    # If already present, keep as-is
    if (
        ps.get("broken_pairs_count") is not None
        and ps.get("broken_pairs_total") is not None
        and ps.get("broken_pairs_ratio") is not None
    ):
        return rec

    orig_edges = init.get("edges") or []
    pert_edges = ps.get("edges") or []
    broken, total, ratio = _compute_broken_pairs_from_edges(orig_edges, pert_edges)
    ps["broken_pairs_count"] = broken
    ps["broken_pairs_total"] = total
    ps["broken_pairs_ratio"] = ratio
    rec["perturbed_solution"] = ps
    return rec

def merge_jsonl(
    out_dir: str,
    operator_type: str,
    runs: int,
    keep: str = "last",
) -> Tuple[str, List[str]]:
    """
    Merge ALL + batch JSONL files into ALL JSONL with de-dup.

    Returns:
      (all_jsonl_path, batch_jsonl_files_used)
    """
    all_jsonl = os.path.join(out_dir, f"{operator_type}_training_data.ALL_r{runs}.jsonl")
    batch_pattern = os.path.join(out_dir, f"{operator_type}_training_data.batch_*_r{runs}.jsonl")
    batch_files = sorted(glob.glob(batch_pattern))

    inputs: List[str] = []
    if os.path.exists(all_jsonl):
        inputs.append(all_jsonl)
    inputs.extend(batch_files)

    if not inputs:
        # create empty ALL jsonl if nothing exists
        os.makedirs(out_dir, exist_ok=True)
        open(all_jsonl, "a", encoding="utf-8").close()
        return all_jsonl, []

    # Dedup: key -> record(dict)
    merged: Dict[Tuple[str, int], dict] = {}
    for path in inputs:
        for rec, line in _iter_jsonl_records(path):
            k = _record_key(rec)
            if k is None:
                continue
            rec = _ensure_broken_pairs_fields(rec)
            if keep == "first":
                merged.setdefault(k, rec)
            else:
                # last wins
                merged[k] = rec

    # Stable order: by (edges_hash, step)
    items = sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    # Atomic write
    os.makedirs(out_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{operator_type}_ALL_", suffix=".jsonl", dir=out_dir)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as w:
            for _, rec in items:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, all_jsonl)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    return all_jsonl, batch_files


def merge_xlsx(out_dir: str, operator_type: str, runs: int, all_jsonl_path: str = None) -> Tuple[str, List[str]]:
    """
    Merge ALL + batch Excel files into ALL Excel with de-dup on original_edges_hash.
    If Excel is missing or empty, rebuild it from JSONL.

    Returns:
      (all_xlsx_path, batch_xlsx_files_used)
    """
    import pandas as pd

    all_xlsx = os.path.join(out_dir, f"{operator_type}_results.ALL_r{runs}.xlsx")
    batch_pattern = os.path.join(out_dir, f"{operator_type}_results.batch_*_r{runs}.xlsx")
    batch_files = sorted(glob.glob(batch_pattern))

    inputs: List[str] = []
    if os.path.exists(all_xlsx):
        inputs.append(all_xlsx)
    inputs.extend(batch_files)

    # Check if we need to rebuild from JSONL
    needs_rebuild = False
    if not inputs:
        needs_rebuild = True
    else:
        # Check if merged Excel is empty or has no data rows
        frames = []
        for p in inputs:
            try:
                df_tmp = pd.read_excel(p)
                if len(df_tmp) > 0:
                    frames.append(df_tmp)
            except Exception:
                continue
        if not frames:
            needs_rebuild = True

    # Rebuild from JSONL if needed
    if needs_rebuild and all_jsonl_path and os.path.exists(all_jsonl_path):
        print(f"  Excel missing or empty; rebuilding from JSONL: {all_jsonl_path}")
        try:
            # Import from perturb module
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from perturb import save_excel_from_training_jsonl
            save_excel_from_training_jsonl(all_jsonl_path, all_xlsx, optima_stats=None)
            return all_xlsx, batch_files
        except Exception as e:
            print(f"  Warning: Failed to rebuild Excel from JSONL: {e}")
            # Fall through to create empty Excel

    if not inputs:
        # create empty ALL xlsx if nothing exists
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame([])
        with pd.ExcelWriter(all_xlsx, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Perturbation Results", index=False)
        return all_xlsx, []

    frames = []
    for p in inputs:
        try:
            frames.append(pd.read_excel(p))
        except Exception:
            continue
    if not frames:
        # fallback empty
        df = pd.DataFrame([])
    else:
        df = pd.concat(frames, ignore_index=True)

    if "original_edges_hash" in df.columns:
        df = df.drop_duplicates(subset=["original_edges_hash"], keep="first")

    # Atomic write
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f".{operator_type}_results.ALL_r{runs}.tmp.xlsx")
    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Perturbation Results", index=False)
        os.replace(tmp_path, all_xlsx)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    return all_xlsx, batch_files


def _build_broken_pairs_map_from_all_jsonl(all_jsonl_path: str) -> Dict[Tuple[str, int], Tuple[int, int, float]]:
    """
    Map (original_edges_hash, step) -> (broken_pairs_count, broken_pairs_total, broken_pairs_ratio)
    using ALL JSONL. Used to backfill ALL.xlsx when older batch excels lack these columns.
    """
    mp: Dict[Tuple[str, int], Tuple[int, int, float]] = {}
    if not os.path.exists(all_jsonl_path):
        return mp
    for rec, _line in _iter_jsonl_records(all_jsonl_path):
        k = _record_key(rec)
        if k is None:
            continue
        ps = rec.get("perturbed_solution") or {}
        c = ps.get("broken_pairs_count")
        t = ps.get("broken_pairs_total")
        r = ps.get("broken_pairs_ratio")
        if c is None or t is None or r is None:
            # compute if still missing
            rec = _ensure_broken_pairs_fields(rec)
            ps = rec.get("perturbed_solution") or {}
            c = ps.get("broken_pairs_count")
            t = ps.get("broken_pairs_total")
            r = ps.get("broken_pairs_ratio")
        try:
            mp[(str(k[0]), int(k[1]))] = (int(c), int(t), float(r))
        except Exception:
            continue
    return mp


def _collect_jsonl_hashes(out_dir: str, operator_type: str, runs: int) -> Set[str]:
    """
    Collect processed original edges_hash from ALL + batch JSONLs.
    """
    all_jsonl = os.path.join(out_dir, f"{operator_type}_training_data.ALL_r{runs}.jsonl")
    batch_pattern = os.path.join(out_dir, f"{operator_type}_training_data.batch_*_r{runs}.jsonl")
    files = []
    if os.path.exists(all_jsonl):
        files.append(all_jsonl)
    files.extend(sorted(glob.glob(batch_pattern)))
    processed = set()
    for p in files:
        for rec, _line in _iter_jsonl_records(p):
            init = rec.get("initial_solution") or {}
            h = init.get("edges_hash")
            if h:
                processed.add(str(h))
    return processed


def _collect_excel_hashes(out_dir: str, operator_type: str, runs: int) -> Set[str]:
    """
    Collect processed original edges_hash from ALL + batch XLSX.
    """
    import pandas as pd

    all_xlsx = os.path.join(out_dir, f"{operator_type}_results.ALL_r{runs}.xlsx")
    batch_pattern = os.path.join(out_dir, f"{operator_type}_results.batch_*_r{runs}.xlsx")
    files = []
    if os.path.exists(all_xlsx):
        files.append(all_xlsx)
    files.extend(sorted(glob.glob(batch_pattern)))
    processed = set()
    for p in files:
        try:
            df = pd.read_excel(p)
        except Exception:
            continue
        if "original_edges_hash" not in df.columns:
            continue
        for h in df["original_edges_hash"].dropna().astype(str).tolist():
            if h:
                processed.add(h)
    return processed


def _count_total_unique_optima(instance_path: str, instance_index: int, basin_base_dir: str) -> Optional[int]:
    """
    Count unique local optima (by edges_hash) from basin_datasets0/<instance_id>/optima.jsonl.
    """
    # Local import to avoid coupling if utils is unavailable in some environments.
    try:
        from utils import get_basin_paths
    except Exception:
        return None

    basin_paths = get_basin_paths(instance_path, instance_index, basin_base_dir)
    optima_jsonl = os.path.join(basin_paths["basin_dir"], "optima.jsonl")
    if not os.path.exists(optima_jsonl):
        return None

    uniq = set()
    with open(optima_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            h = rec.get("edges_hash")
            if h:
                uniq.add(str(h))
    return len(uniq)


def print_resume_stats(
    out_dir: str,
    operator_type: str,
    runs: int,
    instance_path: Optional[str] = None,
    instance_index: Optional[int] = None,
    basin_base_dir: Optional[str] = None,
):
    """
    Print resume/coverage stats:
    - processed_jsonl (source of truth)
    - processed_excel
    - both / jsonl_only / excel_only
    - remaining (if total unique optima is known)
    """
    jsonl_hashes = _collect_jsonl_hashes(out_dir, operator_type, runs)
    excel_hashes = _collect_excel_hashes(out_dir, operator_type, runs)

    both = jsonl_hashes & excel_hashes
    jsonl_only = jsonl_hashes - excel_hashes
    excel_only = excel_hashes - jsonl_hashes

    total_unique = None
    if instance_path is not None and instance_index is not None and basin_base_dir is not None:
        total_unique = _count_total_unique_optima(instance_path, int(instance_index), basin_base_dir)

    print("== Coverage / resume stats ==")
    print(f"  processed (JSONL, source of truth): {len(jsonl_hashes)}")
    print(f"  processed (Excel):                 {len(excel_hashes)}")
    print(f"  both JSONL & Excel:                {len(both)}")
    print(f"  JSONL-only (needs Excel rebuild):  {len(jsonl_only)}")
    print(f"  Excel-only (NOT done, will rerun): {len(excel_only)}")
    if total_unique is not None:
        remaining = max(0, int(total_unique) - len(jsonl_hashes))
        print(f"  total unique optima (optima.jsonl): {total_unique}")
        print(f"  remaining (not in JSONL):           {remaining}")
    else:
        print("  total unique optima:                (unknown; pass --instance_path/--instance_index/--basin_base_dir)")


def safe_delete(files: List[str], out_dir: str):
    """
    Delete only files inside out_dir, and only exact paths provided.
    """
    out_dir_abs = os.path.abspath(out_dir)
    for f in files:
        try:
            f_abs = os.path.abspath(f)
            if not f_abs.startswith(out_dir_abs + os.sep):
                continue
            if not os.path.exists(f_abs):
                continue
            # never delete ALL
            base = os.path.basename(f_abs)
            if ".ALL_" in base or ".ALL" in base:
                continue
            os.remove(f_abs)
        except Exception:
            # best-effort deletion
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output dir containing batch_* files")
    ap.add_argument("--operator_type", required=True, choices=["double_bridge", "remove_and_insert"])
    ap.add_argument("--runs", required=True, type=int)
    ap.add_argument("--keep", choices=["last", "first"], default="last", help="Dedup policy for JSONL")
    ap.add_argument("--delete_batches", action="store_true", help="Delete batch files after successful merge")
    ap.add_argument("--instance_path", type=str, default=None, help="Instance pkl path (for total optima stats)")
    ap.add_argument("--instance_index", type=int, default=None, help="Instance index (for total optima stats)")
    ap.add_argument("--basin_base_dir", type=str, default="basin_datasets0", help="Input basin base dir (for total optima stats)")
    args = ap.parse_args()

    out_dir = args.out_dir
    operator_type = args.operator_type
    runs = int(args.runs)

    print(f"== Merge batches ==")
    print(f"  out_dir:       {out_dir}")
    print(f"  operator_type: {operator_type}")
    print(f"  runs:          {runs}")
    print(f"  dedup keep:    {args.keep}")
    print(f"  delete_batches:{args.delete_batches}")
    print_resume_stats(
        out_dir,
        operator_type,
        runs,
        instance_path=args.instance_path,
        instance_index=args.instance_index,
        basin_base_dir=args.basin_base_dir,
    )

    all_jsonl, used_jsonl_batches = merge_jsonl(out_dir, operator_type, runs, keep=args.keep)
    print(f"  Wrote ALL JSONL: {all_jsonl}")

    all_xlsx, used_xlsx_batches = merge_xlsx(out_dir, operator_type, runs, all_jsonl_path=all_jsonl)
    print(f"  Wrote ALL XLSX:  {all_xlsx}")

    # Backfill broken_pairs_* columns in ALL.xlsx if missing, using ALL.jsonl
    try:
        import pandas as pd

        df = pd.read_excel(all_xlsx) if os.path.exists(all_xlsx) else None
        if df is not None and len(df) > 0 and "original_edges_hash" in df.columns:
            needs = []
            for col in ["step_1_broken_pairs_count", "step_1_broken_pairs_total", "step_1_broken_pairs_ratio"]:
                if col not in df.columns:
                    needs.append(col)
            if needs:
                mp = _build_broken_pairs_map_from_all_jsonl(all_jsonl)
                # Determine max step present in df
                step_cols = [c for c in df.columns if c.startswith("step_") and c.endswith("_hash")]
                steps = []
                for c in step_cols:
                    try:
                        steps.append(int(c.split("_")[1]))
                    except Exception:
                        pass
                max_step = max(steps) if steps else 1
                for step in range(1, max_step + 1):
                    ccol = f"step_{step}_broken_pairs_count"
                    tcol = f"step_{step}_broken_pairs_total"
                    rcol = f"step_{step}_broken_pairs_ratio"
                    if ccol not in df.columns:
                        df[ccol] = None
                    if tcol not in df.columns:
                        df[tcol] = None
                    if rcol not in df.columns:
                        df[rcol] = None

                    for idx, oh in enumerate(df["original_edges_hash"].astype(str).tolist()):
                        key = (oh, step)
                        if key not in mp:
                            continue
                        c, t, r = mp[key]
                        df.at[idx, ccol] = c
                        df.at[idx, tcol] = t
                        df.at[idx, rcol] = r

                tmp_path = os.path.join(out_dir, f".{operator_type}_results.ALL_r{runs}.bpfill.tmp.xlsx")
                with pd.ExcelWriter(tmp_path, engine="openpyxl") as w:
                    df.to_excel(w, sheet_name="Perturbation Results", index=False)
                os.replace(tmp_path, all_xlsx)
                print("  Backfilled broken_pairs_* columns in ALL XLSX from ALL JSONL")
    except Exception as e:
        print(f"  Warning: Failed to backfill broken_pairs_* in ALL XLSX: {e}")

    if args.delete_batches:
        # Only delete batch files, never ALL
        safe_delete(used_jsonl_batches, out_dir)
        safe_delete(used_xlsx_batches, out_dir)
        print(f"  Deleted batch files: jsonl={len(used_jsonl_batches)} xlsx={len(used_xlsx_batches)}")


if __name__ == "__main__":
    main()

