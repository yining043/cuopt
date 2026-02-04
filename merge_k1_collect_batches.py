#!/usr/bin/env python3
import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--operator_type", required=True)
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--delete_batches", action="store_true")
    args = ap.parse_args()

    summary_glob = os.path.join(
        args.out_dir, f"k1_collection_summary_{args.operator_type}.batch_*_r{args.runs}.jsonl"
    )
    results_glob = os.path.join(
        args.out_dir, f"k1_collection_results_{args.operator_type}.batch_*_r{args.runs}.jsonl"
    )
    summary_xlsx_glob = os.path.join(
        args.out_dir, f"k1_collection_summary_{args.operator_type}.batch_*_r{args.runs}.xlsx"
    )

    summary_paths = sorted(glob.glob(summary_glob))
    results_paths = sorted(glob.glob(results_glob))
    summary_xlsx_paths = sorted(glob.glob(summary_xlsx_glob))

    summary_all = os.path.join(args.out_dir, f"k1_collection_summary_{args.operator_type}.ALL_r{args.runs}.jsonl")
    results_all = os.path.join(args.out_dir, f"k1_collection_results_{args.operator_type}.ALL_r{args.runs}.jsonl")
    xlsx_all = os.path.join(args.out_dir, f"k1_collection_summary_{args.operator_type}.ALL_r{args.runs}.xlsx")

    if not summary_paths and not results_paths and not summary_xlsx_paths:
        print("  No batch JSONL/XLSX files found; skipping merge (ALL files left unchanged).")
        return

    # Merge summary JSONL and XLSX, keeping the union of anchors (JSONL rows take precedence).
    seen_anchor = set()
    rows = []

    # 1) Load from JSONL first (authoritative when duplicated with XLSX)
    for p in summary_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                anchor_hash = rec.get("anchor_hash")
                if not anchor_hash or anchor_hash in seen_anchor:
                    continue
                seen_anchor.add(anchor_hash)
                rows.append(rec)

    # 2) Then load from XLSX, only adding anchors not already seen
    if summary_xlsx_paths:
        merged_xlsx = []
        for p in summary_xlsx_paths:
            try:
                df = pd.read_excel(p, engine="openpyxl")
            except Exception:
                continue
            if df is None or df.empty:
                continue
            merged_xlsx.append(df)

        if merged_xlsx:
            df_x = pd.concat(merged_xlsx, ignore_index=True)
            if "anchor_hash" not in df_x.columns:
                for alt in ["anchor", "anchor_edges_hash", "edges_hash"]:
                    if alt in df_x.columns:
                        df_x = df_x.rename(columns={alt: "anchor_hash"})
                        break
            if "anchor_hash" in df_x.columns:
                df_x = df_x.dropna(subset=["anchor_hash"])
                df_x["anchor_hash"] = df_x["anchor_hash"].astype(str)
                # Only keep rows whose anchor_hash not already in JSONL-loaded set
                df_x = df_x[~df_x["anchor_hash"].isin(seen_anchor)]
                for rec in df_x.to_dict(orient="records"):
                    ah = rec.get("anchor_hash")
                    if not ah or ah in seen_anchor:
                        continue
                    seen_anchor.add(ah)
                    rows.append(rec)

    # 3) From combined rows, write ALL JSONL and ALL XLSX (if any)
    if rows:
        df = pd.DataFrame(rows)
        if "anchor_hash" in df.columns:
            df = df.dropna(subset=["anchor_hash"])
            df["anchor_hash"] = df["anchor_hash"].astype(str)
            df = df.drop_duplicates(subset=["anchor_hash"], keep="first")
            if "success" in df.columns:
                df = df.sort_values(["success", "anchor_hash"], ascending=[False, True])
            else:
                df = df.sort_values(["anchor_hash"], ascending=[True])

        with open(summary_all, "w", encoding="utf-8") as out:
            for rec in df.to_dict(orient="records"):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with pd.ExcelWriter(xlsx_all, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Collection Results", index=False)

    if results_paths:
        with open(results_all, "w", encoding="utf-8") as out:
            for p in results_paths:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)

    if args.delete_batches:
        for p in summary_paths + results_paths + summary_xlsx_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()

