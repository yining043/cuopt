#!/usr/bin/env python3
"""
Minimal test: run cuOpt solve with callback (--plot so collect_data=True) and --log,
then check the log for:
  [CUOPT_DEBUG] perform_search: has_callback=...  (confirms binary + callback detection)
  [POP] or [EVOLVE] or [WLOOP]                     (confirms population verbose)

Run from basin_callback dir (after full build):
  python test_population_print.py
  python test_population_print.py --time_limit 30 --start_index 9
"""
import os
import subprocess
import sys


def main():
    basin_callback_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(basin_callback_dir, "curves", "test_population_print")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "baseline_run.log")

    time_limit = 15
    start_index = 0
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--time_limit" and i + 1 < len(argv):
            time_limit = float(argv[i + 1])
        if a == "--start_index" and i + 1 < len(argv):
            start_index = int(argv[i + 1])

    cmd = [
        sys.executable, "-u", os.path.join(basin_callback_dir, "run_cuopt.py"),
        "solve",
        "--time_limit", str(time_limit),
        "--start_index", str(start_index),
        "--n_runs", "1",
        "--log", log_path,
        "--plot",
        "--save_upper_bound_log",
    ]
    print("Running:", " ".join(cmd), flush=True)
    print("(collect_data=True from --plot so callback is registered; C++ stdout goes to log)", flush=True)

    result = subprocess.run(
        cmd,
        cwd=basin_callback_dir,
        capture_output=False,
        timeout=int(time_limit) + 120,
    )

    # Log is written to curves/<start_index>_tl_<time>_<timestamp>/baseline_run.log
    curves_dir = os.path.join(basin_callback_dir, "curves")
    candidates = []
    if os.path.isdir(curves_dir):
        for d in os.listdir(curves_dir):
            p = os.path.join(curves_dir, d, "baseline_run.log")
            if os.path.isfile(p):
                candidates.append((os.path.getmtime(p), p))
    if not candidates:
        print(f"No baseline_run.log found under {curves_dir}", file=sys.stderr)
        sys.exit(2)
    log_path = max(candidates, key=lambda x: x[0])[1]
    print(f"Using log: {log_path}", flush=True)

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    debug_line = "[CUOPT_DEBUG] perform_search:"
    pop_markers = ["[POP]", "[EVOLVE]", "[WLOOP]", "[BASIN]", "[INIT]"]
    has_debug = debug_line in text
    has_pop = any(m in text for m in pop_markers)
    which_pop = [m for m in pop_markers if m in text]

    print()
    print("--- Result ---")
    print(f"  [CUOPT_DEBUG] present: {has_debug}")
    if has_debug:
        for line in text.splitlines():
            if debug_line in line:
                print(f"  -> {line.strip()}")
                break
    print(f"  Population prints ([POP]/[EVOLVE]/...) present: {has_pop}")
    if which_pop:
        print(f"  Found: {which_pop}")
        # show a few sample lines
        for m in which_pop[:2]:
            for line in text.splitlines():
                if m in line:
                    print(f"  e.g. {line.strip()[:80]}...")
                    break
    else:
        print("  None of [POP], [EVOLVE], [WLOOP], [BASIN], [INIT] found in log.")
    print(f"  Full log: {log_path}")

    if not has_debug:
        print("  -> Binary may not be from basin_callback build (no [CUOPT_DEBUG] line).")
    elif not has_pop:
        print("  -> Check [CUOPT_DEBUG] line: if has_callback=0, callback not seen by C++.")
    sys.exit(0 if (has_debug and has_pop) else 1)


if __name__ == "__main__":
    main()
