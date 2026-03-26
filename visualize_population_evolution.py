#!/usr/bin/env python3
"""
Parse baseline_run.log and visualize population evolution as circles (size=cost).
Produces HTML (interactive) and GIF (animated).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_baseline_log(log_path: str | Path) -> tuple[list[float], list[dict]]:
    """
    Parse log and return (initial_population, events).
    Each event: round, step, basin_A, basin_B, crossover_cost, ls_cost,
                 action in ('REJECTED','INSERTED','REPLACED','NO_DESCENDANT','LOGGED','FAILED'),
                 similar_cost?, new_cost?, pop_after (list of costs).
    """
    log_path = Path(log_path)
    initial_pop: list[float] = []
    events: list[dict] = []

    re_improving = re.compile(
        r"\[INIT\] improving island \d+ \(size=(\d+)"
    )
    re_rejected_init = re.compile(
        r"\[POP\] add_solution: REJECTED \(similar exists at cost=([\d.]+)"
    )
    re_round = re.compile(
        r"\[EVOLVE\] === round=(\d+) .* pop_size=(\d+) best=([\d.]+)"
    )
    re_parents = re.compile(
        r"\[EVOLVE\] step=(\d+) parents: basin_A=([\d.]+) basin_B=([\d.]+) similarity=([\d.]+)"
    )
    re_recombiner = re.compile(
        r"\[EVOLVE\]\s+recombiner=(\w+)(?:\s+\(from \d+ options\))?"
    )
    re_child = re.compile(
        r"\[EVOLVE\]\s+crossover→([\d.]+)\s+LS→new_basin=([\d.]+)"
    )
    re_rejected = re.compile(
        r"\[POP\]\s+add_solution:\s+REJECTED\s+.*?pop_size=(\d+)", re.IGNORECASE
    )
    # INSERTED has two formats: evicted (pop_size= single) or pop not full (pop_size= n→m)
    re_inserted_full = re.compile(
        r"\[POP\]\s+add_solution:\s+INSERTED\s+.*?cost=([\d.]+)\s+pop_size=(\d+)(?:→|->)(\d+)", re.IGNORECASE
    )
    re_inserted_evict = re.compile(
        r"\[POP\]\s+add_solution:\s+INSERTED\s+.*?cost=([\d.]+)\s+pop_size=(\d+)\s*$", re.IGNORECASE
    )
    re_inserted_evict_alt = re.compile(
        r"\[POP\]\s+add_solution:\s+INSERTED\s+.*?cost=([\d.]+)\s+pop_size=(\d+)\s"
    )
    re_replaced = re.compile(
        r"\[POP\]\s+add_solution:\s+REPLACED\s+similar\s+\(similar_cost=([\d.]+)\s*(?:→|->)\s*new_cost=([\d.]+)\).*?pop_size=(\d+)(?:→|->)(\d+)", re.IGNORECASE
    )

    current_pop: list[float] = []
    pending: dict | None = None  # round, step, basin_A, basin_B
    current_round = 0
    current_step = 0
    collecting_initial = False
    init_size = 0
    first_improving_seen = False

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[INIT] improving" in line:
                m = re_improving.search(line)
                if m:
                    init_size = int(m.group(1))
                    if not first_improving_seen:
                        initial_pop = []
                        current_pop = []
                        collecting_initial = True
                        first_improving_seen = True
                    else:
                        collecting_initial = False
            if collecting_initial and re_rejected_init.search(line):
                m = re_rejected_init.search(line)
                if m:
                    initial_pop.append(float(m.group(1)))
                    if len(initial_pop) >= init_size:
                        collecting_initial = False
                        current_pop = sorted(initial_pop)[:]
            if "[EVOLVE] === round=" in line:
                m = re_round.search(line)
                if m:
                    current_round = int(m.group(1))
                    pop_size = int(m.group(2))
                    if current_pop and len(current_pop) != pop_size:
                        current_pop = sorted(current_pop)[:pop_size]
                    elif not current_pop and initial_pop:
                        current_pop = sorted(initial_pop)[:]
                continue
            if "[EVOLVE] step=" in line and "parents:" in line:
                m = re_parents.search(line)
                if m:
                    if pending is not None:
                        # Next step started before we saw add_solution for this offspring → no descendant in parsed lineage
                        events.append({**pending, "action": "NO_DESCENDANT", "pop_after": list(current_pop)})
                    current_step = int(m.group(1))
                    basin_A = float(m.group(2))
                    basin_B = float(m.group(3))
                    similarity = float(m.group(4))
                    pending = {
                        "round": current_round,
                        "step": current_step,
                        "basin_A": basin_A,
                        "basin_B": basin_B,
                        "similarity": similarity,
                        "pop_before": list(current_pop),
                        "recombiner": None,
                    }
                continue
            if pending is not None:
                m_recomb = re_recombiner.search(line)
                if m_recomb:
                    pending["recombiner"] = m_recomb.group(1)
            if "[EVOLVE]" in line and "crossover→" in line and "new_basin=" in line:
                m = re_child.search(line)
                if m and pending is not None:
                    crossover_cost = float(m.group(1))
                    ls_cost = float(m.group(2))
                    pending["crossover_cost"] = crossover_cost
                    pending["ls_cost"] = ls_cost
                continue
            if pending is not None and "[POP]" in line and "add_solution:" in line:
                # Match specific outcomes first
                if "REJECTED" in line and "pop_size=" in line:
                    m = re_rejected.search(line)
                    if m:
                        pending["action"] = "REJECTED"
                        pending["pop_after"] = list(current_pop)
                        events.append(pending)
                        pending = None
                        continue
                if "INSERTED" in line:
                    m = re_inserted_full.search(line)
                    if m and pending is not None:
                        new_cost = float(m.group(1))
                        n_after = int(m.group(3))
                        new_pop = sorted(current_pop + [new_cost])[:n_after] if n_after else sorted(current_pop + [new_cost])
                        current_pop = new_pop
                        pending["action"] = "INSERTED"
                        pending["new_cost"] = new_cost
                        pending["pop_after"] = list(current_pop)
                        events.append(pending)
                        pending = None
                        continue
                    m = re_inserted_evict.search(line) or re_inserted_evict_alt.search(line)
                    if m and pending is not None:
                        new_cost = float(m.group(1))
                        n_after = int(m.group(2))
                        new_pop = sorted(current_pop + [new_cost])[:n_after] if n_after else sorted(current_pop + [new_cost])
                        current_pop = new_pop
                        pending["action"] = "INSERTED"
                        pending["new_cost"] = new_cost
                        pending["pop_after"] = list(current_pop)
                        events.append(pending)
                        pending = None
                        continue
                if "REPLACED" in line:
                    m = re_replaced.search(line)
                    if m and pending is not None:
                        similar_cost = float(m.group(1))
                        new_cost = float(m.group(2))
                        n_after = int(m.group(4))  # pop_size= n→m, m is new size
                        cand = [x for x in current_pop if abs(x - similar_cost) > 1e-6] + [new_cost]
                        current_pop = sorted(cand)[:n_after]
                        pending["action"] = "REPLACED"
                        pending["similar_cost"] = similar_cost
                        pending["new_cost"] = new_cost
                        pending["pop_after"] = list(current_pop)
                        events.append(pending)
                        pending = None
                        continue
                # Fallback: any add_solution line we didn't parse → close as LOGGED to avoid NO_DESCENDANT
                pending["action"] = "LOGGED"
                pending["pop_after"] = list(current_pop)
                events.append(pending)
                pending = None
                continue
            if "[EVOLVE]   recombine FAILED" in line or ("SKIPPED" in line and "[EVOLVE]" in line):
                if pending is not None:
                    pending["action"] = "FAILED"
                    pending["pop_after"] = list(current_pop)
                    events.append(pending)
                    pending = None

    if not initial_pop and events:
        initial_pop = events[0].get("pop_before", [])
    elif not initial_pop:
        initial_pop = current_pop

    return (initial_pop, events)


def cost_to_radius(cost: float, cost_min: float, cost_max: float, r_min: float = 8, r_max: float = 32) -> float:
    """Larger cost (worse) -> larger radius."""
    if cost_max <= cost_min:
        return (r_min + r_max) / 2
    t = (cost - cost_min) / (cost_max - cost_min)
    return r_min + t * (r_max - r_min)


def make_html(
    initial_pop: list[float],
    events: list[dict],
    out_path: str | Path,
    max_events: int = 200,
) -> None:
    """Generate interactive HTML: circles = solutions, size = cost."""
    out_path = Path(out_path)
    events = events[:max_events]
    all_costs = initial_pop + [c for e in events for c in e.get("pop_after", [])]
    if events:
        for e in events:
            all_costs.extend(e.get("pop_before", []))
            if "ls_cost" in e:
                all_costs.append(e["ls_cost"])
    cost_min = min(all_costs) if all_costs else 1500
    cost_max = max(all_costs) if all_costs else 1700

    frames = []
    # Frame 0: initial population
    frames.append({
        "title": "Initial population (after init)",
        "pop": list(initial_pop),
        "parents": None,
        "offspring": None,
        "new_cost": None,
        "action": None,
        "similarity": None,
        "recombiner": None,
    })
    for e in events:
        pop_after = e.get("pop_after", [])
        new_cost = e.get("new_cost") if e.get("action") in ("INSERTED", "REPLACED") else None
        sim = e.get("similarity")
        recomb = e.get("recombiner")
        crossover_cost = e.get("crossover_cost")
        ls_cost = e.get("ls_cost")
        similar_cost = e.get("similar_cost")
        action = e.get("action", "")
        if action == "REPLACED" and similar_cost is not None:
            outcome_text = f"REPLACED (replaced cost {similar_cost:.2f})"
        elif action == "NO_DESCENDANT":
            outcome_text = "(outcome not in log; may be in pop)"
        elif action == "LOGGED":
            outcome_text = "add_solution (parse fallback)"
        else:
            outcome_text = action
        title = f"Round {e.get('round', 0)} Step {e.get('step', 0)}"
        frames.append({
            "title": title,
            "pop": list(pop_after),
            "parents": (e.get("basin_A"), e.get("basin_B")) if e.get("basin_A") is not None else None,
            "crossover_cost": crossover_cost,
            "ls_cost": ls_cost,
            "new_cost": new_cost,
            "action": action,
            "similar_cost": similar_cost,
            "similarity": sim,
            "recombiner": recomb,
            "outcome_text": outcome_text,
        })

    data = {
        "cost_min": cost_min,
        "cost_max": cost_max,
        "frames": frames,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Population evolution – pedigree</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
    h1 {{ font-size: 1.2rem; }}
    #controls {{ margin-bottom: 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    #controls button {{ padding: 8px 14px; cursor: pointer; background: #16213e; color: #eee; border: 1px solid #0f3460; border-radius: 6px; }}
    #controls button:hover {{ background: #0f3460; }}
    #frameInfo {{ margin-bottom: 8px; color: #a0a0a0; font-size: 0.95rem; }}
    #legend {{ margin-top: 12px; font-size: 0.85rem; color: #888; }}
    #mainLayout {{ display: flex; gap: 16px; align-items: stretch; }}
    #leftPanel {{ flex-shrink: 0; width: 280px; background: #16213e; border-radius: 8px; padding: 12px; }}
    #leftPanel h3 {{ margin: 0 0 10px 0; font-size: 0.9rem; color: #aaa; }}
    #middlePanel {{ flex: 1; min-width: 380px; background: #16213e; border-radius: 8px; padding: 12px; overflow: auto; }}
    #middlePanel h3 {{ margin: 0 0 10px 0; font-size: 0.9rem; color: #aaa; }}
    #chartPanel {{ flex-shrink: 0; width: 220px; background: #16213e; border-radius: 8px; padding: 16px; display: flex; flex-direction: column; gap: 20px; }}
    #chartPanel h3 {{ margin: 0 0 10px 0; font-size: 0.9rem; color: #aaa; }}
    #outcomeSection {{ flex-shrink: 0; }}
    #recombinerSection {{ flex-shrink: 0; border-top: 1px solid #0f3460; padding-top: 12px; }}
    .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
    .bar-label {{ width: 72px; font-size: 11px; color: #ccc; }}
    .bar-track {{ flex: 1; height: 20px; background: #0f3460; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.25s ease-out; }}
    .bar-fill.reject {{ background: #e94560; }}
    .bar-fill.replace {{ background: #ffab00; }}
    .bar-fill.insert {{ background: #00c853; }}
    .bar-fill.recomb {{ background: #7c4dff; }}
    .bar-num {{ width: 36px; text-align: right; font-size: 12px; color: #eee; }}
    svg {{ background: #16213e; border-radius: 8px; }}
    .circle-pop {{ fill: #4a90d9; stroke: #2d5a87; stroke-width: 1.5; opacity: 0.9; }}
    .circle-pop.changed {{ fill: #e94560; stroke: #c73e54; stroke-width: 2; }}
    .circle-parent {{ fill: #e94560; stroke: #c73e54; stroke-width: 2; }}
    .circle-crossover {{ fill: #b8860b; stroke: #8b6914; stroke-width: 2; }}
    .circle-ls {{ fill: #00d9ff; stroke: #00a8cc; stroke-width: 2; }}
    .pedigree-line {{ stroke: #888; stroke-width: 1.5; fill: none; }}
    .pedigree-arrow {{ stroke: #00d9ff; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }}
    .outcome-reject {{ fill: #e94560; font-weight: bold; }}
    .outcome-insert {{ fill: #00c853; font-weight: bold; }}
    .outcome-replace {{ fill: #ffab00; font-weight: bold; }}
    .outcome-leaf {{ fill: #888; font-weight: normal; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Population evolution – pedigree (circle size = cost, larger = worse)</h1>
  <div id="frameInfo"></div>
  <div id="controls">
    <button id="btnPrev">Prev</button>
    <button id="btnPlay">Play</button>
    <button id="btnNext">Next</button>
    <span id="frameIdx">0 / 0</span>
    <input type="range" id="slider" min="0" max="0" value="0" style="width: 300px;">
  </div>
  <div id="mainLayout">
    <div id="leftPanel">
      <h3>Current population (cost ↑)</h3>
      <div id="leftSvg"></div>
    </div>
    <div id="middlePanel">
      <h3>Pedigree (current step)</h3>
      <div id="middleSvg"></div>
    </div>
    <div id="chartPanel">
      <div id="outcomeSection">
        <h3>Outcome (up to current step)</h3>
        <div class="bar-row"><span class="bar-label">REJECTED</span><div class="bar-track"><div id="barReject" class="bar-fill reject" style="width:0%"></div></div><span id="numReject" class="bar-num">0</span></div>
        <div class="bar-row"><span class="bar-label">REPLACED</span><div class="bar-track"><div id="barReplace" class="bar-fill replace" style="width:0%"></div></div><span id="numReplace" class="bar-num">0</span></div>
        <div class="bar-row"><span class="bar-label">INSERTED</span><div class="bar-track"><div id="barInsert" class="bar-fill insert" style="width:0%"></div></div><span id="numInsert" class="bar-num">0</span></div>
      </div>
      <div id="recombinerSection">
        <h3>Recombiner usage</h3>
        <div id="recombinerBars"></div>
      </div>
    </div>
  </div>
  <div id="legend">Red = parents; Gold = offspring (crossover cost); Cyan = after LS. Lines: parents → crossover → LS. Outcome: REJECTED / INSERTED / REPLACED.</div>

  <script>
    const data = {json.dumps(data)};
    const costMin = data.cost_min;
    const costMax = data.cost_max;
    const frames = data.frames;

    function radius(cost) {{
      const t = (cost - costMin) / (costMax - costMin);
      return 8 + t * 24;
    }}

    let currentFrame = 0;
    let playTimer = null;

    const leftW = 256;
    const leftH = 320;
    const midW = 520;
    const midH = 340;

    function renderLeftPopulation(frame, prevFrame) {{
      const pop = (frame.pop || []).slice().sort((a, b) => a - b);
      const prevSet = prevFrame ? new Set((prevFrame.pop || []).map(c => c.toFixed(2))) : null;
      const gap = 14;
      const margin = 10;
      const maxRowW = leftW - 2 * margin;
      const circles = pop.map(cost => ({{ cost, r: radius(cost), changed: prevSet !== null && !prevSet.has(cost.toFixed(2)) }}));
      const rows = [];
      let row = [];
      let rowW = 0;
      circles.forEach(c => {{
        const need = (row.length ? gap : 0) + 2 * c.r;
        if (row.length > 0 && rowW + need > maxRowW) {{
          rows.push(row);
          row = [];
          rowW = 0;
        }}
        row.push(c);
        rowW += (row.length > 1 ? gap : 0) + 2 * c.r;
      }});
      if (row.length) rows.push(row);
      const nRows = rows.length;
      const rowHeight = 76;
      const padY = Math.max(20, (leftH - nRows * rowHeight) / 2);
      function placeRow(row, rowIndex) {{
        if (row.length === 0) return '';
        const totalW = row.reduce((s, c) => s + 2 * c.r + gap, 0) - gap;
        const maxR = Math.max(...row.map(c => c.r));
        let x0 = leftW/2 - totalW/2;
        const y = padY + rowIndex * rowHeight + 24;
        const textY = y + maxR + 14;
        let s = '';
        row.forEach(c => {{
          x0 += c.r + gap/2;
          const cls = 'circle-pop' + (c.changed ? ' changed' : '');
          s += '<circle cx="' + x0 + '" cy="' + y + '" r="' + c.r + '" class="' + cls + '"/>';
          s += '<text x="' + x0 + '" y="' + textY + '" text-anchor="middle" fill="#e0e0e0" font-size="9">' + c.cost.toFixed(1) + '</text>';
          x0 += c.r + gap/2;
        }});
        return s;
      }}
      let svg = '<svg width="' + leftW + '" height="' + leftH + '" viewBox="0 0 ' + leftW + ' ' + leftH + '">';
      rows.forEach((r, i) => {{ svg += placeRow(r, i); }});
      svg += '</svg>';
      return svg;
    }}

    function renderMiddlePedigree(frame, w, h) {{
      const parents = frame.parents;
      const crossoverCost = frame.crossover_cost;
      const lsCost = frame.ls_cost;
      if (!parents || crossoverCost == null || lsCost == null) {{
        return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '"><text x="' + (w/2) + '" y="' + (h/2) + '" text-anchor="middle" fill="#888" font-size="14">Initial state — see left for population</text></svg>';
      }}
      const outcomeText = frame.outcome_text || frame.action || '';
      const similarity = frame.similarity;
      const recombiner = frame.recombiner || '';

      const cx = w / 2;
      const yParent = 80;
      const yCrossover = 170;
      const yLS = 260;
      const xA = cx - 160;
      const xB = cx + 160;
      const rA = radius(parents[0]);
      const rB = radius(parents[1]);
      const rC = radius(crossoverCost);
      const rL = radius(lsCost);

      let svg = '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">';
      svg += '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#00d9ff"/></marker></defs>';

      svg += '<line x1="' + (xA + rA*0.7) + '" y1="' + (yParent + rA) + '" x2="' + cx + '" y2="' + (yCrossover - rC) + '" class="pedigree-line"/>';
      svg += '<line x1="' + (xB - rB*0.7) + '" y1="' + (yParent + rB) + '" x2="' + cx + '" y2="' + (yCrossover - rC) + '" class="pedigree-line"/>';
      svg += '<line x1="' + cx + '" y1="' + (yCrossover + rC) + '" x2="' + cx + '" y2="' + (yLS - rL) + '" class="pedigree-arrow"/>';

      svg += '<circle cx="' + xA + '" cy="' + yParent + '" r="' + rA + '" class="circle-parent"/>';
      svg += '<text x="' + xA + '" y="' + (yParent + 4) + '" text-anchor="middle" fill="#fff" font-size="10">' + parents[0].toFixed(1) + '</text>';
      svg += '<circle cx="' + xB + '" cy="' + yParent + '" r="' + rB + '" class="circle-parent"/>';
      svg += '<text x="' + xB + '" y="' + (yParent + 4) + '" text-anchor="middle" fill="#fff" font-size="10">' + parents[1].toFixed(1) + '</text>';

      svg += '<circle cx="' + cx + '" cy="' + yCrossover + '" r="' + rC + '" class="circle-crossover"/>';
      svg += '<text x="' + cx + '" y="' + (yCrossover + 4) + '" text-anchor="middle" fill="#fff" font-size="10">' + crossoverCost.toFixed(1) + '</text>';
      let label = '';
      if (similarity != null) label += 'similarity=' + similarity.toFixed(4);
      if (recombiner) label += (label ? '  ' : '') + 'recombiner=' + recombiner;
      if (label) svg += '<text x="' + cx + '" y="' + (yCrossover + rC + 18) + '" text-anchor="middle" fill="#b8860b" font-size="11">' + label + '</text>';

      svg += '<circle cx="' + cx + '" cy="' + yLS + '" r="' + rL + '" class="circle-ls"/>';
      svg += '<text x="' + cx + '" y="' + (yLS + 4) + '" text-anchor="middle" fill="#fff" font-size="10">' + lsCost.toFixed(1) + '</text>';

      const outcomeClass = (frame.action === 'REJECTED') ? 'outcome-reject' : (frame.action === 'INSERTED') ? 'outcome-insert' : (frame.action === 'REPLACED') ? 'outcome-replace' : (frame.action === 'FAILED' || frame.action === 'LOGGED') ? 'outcome-leaf' : 'outcome-leaf';
      svg += '<text x="' + cx + '" y="' + (yLS + rL + 28) + '" text-anchor="middle" font-size="14" class="' + outcomeClass + '">' + outcomeText + '</text>';

      svg += '</svg>';
      return svg;
    }}

    function updateBarChart() {{
      let reject = 0, replace = 0, insert = 0;
      const recombCounts = {{}};
      for (let i = 1; i <= currentFrame && i < frames.length; i++) {{
        const f = frames[i];
        const a = f.action;
        if (a === 'REJECTED') reject++;
        else if (a === 'REPLACED') replace++;
        else if (a === 'INSERTED') insert++;
        const r = f.recombiner;
        if (r) {{ recombCounts[r] = (recombCounts[r] || 0) + 1; }}
      }}
      const maxVal = Math.max(1, reject, replace, insert);
      const pct = (v) => (v / maxVal * 100).toFixed(1);
      document.getElementById("barReject").style.width = pct(reject) + '%';
      document.getElementById("barReplace").style.width = pct(replace) + '%';
      document.getElementById("barInsert").style.width = pct(insert) + '%';
      document.getElementById("numReject").textContent = reject;
      document.getElementById("numReplace").textContent = replace;
      document.getElementById("numInsert").textContent = insert;

      const names = Object.keys(recombCounts).sort();
      const maxR = Math.max(1, ...names.map(n => recombCounts[n]));
      const pctR = (v) => (v / maxR * 100).toFixed(1);
      let html = '';
      names.forEach(n => {{
        const v = recombCounts[n];
        html += '<div class="bar-row"><span class="bar-label">' + n + '</span><div class="bar-track"><div class="bar-fill recomb" style="width:' + pctR(v) + '%"></div></div><span class="bar-num">' + v + '</span></div>';
      }});
      if (names.length === 0) html = '<div style="color:#666;font-size:11px">(none yet)</div>';
      document.getElementById("recombinerBars").innerHTML = html;
    }}

    function render() {{
      const frame = frames[currentFrame];
      if (!frame) return;
      document.getElementById("frameIdx").textContent = (currentFrame + 1) + " / " + frames.length;
      document.getElementById("slider").value = currentFrame;
      document.getElementById("frameInfo").textContent = frame.title || ("Frame " + (currentFrame + 1));

      updateBarChart();

      const prevFrame = currentFrame > 0 ? frames[currentFrame - 1] : null;
      document.getElementById("leftSvg").innerHTML = renderLeftPopulation(frame, prevFrame);
      document.getElementById("middleSvg").innerHTML = renderMiddlePedigree(frame, midW, midH);
    }}

    document.getElementById("btnPrev").onclick = () => {{ currentFrame = Math.max(0, currentFrame - 1); render(); }};
    document.getElementById("btnNext").onclick = () => {{ currentFrame = Math.min(frames.length - 1, currentFrame + 1); render(); }};
    document.getElementById("slider").oninput = (e) => {{ currentFrame = parseInt(e.target.value, 10); render(); }};
    document.getElementById("slider").max = Math.max(0, frames.length - 1);

    document.getElementById("btnPlay").onclick = () => {{
      if (playTimer) {{ clearInterval(playTimer); playTimer = null; document.getElementById("btnPlay").textContent = "Play"; return; }}
      document.getElementById("btnPlay").textContent = "Pause";
      playTimer = setInterval(() => {{
        currentFrame = (currentFrame + 1) % frames.length;
        render();
        if (currentFrame === 0) {{ clearInterval(playTimer); playTimer = null; document.getElementById("btnPlay").textContent = "Play"; }}
      }}, 700);
    }};

    render();
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def make_gif(
    initial_pop: list[float],
    events: list[dict],
    out_path: str | Path,
    max_events: int = 100,
    duration_per_frame: float = 0.5,
) -> None:
    """Generate GIF using matplotlib (optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available; skipping GIF. Install with: pip install matplotlib")
        return

    out_path = Path(out_path)
    events = events[:max_events]
    all_costs = initial_pop + [c for e in events for c in e.get("pop_after", [])]
    for e in events:
        all_costs.extend(e.get("pop_before", []))
        if "ls_cost" in e:
            all_costs.append(e["ls_cost"])
    cost_min = min(all_costs) if all_costs else 1500
    cost_max = max(all_costs) if all_costs else 1700

    def radius(cost: float) -> float:
        """Larger cost (worse) -> larger radius."""
        t = (cost - cost_min) / (cost_max - cost_min) if cost_max > cost_min else 0.5
        return 8 + t * 24

    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("imageio not available; skipping GIF. Install with: pip install imageio")
            return
    from io import BytesIO

    frames_list = [{"title": "Initial population", "pop": list(initial_pop), "parents": None, "new_cost": None, "similarity": None, "recombiner": None}]
    for e in events:
        sim = e.get("similarity")
        recomb = e.get("recombiner")
        title = f"R{e.get('round',0)} S{e.get('step',0)} {e.get('action','')}"
        if recomb:
            title += f"  recombiner={recomb}"
        if sim is not None:
            title += f"  similarity={sim:.4f}"
        frames_list.append({
            "title": title,
            "pop": list(e.get("pop_after", [])),
            "parents": (e.get("basin_A"), e.get("basin_B")) if e.get("basin_A") is not None else None,
            "new_cost": e.get("new_cost") if e.get("action") in ("INSERTED", "REPLACED") else None,
            "similarity": sim,
            "recombiner": recomb,
        })

    images = []
    for fi, frame in enumerate(frames_list):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_facecolor("#16213e")
        fig.patch.set_facecolor("#1a1a2e")
        pop = frame["pop"]
        parents = frame.get("parents")
        new_cost = frame.get("new_cost")
        circles = []
        for c in pop:
            is_parent = parents and (abs(c - parents[0]) < 0.1 or abs(c - parents[1]) < 0.1)
            is_new = new_cost is not None and abs(c - new_cost) < 0.01
            circles.append({"cost": c, "r": radius(c), "is_parent": is_parent, "is_new": is_new})
        circles.sort(key=lambda x: x["cost"])
        x = 8
        for c in circles:
            x += c["r"] + 5
            color = "#e94560" if c["is_parent"] else "#00d9ff" if c["is_new"] else "#4a90d9"
            circ = mpatches.Circle((x, 2), c["r"], facecolor=color, edgecolor="#2d5a87", linewidth=1.5)
            ax.add_patch(circ)
            ax.text(x, 2, f"{c['cost']:.1f}", ha="center", va="center", fontsize=8, color="white")
            x += c["r"] + 5
        ax.set_xlim(0, x + 20)
        ax.set_ylim(-5, 9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(frame.get("title", f"Frame {fi+1}"), color="white", fontsize=10)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=80, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()

    if not images:
        return
    imageio.mimsave(out_path, images, duration=duration_per_frame, loop=0)
    print(f"Wrote {out_path}")


def plot_similarity_over_time(
    events: list[dict],
    out_path: str | Path,
    max_events: int | None = None,
) -> None:
    """Extract parent similarity from events and plot as line chart vs step index t."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping similarity plot. Install with: pip install matplotlib")
        return

    out_path = Path(out_path)
    ev = events if max_events is None else events[:max_events]
    t_vals: list[int] = []
    sim_vals: list[float] = []
    for i, e in enumerate(ev):
        sim = e.get("similarity")
        if sim is not None:
            t_vals.append(i)
            sim_vals.append(sim)

    if not t_vals:
        print("No parent similarity data found in events; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_vals, sim_vals, color="#4a90d9", linewidth=1.2, marker=".", markersize=3)
    ax.set_xlabel("Step t (event index)")
    ax.set_ylabel("Parent similarity")
    ax.set_title("Parent similarity over time")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    log_path = Path(__file__).parent / "cvrp100_uniform_3_tl_600_20260224_041411" / "baseline_run.log"
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        sys.exit(1)

    out_dir = log_path.parent
    initial_pop, events = parse_baseline_log(log_path)
    print(f"Initial population size: {len(initial_pop)}, events: {len(events)}")
    if initial_pop:
        print(f"Initial costs (sorted): {sorted(initial_pop)}")

    make_html(initial_pop, events, out_dir / "population_evolution.html", max_events=-1)
    plot_similarity_over_time(events, out_dir / "parent_similarity_vs_t.png", max_events=None)
    # make_gif(initial_pop, events, out_dir / "population_evolution.gif", max_events=80, duration_per_frame=0.45)


if __name__ == "__main__":
    main()
