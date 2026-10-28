"""
make_report.py  --  the results tables, built from a run folder's JSON files.

Every script writes what it printed to <run>/results/*.json (core/results.py).
This turns them into RESULTS.md inside the run folder and, with
--update-readme, refreshes the marked blocks in README.md and fleet/README.md:

    <!-- results:phase1_models -->
    ...generated, do not edit by hand...
    <!-- /results:phase1_models -->

The prose around a block is written by a person against one run; the block
itself is always the numbers of the run it names.

    python run.py report                               # latest run in runs/
    python run.py report --run runs/2026-10-28_0400 --update-readme
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import config
from core import results

README = config.ROOT_DIR / "README.md"
FLEET_README = config.ROOT_DIR / "fleet" / "README.md"


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #
def f(x, fmt="{:.2f}", none="–"):
    return none if x is None else fmt.format(x)


def pct(x, signed=True, digits=1):
    if x is None:
        return "–"
    if abs(x) < 0.5 * 10 ** -digits:
        x = 0.0                                   # no "-0.0%" from float noise
    return f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


class Report:
    def __init__(self, run: Path, base: Path):
        self.run = run
        self.base = base                 # where the markdown will live (for image links)
        self.R = results.load_all(run)

    def img(self, name, alt, sub=""):
        hits = sorted(self.run.rglob(name))
        if not hits:
            return ""
        path = Path(os.path.relpath(hits[0], self.base)).as_posix()
        alt = alt.replace("*", "").split(" — ")[0]
        return f"\n\n![{alt}]({path}){sub}\n"       # blank line: else GFM reads it as a table row

    def get(self, prefix):
        return {k: v for k, v in self.R.items() if k.startswith(prefix)}

    # ----------------------------------------------------------------------- #
    # Sections                                                                 #
    # ----------------------------------------------------------------------- #
    def phase1_models(self):
        d = self.R.get("phase1")
        if not d:
            return None
        rows = [[("**" + m["Model"] + "**") if m["Model"] == d["best_model"] else m["Model"],
                 f(m["CV_R2_mean"], "{:.3f}"), f(m["Test_R2"], "{:.3f}"),
                 f(m["Test_MAE"]), f(m["Test_RMSE"]), pct(m["Test_MAPE%"], False),
                 pct(m.get("Regret_mean%"), False, 2), pct(m.get("Regret_max%"), False, 2)]
                for m in d["models"]]
        s = table(["Model", "CV R²", "Test R²", "MAE", "RMSE", "MAPE",
                   "Regret mean", "Regret max"], rows)
        s += (f"\n\n{d['rows']:,} rows, 20% held-out test + 5-fold CV. Regret = extra "
              "TRUE fuel burned at the speed each model recommends, over 12 voyages "
              f"(`optimizer.regret_panel`). Saved model: **{d['best_model']}** (best test R²).")
        if d.get("physics"):
            s += ("\nFitted physics: " + ", ".join(f"{k} = {v:.4g}"
                                                   for k, v in d["physics"].items()) + ".")
        return s

    def phase1_speed(self):
        d = self.R.get("phase1")
        if not d:
            return None
        rows = []
        for title, r in d["voyages"].items():
            if not r.get("feasible"):
                rows.append([title, "infeasible", "", "", "", "", ""])
                continue
            rows.append([title, f"{r['opt_speed']:.2f} kn", f"{r['opt_fuel']:.1f} t",
                         pct(-r["fuel_saving_pct"]), f"{r['co2_saved_t']:.1f} t",
                         f"{r['true_opt_speed']:.2f} kn / {r['true_opt_fuel']:.1f} t",
                         pct(r["regret_pct"], digits=2)])
        s = table(["Scenario", "Optimal speed", "Fuel (model)", "vs baseline",
                   "CO₂ saved", "True optimum", "Decision regret"], rows)
        return s + self.img(d["plot"], "Total voyage fuel vs speed")

    def speed_qubo(self):
        runs = self.get("speed_qubo_")
        if not runs:
            return None
        rows = [[f"{d['deadline_h']:.0f} h", d["levels"],
                 f"{d['qubo']['speed']:.2f} kn / {d['qubo']['fuel']:.2f} t",
                 f"{d['sa_speed']:.2f} kn / {d['sa_fuel']:.2f} t",
                 f"{d['grid_speed']:.2f} kn / {d['grid_fuel']:.2f} t",
                 "yes" if d["qubo"]["valid_one_hot"] else "NO"]
                for d in runs.values()]
        return table(["Deadline", "Levels", "QUBO (neal)", "Simulated annealing",
                      "Grid (exact)", "One-hot valid"], rows)

    def route_qubo(self):
        runs = self.get("route_qubo_")
        if not runs:
            return None
        rows = [[d["grid"], d["storms"] or "1 centred", d["seed"],
                 "yes" if d["presolve"] else "no", d["n_vars"], f(d["naive_fuel"]),
                 f(d["qubo_fuel"]) if d["valid"] else "invalid", f(d["dp_fuel"]),
                 pct(d["saving_pct"]), pct(d["gap_pct"], digits=2),
                 pct(d["regret_pct"], digits=2)]
                for d in sorted(runs.values(), key=lambda d: (d["n_vars"], d["presolve"]))]
        s = table(["Grid", "Storms", "Seed", "Presolve", "Vars", "Naive (t)", "QUBO (t)",
                   "Exact DP (t)", "Saved vs naive", "QUBO vs exact", "Decision regret"], rows)
        first = next((d for d in runs.values() if d["grid"] == "5x7" and not d["presolve"]), None)
        return s + (self.img(first["plot"], "Weather routing QUBO") if first else "")

    def quantum_gate(self):
        runs = self.get("compare_")
        if not runs:
            return None
        parts = []
        for name, d in sorted(runs.items()):
            rows = [[r["label"], f(r["best_fuel"]),
                     "–" if r["label"].startswith("Exact") else pct(r["gap"]),
                     f"{r['feasible']}/{r['trials']}", f"{r['successes']}/{r['trials']}",
                     f"{r['avg_time']:.2f} s"] for r in d["records"]]
            parts.append(f"**{d['title']}** — {d['n_qubits']} qubits, Aer on {d['device']}, "
                         f"{d['trials']} trials\n\n"
                         + table(["Method", "Best fuel (t)", "Gap", "Feasible", "Hit optimum",
                                  "Time/run"], rows)
                         + self.img(d["plot"], d["title"]))
        return "\n\n".join(parts)

    def audit(self):
        d = self.R.get("audit_seeds")
        if not d:
            return None
        rows = [[r["solver"], f"{r['hits']}/{r['n']}",
                 f"{r['hit_ci'][0]:.0%}–{r['hit_ci'][1]:.0%}", f"{r['feasible']}/{r['n']}",
                 pct(r["mean_gap"]), pct(r["worst_gap"])] for r in d["rows"]]
        return (f"3×4 route QUBO, exact optimum {d['exact_fuel']:.2f} t, {d['seeds']} seeds "
                f"per solver, every run counted. CVaR α = {d['cvar_alpha']}.\n\n"
                + table(["Solver", "Hit optimum", "95% CI", "Feasible", "Mean gap",
                         "Worst gap"], rows))

    def benchmark(self):
        parts = []
        for tag, title in (("", "raw QUBO"), ("_presolve", "presolved QUBO"),
                           ("_gate", "gate-model, presolved")):
            d = self.R.get(f"benchmark{tag}")
            if not d:
                continue
            solvers = [s for s in d["solvers"] if any(s in c for c in d["size"])]
            rows = []
            for cell in d["size"]:
                row = [cell["grid"], cell["n_vars"]]
                for s in solvers:
                    r = cell.get(s)
                    row.append("–" if not r else
                               f"{r['opt_pct']:.0f}% / {pct(r['mean_gap']) if r['n_feasible'] else 'infeasible'}")
                rows.append(row)
            parts.append(f"**Study A, {title}** ({d['instances']} instances × {d['seeds']} "
                         f"seeds; cell = % optimal / mean gap; intervals and TTS99 in "
                         f"`BENCHMARK{tag}.md`)\n\n"
                         + table(["Grid", "Vars", *solvers], rows)
                         + self.img(f"bench_size{tag}.png", f"Reliability vs size ({title})"))
            if d["penalty"]:
                prow = [[f"{pm}×", *[f"{d['penalty'][pm][s]['opt_pct']:.0f}% / "
                                     f"{pct(d['penalty'][pm][s]['mean_gap'])}"
                                     for s in solvers if s in d["penalty"][pm]]]
                        for pm in d["penalty"]]
                parts.append(f"**Study B, {title}** — penalty weight on 4×6\n\n"
                             + table(["Penalty", *[s for s in solvers
                                                   if s in next(iter(d['penalty'].values()))]],
                                     prow)
                             + self.img(f"bench_penalty{tag}.png", f"Penalty sensitivity ({title})"))
        return "\n\n".join(parts) or None

    def speed_compare(self):
        runs = self.get("speed_compare_")
        if not runs:
            return None
        parts = []
        for name, d in sorted(runs.items(), key=lambda kv: (kv[1]["legs"], kv[1]["deadline_h"])):
            rows = [[r["method"], f(r["fuel"]), f(r["hours"], "{:.1f}"), pct(r["vs_ref_pct"]),
                     f"{r['seconds']:.1f} s", " ".join(f"{v:.1f}" for v in r["speeds"])]
                    for r in d["rows"]]
            head = (f"**{d['distance_nm']:.0f} nm, {d['legs']} leg(s), deadline "
                    f"{d['deadline_h']:.0f} h** — {d['evals']:,} evaluations per optimizer")
            if d.get("lower_bound") is not None:
                head += (f"; reference {d['reference_fuel']:.2f} t, lower bound "
                         f"{d['lower_bound']:.2f} t")
            parts.append(head + "\n\n" + table(["Method", "Fuel (t)", "Hours", "vs ref",
                                                "Time", "Speed per leg (kn)"], rows)
                         + self.img(d["plot"], head.strip("*")))
        return "\n\n".join(parts)

    def joint(self):
        runs = self.get("joint_")
        if not runs:
            return None
        rows = [[d["grid"], " ".join(f"{v:.0f}" for v in d["levels"]), d["n_vars"],
                 f"{d['deadline_h']:.1f} h", f(d["naive_fuel"]), f(d["qubo_fuel"]),
                 f(d["exact_fuel"]),
                 pct((d["qubo_fuel"] / d["exact_fuel"] - 1) * 100
                     if d["qubo_fuel"] and d["exact_fuel"] else None, digits=2),
                 d["qubo_solves"]] for d in runs.values()]
        first = next(iter(runs.values()))
        return (table(["Grid", "Speeds (kn)", "Vars", "Deadline", "Naive (t)",
                       "QUBO + price (t)", "Exact (t)", "Gap", "QUBO solves"], rows)
                + self.img(first["plot"], "Joint speed + route"))

    def fleet_benchmark(self, tag=""):
        d = self.R.get(f"fleet_benchmark{tag}")
        if not d or not d["n_valid"]:
            return None
        n = d["n_valid"]
        rows = []
        for s in d["solvers"]:
            r = d["summary"][s]
            ci = ("" if r["mean_gap"] is None
                  else f"[{r['gap_ci'][0]:+.1f}, {r['gap_ci'][1]:+.1f}]")
            rows.append([s, pct(r["mean_gap"]), ci, f(r["std_gap"], "{:.1f}"),
                         f"{r['optimal']}/{n}", f"{r['opt_ci'][0]:.0%}–{r['opt_ci'][1]:.0%}",
                         r["failed"], f"{r['mean_seconds']:.2f} s"])
        s = (f"{d['ships']} ships × {d['missions']} missions, {n} feasible of {d['seeds']} "
             f"instances (horizon {d['horizon']} slots). Budgets: {d['iters']:,} iterations "
             f"(SA/ALNS), {d['num_reads']:,} neal reads per QUBO solve, {d['budget']:,} "
             f"distinct exact scorings (rand+exact / ga / hybrid).\n\n"
             + table(["Solver", "Mean gap", "95% CI", "Std", "Optimal", "95% CI",
                      "Failed", "Time"], rows))
        any_rows = [[s, *[f"{pct(v['mean_gap'])} ({v['optimal']}/{n})"
                          for v in d["anytime"][s].values()]] for s in d["anytime"]]
        s += ("\n\nAt equal numbers of exact scorings (mean gap, optimal count):\n\n"
              + table(["Proposals from", *[f"@{b}" for b in next(iter(d['anytime'].values()))]],
                      any_rows))
        t_rows = [[k, f(t["mean_diff"], "{:+.2f}"), f(t["wilcoxon_p"], "{:.3f}"),
                   f"{t['only_a_opt']} / {t['only_b_opt']}", f(t["mcnemar_p"], "{:.3f}")]
                  for k, t in d["paired_tests"].items()]
        s += ("\n\nPaired tests on the same instances (p < 0.05 = a difference the data "
              "supports):\n\n" + table(["A vs B", "Mean gap diff (pts)", "Wilcoxon p",
                                        "Optimal only A / only B", "McNemar p"], t_rows))
        lines = []
        for inst in d["instances"]:
            if not inst["feasible"]:
                lines.append(f"seed {inst['seed']}: infeasible (excluded)")
                continue
            lines.append(f"seed {inst['seed']}: exact {inst['exact']:.2f}t | " + "  ".join(
                f"{k} " + ("FAIL" if inst[k]["gap"] is None else f"{inst[k]['gap']:+.1f}%")
                for k in d["solvers"]))
        s += "\n\nEvery instance:\n\n```\n" + "\n".join(lines) + "\n```"
        s += self.img(d["plots"][0], "Fleet benchmark") + self.img(d["plots"][1], "Anytime")
        return s

    def fleet_scale(self):
        d = self.R.get("fleet_scale")
        if not d:
            return None
        key = ["greedy", "alns_fixed", "alns", "rand+exact", "ga", "hybrid_v1", "hybrid", "hybrid+alns"]
        rows = []
        for z in d["sizes"]:
            if not z["n_valid"]:
                rows.append([f"{z['ships']}×{z['missions']}", f"0/{z['seeds']}",
                             *["–"] * len(key)])
                continue
            rows.append([f"{z['ships']}×{z['missions']}", f"{z['n_valid']}/{z['seeds']}",
                         *[f"{pct(z['summary'][s]['mean_gap'])} ({z['summary'][s]['optimal']})"
                           for s in key]])
        s = ("Cell = mean gap (instances solved to optimality); exact subset-DP ground "
             "truth at every size.\n\n"
             + table(["Size", "Feasible", *key], rows)
             + self.img(d["plot"], "Fleet scale study"))

        # the per-size benchmark files hold the anytime curves and the paired tests
        per = []
        for z in d["sizes"]:
            b = (self.R.get(f"fleet_benchmark_{z['ships']}x{z['missions']}")
                 or self.R.get("fleet_benchmark"))
            if z["n_valid"] and b and (b["ships"], b["missions"]) == (z["ships"], z["missions"]):
                per.append((f"{z['ships']}×{z['missions']}", b))
        if not per:
            return s
        anytime = list(per[0][1]["anytime"])
        a_rows = [[size, f"@{n}", *[f"{pct(b['anytime'][m][n]['mean_gap'])} "
                                    f"({b['anytime'][m][n]['optimal']})" for m in anytime]]
                  for size, b in per for n in b["anytime"][anytime[0]]]
        s += ("\n\nThe proposal methods at equal numbers of distinct exact scorings "
              "(mean gap, instances solved to optimality):\n\n"
              + table(["Size", "Scorings", *anytime], a_rows))
        pairs = [k for k in per[0][1]["paired_tests"] if "@" not in k]
        t_rows = []
        for size, b in per:
            cells = []
            for k in pairs:
                t = b["paired_tests"].get(k)
                cells.append("–" if not t or t["mean_diff"] is None else
                             f"{t['mean_diff']:+.1f} (p={t['wilcoxon_p']:.3f}; "
                             f"{t['only_a_opt']}/{t['only_b_opt']})")
            t_rows.append([size, *cells])
        s += ("\n\nPaired tests at the final budget, per size. Cell = mean gap difference "
              "A − B in points (Wilcoxon p; instances only A / only B solved to "
              "optimality). Negative = A better; p < 0.05 = a difference the data "
              "supports.\n\n" + table(["Size", *pairs], t_rows))
        return s

    def fleet_plans(self):
        runs = self.get("fleet_plan_seed")
        if not runs:
            return None
        parts = []
        for name, d in sorted(runs.items()):
            rows = [[r["solver"], f(r["fuel"]),
                     pct((r["fuel"] / d["exact"] - 1) * 100 if r["fuel"] and d["exact"] else None),
                     f"{r['seconds']:.2f} s"] for r in d["rows"]]
            plan = "; ".join(f"{k}: {' → '.join(v) or 'idle'}" for k, v in d["alns_plan"].items())
            parts.append(f"**Seed {d['seed']}** ({d['ships']} ships × {d['missions']} missions)"
                         f" — ALNS plan: {plan}\n\n"
                         + table(["Solver", "Fuel (t)", "Gap", "Time"], rows)
                         + self.img(d["plot"], f"Fleet plan seed {d['seed']}"))
        return "\n\n".join(parts)

    def fleet_qaoa(self):
        runs = self.get("fleet_qaoa_")
        if not runs:
            return None
        def backend(b):
            return ("Aer GPU" if "GPU" in b else "Aer CPU") if b.startswith("Aer") else b

        rows = [[f"{d['ships']}×{d['missions']}", d["n_qubits"], backend(d["backend"]),
                 "CVaR " + str(d["cvar"]) if d["cvar"] else "mean", d["reps"],
                 f"{d.get('assignments_scored', '–')} of {d['ships'] ** d['missions']:,}",
                 f(d["qaoa_fuel"]), f(d["neal_hybrid_fuel"]), f(d["exact_fuel"]),
                 pct(d["gap_pct"]), f"{d['seconds']:.1f} s"]
                for d in sorted(runs.values(), key=lambda d: (d["n_qubits"], d["cvar"] or 0))]
        return table(["Size", "Qubits", "Simulator", "Objective", "Reps",
                      "Assignments scored", "QAOA hybrid (t)", "neal hybrid v1 (t)",
                      "Exact (t)", "QAOA gap", "Time"], rows)

    def manifest(self):
        p = self.run / "manifest.json"
        if not p.exists():
            return None
        m = json.loads(p.read_text(encoding="utf-8"))
        s = (f"Run `{self.run.name}` — started {m['started']} on {m['host']} "
             f"({m['cpu']}, {m['cores']} cores; GPU: {m['gpu'] or 'none'}), "
             f"FUEL_JOBS={m['env'].get('FUEL_JOBS')}. Packages: "
             + ", ".join(f"{k} {v}" for k, v in m["packages"].items()) + ".")
        tp = self.run / "timings.tsv"
        if tp.exists():
            steps = [ln.split("\t") for ln in tp.read_text().splitlines() if ln.strip()]
            total = sum(int(t[1]) for t in steps if len(t) > 1)
            failed = [t[0] for t in steps if len(t) > 2 and t[2] != "0"]
            s += (f" {len(steps)} steps, {total / 3600:.1f} h in total"
                  + (f"; FAILED: {', '.join(failed)}" if failed else "; all steps succeeded")
                  + ".")
        return s


SECTIONS = {   # block name -> (method, heading in RESULTS.md)
    "run_info": ("manifest", "Run"),
    "phase1_models": ("phase1_models", "Phase 1 — fuel-rate models"),
    "phase1_speed": ("phase1_speed", "Phase 1 — slow-steaming speed"),
    "speed_qubo": ("speed_qubo", "Phase 2a — speed as a QUBO"),
    "route_qubo": ("route_qubo", "Phase 2b — weather routing as a QUBO"),
    "quantum_gate": ("quantum_gate", "Phase 2c — gate-model QAOA / VQE"),
    "audit": ("audit", "Seed audit"),
    "benchmark": ("benchmark", "Route-QUBO solver benchmark"),
    "speed_compare": ("speed_compare", "SA vs GA vs GSA on speed"),
    "joint": ("joint", "Joint speed + route QUBO"),
    "fleet_benchmark": ("fleet_benchmark", "Fleet benchmark (3 × 6)"),
    "fleet_scale": ("fleet_scale", "Fleet scale study"),
    "fleet_plans": ("fleet_plans", "Fleet demo plans"),
    "fleet_qaoa": ("fleet_qaoa", "Fleet assignment on QAOA"),
}


def render(rep, name):
    return getattr(rep, SECTIONS[name][0])()


def update_readme(path: Path, run: Path) -> list[str]:
    """Replace every <!-- results:NAME --> block in `path`; returns the names done."""
    rep = Report(run, path.parent)
    text = path.read_text(encoding="utf-8")
    done = []

    def sub(m):
        name = m.group(1)
        if name not in SECTIONS:
            return m.group(0)
        body = render(rep, name)
        if body is None:
            return m.group(0)
        done.append(name)
        return (f"<!-- results:{name} -->\n<!-- generated by tools/make_report.py from "
                f"{Path(os.path.relpath(run, path.parent)).as_posix()} -- do not edit "
                f"by hand -->\n"
                f"{body.strip()}\n<!-- /results:{name} -->")

    text = re.sub(r"<!-- results:(\w+) -->.*?<!-- /results:\1 -->", sub, text, flags=re.S)
    path.write_text(text, encoding="utf-8")
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Results tables from a run folder.")
    p.add_argument("--run", default=None, help="run folder (default: latest in runs/)")
    p.add_argument("--update-readme", action="store_true",
                   help="also refresh the marked blocks in README.md and fleet/README.md")
    args = p.parse_args()
    if args.run:
        run = Path(args.run).resolve()
    else:
        runs = sorted((config.ROOT_DIR / "runs").glob("*/"))
        if not runs:
            raise SystemExit("no runs/ folder yet -- run run_all.sh first, or pass --run")
        run = runs[-1].resolve()

    rep = Report(run, run)
    out = [f"# Results — {run.name}\n",
           "Generated by `tools/make_report.py` from the JSON files in this folder.\n"]
    for name, (_, heading) in SECTIONS.items():
        body = render(rep, name)
        if body:
            out.append(f"## {heading}\n\n{body.strip()}\n")
    (run / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print(f"RESULTS.md -> {run / 'RESULTS.md'}  ({len(rep.R)} result files)")
    if args.update_readme:
        for path in (README, FLEET_README):
            done = update_readme(path, run)
            print(f"{path.relative_to(config.ROOT_DIR)}: refreshed {', '.join(done) or 'nothing'}")


if __name__ == "__main__":
    main()
