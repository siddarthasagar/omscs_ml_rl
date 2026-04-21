"""Phase 8 — Report Tables, Macros, and Repro Artifacts.

Reads all upstream phase checkpoints and writes:

  artifacts/tables/
    tab_bj_dp.tex          — VI / PI Blackjack results
    tab_cp_dp.tex          — VI / PI CartPole results (per grid)
    tab_bj_mf.tex          — SARSA / Q-Learning Blackjack (controlled vs tuned)
    tab_cp_mf.tex          — SARSA / Q-Learning CartPole (controlled vs tuned)
    tab_cp_disc.tex        — CartPole discretization study
    tab_dqn_ec.tex         — DQN extra credit (written only if phase7.json exists)
    report_numbers.tex     — \\newcommand macros for every inline number

  artifacts/repro/
    runbook.md             — exact reproduction commands and expected outputs
    submission_checklist.md — deliverable tracker

  artifacts/metadata/phase8.json — phase checkpoint

All table bodies are bare tabulars — no \\begin{table} wrapper — so that the
report does \\input{tables/tab_bj_dp} at each table site.

Upstream inputs:
    artifacts/metadata/phase1.json
    artifacts/metadata/phase2.json
    artifacts/metadata/phase3.json
    artifacts/metadata/phase4.json
    artifacts/metadata/phase5.json
    artifacts/metadata/phase6.json
    artifacts/metadata/phase7.json  (optional — DQN extra credit)

Usage:
    make phase8
"""

from __future__ import annotations

import time
from pathlib import Path

from src.config import (
    ARTIFACTS_DIR,
    METADATA_DIR,
    SEEDS,
)
from src.utils.logger import configure_logger
from src.utils.phase_artifacts import (
    SCHEMA_VERSION,
    load_checkpoint_json,
    write_checkpoint_json,
)

# ── Directory constants ───────────────────────────────────────────────────────

TABLES_DIR = ARTIFACTS_DIR / "tables"
REPRO_DIR = ARTIFACTS_DIR / "repro"


# ── Formatting helpers ────────────────────────────────────────────────────────


def _fmt(val: float | None, decimals: int = 2) -> str:
    """Format a float or None for LaTeX output."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def _fmt_int(val: float | int | None) -> str:
    """Format an integer-valued float (e.g. convergence episode) with commas."""
    if val is None:
        return "--"
    return f"{int(val):,}"


def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    """Format a fraction (0–1) as a percentage string."""
    if val is None:
        return "--"
    return f"{val * 100:.{decimals}f}\\%"


def _macro(name: str, val: str) -> str:
    """Return a single \\newcommand line."""
    return f"\\newcommand{{\\{name}}}{{{val}}}"


# ── Table generators ──────────────────────────────────────────────────────────


def _tab_bj_dp(p2: dict) -> str:
    """Tabular body: VI / PI results on Blackjack."""
    vi = p2["vi"]
    pi = p2["pi"]

    rows = [
        r"\hline",
        r"\textbf{Algorithm} & \textbf{Iters} & \textbf{Wall-clock (s)}"
        r" & \textbf{Eval Return} & \textbf{IQR} \\",
        r"\hline",
        f"VI & {vi['convergence_iter']} & {_fmt(vi['wall_clock_s'], 3)}"
        f" & {_fmt(vi['mean_eval_return'], 4)} & {_fmt(vi['eval_return_iqr'], 3)} \\\\",
        f"PI & {pi['iterations']} & {_fmt(pi['wall_clock_s'], 3)}"
        f" & {_fmt(pi['mean_eval_return'], 4)} & {_fmt(pi['eval_return_iqr'], 3)} \\\\",
        r"\hline",
    ]
    cols = "lrrrr"
    return f"\\begin{{tabular}}{{{cols}}}\n" + "\n".join(rows) + "\n\\end{tabular}\n"


def _tab_cp_dp(p3: dict) -> str:
    """Tabular body: VI / PI results on CartPole (all three grids)."""
    grid_labels = {"coarse": "Coarse", "default": "Default", "fine": "Fine"}
    rows = [
        r"\hline",
        r"\textbf{Grid} & \textbf{Alg.} & \textbf{States}"
        r" & \textbf{Coverage} & \textbf{Iters}"
        r" & \textbf{Wall (s)} & \textbf{Ep-len} & \textbf{IQR} \\",
        r"\hline",
    ]
    for grid_key in ("coarse", "default", "fine"):
        g = p3[grid_key]
        label = grid_labels[grid_key]
        cov = _fmt_pct(g["coverage_pct"], 1)
        for algo_key, algo_label in (("vi", "VI"), ("pi", "PI")):
            a = g[algo_key]
            iters = a["iterations"]
            wall = _fmt(a["wall_clock_s"], 3)
            ep = _fmt(a["mean_episode_len"], 1)
            iqr = _fmt(a["eval_episode_len_iqr"], 2)
            grid_col = label if algo_key == "vi" else ""
            cov_col = cov if algo_key == "vi" else ""
            rows.append(
                f"{grid_col} & {algo_label} & {g['n_states'] if algo_key == 'vi' else ''}"
                f" & {cov_col} & {iters} & {wall} & {ep} & {iqr} \\\\"
            )
        rows.append(r"\hline")

    cols = "llrrrrrr"
    return f"\\begin{{tabular}}{{{cols}}}\n" + "\n".join(rows) + "\n\\end{tabular}\n"


def _tab_bj_mf(p4: dict) -> str:
    """Tabular body: SARSA / Q-Learning on Blackjack (controlled vs tuned)."""
    algo_labels = {"sarsa": "SARSA", "qlearning": "Q-Learning"}
    rows = [
        r"\hline",
        r"\textbf{Algorithm} & \textbf{Regime}"
        r" & \textbf{Eval Return} & \textbf{IQR}"
        r" & \textbf{Final Win Rate} & \textbf{Conv. Episode} & \textbf{Conv. IQR} \\",
        r"\hline",
    ]
    for regime in ("controlled", "tuned"):
        regime_label = regime.capitalize()
        for algo_key in ("sarsa", "qlearning"):
            a = p4[regime][algo_key]
            rows.append(
                f"{algo_labels[algo_key]} & {regime_label}"
                f" & {_fmt(a['mean_return'], 4)}"
                f" & {_fmt(a['iqr_return'], 3)}"
                f" & {_fmt(a['mean_win_rate'], 3)}"
                f" & {_fmt_int(a['mean_convergence_episode'])}"
                f" & {_fmt_int(a['convergence_episode_iqr'])} \\\\"
            )
        rows.append(r"\hline")

    cols = "llrrrrrr"  # noqa: E501
    return f"\\begin{{tabular}}{{{cols}}}\n" + "\n".join(rows) + "\n\\end{tabular}\n"


def _tab_cp_mf(p5: dict) -> str:
    """Tabular body: SARSA / Q-Learning on CartPole (controlled vs tuned)."""
    algo_labels = {"sarsa": "SARSA", "qlearning": "Q-Learning"}
    rows = [
        r"\hline",
        r"\textbf{Algorithm} & \textbf{Regime}"
        r" & \textbf{Ep-len} & \textbf{IQR}"
        r" & \textbf{Conv. Episode} & \textbf{Conv. IQR} \\",
        r"\hline",
    ]
    for regime in ("controlled", "tuned"):
        regime_label = regime.capitalize()
        for algo_key in ("sarsa", "qlearning"):
            a = p5[regime][algo_key]
            rows.append(
                f"{algo_labels[algo_key]} & {regime_label}"
                f" & {_fmt(a['mean_episode_len'], 1)}"
                f" & {_fmt(a['iqr_episode_len'], 1)}"
                f" & {_fmt_int(a['mean_convergence_episode'])}"
                f" & {_fmt_int(a['convergence_episode_iqr'])} \\\\"
            )
        rows.append(r"\hline")

    cols = "llrrrr"
    return f"\\begin{{tabular}}{{{cols}}}\n" + "\n".join(rows) + "\n\\end{tabular}\n"


def _tab_cp_disc(p5: dict) -> str:
    """Tabular body: CartPole discretization study (tuned HP applied to each grid)."""
    disc = p5["discretization"]
    grid_labels = {"coarse": "Coarse", "default": "Default", "fine": "Fine"}
    rows = [
        r"\hline",
        r"\textbf{Grid} & \textbf{SARSA Ep-len} & \textbf{Q-Learning Ep-len} \\",
        r"\hline",
    ]
    for grid_key in ("coarse", "default", "fine"):
        g = disc[grid_key]
        rows.append(
            f"{grid_labels[grid_key]}"
            f" & {_fmt(g['sarsa']['mean_episode_len'], 1)}"
            f" & {_fmt(g['qlearning']['mean_episode_len'], 1)} \\\\"
        )
    rows.append(r"\hline")
    cols = "lrr"
    return f"\\begin{{tabular}}{{{cols}}}\n" + "\n".join(rows) + "\n\\end{tabular}\n"


def _tab_dqn_ec(p7: dict) -> str:
    """Tabular body: DQN extra-credit results (greedy eval vs tuned tabular)."""
    variants = p7["variants"]
    tc = p7["tabular_comparison"]
    rows = [
        r"\hline",
        r"\textbf{Method} & \textbf{Greedy Eval Ep-len} & \textbf{IQR}"
        r" & \textbf{Converged (2\,k ep)} \\",
        r"\hline",
        f"Vanilla DQN"
        f" & {_fmt(variants['vanilla_dqn']['mean_eval_ep_len'], 1)}"
        f" & {_fmt(variants['vanilla_dqn']['eval_ep_len_iqr'], 1)}"
        f" & No \\\\",
        f"Double DQN"
        f" & {_fmt(variants['double_dqn']['mean_eval_ep_len'], 1)}"
        f" & {_fmt(variants['double_dqn']['eval_ep_len_iqr'], 1)}"
        f" & No \\\\",
        r"\hline",
        r"\textit{Tuned SARSA}"
        f" & {_fmt(tc['sarsa_tuned_mean_ep_len'], 1)}"
        f" & {_fmt(tc['sarsa_tuned_ep_len_iqr'], 1)}"
        r" & -- \\",
        r"\textit{Tuned Q-Learning}"
        f" & {_fmt(tc['qlearning_tuned_mean_ep_len'], 1)}"
        f" & {_fmt(tc['qlearning_tuned_ep_len_iqr'], 1)}"
        r" & -- \\",
        r"\hline",
    ]
    cols = "lrrr"
    return f"\\begin{{tabular}}{{{cols}}}\n" + "\n".join(rows) + "\n\\end{tabular}\n"


# ── Macro generator ───────────────────────────────────────────────────────────


def _build_report_numbers(
    p1: dict,
    p2: dict,
    p3: dict,
    p4: dict,
    p5: dict,
    p6: dict,
    p7: dict | None,
) -> str:
    """Return the full content of report_numbers.tex."""
    lines: list[str] = [
        "% report_numbers.tex — auto-generated by scripts/run_phase_8_report_tables.py",
        "% DO NOT EDIT BY HAND — regenerate with: make phase8",
        "%",
        "% Usage in report preamble:  \\input{../artifacts/tables/report_numbers}",
        "",
    ]

    # ── Seeds ──────────────────────────────────────────────────────────────────
    seed_str = ", ".join(str(s) for s in SEEDS)
    lines.append("% Seeds")
    lines.append(_macro("SeedList", seed_str))
    lines.append(_macro("NumSeeds", str(len(SEEDS))))
    lines.append("")

    # ── Phase 1 — environment setup ───────────────────────────────────────────
    cp1 = p1.get("cartpole", {})
    bj1 = p1.get("blackjack", {})
    lines.append("% Phase 1 — environment models")
    lines.append(_macro("CPNStates", f"{cp1.get('n_states', 864):,}"))
    lines.append(_macro("CPModelBuildWallClock", _fmt(cp1.get("wall_clock_s"), 1)))
    lines.append(_macro("BJNStates", f"{bj1.get('n_states', 290):,}"))
    lines.append("")

    # ── Phase 2 — Blackjack VI / PI ───────────────────────────────────────────
    vi2, pi2 = p2["vi"], p2["pi"]
    lines.append("% Phase 2 — Blackjack VI / PI")
    lines.append(_macro("BJVIIters", str(vi2["convergence_iter"])))
    lines.append(_macro("BJVIWallClock", _fmt(vi2["wall_clock_s"], 3)))
    lines.append(_macro("BJVIReturn", _fmt(vi2["mean_eval_return"], 4)))
    lines.append(_macro("BJVIReturnIQR", _fmt(vi2["eval_return_iqr"], 3)))
    lines.append(_macro("BJPIIters", str(pi2["iterations"])))
    lines.append(_macro("BJPIWallClock", _fmt(pi2["wall_clock_s"], 3)))
    lines.append(_macro("BJPIReturn", _fmt(pi2["mean_eval_return"], 4)))
    lines.append(_macro("BJPIReturnIQR", _fmt(pi2["eval_return_iqr"], 3)))
    lines.append(
        _macro("BJVIPolicyMatch", _fmt(pi2["policy_match_vi"] * 100, 1) + r"\%")
    )
    lines.append("")

    # ── Phase 3 — CartPole VI / PI ────────────────────────────────────────────
    lines.append("% Phase 3 — CartPole VI / PI (default grid unless noted)")
    for grid_key in ("coarse", "default", "fine"):
        g = p3[grid_key]
        prefix = "CP" + grid_key.capitalize()
        lines.append(_macro(f"{prefix}NStates", str(g["n_states"])))
        lines.append(
            _macro(f"{prefix}Coverage", _fmt(g["coverage_pct"] * 100, 1) + r"\%")
        )
        for algo_key, algo_label in (("vi", "VI"), ("pi", "PI")):
            a = g[algo_key]
            ap = f"{prefix}{algo_label}"
            lines.append(_macro(f"{ap}Iters", str(a["iterations"])))
            lines.append(_macro(f"{ap}WallClock", _fmt(a["wall_clock_s"], 3)))
            lines.append(_macro(f"{ap}EpLen", _fmt(a["mean_episode_len"], 1)))
            lines.append(_macro(f"{ap}EpLenIQR", _fmt(a["eval_episode_len_iqr"], 2)))
        lines.append(
            _macro(
                f"{prefix}PolicyAgreement", _fmt(g["policy_agreement_pct"], 1) + r"\%"
            )
        )
    lines.append("")

    # ── Phase 4 — Blackjack SARSA / Q-Learning ────────────────────────────────
    lines.append("% Phase 4 — Blackjack SARSA / Q-Learning")
    for regime in ("controlled", "tuned"):
        rl = regime.capitalize()
        for algo_key, algo_label in (("sarsa", "Sarsa"), ("qlearning", "QL")):
            a = p4[regime][algo_key]
            ap = f"BJ{algo_label}{rl}"
            lines.append(_macro(f"{ap}Return", _fmt(a["mean_return"], 4)))
            lines.append(_macro(f"{ap}ReturnIQR", _fmt(a["iqr_return"], 3)))
            lines.append(_macro(f"{ap}WinRate", _fmt(a["mean_win_rate"], 3)))
            lines.append(_macro(f"{ap}ConvEp", _fmt_int(a["mean_convergence_episode"])))
            lines.append(_macro(f"{ap}ConvIQR", _fmt_int(a["convergence_episode_iqr"])))
    ft4 = p4.get("final_training", {})
    lines.append(_macro("BJMFTrainWallClock", _fmt(ft4.get("train_wall_clock_s"), 1)))
    lines.append("")

    # ── Phase 5 — CartPole SARSA / Q-Learning ────────────────────────────────
    lines.append("% Phase 5 — CartPole SARSA / Q-Learning")
    for regime in ("controlled", "tuned"):
        rl = regime.capitalize()
        for algo_key, algo_label in (("sarsa", "Sarsa"), ("qlearning", "QL")):
            a = p5[regime][algo_key]
            ap = f"CP{algo_label}{rl}"
            lines.append(_macro(f"{ap}EpLen", _fmt(a["mean_episode_len"], 1)))
            lines.append(_macro(f"{ap}EpLenIQR", _fmt(a["iqr_episode_len"], 1)))
            lines.append(_macro(f"{ap}ConvEp", _fmt_int(a["mean_convergence_episode"])))
            lines.append(_macro(f"{ap}ConvIQR", _fmt_int(a["convergence_episode_iqr"])))
    ft5 = p5.get("final_training", {})
    lines.append(_macro("CPMFTrainWallClock", _fmt(ft5.get("train_wall_clock_s"), 1)))
    lines.append(_macro("CPDiscWallClock", _fmt(ft5.get("disc_wall_clock_s"), 1)))
    # Discretization study
    lines.append("% Phase 5 — CartPole discretization study")
    for grid_key in ("coarse", "default", "fine"):
        g = p5["discretization"][grid_key]
        gp = f"CPDisc{grid_key.capitalize()}"
        lines.append(_macro(f"{gp}SarsaEpLen", _fmt(g["sarsa"]["mean_episode_len"], 1)))
        lines.append(
            _macro(f"{gp}QLEpLen", _fmt(g["qlearning"]["mean_episode_len"], 1))
        )
    lines.append("")

    # ── Phase 6 — wall-clock totals ───────────────────────────────────────────
    lines.append("% Phase 6 — wall-clock summary")
    wc = p6.get("wall_clock", {})
    bj_wc = wc.get("blackjack", {})
    cp_wc = wc.get("cartpole", {})
    lines.append(_macro("BJVIWallClockPhaseSix", _fmt(bj_wc.get("vi"), 3)))
    lines.append(_macro("BJPIWallClockPhaseSix", _fmt(bj_wc.get("pi"), 3)))
    lines.append(_macro("BJSarsaWallClock", _fmt(bj_wc.get("sarsa"), 1)))
    lines.append(_macro("BJQLWallClock", _fmt(bj_wc.get("qlearning"), 1)))
    lines.append(
        _macro("CPModelBuildWallClockPhaseSix", _fmt(cp_wc.get("model_build_s"), 1))
    )
    lines.append(_macro("CPVIWallClock", _fmt(cp_wc.get("vi"), 3)))
    lines.append(_macro("CPPIWallClock", _fmt(cp_wc.get("pi"), 3)))
    lines.append(_macro("CPSarsaWallClock", _fmt(cp_wc.get("sarsa"), 1)))
    lines.append(_macro("CPQLWallClock", _fmt(cp_wc.get("qlearning"), 1)))

    # Total wall clock (sum of all method wall clocks — model-free phases dominate)
    total_wc = sum(
        v
        for v in [
            bj_wc.get("vi", 0),
            bj_wc.get("pi", 0),
            bj_wc.get("sarsa", 0),
            bj_wc.get("qlearning", 0),
            cp_wc.get("vi", 0),
            cp_wc.get("pi", 0),
            cp_wc.get("sarsa", 0),
            cp_wc.get("qlearning", 0),
            cp_wc.get("model_build_s", 0),
        ]
    )
    lines.append(_macro("WallClockTotal", _fmt(total_wc / 60, 1) + "~min"))
    lines.append("")

    # ── Phase 7 — DQN EC (optional) ──────────────────────────────────────────
    if p7 is not None:
        lines.append("% Phase 7 — DQN extra credit")
        v7 = p7["variants"]
        tc7 = p7["tabular_comparison"]
        for var_key, var_label in (
            ("vanilla_dqn", "VanillaDQN"),
            ("double_dqn", "DoubleDQN"),
        ):
            v = v7[var_key]
            lines.append(
                _macro(f"{var_label}EvalEpLen", _fmt(v["mean_eval_ep_len"], 1))
            )
            lines.append(_macro(f"{var_label}EvalIQR", _fmt(v["eval_ep_len_iqr"], 1)))
            lines.append(
                _macro(f"{var_label}FinalEpLen", _fmt(v["mean_final_ep_len"], 1))
            )
        lines.append(
            _macro("DQNSarsaBaseline", _fmt(tc7["sarsa_tuned_mean_ep_len"], 1))
        )
        lines.append(
            _macro("DQNQLBaseline", _fmt(tc7["qlearning_tuned_mean_ep_len"], 1))
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Repro artifact generators ─────────────────────────────────────────────────


def _build_runbook(p7_exists: bool) -> str:
    seed_str = ", ".join(str(s) for s in SEEDS)
    # Mirror the literal Makefile pipeline target:
    #   pipeline: phase1 gate1 phase2 phase3 phase4 phase5 phase6 phase8
    # Phase 7 (DQN EC) is always optional; it is never part of `make pipeline`.
    pipeline_target = "phase1 gate1 phase2 phase3 phase4 phase5 phase6 phase8"
    return f"""# Runbook — CS7641 RL Spring 2026

Auto-generated by `scripts/run_phase_8_report_tables.py`.

## Environment

- Python 3.13 via [uv](https://docs.astral.sh/uv/)
- All dependencies managed by `uv` (see `pyproject.toml`)
- No external datasets — both environments provided by `gymnasium`

## Setup

```bash
git clone <repo-url>
cd omscs_ml_rl
make dev          # creates .venv, installs all dependencies
```

## Full pipeline

```bash
make pipeline     # runs: {pipeline_target}
```

`gate1` runs `pytest tests/test_envs.py` after Phase 1 as a correctness gate.

Or individually (use `--detach` for phases expected to take > 2 min):

```bash
make phase1                                    # CartPole model estimation (~40 s)
make gate1                                     # env test gate (required before Phase 2)
make phase2                                    # VI/PI on Blackjack (< 1 s)
make phase3                                    # VI/PI on CartPole (~70 s)
make phase4                                    # SARSA/Q-Learning on Blackjack (~85 s)
make phase5                                    # SARSA/Q-Learning on CartPole (~960 s)
make phase6                                    # Cross-method comparison (< 5 s)
# Phase 7 is optional extra credit — not part of make pipeline:
{"make phase7                                    # DQN EC (~95 s, optional)" if p7_exists else "# make phase7                                  # DQN EC (~95 s, optional — not yet run)"}
make phase8                                    # Tables and repro artifacts (< 1 s)
```

For long-running phases:

```bash
bash ml_run.sh --detach "make phase5" phase5   # background tmux/screen session
```

## Seeds

Fixed across all model-free training runs: **{seed_str}**

Constant: `SEEDS = {list(SEEDS)}` in `src/config.py`

## Figures

```bash
make viz          # re-renders all figures from saved checkpoints
```

Or a single phase:

```bash
uv run python scripts/visualize_all.py --phase phase5
```

## Expected outputs

All outputs are written under `artifacts/` (git-ignored):

| Path | Description |
|------|-------------|
| `artifacts/metadata/phase{{N}}.json` | Checkpoint per phase |
| `artifacts/metrics/phase{{N}}_*/` | CSVs per phase |
| `artifacts/figures/phase{{N}}_*/` | PNG figures per phase |
| `artifacts/logs/phase{{N}}.log` | Execution log per phase |
| `artifacts/tables/report_numbers.tex` | Inline number macros |
| `artifacts/tables/tab_*.tex` | Table bodies |
| `artifacts/repro/runbook.md` | This file |
| `artifacts/repro/submission_checklist.md` | Submission tracker |

## Notes

- `artifacts/repro/overleaf_link.md` — written manually with the READ-ONLY Overleaf URL
- `artifacts/repro/ai_use_statement.md` — written manually before submission
"""


def _build_checklist(p7_exists: bool) -> str:
    dqn_line = (
        "- [x] Phase 7 (DQN EC): `artifacts/metadata/phase7.json` exists"
        if p7_exists
        else "- [ ] Phase 7 (DQN EC): optional — skip or run `make phase7`"
    )
    return f"""# Submission Checklist — CS7641 RL Spring 2026

Auto-generated by `scripts/run_phase_8_report_tables.py`.

## Pipeline artifacts

- [ ] Phase 1: `artifacts/metadata/phase1.json` exists
- [ ] Phase 2: `artifacts/metadata/phase2.json` exists
- [ ] Phase 3: `artifacts/metadata/phase3.json` exists
- [ ] Phase 4: `artifacts/metadata/phase4.json` exists
- [ ] Phase 5: `artifacts/metadata/phase5.json` exists
- [ ] Phase 6: `artifacts/metadata/phase6.json` exists
{dqn_line}
- [ ] Phase 8: `artifacts/tables/report_numbers.tex` exists

## Tables

- [ ] `artifacts/tables/tab_bj_dp.tex`
- [ ] `artifacts/tables/tab_cp_dp.tex`
- [ ] `artifacts/tables/tab_bj_mf.tex`
- [ ] `artifacts/tables/tab_cp_mf.tex`
- [ ] `artifacts/tables/tab_cp_disc.tex`
{"- [ ] `artifacts/tables/tab_dqn_ec.tex`" if p7_exists else ""}

## Figures

- [ ] `artifacts/figures/phase2_vi_pi_blackjack/` — Blackjack VI/PI figures
- [ ] `artifacts/figures/phase3_vi_pi_cartpole/` — CartPole VI/PI figures
- [ ] `artifacts/figures/phase4_model_free_blackjack/` — Blackjack MF figures
- [ ] `artifacts/figures/phase5_model_free_cartpole/` — CartPole MF figures
- [ ] `artifacts/figures/phase6_comparison/` — Cross-method comparison figures
{"- [ ] `artifacts/figures/phase7_dqn_ec/` — DQN EC figures" if p7_exists else ""}

## Report

- [ ] Report PDF compiled from `REPORT_RL/RL_Report_schinne3.tex`
- [ ] All `\\input{{../artifacts/tables/tab_*}}` statements match existing files
- [ ] `report_numbers.tex` input in preamble: `\\input{{../artifacts/tables/report_numbers}}`
- [ ] All inline numbers in report cross-checked against macros
- [ ] Page count: ≤ 8 pages (IEEE conference format)
- [ ] Mandatory citation: Sutton \\& Barto 2018 (Blackjack)
- [ ] Mandatory citation: Barto et al. 1983 (CartPole)
- [ ] AI Use Statement present at end of report

## Reproducibility sheet

- [ ] Seed list: {", ".join(str(s) for s in SEEDS)}
- [ ] Total wall-clock reported
- [ ] CartPole bin edges reported verbatim
- [ ] GitHub commit hash recorded

## Submission

- [ ] `artifacts/repro/overleaf_link.md` — READ-ONLY Overleaf URL recorded
- [ ] `artifacts/repro/ai_use_statement.md` — AI-use disclosure written
- [ ] GitHub commit hash matches submitted Overleaf snapshot
- [ ] Overleaf link submitted on Canvas
"""


# ── Main lifecycle ────────────────────────────────────────────────────────────


def run() -> Path:
    logger = configure_logger("phase8")
    t0 = time.perf_counter()

    # ── Load upstream checkpoints ─────────────────────────────────────────────
    logger.info("Loading upstream phase checkpoints")
    p1 = load_checkpoint_json(METADATA_DIR / "phase1.json")
    p2_full = load_checkpoint_json(METADATA_DIR / "phase2.json")
    p3_full = load_checkpoint_json(METADATA_DIR / "phase3.json")
    p4_full = load_checkpoint_json(METADATA_DIR / "phase4.json")
    p5_full = load_checkpoint_json(METADATA_DIR / "phase5.json")
    p6_full = load_checkpoint_json(METADATA_DIR / "phase6.json")

    p7_path = METADATA_DIR / "phase7.json"
    p7_full = load_checkpoint_json(p7_path) if p7_path.exists() else None
    if p7_full is not None:
        logger.info("Phase 7 checkpoint found — DQN EC table will be generated")
    else:
        logger.info("Phase 7 checkpoint not found — skipping DQN EC table")

    # Unwrap summaries
    p2 = p2_full["summary"]
    p3 = p3_full["summary"]
    p4 = p4_full["summary"]
    p5 = p5_full["summary"]
    p6 = p6_full["summary"]
    p7 = p7_full["summary"] if p7_full is not None else None

    # ── Create output directories ─────────────────────────────────────────────
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output dirs: %s  %s", TABLES_DIR, REPRO_DIR)

    # ── Write LaTeX tables ────────────────────────────────────────────────────
    tables_written: list[str] = []

    def _write_table(filename: str, content: str) -> None:
        path = TABLES_DIR / filename
        path.write_text(content)
        tables_written.append(str(path))
        logger.info("Wrote %s", path)

    _write_table("tab_bj_dp.tex", _tab_bj_dp(p2))
    _write_table("tab_cp_dp.tex", _tab_cp_dp(p3))
    _write_table("tab_bj_mf.tex", _tab_bj_mf(p4))
    _write_table("tab_cp_mf.tex", _tab_cp_mf(p5))
    _write_table("tab_cp_disc.tex", _tab_cp_disc(p5))
    if p7 is not None:
        _write_table("tab_dqn_ec.tex", _tab_dqn_ec(p7))

    # ── Write report_numbers.tex ──────────────────────────────────────────────
    report_numbers_content = _build_report_numbers(p1, p2, p3, p4, p5, p6, p7)
    report_numbers_path = TABLES_DIR / "report_numbers.tex"
    report_numbers_path.write_text(report_numbers_content)
    logger.info(
        "Wrote %s (%d macros)",
        report_numbers_path,
        report_numbers_content.count(r"\newcommand"),
    )

    # ── Write repro artifacts ─────────────────────────────────────────────────
    p7_exists = p7 is not None
    runbook_path = REPRO_DIR / "runbook.md"
    runbook_path.write_text(_build_runbook(p7_exists))
    logger.info("Wrote %s", runbook_path)

    checklist_path = REPRO_DIR / "submission_checklist.md"
    checklist_path.write_text(_build_checklist(p7_exists))
    logger.info("Wrote %s", checklist_path)

    # ── Write phase8 checkpoint ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    checkpoint: dict = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase8",
        "slug": "report_tables",
        "upstream_inputs": [
            "artifacts/metadata/phase1.json",
            "artifacts/metadata/phase2.json",
            "artifacts/metadata/phase3.json",
            "artifacts/metadata/phase4.json",
            "artifacts/metadata/phase5.json",
            "artifacts/metadata/phase6.json",
        ]
        + (["artifacts/metadata/phase7.json"] if p7_exists else []),
        "outputs": {
            "tables_dir": str(TABLES_DIR),
            "repro_dir": str(REPRO_DIR),
            "tables_written": tables_written
            + [
                str(report_numbers_path),
                str(runbook_path),
                str(checklist_path),
            ],
        },
        "config_snapshot": {
            "seeds": list(SEEDS),
            "phase7_included": p7_exists,
        },
        "summary": {
            "n_tables": len(tables_written),
            "n_macros": report_numbers_content.count(r"\newcommand"),
            "wall_clock_s": round(elapsed, 3),
        },
    }

    ckpt_path = METADATA_DIR / "phase8.json"
    write_checkpoint_json(checkpoint, ckpt_path)
    logger.info("Phase 8 complete in %.2f s — checkpoint: %s", elapsed, ckpt_path)

    return ckpt_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
