"""Experiments of Q2.(a), Q2.(b) and Q2.(d): one function per part, each saves CSV + PNG to ``out``.

Run through ``python main.py --question Q2_linear`` (2.b), ``--q2a`` or ``--q2d``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import InputData, load_question
from .model_q2b import LinearUtilityModel, UtilityAsDisutilityModel
from .plotting import plot_duals, plot_inputs, plot_schedule
from .q2_analysis import (EPS, daily_metrics, linear_thresholds, marginal_costs, predict_linear_load,
                          solve_with_metrics, sweep)
from .q2_plots import (plot_hourly_response, plot_load_heatmap, plot_loads_vs_reference, plot_model_metrics,
                       plot_sweep_metrics)


def run_part_a(out: Path) -> dict[str, float]:
    """2.(a): on Q1_caseA, Question 1 and the disutility model with l_ref = L_max, D = u (L_max - L) give the
    same load and objectives that differ by the constant 24 u L_max. Saves q2a_special_case_check.csv."""
    data = load_question("Q1_caseA")
    r_q1 = LinearUtilityModel(data, reference=np.zeros(data.n_hours)).build().solve()
    r_d = UtilityAsDisutilityModel(data).build().solve()
    check = {"Q1_objective": r_q1.objective, "disutility_objective": r_d.objective,
             "difference": r_d.objective - r_q1.objective,
             "24_u_Lmax": data.n_hours * data.consumption_utility * data.load_max_kWh,
             "max_load_gap_kWh": float(np.abs(r_q1.hourly["load"] - r_d.hourly["load"]).max())}
    pd.Series(check).to_csv(out / "q2a_special_case_check.csv", header=["value"])
    print("2.(a) check:", {k: round(v, 4) for k, v in check.items()}, "\n")
    return check


def cL_sweep_values(d: InputData) -> np.ndarray:
    """One c^L per regime: midpoints between consecutive thresholds, one value below the smallest and one
    above the largest, plus the base value - every regime of 2.(b).ii is visited without hitting a tie."""
    th = np.array(linear_thresholds(d))
    return np.unique(np.round(np.r_[th[0] / 2, (th[:-1] + th[1:]) / 2, th[-1] + 0.5, d.linear_disutility], 4))


def run_part_b(out: Path) -> pd.DataFrame:
    """2.(b).iv: base case of Q2_linear (schedule, duals, metrics, check of the 2.(b).ii prediction) and a
    sweep of c^L (DKK/kWh). Saves Q2_linear_hourly.csv / _summary.txt, q2b_base_metrics.csv, sweep_cL.csv,
    sweep_cL_load.csv, cL_thresholds.csv and the figures."""
    data = load_question("Q2_linear")
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    res, met = solve_with_metrics("linear", data)
    gap = np.abs(res.hourly["load"].to_numpy() - predict_linear_load(data, data.linear_disutility)).max()
    print(res, "\n", {k: round(v, 3) for k, v in met.items()})
    print(f"max |solver load - predicted load 2.(b).ii| = {gap:.2e} kWh/h\n")
    res.save(out)
    pd.Series(met).to_csv(out / "q2b_base_metrics.csv", header=["value"])
    plot_schedule(res, data, save_to=out / "schedule.png")
    plot_duals(res, data, save_to=out / "duals.png")
    plot_loads_vs_reference({f"optimal load, $c^L$={data.linear_disutility:g}": res.hourly["load"].to_numpy()},
                            data.reference_load, data.pv_available, "2.(b) linear disutility - base case",
                            save_to=out / "load_vs_reference.png")

    # sweep of c^L over all regimes
    df, loads = sweep("linear", data, "linear_disutility", cL_sweep_values(data), "cL")
    df["matches_prediction"] = [bool(np.allclose(loads[c], predict_linear_load(data, c), atol=1e-5))
                                for c in df["cL"]]
    th = linear_thresholds(data)
    m_low, m_high = marginal_costs(data)
    ref, pv, lmin = data.reference_load, data.pv_available, data.load_min_kWh
    has_low = np.minimum(ref, pv) - lmin > EPS          # hours with a PV-served part of the reference
    has_high = ref - np.maximum(pv, lmin) > EPS         # hours with an imported part of the reference
    hours_at = lambda m, mask, v: ", ".join(str(t) for t in np.flatnonzero(np.isclose(m, v) & mask))
    pd.DataFrame({"cL_threshold": th,
                  "hours_PV_layer (max(c_PV, p_exp))": [hours_at(m_low, has_low, v) for v in th],
                  "hours_import_layer (p_imp)": [hours_at(m_high, has_high, v) for v in th]}
                 ).to_csv(out / "cL_thresholds.csv", index=False)
    df.to_csv(out / "sweep_cL.csv", index=False)
    loads.rename(columns=lambda c: f"cL={c:.3g}").to_csv(out / "sweep_cL_load.csv")
    print(df.round(3).to_string(index=False))
    print(f"\npredicted c^L thresholds (DKK/kWh): {th}\n")
    plot_sweep_metrics(df, "cL", "$c^L$ [DKK/kWh]", "2.(b).iv sweep of the linear disutility coefficient",
                       th, save_to=out / "sweep_cL.png")
    plot_load_heatmap(loads, data.reference_load, "$c^L$ [DKK/kWh]",
                      "2.(b).iv optimal load / reference per hour and $c^L$", save_to=out / "sweep_cL_heatmap.png")
    return df


def run_part_d(out: Path, cL_values=(0.5, 1.43, 3.5), cQ_values=(0.1, 1.0, 10.0),
               response_hours=(8, 10, 13)) -> pd.DataFrame:
    """2.(d): compare Question 1 (linear utility, Q1_caseA), the linear (2.(b)) and the quadratic (2.(c))
    disutility models. Scenarios = base case of each model plus ``cL_values`` / ``cQ_values`` (one value per
    regime of the sweeps: below, around and above the effective-price thresholds). Deviations of the
    Question 1 model are measured against the Question 2 reference. Saves q2d_comparison.csv, the profiles,
    the metric bars and the response of ``response_hours`` to c^L and c^Q."""
    lin, quad, q1 = load_question("Q2_linear"), load_question("Q2_quadratic"), load_question("Q1_caseA")
    ref, pv = lin.reference_load, lin.pv_available

    res_q1, met_q1 = solve_with_metrics("utility", q1, reference=ref)
    rows = [{"model": "Q1 utility", "scenario": f"Q1 u={q1.consumption_utility:g}", **met_q1}]
    profiles: dict[str, dict[str, np.ndarray]] = {"linear": {}, "quadratic": {}}
    for kind, base, attr, values in [("linear", lin, "linear_disutility", cL_values),
                                     ("quadratic", quad, "quadratic_disutility", cQ_values)]:
        for c in values:
            res, met = solve_with_metrics(kind, replace(base, **{attr: float(c)}))
            tag = f"{kind} c={c:g}" + (" (base)" if np.isclose(c, getattr(base, attr)) else "")
            rows.append({"model": kind, "scenario": tag, **met})
            profiles[kind][tag] = res.hourly["load"].to_numpy()
    df = pd.DataFrame(rows)
    df.to_csv(out / "q2d_comparison.csv", index=False)
    print(df.round(3).to_string(index=False), "\n")

    for kind in profiles:
        plot_loads_vs_reference(profiles[kind], ref, pv, f"2.(d) {kind} disutility: load profiles",
                                save_to=out / f"q2d_profiles_{kind}.png")
    base_profiles = {"Q1 utility (no reference)": res_q1.hourly["load"].to_numpy(),
                     **{k: v for p in profiles.values() for k, v in p.items() if "(base)" in k}}
    plot_loads_vs_reference(base_profiles, ref, pv, "2.(d) base cases of the three objectives",
                            save_to=out / "q2d_profiles_base.png")
    plot_model_metrics(df, save_to=out / "q2d_metrics.png")

    # response of single hours to the coefficient: jumps (linear) vs gradual (quadratic)
    _, loads_l = sweep("linear", lin, "linear_disutility", np.logspace(-1, np.log10(4), 60), "cL")
    _, loads_q = sweep("quadratic", quad, "quadratic_disutility", np.logspace(-2, 1.5, 60), "cQ")
    hours = list(response_hours)
    plot_hourly_response({"linear disutility|$c^L$ [DKK/kWh]": loads_l.loc[hours].T,
                          "quadratic disutility|$c^Q$ [DKK/kWh$^2$]": loads_q.loc[hours].T},
                         hours, save_to=out / "q2d_hourly_response.png")
    return df
