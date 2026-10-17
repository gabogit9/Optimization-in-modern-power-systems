"""Analysis helpers shared by Q2 (and reusable for Q3): closed-form hypotheses, daily metrics, sweeps.

* ``predict_linear_load`` / ``linear_thresholds``: 2.(b).ii closed form of the optimal load.
* ``daily_metrics``: the metrics table of 2.(b).iv, 2.(c).iv, 2.(d) (same columns for every model).
* ``solve_with_metrics`` / ``sweep``: solve one model kind (linear / quadratic / utility / cumulative),
  or sweep one coefficient of the input data.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .data_loader import InputData
from .model import Results
from .model_q2b import LinearDisutilityModel, LinearUtilityModel, effective_prices
from .model_q2c import QuadraticDisutilityModel
from .model_q2e import SustainedDisutilityModel

EPS = 1e-6   # kWh/h, tolerance used to classify hours (at a bound, deviating, ...)


def quadratic_disutility_values(d: InputData, load: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """D_t = c^Q (L_t - l_ref_t)^2 of the teammate's 2.(c) model (DKK per hour)."""
    return d.quadratic_disutility * (load - ref)**2


# ============================================================ 2.(b).ii closed-form hypotheses
def marginal_costs(d: InputData) -> tuple[np.ndarray, np.ndarray]:
    """Hourly marginal cost (DKK/kWh) of serving one more kWh of load.

    m_low:  load below the PV availability: that PV kWh would otherwise be curtailed (saves c_PV) or
            exported (earns p_exp), so it costs max(c_PV, p_exp), capped at p_imp (PV dearer than
            imports is never used).
    m_high: load above the PV availability is imported: p_imp.
    """
    p_imp, p_exp = effective_prices(d)
    return np.minimum(p_imp, np.maximum(d.pv_marginal_cost, p_exp)), p_imp


def predict_linear_load(d: InputData, c_lin: float) -> np.ndarray:
    """2.(b).ii closed form of the optimal hourly load for D_t = c^L |L_t - l_ref_t| (kWh/h).

    Starting from l_ref, a kWh is shed iff its marginal cost exceeds c^L. Consuming above l_ref never pays
    (marginal cost > 0 > -c^L), so the load jumps between three candidates:
    l_ref (c^L > p_imp), max(L_min, min(l_ref, PV_max)) (m_low < c^L < p_imp), L_min (c^L < m_low).
    Ties (c^L equal to a threshold) give a non-unique optimum and are not covered.
    """
    m_low, m_high = marginal_costs(d)
    ref, lmin, pv = d.reference_load, d.load_min_kWh, d.pv_available
    low_part = np.where(c_lin > m_low, np.maximum(0.0, np.minimum(ref, pv) - lmin), 0.0)
    high_part = np.where(c_lin > m_high, np.maximum(0.0, ref - np.maximum(pv, lmin)), 0.0)
    return lmin + low_part + high_part


def linear_thresholds(d: InputData) -> list[float]:
    """Sorted values of c^L (DKK/kWh) at which the optimal load of at least one hour jumps."""
    m_low, m_high = marginal_costs(d)
    ref, lmin, pv = d.reference_load, d.load_min_kWh, d.pv_available
    has_low = np.minimum(ref, pv) - lmin > EPS          # some reference load can be served from PV
    has_high = ref - np.maximum(pv, lmin) > EPS         # some reference load must be imported
    return sorted({float(v) for v in np.round(np.r_[m_low[has_low], m_high[has_high]], 6)})


def daily_metrics(d: InputData, hourly: pd.DataFrame, disutility: np.ndarray, ref: np.ndarray) -> dict:
    """Daily metrics of 2.(b).iv / 2.(c).iv / 2.(d).

    procurement_cost, disutility, total (DKK); energy_kWh, abs_deviation_kWh (kWh);
    hours_deviating: hours with |L - l_ref| > 0;
    hours_dev_bound_binding: hours where the deviation reaches its bound, i.e. the load is fully shed
    (L = L_min < l_ref) or pushed to L_max > l_ref.
    """
    p_imp, p_exp = effective_prices(d)
    load = hourly["load"].to_numpy()
    cost = float((p_imp*hourly["import"] - p_exp*hourly["export"] + d.pv_marginal_cost*hourly["pv"]).sum())
    dev = load - ref
    at_min, at_max = load <= d.load_min_kWh + EPS, load >= d.load_max_kWh - EPS
    return {"procurement_cost": cost, "disutility": float(disutility.sum()), "total": cost + float(disutility.sum()),
            "energy_kWh": float(load.sum()), "abs_deviation_kWh": float(np.abs(dev).sum()),
            "hours_deviating": int((np.abs(dev) > EPS).sum()),
            "hours_dev_bound_binding": int(((at_min & (ref > d.load_min_kWh + EPS))
                                            | (at_max & (ref < d.load_max_kWh - EPS))).sum())}


def solve_with_metrics(kind: str, data: InputData, reference: np.ndarray | None = None,
                       **model_kwargs) -> tuple[Results, dict]:
    """Build and solve one model and return (results, metrics).

    ``kind``: "linear" (2.b), "quadratic" (2.c), "cumulative" (2.e; ``model_kwargs`` = rho, c_S),
    "utility" (Question 1 utility, deviations measured against ``reference``).
    """
    ref = np.asarray(data.reference_load if reference is None else reference, dtype=float)
    if kind == "quadratic":
        res = QuadraticDisutilityModel(data, **model_kwargs).build().solve()
        disutil = quadratic_disutility_values(data, res.hourly["load"].to_numpy(), ref)
    elif kind == "cumulative":
        model = SustainedDisutilityModel(data, **model_kwargs).build()
        res = model.solve()
        disutil = model.c_S * res.hourly["cumulative_deviation"].to_numpy() ** 2
    else:
        model = {"linear": LinearDisutilityModel, "utility": LinearUtilityModel}[kind](data, reference=ref, **model_kwargs).build()
        res = model.solve()
        disutil = model.disutility_values(res.hourly["load"].to_numpy())
    return res, daily_metrics(data, res.hourly, disutil, ref)

def sweep(kind: str, base: InputData, attr: str, values, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solve model ``kind`` for every value of the coefficient ``attr`` of ``base``.

    Returns (metrics, loads): one row of ``daily_metrics`` per value (column ``label``), and the hourly
    load of every run (hours x values, kWh/h).
    """
    rows, loads = [], {}
    for c in values:
        res, met = solve_with_metrics(kind, replace(base, **{attr: float(c)}))
        rows.append({label: float(c), **met})
        loads[float(c)] = res.hourly["load"].to_numpy()
    return pd.DataFrame(rows), pd.DataFrame(loads, index=pd.Index(range(base.n_hours), name="hour"))
