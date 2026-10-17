"""Q2.(a) and Q2.(b): separable-disutility models of the flexible consumer (gurobipy).

    SeparableDisutilityModel       2.(a)  cost(I, E, PV) + sum_t D_t(L_t)   (generic; D_t given by a subclass)
      |- LinearDisutilityModel     2.(b)  D_t = c^L |L_t - l_ref_t|          (LP via dev_up, dev_down >= 0)
      |- UtilityAsDisutilityModel  2.(a)  D_t = U (L_max - L_t)             (Question 1 as a special case)
      |- LinearUtilityModel        Q1     D_t = -U L_t                      (Question 1, reference for 2.(d))

All minimise a daily cost in DKK with cost_t = p_imp_t I_t - p_exp_t E_t + c_PV PV_t,
p_imp = p + tau_imp, p_exp = p - tau_exp. Loads, PV, imports, exports in kWh/h.
The quadratic (2.(c)) and cumulative (2.(e)) models live in model_q2c.py / model_q2e.py.
"""
from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .data_loader import InputData
from .model import FlexibleConsumerModel


def effective_prices(d: InputData) -> tuple[np.ndarray, np.ndarray]:
    """Effective import and export prices p_imp = p + tau_imp, p_exp = p - tau_exp (DKK/kWh, 24 values)."""
    return d.energy_price + d.import_tariff, d.energy_price - d.export_tariff


class SeparableDisutilityModel(FlexibleConsumerModel):
    """2.(a): min sum_t [p_imp I - p_exp E + c_PV PV + D_t(L_t)] subject to the constraints of Question 1.

    Subclasses implement ``_disutility(t)`` (gurobipy expression, DKK) and ``disutility_values(load)``
    (numpy, DKK per hour), and may add auxiliary variables/constraints in ``_add_auxiliary()``.
    ``reference`` (kWh/h, 24 values) overrides ``data.reference_load`` when given.
    """

    def __init__(self, data: InputData, reference: np.ndarray | None = None, **kwargs):
        super().__init__(data, **kwargs)
        self.ref = np.asarray(data.reference_load if reference is None else reference, dtype=float)

    def _add_auxiliary(self) -> None:
        """Hook for auxiliary variables/constraints (none for a differentiable disutility)."""

    def _disutility(self, t: int):
        raise NotImplementedError

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def build(self) -> "SeparableDisutilityModel":
        d, m, T = self.data, self.m, self.T
        p_imp, p_exp = effective_prices(d)

        # variables: I, E, PV >= 0 as variable bounds; the load bounds are explicit constraints (duals)
        self.var["import"] = m.addVars(T, lb=0.0, name="import")
        self.var["export"] = m.addVars(T, lb=0.0, name="export")
        self.var["load"]   = m.addVars(T, lb=-GRB.INFINITY, name="load")
        self.var["pv"]     = m.addVars(T, lb=0.0, name="pv")
        I, E, L, PV = (self.var[k] for k in ("import", "export", "load", "pv"))
        self._add_auxiliary()

        # objective: procurement cost + disutility
        m.setObjective(gp.quicksum(p_imp[t]*I[t] - p_exp[t]*E[t] + d.pv_marginal_cost*PV[t]
                                   + self._disutility(t) for t in T), GRB.MINIMIZE)

        # constraints (explicit, so their duals are returned as dual_<name>)
        self.con["balance"]  = m.addConstrs((L[t] - PV[t] - I[t] + E[t] == 0 for t in T), name="balance")
        self.con["pv_max"]   = m.addConstrs((PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")
        self.con["load_max"] = m.addConstrs((L[t] - d.load_max_kWh <= 0 for t in T), name="load_max")
        self.con["load_min"] = m.addConstrs((d.load_min_kWh - L[t] <= 0 for t in T), name="load_min")
        m.update()
        return self


class LinearDisutilityModel(SeparableDisutilityModel):
    """2.(b): D_t = c^L |L_t - l_ref_t|, reformulated as an LP.

    L_t - l_ref_t = dev_up_t - dev_down_t,  dev_up_t, dev_down_t >= 0,  D_t = c^L (dev_up_t + dev_down_t).
    With c^L > 0 at most one of dev_up_t, dev_down_t is positive at the optimum, so their sum is |L - l_ref|.
    """

    def _add_auxiliary(self) -> None:
        m, T, L = self.m, self.T, self.var["load"]
        self.var["dev_up"]   = m.addVars(T, lb=0.0, name="dev_up")     # kWh/h above the reference
        self.var["dev_down"] = m.addVars(T, lb=0.0, name="dev_down")   # kWh/h below the reference
        up, dn = self.var["dev_up"], self.var["dev_down"]
        self.con["deviation"] = m.addConstrs((L[t] - self.ref[t] - up[t] + dn[t] == 0 for t in T),
                                             name="deviation")

    def _disutility(self, t: int):
        return self.data.linear_disutility * (self.var["dev_up"][t] + self.var["dev_down"][t])

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        return self.data.linear_disutility * np.abs(load - self.ref)


class UtilityAsDisutilityModel(SeparableDisutilityModel):
    """2.(a): Question 1 as a special case: l_ref_t = L_max and D_t = u (L_max - L_t) >= 0.

    Its objective equals the Question 1 objective (cost - u sum_t L_t) plus the constant 24 u L_max.
    """

    def __init__(self, data: InputData, **kwargs):
        super().__init__(data, reference=np.full(data.n_hours, data.load_max_kWh), **kwargs)

    def _disutility(self, t: int):
        return self.data.consumption_utility * (self.ref[t] - self.var["load"][t])

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        return self.data.consumption_utility * (self.ref - load)


class LinearUtilityModel(SeparableDisutilityModel):
    """Question 1: min sum_t cost_t - u L_t (constant marginal utility u, no reference profile).

    ``reference`` (e.g. the Question 2 profile) is only used to measure deviations in the metrics; the
    reported "disutility" is then minus the utility, -u L_t (DKK).
    """

    def _disutility(self, t: int):
        return -self.data.consumption_utility * self.var["load"][t]

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        return -self.data.consumption_utility * load
