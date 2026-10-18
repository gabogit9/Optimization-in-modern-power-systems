
from __future__ import annotations

from gurobipy import GRB
from .model_q2c import QuadraticDisutilityModel


class BatteryModel(QuadraticDisutilityModel):
    """Q3(g): quadratic-disutility model with battery storage."""

    def build(self) -> "BatteryModel":
        super().build()
        d, m, T = self.data, self.m, self.T

        if d.battery_capacity_kWh is None:
            raise ValueError("Battery parameters are missing.")

        # Replace the original power balance with one including the battery.
        m.remove(list(self.con["balance"].values()))
        m.update()

        C = m.addVars(T, lb=0, ub=d.battery_max_charge_kW, name="charge")
        D = m.addVars(T, lb=0, ub=d.battery_max_discharge_kW, name="discharge")
        S = m.addVars(range(d.n_hours + 1), lb=0,
                      ub=d.battery_capacity_kWh, name="soc")
        self.var.update(charge=C, discharge=D, soc=S)

        L, PV, I, E = (self.var[k] for k in ("load", "pv", "import", "export"))

        self.con["balance"] = m.addConstrs(
            (L[t] - PV[t] - I[t] + E[t] + C[t] - D[t] == 0
             for t in T), name="balance"
        )
        self.con["soc_initial"] = m.addConstr(
            S[0] == d.battery_initial_soc_kWh, name="soc_initial"
        )
        self.con["soc_dynamics"] = m.addConstrs(
            (S[t + 1] == S[t]
             + d.battery_charging_efficiency * C[t]
             - D[t] / d.battery_discharging_efficiency
             for t in T), name="soc_dynamics"
        )
        self.con["soc_final"] = m.addConstr(
            S[d.n_hours] == d.battery_initial_soc_kWh, name="soc_final"
        )

        m.update()
        return self
