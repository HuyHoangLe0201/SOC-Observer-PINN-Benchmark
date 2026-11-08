"""
Loader for the NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery
Aging Dataset (B. Saha and K. Goebel, 2007). Commercial 18650 cells, 2.0 Ah rated.

Each .mat file contains a 'cycle' struct array with 'charge', 'discharge' and
'impedance' operations. This module extracts measured discharge time series and
EIS-derived resistances (Re, Rct) without any synthetic modification.
"""

import os
import numpy as np
import scipy.io as sio

DATA_ROOT = r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\NASABattery\5. Battery Data Set"

CELL_DIRS = {
    "B0005": "1. BatteryAgingARC-FY08Q4", "B0006": "1. BatteryAgingARC-FY08Q4",
    "B0007": "1. BatteryAgingARC-FY08Q4", "B0018": "1. BatteryAgingARC-FY08Q4",
}
for _c in ["B0025", "B0026", "B0027", "B0028", "B0029", "B0030", "B0031", "B0032",
           "B0033", "B0034", "B0036", "B0038", "B0039", "B0040",
           "B0041", "B0042", "B0043", "B0044"]:
    CELL_DIRS[_c] = "3. BatteryAgingARC_25-44"

RATED_CAPACITY_AH = 2.0
_cache = {}


def _load(cell_id):
    if cell_id in _cache:
        return _cache[cell_id]
    path = os.path.join(DATA_ROOT, CELL_DIRS[cell_id], f"{cell_id}.mat")
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    cycles = mat[cell_id].cycle
    _cache[cell_id] = cycles
    return cycles


def discharge_cycles(cell_id, min_duration_s=300.0):
    """
    Returns a list of dicts for every valid discharge cycle:
      t [s], v [V], i [A, positive = discharge], temp [degC], ambient [degC],
      capacity_ah (coulomb-counted to the end of the record), idx (discharge index)
    """
    out = []
    k = 0
    for c in _load(cell_id):
        if str(c.type).strip() != "discharge":
            continue
        d = c.data
        t = np.atleast_1d(np.asarray(d.Time, dtype=float))
        v = np.atleast_1d(np.asarray(d.Voltage_measured, dtype=float))
        i = -np.atleast_1d(np.asarray(d.Current_measured, dtype=float))  # NASA: negative = discharge
        temp = np.atleast_1d(np.asarray(d.Temperature_measured, dtype=float))
        if t.size < 20 or (t[-1] - t[0]) < min_duration_s:
            k += 1
            continue
        order = np.argsort(t)
        t, v, i, temp = t[order], v[order], i[order], temp[order]
        dt = np.diff(t, prepend=t[0])
        q_ah = np.cumsum(i * dt) / 3600.0
        out.append({
            "idx": k, "t": t - t[0], "v": v, "i": i, "temp": temp,
            "ambient": float(c.ambient_temperature),
            "capacity_ah": float(q_ah[-1]),
            "capacity_nasa": float(np.atleast_1d(getattr(d, "Capacity", np.nan))[0]),
        })
        k += 1
    return out


def impedance_records(cell_id):
    """Returns list of (order, Re [Ohm], Rct [Ohm], complex spectrum)."""
    out = []
    for n, c in enumerate(_load(cell_id)):
        if str(c.type).strip() != "impedance":
            continue
        d = c.data
        try:
            re_, rct = float(np.real(d.Re)), float(np.real(d.Rct))
        except Exception:
            continue
        z = np.atleast_1d(np.asarray(getattr(d, "Rectified_Impedance", np.array([])), dtype=complex))
        out.append({"order": n, "Re": re_, "Rct": rct, "Z": z})
    return out


def resample_uniform(rec, dt=1.0):
    """Linear resampling of a discharge record onto a uniform grid (zero-order for current)."""
    t = rec["t"]
    tg = np.arange(0.0, t[-1], dt)
    idx = np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 1)
    return {
        "t": tg,
        "v": np.interp(tg, t, rec["v"]),
        "i": rec["i"][idx],
        "temp": np.interp(tg, t, rec["temp"]),
        "ambient": rec["ambient"],
        "capacity_ah": rec["capacity_ah"],
        "idx": rec["idx"],
    }


if __name__ == "__main__":
    for cid in ["B0005", "B0018", "B0025", "B0029", "B0033", "B0041"]:
        recs = discharge_cycles(cid)
        caps = [r["capacity_ah"] for r in recs]
        dts = np.median(np.diff(recs[0]["t"]))
        imp = impedance_records(cid)
        print(f"{cid}: {len(recs)} discharges, amb={recs[0]['ambient']}C, "
              f"Q first={caps[0]:.3f} last={caps[-1]:.3f} Ah, NASA Q first={recs[0]['capacity_nasa']:.3f}, "
              f"median dt={dts:.1f}s, len0={len(recs[0]['t'])}, I range=[{recs[0]['i'].min():.2f},{recs[0]['i'].max():.2f}] "
              f"V range=[{recs[0]['v'].min():.2f},{recs[0]['v'].max():.2f}], EIS n={len(imp)}"
              + (f" Re0={imp[0]['Re']:.4f} Rct0={imp[0]['Rct']:.4f}" if imp else ""))
