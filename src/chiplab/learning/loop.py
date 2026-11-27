"""Learning loop: improve ChipLab's model predictions and recommendations from validated experiments.

Flow
1. A design tool records a *prediction* before anything is built (``record_prediction``): model
   id, metric, value, unit, and the design features it depended on. Predictions are immutable, so
   a measurement can only ever be compared with what ChipLab said beforehand.
2. A researcher submits a *measurement* against that prediction (``submit_measurement``). It is
   validated: real-world source kind (bench / silicon / fpga / published), method, date, matching
   unit, a stated uncertainty or >= 3 repeats, physical bounds, and an outlier check against the
   current model. ChipLab's own simulation output is refused, so the loop never trains on itself.
   Anything failing is stored as rejected or needs_review with reasons, never used.
3. ``refit`` learns a correction per (model, metric): Bayesian linear regression of the residual
   (log ratio for positive metrics, difference otherwise) on the design features, with a prior
   centred on "no correction" and per-point noise from the stated uncertainty. A new version is
   *accepted* only if leave-one-out error beats the uncorrected model on >= 3 validated points;
   otherwise it stays a candidate. Every version is kept, nothing is overwritten, and rollback is
   one call.
4. ``corrected`` applies the accepted correction to a fresh prediction and returns the value with
   a predictive interval and its provenance. ``recommend`` ranks candidate designs on corrected
   predictions with uncertainty and suggests the most informative next experiment.

Limitation stated plainly: the loop cannot tell a real measurement from a fabricated one. It
relies on the submitter's attestation and records who submitted what.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# "signoff": reports from a real P&R/STA flow (e.g. LibreLane), higher fidelity than ChipLab
REAL_SOURCES = ("bench", "silicon", "fpga", "published", "signoff")
REFUSED_SOURCES = ("simulation", "chiplab", "synthetic", "model")
MIN_POINTS = 3
DEFAULT_DB = Path(os.environ.get("CHIPLAB_LEARNING_DB",
                                 Path.home() / ".local" / "share" / "chiplab" / "learning.db"))


class LearningError(ValueError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LearningLoop:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, created TEXT, model_id TEXT, metric TEXT, value REAL,
            unit TEXT, scale TEXT, design_id TEXT, family TEXT, features TEXT, provenance TEXT);
        CREATE TABLE IF NOT EXISTS measurements (
            id TEXT PRIMARY KEY, created TEXT, prediction_id TEXT, value REAL, unit TEXT,
            uncertainty REAL, repeats TEXT, source_kind TEXT, method TEXT, measured_on TEXT,
            submitted_by TEXT, reference TEXT, attested_real INTEGER, status TEXT, reasons TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id));
        CREATE TABLE IF NOT EXISTS corrections (
            model_id TEXT, metric TEXT, version INTEGER, created TEXT, status TEXT,
            payload TEXT, PRIMARY KEY(model_id, metric, version));
        CREATE TABLE IF NOT EXISTS builds (
            id TEXT PRIMARY KEY, created TEXT, family TEXT, design_id TEXT, worked INTEGER,
            notes TEXT, submitted_by TEXT);
        """)
        self._db.commit()

    # ------------------------------------------------------------------ predictions
    def record_prediction(self, model_id: str, metric: str, value: float, unit: str,
                          features: dict[str, float] | None = None, design_id: str = "",
                          family: str = "", provenance: str = "") -> str:
        if not (model_id and metric and unit):
            raise LearningError("model_id, metric and unit are required")
        if not math.isfinite(float(value)):
            raise LearningError("prediction value must be finite")
        feats = {k: float(v) for k, v in (features or {}).items()}
        if not all(math.isfinite(v) for v in feats.values()):
            raise LearningError("features must be finite numbers")
        pid = "pred_" + uuid.uuid4().hex[:12]
        scale = "log" if value > 0 else "linear"
        self._db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (pid, _now(), model_id, metric, float(value), unit, scale, design_id,
                          family, json.dumps(feats, sort_keys=True), provenance))
        self._db.commit()
        return pid

    def prediction(self, pid: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
        if row is None:
            raise LearningError(f"unknown prediction {pid!r}")
        d = dict(row)
        d["features"] = json.loads(d["features"])
        return d

    # ------------------------------------------------------------------ measurements
    def submit_measurement(self, prediction_id: str, value: float, unit: str, *,
                           source_kind: str, method: str, measured_on: str, submitted_by: str,
                           attested_real: bool, uncertainty: float | None = None,
                           repeats: list[float] | None = None, reference: str = "",
                           confirm_outlier: bool = False) -> dict[str, Any]:
        """Validate and store one measurement; returns its status and the reasons."""
        pred = self.prediction(prediction_id)
        reasons: list[str] = []
        review: list[str] = []
        kind = (source_kind or "").lower()
        if kind in REFUSED_SOURCES:
            reasons.append(f"source_kind {kind!r} is model output; the loop learns only from "
                           "real measurements")
        elif kind not in REAL_SOURCES:
            reasons.append(f"source_kind must be one of {REAL_SOURCES}")
        if not attested_real:
            reasons.append("submitter did not attest the data are real measurements")
        if not (method or "").strip():
            reasons.append("method (instrument / procedure) is required")
        if not (submitted_by or "").strip():
            reasons.append("submitted_by is required")
        if not (measured_on or "").strip():
            reasons.append("measured_on date is required")
        if kind == "published" and not reference.strip():
            reasons.append("published data need a reference (DOI or URL)")
        if unit != pred["unit"]:
            reasons.append(f"unit {unit!r} does not match the prediction's {pred['unit']!r}")
        v = float(value)
        if not math.isfinite(v):
            reasons.append("value must be finite")
        if repeats:
            rep = [float(r) for r in repeats]
            if len(rep) >= 3 and all(math.isfinite(r) for r in rep):
                if uncertainty is None:
                    uncertainty = float(np.std(rep, ddof=1) / math.sqrt(len(rep)))
                if abs(np.mean(rep) - v) > 3 * (uncertainty or 0) + 1e-12 * abs(v):
                    reasons.append("value is not consistent with the mean of the repeats")
            else:
                reasons.append("repeats need >= 3 finite values")
        if uncertainty is None or not (uncertainty > 0 and math.isfinite(uncertainty)):
            reasons.append("a positive uncertainty or >= 3 repeats is required")
        if pred["scale"] == "log" and math.isfinite(v) and v <= 0:
            reasons.append("value must be positive for this metric")
        if not reasons:
            z = self._outlier_z(pred, v, float(uncertainty))
            if z is not None and z > 5 and not confirm_outlier:
                review.append(f"{z:.1f} sigma from the current model; resubmit with "
                              "confirm_outlier=True after checking the setup")
        status = "rejected" if reasons else "needs_review" if review else "validated"
        mid = "meas_" + uuid.uuid4().hex[:12]
        self._db.execute(
            "INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, _now(), prediction_id, v, unit, uncertainty, json.dumps(repeats or []), kind,
             method, measured_on, submitted_by, reference, int(bool(attested_real)), status,
             json.dumps(reasons + review)))
        self._db.commit()
        return {"measurement_id": mid, "status": status, "reasons": reasons + review,
                "used_for_learning": status == "validated"}

    def _outlier_z(self, pred: dict[str, Any], value: float, unc: float) -> float | None:
        """Distance (in sigma) of a new point from the accepted correction, if there is one."""
        act = self._active(pred["model_id"], pred["metric"])
        if act is None:
            return None
        c = self.corrected(pred["model_id"], pred["metric"], pred["value"], pred["features"])
        r, u = self._residual(value, pred["value"], unc, pred["scale"])
        center, _ = self._residual(c["value"], pred["value"], 0.0, pred["scale"])
        lo, hi = c["interval_90"]
        sd = (math.log(hi / lo) if pred["scale"] == "log" and lo > 0 else hi - lo) / (2 * 1.645)
        return abs(r - center) / math.sqrt(sd ** 2 + u ** 2)

    def _validated(self, model_id: str, metric: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT m.value AS meas, m.uncertainty AS unc, p.value AS pred, p.scale, p.features,"
            " p.design_id FROM measurements m JOIN predictions p ON m.prediction_id = p.id "
            "WHERE p.model_id=? AND p.metric=? AND m.status='validated' ORDER BY m.created",
            (model_id, metric)).fetchall()
        return [dict(r) | {"features": json.loads(r["features"])} for r in rows]

    # ------------------------------------------------------------------ correction model
    @staticmethod
    def _residual(meas: float, pred: float, unc: float, scale: str) -> tuple[float, float]:
        if scale == "log":
            return math.log(meas / pred), unc / meas
        return meas - pred, unc

    @staticmethod
    def _design(rows: list[dict[str, Any]], names: list[str], mu: np.ndarray, sd: np.ndarray
                ) -> np.ndarray:
        x = np.array([[r["features"].get(n, 0.0) for n in names] for r in rows], float)
        x = (x - mu) / sd if names else x.reshape(len(rows), 0)
        return np.hstack([np.ones((len(rows), 1)), x])

    @staticmethod
    def _posterior(x: np.ndarray, r: np.ndarray, var: np.ndarray, prior: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
        a = np.diag(1 / prior) + x.T @ (x / var[:, None])
        cov = np.linalg.inv(a)
        return cov @ (x.T @ (r / var)), cov

    def _fit(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        names = sorted({k for r in rows for k in r["features"]})
        if len(rows) < len(names) + 3:            # too few points for feature slopes
            names = []
        xs = np.array([[r["features"].get(n, 0.0) for n in names] for r in rows], float)
        mu = xs.mean(0) if names else np.zeros(0)
        sd = xs.std(0) if names else np.zeros(0)
        sd[sd == 0] = 1.0
        res = [self._residual(r["meas"], r["pred"], r["unc"], r["scale"]) for r in rows]
        r = np.array([a for a, _ in res])
        u2 = np.array([b for _, b in res]) ** 2
        x = self._design(rows, names, mu, sd)
        prior = np.array([1.0] + [0.25] * len(names))
        # extra scatter s^2 the stated uncertainties do not explain: max marginal likelihood
        best = None
        for s2 in [0.0] + list(np.geomspace(1e-6, 4.0, 40) * max(1e-12, float(np.var(r)) + 1e-12)):
            var = u2 + s2
            cov_y = x @ np.diag(prior) @ x.T + np.diag(var)
            sign, logdet = np.linalg.slogdet(cov_y)
            ll = -0.5 * (logdet + r @ np.linalg.solve(cov_y, r))
            if best is None or ll > best[0]:
                best = (ll, s2)
        s2 = best[1]
        w, cov = self._posterior(x, r, u2 + s2, prior)
        # leave-one-out: corrected vs uncorrected squared residual (in residual space)
        loo_c, loo_0 = [], []
        for i in range(len(rows)):
            keep = np.arange(len(rows)) != i
            wi, _ = self._posterior(x[keep], r[keep], (u2 + s2)[keep], prior)
            loo_c.append(float((r[i] - x[i] @ wi) ** 2))
            loo_0.append(float(r[i] ** 2))
        return {"features": names, "mu": mu.tolist(), "sd": sd.tolist(), "weights": w.tolist(),
                "cov": cov.tolist(), "extra_var": float(s2), "scale": rows[-1]["scale"],
                "n": len(rows), "loo_mse_corrected": float(np.mean(loo_c)),
                "loo_mse_uncorrected": float(np.mean(loo_0))}

    def refit(self, model_id: str, metric: str) -> dict[str, Any]:
        rows = self._validated(model_id, metric)
        if not rows:
            return {"model_id": model_id, "metric": metric, "status": "no_data", "n": 0}
        fit = self._fit(rows)
        better = fit["loo_mse_corrected"] < fit["loo_mse_uncorrected"]
        status = "accepted" if fit["n"] >= MIN_POINTS and better else "candidate"
        fit["why"] = ("leave-one-out error improved" if status == "accepted" else
                      f"needs >= {MIN_POINTS} validated points" if fit["n"] < MIN_POINTS else
                      "correction did not beat the uncorrected model out of sample")
        prev = self._db.execute("SELECT MAX(version) FROM corrections WHERE model_id=? AND "
                                "metric=?", (model_id, metric)).fetchone()[0] or 0
        self._db.execute("INSERT INTO corrections VALUES (?,?,?,?,?,?)",
                         (model_id, metric, prev + 1, _now(), status, json.dumps(fit)))
        self._db.commit()
        return {"model_id": model_id, "metric": metric, "version": prev + 1, "status": status,
                "n": fit["n"], "why": fit["why"], "loo_mse_corrected": fit["loo_mse_corrected"],
                "loo_mse_uncorrected": fit["loo_mse_uncorrected"],
                "bias_factor" if fit["scale"] == "log" else "bias_offset":
                    math.exp(fit["weights"][0]) if fit["scale"] == "log" else fit["weights"][0]}

    def rollback(self, model_id: str, metric: str, to_version: int) -> dict[str, Any]:
        """Re-issue an earlier version as the newest one (history is never rewritten)."""
        row = self._db.execute("SELECT status, payload FROM corrections WHERE model_id=? AND "
                               "metric=? AND version=?", (model_id, metric, to_version)).fetchone()
        if row is None:
            raise LearningError("no such version")
        prev = self._db.execute("SELECT MAX(version) FROM corrections WHERE model_id=? AND "
                                "metric=?", (model_id, metric)).fetchone()[0]
        payload = json.loads(row["payload"]) | {"rollback_of": to_version}
        self._db.execute("INSERT INTO corrections VALUES (?,?,?,?,?,?)",
                         (model_id, metric, prev + 1, _now(), row["status"], json.dumps(payload)))
        self._db.commit()
        return {"model_id": model_id, "metric": metric, "version": prev + 1,
                "restored": to_version, "status": row["status"]}

    def _active(self, model_id: str, metric: str) -> tuple[int, dict[str, Any]] | None:
        row = self._db.execute("SELECT version, status, payload FROM corrections WHERE "
                               "model_id=? AND metric=? ORDER BY version DESC LIMIT 1",
                               (model_id, metric)).fetchone()
        if row is None or row["status"] != "accepted":
            return None
        return row["version"], json.loads(row["payload"])

    def corrected(self, model_id: str, metric: str, value: float,
                  features: dict[str, float] | None = None) -> dict[str, Any]:
        """Apply the accepted correction (if any) with a 90 % predictive interval."""
        act = self._active(model_id, metric)
        if act is None:
            return {"value": value, "corrected": False, "interval_90": None,
                    "provenance": "uncorrected model output (no accepted correction yet)"}
        version, fit = act
        names = fit["features"]
        f = features or {}
        x = np.array([1.0] + [(f.get(n, 0.0) - m) / s
                              for n, m, s in zip(names, fit["mu"], fit["sd"])])
        mean = float(x @ np.array(fit["weights"]))
        sd = math.sqrt(float(x @ np.array(fit["cov"]) @ x) + fit["extra_var"])
        if fit["scale"] == "log" and value > 0:
            val, lo, hi = (value * math.exp(mean), value * math.exp(mean - 1.645 * sd),
                           value * math.exp(mean + 1.645 * sd))
        else:
            val, lo, hi = value + mean, value + mean - 1.645 * sd, value + mean + 1.645 * sd
        return {"value": val, "raw_value": value, "corrected": True, "interval_90": [lo, hi],
                "relative_sd": sd if fit["scale"] == "log" else None,
                "provenance": f"learning-loop correction v{version} from {fit['n']} validated "
                              "measurements"}

    # ------------------------------------------------------------------ builds + recommendations
    def record_build(self, family: str, design_id: str, worked: bool, submitted_by: str,
                     notes: str = "") -> str:
        if not (family and submitted_by):
            raise LearningError("family and submitted_by are required")
        bid = "build_" + uuid.uuid4().hex[:12]
        self._db.execute("INSERT INTO builds VALUES (?,?,?,?,?,?,?)",
                         (bid, _now(), family, design_id, int(bool(worked)), notes, submitted_by))
        self._db.commit()
        return bid

    def build_success(self, family: str) -> dict[str, Any]:
        """Beta(1, 1) posterior on 'a build of this family works first time'."""
        ok, n = self._db.execute("SELECT COALESCE(SUM(worked), 0), COUNT(*) FROM builds WHERE "
                                 "family=?", (family,)).fetchone()
        a, b = 1 + ok, 1 + n - ok
        return {"family": family, "builds": n, "worked": ok, "p_success_mean": a / (a + b)}

    def recommend(self, candidates: list[dict[str, Any]], objective: str,
                  minimize: bool = True, constraints: dict[str, tuple[float | None, float | None]]
                  | None = None) -> dict[str, Any]:
        """Rank candidates on corrected predictions.

        Each candidate: {"id", "family", "model_prefix", "features", "metrics": {name: value}}.
        Score = corrected objective; a 90 % interval violating a constraint marks it risky.
        The next experiment is the feasible candidate whose objective is least certain among
        the top three (largest interval relative to its value).
        """
        if not candidates:
            raise LearningError("no candidates")
        cons = constraints or {}
        ranked = []
        for c in candidates:
            if objective not in c["metrics"]:
                raise LearningError(f"candidate {c.get('id')} has no metric {objective!r}")
            mets = {k: self.corrected(f"{c['model_prefix']}.{k}", k, float(v), c.get("features"))
                    for k, v in c["metrics"].items()}
            feasible, risky = True, []
            for k, (lo, hi) in cons.items():
                m = mets.get(k)
                if m is None:
                    continue
                if (lo is not None and m["value"] < lo) or (hi is not None and m["value"] > hi):
                    feasible = False
                iv = m["interval_90"]
                if iv and ((lo is not None and iv[0] < lo) or (hi is not None and iv[1] > hi)):
                    risky.append(k)
            obj = mets[objective]
            width = (obj["interval_90"][1] - obj["interval_90"][0]) / abs(obj["value"]) \
                if obj["interval_90"] and obj["value"] else None
            ranked.append({"id": c["id"], "family": c.get("family", ""), "feasible": feasible,
                           "constraint_risk": risky, "objective": obj["value"],
                           "objective_interval_90": obj["interval_90"],
                           "relative_uncertainty": width, "corrected": obj["corrected"],
                           "build_success": self.build_success(c.get("family", "")),
                           "metrics": {k: m["value"] for k, m in mets.items()}})
        ranked.sort(key=lambda r: (not r["feasible"], r["objective"] if minimize
                                   else -r["objective"]))
        top = [r for r in ranked if r["feasible"]][:3]
        nxt = None
        if top:
            # uncorrected predictions have unknown error: measuring one teaches the most
            nxt = max(top, key=lambda r: (not r["corrected"], r["relative_uncertainty"] or 0))
        return {"objective": objective, "minimize": minimize, "ranking": ranked,
                "best": top[0]["id"] if top else None,
                "next_experiment": nxt and {"id": nxt["id"], "why": (
                    "no validated measurement has corrected this model yet" if not nxt["corrected"]
                    else "widest predictive interval among the top candidates")}}

    def status(self) -> dict[str, Any]:
        q = self._db.execute
        counts = {s: n for s, n in q("SELECT status, COUNT(*) FROM measurements GROUP BY status")}
        models = []
        for model_id, metric in q("SELECT DISTINCT model_id, metric FROM corrections"):
            row = q("SELECT version, status, payload FROM corrections WHERE model_id=? AND "
                    "metric=? ORDER BY version DESC LIMIT 1", (model_id, metric)).fetchone()
            fit = json.loads(row["payload"])
            models.append({"model_id": model_id, "metric": metric, "version": row["version"],
                           "status": row["status"], "n": fit["n"], "why": fit.get("why")})
        return {"predictions": q("SELECT COUNT(*) FROM predictions").fetchone()[0],
                "measurements": counts, "corrections": models,
                "builds": q("SELECT COUNT(*) FROM builds").fetchone()[0],
                "limitation": "cannot distinguish real from fabricated data; relies on the "
                              "submitter's attestation and records who submitted each point"}
