#!/usr/bin/env python3
"""
IEM-OS & QID-OS V55.3.9 Definitive Reference Kernel (Strictly Hardened & Semantic-Locked Edition)
Formally Specified Adaptive Control Kernel with Strict Interval Feasibility Semantics,
Decoupled Safety-Cache Hashing, Separated Monotonicity Status, and Audit-Locked Discipline (V55.3.9)

Author: Tadayuki M. / Collaborating AI (Gemini)

========================================================================================
V55.3.9 ARCHITECTURAL & AUDIT SPECIFICATIONS
========================================================================================
1. Chronological Closed-Loop Timing & Closed-Loop Invariant Closure:
   - Step t: Observe x_t -> Observe y_t -> Analytical SMI Update (\Lambda_t -> \Lambda_t^+) ->
     Certified Invariant Synthesis K^*(\Lambda_t^+) -> Runtime Authorization:
       a) Check Non-Empty Parameter Set (\Lambda_t^+ \neq \emptyset)
       b) Check Candidate Action ∈ Certificate Action Space Bound
       c) Check B_t \subseteq K^*(\Lambda_t^+) (Belief Containment)
       d) Check \Lambda_t^+ \subseteq \Lambda_{\rm cert} (Parameter Set Binding)
       e) Check Post(B_t, u, \Lambda_t^+, W) \subseteq K^*(\Lambda_t^+) (Robust Post-Condition)
     -> Optimal Control Selection u_t -> Actuation -> x_{t+1}.
2. Scientific Claim Discipline:
   - Strictly framed as a "Finite-Step Certified Robust Controlled-Invariant Inner Approximation"
     with component-wise sufficient union containment and tamper-evident integrity binding.
========================================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.optimize import linprog

# ================================================================
# Version Metadata & Scientific Naming Discipline
# ================================================================

MODEL_VERSION = "IEM-OS-V55.3.9-DefinitiveReferenceKernel"
IDENTIFICATION_VERSION = "Analytical-SMI-ScalarAffineObservation-V2"
CACHE_VERSION = "ConfigurationBoundMonotonicCache-V15"
ALGORITHM_VERSION = "RobustUnionInvariant-BiDirectionalFixedPoint-Witness-V12"

STATUS_CERTIFIED = "CERTIFIED_CONVERGED"
STATUS_NOT_CONVERGED = "NOT_CONVERGED"
STATUS_EMPTY = "EMPTY_SET"
STATUS_MONOTONICITY_VIOLATION = "MONOTONICITY_VIOLATION"
STATUS_UNKNOWN = "UNKNOWN"

ID_UPDATED = "UPDATED"
ID_UNCHANGED = "UNCHANGED"
ID_MISMATCH = "MODEL_MISMATCH"


# ================================================================
# Utilities & Environment Provenance Binding
# ================================================================

def _as_float_vector(x, name: str, expected_dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if expected_dim is not None and arr.shape != (expected_dim,):
        raise ValueError(f"{name} shape mismatch: expected ({expected_dim},), got {arr.shape}.")
    return arr


def _as_float_matrix(x, name: str, expected_rows=None, expected_cols=None) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise ValueError(f"{name} row dimension mismatch: {arr.shape[0]} != {expected_rows}")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise ValueError(f"{name} column dimension mismatch: {arr.shape[1]} != {expected_cols}")
    return arr


def _canonical_float(x):
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            raise ValueError("Non-finite value cannot enter provenance payload.")
        return float(format(float(x), ".17g"))
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_canonical_float(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_canonical_float(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _canonical_float(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    return x


def _sha256_canonical(obj) -> str:
    canonical = _canonical_float(obj)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ================================================================
# Data Structures & Witness Metadata
# ================================================================

@dataclass(frozen=True)
class LPResult:
    status: str
    objective: Optional[float] = None
    raw_status: Optional[int] = None
    message: str = ""


@dataclass(frozen=True)
class ClosedInterval:
    low: float
    high: float

    def __post_init__(self):
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("Interval bounds must be finite.")
        if self.low > self.high:
            raise ValueError(f"Invalid interval bounds: [{self.low}, {self.high}]")

    def contains(self, other: "ClosedInterval", atol=1e-12) -> bool:
        return self.low <= other.low + atol and self.high >= other.high - atol

    def subset_of(self, other: "ClosedInterval", atol=1e-12) -> bool:
        return other.contains(self, atol=atol)

    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class PolytopeWitness:
    component_id: int
    parent_component_id: Optional[int]
    action_u_c: Optional[Tuple[float, ...]]
    action_u_q: Optional[Tuple[float, ...]]
    parameter_uncertainty_set: Tuple[Tuple[float, float], ...]
    generation_iteration: int
    generation_reason: str = "INVARIANT_SYNTHESIS"


# ================================================================
# Analytical Set-Membership Identifier (Scalar Affine Observation)
# ================================================================

class AnalyticalScalarSetMembershipIdentifier:
    def __init__(self, v_max: float, tolerance: float = 1e-8):
        if v_max < 0 or not np.isfinite(v_max):
            raise ValueError("v_max must be finite and non-negative.")
        self.v_max = float(v_max)
        self.tolerance = float(tolerance)

    def _safe_interval(self, lo: float, hi: float) -> Optional[ClosedInterval]:
        # V55.3.9 Strict Hardening: Eliminate false-feasible tolerance inflation.
        # If lo > hi, it is strictly empty set. Fail-Closed adherence.
        if lo > hi:
            return None
        return ClosedInterval(lo, hi)

    def _apply_leq(self, interval: ClosedInterval, a: float, b: float) -> Optional[ClosedInterval]:
        lo, hi = interval.low, interval.high
        if abs(a) <= self.tolerance:
            return interval if 0.0 <= b + self.tolerance else None
        boundary = b / a
        if a > 0:
            hi = min(hi, boundary)
        else:
            lo = max(lo, boundary)
        return self._safe_interval(lo, hi)

    def _apply_geq(self, interval: ClosedInterval, a: float, b: float) -> Optional[ClosedInterval]:
        lo, hi = interval.low, interval.high
        if abs(a) <= self.tolerance:
            return interval if 0.0 >= b - self.tolerance else None
        boundary = b / a
        if a > 0:
            lo = max(lo, boundary)
        else:
            hi = min(hi, boundary)
        return self._safe_interval(lo, hi)

    def _nonnegative_lambda_interval(self, current: ClosedInterval, phi_min: float, phi_max: float, y_lo: float, y_hi: float) -> Optional[ClosedInterval]:
        lo = max(current.low, 0.0)
        hi = current.high
        if lo > hi:
            return None
        res = ClosedInterval(lo, hi)
        res = self._apply_leq(res, phi_min, y_hi)
        if res is None:
            return None
        return self._apply_geq(res, phi_max, y_lo)

    def _nonpositive_lambda_interval(self, current: ClosedInterval, phi_min: float, phi_max: float, y_lo: float, y_hi: float) -> Optional[ClosedInterval]:
        lo = current.low
        hi = min(current.high, 0.0)
        if lo > hi:
            return None
        res = ClosedInterval(lo, hi)
        res = self._apply_leq(res, phi_max, y_hi)
        if res is None:
            return None
        return self._apply_geq(res, phi_min, y_lo)

    @staticmethod
    def merge_intervals(intervals: Sequence[ClosedInterval], atol: float = 1e-12) -> List[ClosedInterval]:
        if not intervals:
            return []
        ordered = sorted(intervals, key=lambda x: (x.low, x.high))
        merged = [ordered[0]]
        for curr in ordered[1:]:
            prev = merged[-1]
            if curr.low <= prev.high + atol:
                merged[-1] = ClosedInterval(prev.low, max(prev.high, curr.high))
            else:
                merged.append(curr)
        return merged

    def feasible_lambda_intervals(self, current: ClosedInterval, phi_min: float, phi_max: float, y_measured: float) -> List[ClosedInterval]:
        if not np.isfinite(y_measured):
            return []
        if phi_min > phi_max:
            phi_min, phi_max = phi_max, phi_min
        y_lo = y_measured - self.v_max
        y_hi = y_measured + self.v_max
        
        intervals = []
        pos = self._nonnegative_lambda_interval(current, phi_min, phi_max, y_lo, y_hi)
        if pos is not None:
            intervals.append(pos)
        neg = self._nonpositive_lambda_interval(current, phi_min, phi_max, y_lo, y_hi)
        if neg is not None:
            intervals.append(neg)
        return self.merge_intervals(intervals, atol=self.tolerance)


# ================================================================
# Pure Safety Provenance Container (Tamper-Evident Integrity Binding)
# ================================================================

class V5539Certificate:
    def __init__(self, polytope_list, witness_list, admissible_actions, status, iterations, lambda_intervals, H_safe, h_safe, action_space, w_max, v_max, certificate_comparison_tolerance, observation_model, dynamics_model, solver_provenance, algorithm_metadata):
        self.polytope_list = polytope_list
        self.witness_list = witness_list
        self.admissible_actions = admissible_actions
        self.status = status
        self.iterations = iterations
        self.lambda_intervals = [[float(i.low), float(i.high)] for i in lambda_intervals]
        self.H_safe = np.asarray(H_safe, dtype=float)
        self.h_safe = np.asarray(h_safe, dtype=float)
        self.action_space_canonical = {
            "u_c": [np.asarray(u, dtype=float).tolist() for u in action_space["u_c"]],
            "u_q": [np.asarray(u, dtype=float).tolist() for u in action_space["u_q"]],
        }
        self.w_max = float(w_max)
        self.v_max = float(v_max)
        self.certificate_comparison_tolerance = float(certificate_comparison_tolerance)
        self.observation_model = observation_model
        self.dynamics_model = dynamics_model
        self.solver_provenance = solver_provenance
        self.algorithm_metadata = algorithm_metadata
        self.model_version = MODEL_VERSION
        self.provenance_hash = self._compute_provenance_hash()

    def _canonical_polytopes(self):
        res = [{"H": np.asarray(H, dtype=float).tolist(), "h": np.asarray(h, dtype=float).tolist()} for H, h in self.polytope_list]
        return sorted(res, key=lambda x: json.dumps(x, sort_keys=True))

    def _canonical_witnesses(self):
        return [
            {
                "component_id": w.component_id,
                "parent_component_id": w.parent_component_id,
                "action_u_c": list(w.action_u_c) if w.action_u_c is not None else None,
                "action_u_q": list(w.action_u_q) if w.action_u_q is not None else None,
                "parameter_uncertainty_set": [list(iv) for iv in w.parameter_uncertainty_set],
                "generation_iteration": w.generation_iteration,
                "generation_reason": w.generation_reason,
            }
            for w in self.witness_list
        ]

    def safety_provenance_payload(self) -> Dict:
        return {
            "model_version": self.model_version,
            "status": self.status,
            "iterations": self.iterations,
            "lambda_intervals": self.lambda_intervals,
            "H_safe": self.H_safe.tolist(),
            "h_safe": self.h_safe.tolist(),
            "action_space": self.action_space_canonical,
            "polytope_list": self._canonical_polytopes(),
            "witness_list": self._canonical_witnesses(),
            "w_max": self.w_max,
            "v_max": self.v_max,
            "certificate_comparison_tolerance": self.certificate_comparison_tolerance,
            "observation_model": self.observation_model,
            "dynamics_model": self.dynamics_model,
            "solver_provenance": self.solver_provenance,
            "algorithm_metadata": self.algorithm_metadata,
        }

    def _compute_provenance_hash(self) -> str:
        return _sha256_canonical(self.safety_provenance_payload())

    def verify_provenance(self) -> bool:
        return self._compute_provenance_hash() == self.provenance_hash


# ================================================================
# Main IEM-OS V55.3.9 Reference Kernel (Audit-Locked & Hardened)
# ================================================================

class IEMOSV5539Kernel:
    def __init__(self, initial_lambda_interval, dim_state=4, w_max=0.01, v_max=0.005, tolerance=1e-8, cache_lambda_atol=1e-10):
        self.dim_state = int(dim_state)
        if self.dim_state <= 0:
            raise ValueError("dim_state must be positive.")
        self.w_max = float(w_max)
        self.v_max = float(v_max)
        self.tolerance = float(tolerance)
        self.cache_lambda_atol = float(cache_lambda_atol)

        if self.w_max < 0 or self.v_max < 0:
            raise ValueError("Disturbance and noise bounds must be non-negative.")

        self.lambda_intervals = self._validate_lambda_intervals(initial_lambda_interval)
        self.identifier = AnalyticalScalarSetMembershipIdentifier(v_max=self.v_max, tolerance=self.tolerance)

        self.cached_certificate = None
        self.cached_config_key = None
        self.last_identification_status = ID_UNCHANGED
        self.last_identification_intervals = self.lambda_intervals
        self.mock_solver_status = "NORMAL"

    def get_solver_provenance(self) -> Dict:
        return {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "lp_method": "highs",
            "solver_backend": "scipy.optimize.linprog (HiGHS backend)",
            "reproducibility_class": "environment-bound metadata verified",
        }

    def get_performance_provenance(self, cost_func_name: str) -> Dict:
        return {
            "performance_model_id": f"QuadraticControlCost-{cost_func_name}-V1",
            "performance_model_hash": _sha256_canonical({"cost_function": cost_func_name}),
            "decoupled_from_safety": True,
        }

    def get_adaptive_dynamics(self, lam):
        lam = _as_float_vector(lam, "lambda", expected_dim=1)
        A = np.eye(self.dim_state)
        A[0, 1] = 0.1 * lam[0]
        B_c = np.array([[0.1, 0.0], [0.2, 0.1], [0.0, 0.1], [0.0, 0.0]], dtype=float)
        B_q = np.array([[0.0, 0.1], [0.1, 0.0], [0.0, 0.0], [0.1, 0.1]], dtype=float)
        return A, B_c, B_q

    def observation_model_metadata(self) -> Dict:
        return {
            "model_id": IDENTIFICATION_VERSION,
            "equation": "y_t = phi(x_t) * lambda + v_t",
            "phi_definition": "phi = 0.1 * x[1]",
            "noise_set": [-self.v_max, self.v_max],
            "timing_semantics": "Chronological: observe(x_t) -> observe(y_t) -> update(Lambda_t^+) -> certify -> authorize(u_t) -> x_{t+1}",
        }

    def dynamics_model_metadata(self) -> Dict:
        A0 = np.eye(self.dim_state)
        A1 = np.zeros((self.dim_state, self.dim_state))
        A1[0, 1] = 0.1
        B_c = np.array([[0.1, 0.0], [0.2, 0.1], [0.0, 0.1], [0.0, 0.0]], dtype=float)
        B_q = np.array([[0.0, 0.1], [0.1, 0.0], [0.0, 0.0], [0.1, 0.1]], dtype=float)
        return {
            "model_id": "AffineScalarLambdaDynamics-V1",
            "A0": A0.tolist(), "A1": A1.tolist(),
            "B_c": B_c.tolist(), "B_q": B_q.tolist(),
            "equation": "x_{t+1} = (A0 + lambda*A1)x_t + Bc*uc + Bq*uq + w_t",
            "disturbance_set": "box",
        }

    def _validate_lambda_intervals(self, intervals) -> List[ClosedInterval]:
        if isinstance(intervals, tuple) and len(intervals) == 2 and np.asarray(intervals[0]).ndim == 1:
            low = _as_float_vector(intervals[0], "lambda_min", expected_dim=1)
            high = _as_float_vector(intervals[1], "lambda_max", expected_dim=1)
            if np.any(low > high):
                raise ValueError("Invalid lambda interval bounds.")
            return [ClosedInterval(float(low[0]), float(high[0]))]
        if isinstance(intervals, Sequence):
            res = []
            for item in intervals:
                if len(item) != 2:
                    raise ValueError("Lambda interval must contain pairs.")
                res.append(ClosedInterval(float(item[0]), float(item[1])))
            if not res:
                raise ValueError("At least one lambda interval required.")
            return res
        raise ValueError("Invalid lambda interval representation.")

    def validate_belief(self, belief_bounds) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(belief_bounds, (tuple, list)) or len(belief_bounds) != 2:
            return False, None, None
        try:
            lower = _as_float_vector(belief_bounds[0], "belief_lower", expected_dim=self.dim_state)
            upper = _as_float_vector(belief_bounds[1], "belief_upper", expected_dim=self.dim_state)
        except ValueError:
            return False, None, None
        if np.any(lower > upper):
            return False, None, None
        return True, lower, upper

    def validate_safe_set(self, H_safe, h_safe) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            H = _as_float_matrix(H_safe, "H_safe", expected_cols=self.dim_state)
            h = _as_float_vector(h_safe, "h_safe", expected_dim=H.shape[0])
        except ValueError:
            return False, None, None
        return True, H, h

    def validate_action_space(self, action_space) -> bool:
        if not isinstance(action_space, dict) or "u_c" not in action_space or "u_q" not in action_space:
            return False
        if not action_space["u_c"] or not action_space["u_q"]:
            return False
        try:
            for u in action_space["u_c"]:
                _as_float_vector(u, "u_c", expected_dim=2)
            for u in action_space["u_q"]:
                _as_float_vector(u, "u_q", expected_dim=2)
        except ValueError:
            return False
        return True

    def _compute_configuration_key(self, H_safe, h_safe, action_space, max_iterations) -> str:
        # V55.3.9 Hardening: cost_func_name is completely decoupled from safety cache key.
        payload = {
            "cache_version": CACHE_VERSION,
            "model_version": MODEL_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "dim_state": self.dim_state,
            "H_safe": np.asarray(H_safe).tolist(),
            "h_safe": np.asarray(h_safe).tolist(),
            "action_space": {
                "u_c": [np.asarray(u).tolist() for u in action_space["u_c"]],
                "u_q": [np.asarray(u).tolist() for u in action_space["u_q"]],
            },
            "w_max": self.w_max,
            "v_max": self.v_max,
            "certificate_comparison_tolerance": self.tolerance,
            "max_iterations": int(max_iterations),
            "observation_model": self.observation_model_metadata(),
            "dynamics_model": self.dynamics_model_metadata(),
            "solver_provenance": self.get_solver_provenance(),
            "safety_provenance_isolated": True,
        }
        return _sha256_canonical(payload)

    def check_polytope_feasibility(self, H, h) -> LPResult:
        if self.mock_solver_status in ("FORCE_UNKNOWN", "FORCE_UNKNOWN_FEASIBILITY"):
            return LPResult(STATUS_UNKNOWN, message="Mocked solver feasibility failure.")
        try:
            H = np.asarray(H, dtype=float)
            h = np.asarray(h, dtype=float)
            if H.ndim != 2 or h.ndim != 1 or H.shape[0] != h.shape[0]:
                return LPResult(STATUS_UNKNOWN, message="Dimension mismatch in polytope.")
            if H.shape[1] != self.dim_state:
                return LPResult(STATUS_UNKNOWN, message="State dimension mismatch.")
            if H.shape[0] == 0:
                return LPResult(STATUS_EMPTY, raw_status=2, message="Empty constraint set.")

            res = linprog(np.zeros(self.dim_state), A_ub=H, b_ub=h, bounds=(None, None), method="highs")
            if res.success:
                return LPResult("FEASIBLE", objective=res.fun, raw_status=res.status, message=res.message)
            if res.status == 2:
                return LPResult(STATUS_EMPTY, raw_status=res.status, message=res.message)
            return LPResult(STATUS_UNKNOWN, raw_status=res.status, message=res.message)
        except Exception as exc:
            return LPResult(STATUS_UNKNOWN, message=f"Solver exception trapped: {type(exc).__name__}: {exc}")

    def check_polytope_containment(self, H_sub, h_sub, H_super, h_super) -> str:
        if self.mock_solver_status in ("FORCE_UNKNOWN", "FORCE_UNKNOWN_CONTAINMENT"):
            return STATUS_UNKNOWN
        try:
            H_sub = np.asarray(H_sub, dtype=float)
            h_sub = np.asarray(h_sub, dtype=float)
            H_super = np.asarray(H_super, dtype=float)
            h_super = np.asarray(h_super, dtype=float)

            if H_sub.shape[1] != self.dim_state:
                return STATUS_UNKNOWN

            for i in range(H_super.shape[0]):
                res = linprog(-H_super[i], A_ub=H_sub, b_ub=h_sub, bounds=(None, None), method="highs")
                if res.status == 2:
                    return STATUS_EMPTY
                if not res.success:
                    return STATUS_UNKNOWN
                if -res.fun > h_super[i] + self.tolerance:
                    return "VIOLATION"
            return "CERTIFIED"
        except Exception:
            return STATUS_UNKNOWN

    def check_belief_containment_in_certificate(self, belief_lower, belief_upper, cert: V5539Certificate) -> str:
        if cert is None or cert.status != STATUS_CERTIFIED:
            return "VIOLATION"
        H_box = np.vstack([np.eye(self.dim_state), -np.eye(self.dim_state)])
        h_box = np.concatenate([belief_upper, -belief_lower])
        
        for H_k, h_k in cert.polytope_list:
            status = self.check_polytope_containment(H_box, h_box, H_k, h_k)
            if status == STATUS_UNKNOWN:
                return STATUS_UNKNOWN
            if status == "CERTIFIED":
                return "CERTIFIED"
        return "VIOLATION"

    def parameter_set_subset(self, current_intervals: Sequence[ClosedInterval], certified_intervals: Sequence[ClosedInterval]) -> bool:
        if not current_intervals:
            return False
        for curr in current_intervals:
            if not any(curr.subset_of(cert_iv, atol=self.cache_lambda_atol) for cert_iv in certified_intervals):
                return False
        return True

    def update_exact_set_membership_identification(self, y_measured, b_lower, b_upper) -> Tuple[List[ClosedInterval], str]:
        y_arr = np.asarray(y_measured, dtype=float).reshape(-1)
        if y_arr.size != 1:
            return self.lambda_intervals, ID_MISMATCH
        y = float(y_arr[0])
        if not np.isfinite(y):
            return self.lambda_intervals, ID_MISMATCH

        phi_min = 0.1 * float(b_lower[1])
        phi_max = 0.1 * float(b_upper[1])

        cand_intervals = []
        for curr in self.lambda_intervals:
            feas = self.identifier.feasible_lambda_intervals(curr, phi_min, phi_max, y)
            cand_intervals.extend(feas)

        cand_intervals = self.identifier.merge_intervals(cand_intervals, atol=self.tolerance)
        if not cand_intervals:
            self.last_identification_status = ID_MISMATCH
            return self.lambda_intervals, ID_MISMATCH

        old_w = sum(i.width() for i in self.lambda_intervals)
        new_w = sum(i.width() for i in cand_intervals)

        status = ID_UPDATED if new_w < old_w - self.tolerance else ID_UNCHANGED
        self.lambda_intervals = cand_intervals
        self.last_identification_intervals = cand_intervals
        self.last_identification_status = status
        return cand_intervals, status

    def compute_set_valued_robust_predecessor(self, H, h, u_c, u_q, lambda_intervals):
        u_c = _as_float_vector(u_c, "u_c", expected_dim=2)
        u_q = _as_float_vector(u_q, "u_q", expected_dim=2)

        matrices, rhs = [], []
        for interval in lambda_intervals:
            A_min, B_c, B_q = self.get_adaptive_dynamics(np.array([interval.low]))
            A_max, _, _ = self.get_adaptive_dynamics(np.array([interval.high]))

            control_effect = B_c @ u_c + B_q @ u_q
            HBu = H @ control_effect
            rho_W = np.sum(np.abs(H), axis=1) * self.w_max
            h_base = h - HBu - rho_W

            matrices.extend([H @ A_min, H @ A_max])
            rhs.extend([h_base, h_base])

        return np.vstack(matrices), np.concatenate(rhs)

    def remove_redundant_polytopes_with_witness(self, polytope_witness_list):
        if self.mock_solver_status == "FORCE_UNKNOWN_REDUNDANCY":
            return [], STATUS_UNKNOWN
        if not polytope_witness_list:
            return [], "CERTIFIED"
        keep = [True] * len(polytope_witness_list)
        for i, (pw_i, (H_i, h_i)) in enumerate(polytope_witness_list):
            if not keep[i]:
                continue
            for j, (pw_j, (H_j, h_j)) in enumerate(polytope_witness_list):
                if i == j or not keep[j]:
                    continue
                status = self.check_polytope_containment(H_i, h_i, H_j, h_j)
                if status == STATUS_UNKNOWN:
                    return [], STATUS_UNKNOWN
                if status == "CERTIFIED":
                    keep[i] = False
                    break
        filtered = [polytope_witness_list[i] for i in range(len(polytope_witness_list)) if keep[i]]
        return filtered, "CERTIFIED"

    def check_union_containment_sufficient(self, list_A, list_B) -> str:
        if not list_A:
            return "CERTIFIED"
        if not list_B:
            return "VIOLATION"
        for H_a, h_a in list_A:
            covered = False
            unknown_seen = False
            for H_b, h_b in list_B:
                status = self.check_polytope_containment(H_a, h_a, H_b, h_b)
                if status == "CERTIFIED":
                    covered = True
                    break
                if status == STATUS_UNKNOWN:
                    unknown_seen = True
            if not covered:
                return STATUS_UNKNOWN if unknown_seen else "VIOLATION"
        return "CERTIFIED"

    def synthesize_set_valued_invariant(self, H_safe, h_safe, action_space, lambda_intervals, max_iterations=8, cost_func_name="Default"):
        if not lambda_intervals:
            return None, "EMPTY_PARAMETER_SET"

        valid_safe, H_safe, h_safe = self.validate_safe_set(H_safe, h_safe)
        if not valid_safe or not self.validate_action_space(action_space):
            return None, "INVALID_INPUT"

        current_config_key = self._compute_configuration_key(H_safe, h_safe, action_space, max_iterations)

        if (self.cached_certificate is not None and 
            self.cached_config_key == current_config_key and 
            self.cached_certificate.status == STATUS_CERTIFIED):
            cached_intervals = [ClosedInterval(i[0], i[1]) for i in self.cached_certificate.lambda_intervals]
            if all(any(curr.subset_of(cached, atol=self.cache_lambda_atol) for cached in cached_intervals) for curr in lambda_intervals):
                return self.cached_certificate, "CACHE_HIT"

        safe_status = self.check_polytope_feasibility(H_safe, h_safe)
        if safe_status.status == STATUS_UNKNOWN:
            cert = V5539Certificate(
                [], [], [], STATUS_UNKNOWN, 0,
                lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance,
                self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(),
                {"algorithm_version": ALGORITHM_VERSION}
            )
            return cert, "SOLVER_UNKNOWN"

        flat_actions = [(tuple(uc), tuple(uq)) for uc in action_space["u_c"] for uq in action_space["u_q"]]
        param_uncertainty_tuple = tuple((float(i.low), float(i.high)) for i in lambda_intervals)

        if safe_status.status != "FEASIBLE":
            cert = V5539Certificate(
                [], [], [], STATUS_EMPTY if safe_status.status == STATUS_EMPTY else STATUS_UNKNOWN, 0,
                lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance,
                self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(),
                {"algorithm_version": ALGORITHM_VERSION}
            )
            return cert, "CACHE_MISS_SAFE_SET"

        root_witness = PolytopeWitness(component_id=0, parent_component_id=None, action_u_c=None, action_u_q=None, parameter_uncertainty_set=param_uncertainty_tuple, generation_iteration=0, generation_reason="SAFE_SET_INITIALIZATION")
        current_pw_list = [(root_witness, (H_safe.copy(), h_safe.copy()))]
        
        status = STATUS_NOT_CONVERGED
        final_iter = 0
        next_id_counter = 1
        monotonic_descent_flag = False

        for iteration in range(max_iterations):
            final_iter = iteration + 1
            next_pw_list = []
            
            for uc in action_space["u_c"]:
                for uq in action_space["u_q"]:
                    for parent_pw, (H_k, h_k) in current_pw_list:
                        H_pre, h_pre = self.compute_set_valued_robust_predecessor(H_k, h_k, uc, uq, lambda_intervals)
                        
                        H_q = np.vstack([H_k, H_safe, H_pre])
                        h_q = np.concatenate([h_k, h_safe, h_pre])
                        
                        feas = self.check_polytope_feasibility(H_q, h_q)
                        
                        if feas.status == "FEASIBLE":
                            w_item = PolytopeWitness(
                                component_id=next_id_counter,
                                parent_component_id=parent_pw.component_id,
                                action_u_c=tuple(float(v) for v in uc),
                                action_u_q=tuple(float(v) for v in uq),
                                parameter_uncertainty_set=param_uncertainty_tuple,
                                generation_iteration=final_iter,
                                generation_reason="INVARIANT_SYNTHESIS",
                            )
                            next_id_counter += 1
                            next_pw_list.append((w_item, (H_q, h_q)))
                        elif feas.status == STATUS_UNKNOWN:
                            polytopes_only = [p[1] for p in current_pw_list]
                            witnesses_only = [p[0] for p in current_pw_list]
                            cert = V5539Certificate(polytopes_only, witnesses_only, flat_actions, STATUS_UNKNOWN, final_iter, lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance, self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(), {"algorithm_version": ALGORITHM_VERSION})
                            return cert, "SOLVER_UNKNOWN"

            if not next_pw_list:
                cert = V5539Certificate([], [], flat_actions, STATUS_EMPTY, final_iter, lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance, self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(), {"algorithm_version": ALGORITHM_VERSION})
                return cert, "EMPTY_INVARIANT"

            next_pw_list, red_status = self.remove_redundant_polytopes_with_witness(next_pw_list)
            if red_status == STATUS_UNKNOWN:
                cert = V5539Certificate([], [], flat_actions, STATUS_UNKNOWN, final_iter, lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance, self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(), {"algorithm_version": ALGORITHM_VERSION})
                return cert, "SOLVER_UNKNOWN"

            curr_polytopes_only = [p[1] for p in current_pw_list]
            next_polytopes_only = [p[1] for p in next_pw_list]
            
            forward_containment = self.check_union_containment_sufficient(next_polytopes_only, curr_polytopes_only)
            
            if forward_containment == STATUS_UNKNOWN:
                cert = V5539Certificate(curr_polytopes_only, [p[0] for p in current_pw_list], flat_actions, STATUS_UNKNOWN, final_iter, lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance, self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(), {"algorithm_version": ALGORITHM_VERSION})
                return cert, "MONOTONICITY_UNKNOWN"
            
            if forward_containment != "CERTIFIED":
                # V55.3.9 Hardening: Separated MONOTONICITY_VIOLATION from STATUS_EMPTY
                cert = V5539Certificate(curr_polytopes_only, [p[0] for p in current_pw_list], flat_actions, STATUS_MONOTONICITY_VIOLATION, final_iter, lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance, self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(), {"algorithm_version": ALGORITHM_VERSION, "fail_closed_reason": "MONOTONICITY_VIOLATION"})
                return cert, STATUS_MONOTONICITY_VIOLATION

            backward_containment = self.check_union_containment_sufficient(curr_polytopes_only, next_polytopes_only)
            if backward_containment == STATUS_UNKNOWN:
                cert = V5539Certificate(curr_polytopes_only, [p[0] for p in current_pw_list], flat_actions, STATUS_UNKNOWN, final_iter, lambda_intervals, H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance, self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(), {"algorithm_version": ALGORITHM_VERSION})
                return cert, "UNION_CONTAINMENT_UNKNOWN"

            if backward_containment == "CERTIFIED":
                monotonic_descent_flag = True
                current_pw_list = next_pw_list
                status = STATUS_CERTIFIED
                break
            
            current_pw_list = next_pw_list

        final_polytopes = [p[1] for p in current_pw_list]
        final_witnesses = [p[0] for p in current_pw_list]

        cert = V5539Certificate(
            final_polytopes, final_witnesses, flat_actions, status, final_iter, lambda_intervals,
            H_safe, h_safe, action_space, self.w_max, self.v_max, self.tolerance,
            self.observation_model_metadata(), self.dynamics_model_metadata(), self.get_solver_provenance(),
            {"algorithm_version": ALGORITHM_VERSION, "max_iterations": max_iterations, "monotonic_descent_certified": monotonic_descent_flag}
        )

        if cert.status == STATUS_CERTIFIED:
            self.cached_certificate = cert
            self.cached_config_key = current_config_key
            return cert, "CACHE_MISS_SYNTHESIZED"
        return cert, "SYNTHESIS_NOT_CERTIFIED"

    def _robust_post_box(self, belief_lower, belief_upper, uc, uq, lambda_intervals):
        center = 0.5 * (belief_lower + belief_upper)
        radius = 0.5 * (belief_upper - belief_lower)

        lower_c, upper_c, abs_A = [], [], []
        for interval in lambda_intervals:
            A_low, B_c, B_q = self.get_adaptive_dynamics(np.array([interval.low]))
            A_high, _, _ = self.get_adaptive_dynamics(np.array([interval.high]))
            control = B_c @ uc + B_q @ uq
            lower_c.append(A_low @ center + control)
            upper_c.append(A_high @ center + control)
            abs_A.append(np.maximum(np.abs(A_low), np.abs(A_high)))

        lower_c, upper_c, abs_A = np.asarray(lower_c), np.asarray(upper_c), np.asarray(abs_A)
        center_min = np.min(np.vstack([lower_c, upper_c]), axis=0)
        center_max = np.max(np.vstack([lower_c, upper_c]), axis=0)
        max_abs_A = np.max(abs_A, axis=0)
        radius_effect = max_abs_A @ radius

        return center_min - radius_effect - self.w_max, center_max + radius_effect + self.w_max

    def authorize_candidate_action(self, cert: V5539Certificate, belief_bounds, candidate_uc, candidate_uq, lambda_intervals) -> Tuple[bool, str]:
        if not lambda_intervals:
            return False, "FAIL_CLOSED_EMPTY_PARAMETER_SET"

        if cert is None or cert.status != STATUS_CERTIFIED:
            return False, "FAIL_CLOSED_INVALID_CERTIFICATE"

        if not cert.verify_provenance():
            return False, "FAIL_CLOSED_PROVENANCE_FAILURE"

        try:
            uc = _as_float_vector(candidate_uc, "candidate_uc", expected_dim=2)
            uq = _as_float_vector(candidate_uq, "candidate_uq", expected_dim=2)
        except ValueError:
            return False, "FAIL_CLOSED_INVALID_CANDIDATE_ACTION"

        action_bound_verified = any(
            np.allclose(uc, bound_uc, atol=self.tolerance) and np.allclose(uq, bound_uq, atol=self.tolerance)
            for bound_uc in cert.action_space_canonical["u_c"]
            for bound_uq in cert.action_space_canonical["u_q"]
        )
        if not action_bound_verified:
            return False, "FAIL_CLOSED_CANDIDATE_ACTION_OUT_OF_BOUNDS"

        certified_lambda_intervals = [ClosedInterval(i[0], i[1]) for i in cert.lambda_intervals]
        if not self.parameter_set_subset(lambda_intervals, certified_lambda_intervals):
            return False, "FAIL_CLOSED_PARAMETER_SET_BINDING"

        valid, b_lower, b_upper = self.validate_belief(belief_bounds)
        if not valid:
            return False, "FAIL_CLOSED_INVALID_BELIEF"

        belief_containment_status = self.check_belief_containment_in_certificate(b_lower, b_upper, cert)
        if belief_containment_status == STATUS_UNKNOWN:
            return False, "FAIL_CLOSED_BELIEF_CONTAINMENT_UNKNOWN"
        if belief_containment_status != "CERTIFIED":
            return False, "FAIL_CLOSED_BELIEF_OUTSIDE_CERTIFIED_SET"

        box_low, box_high = self._robust_post_box(b_lower, b_upper, uc, uq, lambda_intervals)
        h_post_box = np.concatenate([box_high, -box_low])
        H_post_box = np.vstack([np.eye(self.dim_state), -np.eye(self.dim_state)])

        action_certified = False
        for H_k, h_k in cert.polytope_list:
            containment = self.check_polytope_containment(H_post_box, h_post_box, H_k, h_k)
            if containment == STATUS_UNKNOWN:
                return False, "FAIL_CLOSED_SOLVER_UNKNOWN"
            if containment == "CERTIFIED":
                action_certified = True
                break

        if not action_certified:
            return False, "FAIL_CLOSED_NO_CERTIFIED_ACTION"

        return True, "CERTIFIED_ALLOW"

    def authorize_runtime_action(self, cert: V5539Certificate, belief_bounds, runtime_action_space, lambda_intervals) -> Tuple[List[Tuple[Tuple[float, ...], Tuple[float, ...]]], str]:
        if not lambda_intervals:
            return [], "FAIL_CLOSED_EMPTY_PARAMETER_SET"

        if cert is None or cert.status != STATUS_CERTIFIED:
            return [], "FAIL_CLOSED_INVALID_CERTIFICATE"

        if not cert.verify_provenance():
            return [], "FAIL_CLOSED_PROVENANCE_FAILURE"

        if not self.validate_action_space(runtime_action_space):
            return [], "FAIL_CLOSED_INVALID_RUNTIME_ACTION_SPACE"

        runtime_action_canonical = {
            "u_c": [np.asarray(u, dtype=float).tolist() for u in runtime_action_space["u_c"]],
            "u_q": [np.asarray(u, dtype=float).tolist() for u in runtime_action_space["u_q"]],
        }
        if runtime_action_canonical != cert.action_space_canonical:
            return [], "FAIL_CLOSED_ACTION_SPACE_BINDING"

        certified_lambda_intervals = [ClosedInterval(i[0], i[1]) for i in cert.lambda_intervals]
        if not self.parameter_set_subset(lambda_intervals, certified_lambda_intervals):
            return [], "FAIL_CLOSED_PARAMETER_SET_BINDING"

        valid, b_lower, b_upper = self.validate_belief(belief_bounds)
        if not valid:
            return [], "FAIL_CLOSED_INVALID_BELIEF"

        belief_containment_status = self.check_belief_containment_in_certificate(b_lower, b_upper, cert)
        if belief_containment_status == STATUS_UNKNOWN:
            return [], "FAIL_CLOSED_BELIEF_CONTAINMENT_UNKNOWN"
        if belief_containment_status != "CERTIFIED":
            return [], "FAIL_CLOSED_BELIEF_OUTSIDE_CERTIFIED_SET"

        certified_actions = []
        for uc in runtime_action_space["u_c"]:
            for uq in runtime_action_space["u_q"]:
                allowed, _ = self.authorize_candidate_action(cert, belief_bounds, uc, uq, lambda_intervals)
                if allowed:
                    certified_actions.append((tuple(float(v) for v in uc), tuple(float(v) for v in uq)))

        if not certified_actions:
            return [], "FAIL_CLOSED_NO_CERTIFIED_ACTION"

        return certified_actions, "CERTIFIED_ALLOW"

    def synthesize_governance_optimal_policy(self, y_measured, belief_bounds, action_space, H_safe, h_safe, performance_cost_func: Optional[Callable] = None, cost_func_name="Default", max_iterations=8):
        telemetry = {
            "model_version": MODEL_VERSION,
            "identification_version": IDENTIFICATION_VERSION,
            "cache_version": CACHE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "timing_contract": "Chronological: observe(x_t) -> observe(y_t) -> update(Lambda_t^+) -> certify -> authorize(u_t) -> x_{t+1}",
        }

        valid, b_lower, b_upper = self.validate_belief(belief_bounds)
        if not valid:
            telemetry["decision"] = "FAIL_CLOSED_INVALID_BELIEF"
            return None, self.lambda_intervals, telemetry

        valid_safe, H_safe, h_safe = self.validate_safe_set(H_safe, h_safe)
        if not valid_safe:
            telemetry["decision"] = "FAIL_CLOSED_INVALID_SAFE_SET"
            return None, self.lambda_intervals, telemetry

        if not self.validate_action_space(action_space):
            telemetry["decision"] = "FAIL_CLOSED_INVALID_ACTION_SPACE"
            return None, self.lambda_intervals, telemetry

        # Step 1: Analytical Set-Membership Identification (Posterior \Lambda_t^+)
        lambda_intervals, id_status = self.update_exact_set_membership_identification(y_measured, b_lower, b_upper)
        telemetry["identification_status"] = id_status
        telemetry["lambda_intervals"] = [[i.low, i.high] for i in lambda_intervals]

        if id_status == ID_MISMATCH or not lambda_intervals:
            telemetry["decision"] = "FAIL_CLOSED_MODEL_MISMATCH" if id_status == ID_MISMATCH else "FAIL_CLOSED_EMPTY_PARAMETER_SET"
            return None, lambda_intervals, telemetry

        # Step 2: Certified Robust Controlled-Invariant Inner Approximation (Synthesis)
        cert, cache_status = self.synthesize_set_valued_invariant(H_safe, h_safe, action_space, lambda_intervals, max_iterations=max_iterations, cost_func_name=cost_func_name)
        telemetry["cache_status"] = cache_status
        if cert is None or cert.status != STATUS_CERTIFIED:
            if cache_status == STATUS_MONOTONICITY_VIOLATION:
                telemetry["decision"] = "FAIL_CLOSED_MONOTONICITY_VIOLATION"
            elif cache_status == "EMPTY_PARAMETER_SET":
                telemetry["decision"] = "FAIL_CLOSED_EMPTY_PARAMETER_SET"
            else:
                telemetry["decision"] = f"FAIL_CLOSED_CERT_NOT_CERTIFIED_{cert.status if cert else 'NONE'}"
            return None, lambda_intervals, telemetry

        # Step 3: Decoupled Runtime Authorization (with B_t \subseteq K_cert and \Lambda_t^+ \subseteq \Lambda_cert)
        certified_actions, auth_status = self.authorize_runtime_action(cert, belief_bounds, action_space, lambda_intervals)
        telemetry["authorization_status"] = auth_status
        telemetry["certified_action_count"] = len(certified_actions)

        if auth_status != "CERTIFIED_ALLOW" or not certified_actions:
            telemetry["decision"] = auth_status
            return None, lambda_intervals, telemetry

        # Step 4: Safety-First Performance Optimization (Decoupled Performance Provenance)
        telemetry["performance_provenance"] = self.get_performance_provenance(cost_func_name)

        if performance_cost_func is None:
            selected = certified_actions[0]
            telemetry["decision"] = "CERTIFIED_ALLOW"
            telemetry["selected_action"] = selected
            return selected, lambda_intervals, telemetry

        best_action, best_cost = None, float("inf")
        for action in certified_actions:
            uc, uq = np.asarray(action[0], dtype=float), np.asarray(action[1], dtype=float)
            try:
                cost = float(performance_cost_func(belief_bounds, uc, uq))
            except Exception:
                continue
            if np.isfinite(cost) and cost < best_cost:
                best_cost, best_action = cost, action

        if best_action is None:
            telemetry["decision"] = "FAIL_CLOSED_PERFORMANCE_FAILURE"
            return None, lambda_intervals, telemetry

        telemetry["decision"] = "CERTIFIED_OPTIMAL_ALLOW"
        telemetry["selected_action"] = best_action
        telemetry["selected_cost"] = best_cost
        return best_action, lambda_intervals, telemetry


# ================================================================
# Extended Definitive Reference Verification Suite (V55.3.9)
# ================================================================

def run_v5539_definitive_reference_suite():
    print("=" * 72)
    print("V55.3.9 DEFINITIVE REFERENCE VERIFICATION SUITE (AUDIT-LOCKED & SEMANTIC-LOCKED)")
    print("=" * 72)

    H_safe = np.array([
        [1, 0, 0, 0], [-1, 0, 0, 0],
        [0, 1, 0, 0], [0, -1, 0, 0],
        [0, 0, 1, 0], [0, 0, -1, 0],
        [0, 0, 0, 1], [0, 0, 0, -1],
    ], dtype=float)
    h_safe = np.array([5, 5, 2, 2, 1, 0, 1, 0], dtype=float)
    action_space = {
        "u_c": [np.array([0.0, 0.0]), np.array([0.01, 0.005])],
        "u_q": [np.array([0.0, 0.0])],
    }
    def demo_cost(b, uc, uq): return float(np.dot(uc, uc))

    # Test 1: 1000-Step Sequential Stochastic Truth-Containment Stress Test
    print("\n[Test 1] 1000-Step Sequential Stochastic Truth-Containment Stress Test")
    kernel_t1 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    rng = np.random.default_rng(42)
    lambda_true = 1.03
    belief = (np.array([-0.20, 0.40, 0.40, 0.50]), np.array([0.20, 0.60, 0.60, 0.50]))
    
    phi_min_bound = 0.1 * belief[0][1]
    phi_max_bound = 0.1 * belief[1][1]

    for r_idx in range(1000):
        phi_true = 0.1 * rng.uniform(0.40, 0.60)
        v_true = rng.uniform(-0.005, 0.005)
        y_meas = phi_true * lambda_true + v_true
        
        assert phi_min_bound <= phi_true <= phi_max_bound
        assert -kernel_t1.v_max <= v_true <= kernel_t1.v_max

        intervals, id_status = kernel_t1.update_exact_set_membership_identification(np.array([y_meas]), belief[0], belief[1])
        assert any(i.low <= lambda_true <= i.high for i in intervals), f"Test 1 failed at step {r_idx}"
    print("  -> PASSED: 1000-step sequential stochastic stress test validated phi, v, and lambda_true containment.")

    # Test 2: Model Mismatch Rejection
    print("\n[Test 2] Model Mismatch Rejection & Fail-Closed")
    kernel_t2 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    _, _, tel_b = kernel_t2.synthesize_governance_optimal_policy(np.array([0.50]), belief, action_space, H_safe, h_safe, demo_cost)
    assert tel_b["decision"] == "FAIL_CLOSED_MODEL_MISMATCH", "Test 2 failed"
    print("  -> PASSED: Model mismatch correctly triggered Fail-Closed.")

    # Test 3: Fixed-Point Monotonic Descent & Inner Approximation Certification
    print("\n[Test 3] Fixed-Point Monotonic Descent & Inner Approximation Certification")
    kernel_t3 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    cert_c, _ = kernel_t3.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t3.lambda_intervals)
    assert cert_c.status == STATUS_CERTIFIED and cert_c.algorithm_metadata.get("monotonic_descent_certified"), "Test 3 failed"
    print("  -> PASSED: Construction-level and verification-level monotonic descent successfully certified.")

    # Test 4: Monotonic Cache Reuse & Invalidation (Independent of Performance Cost)
    print("\n[Test 4] Monotonic Cache Reuse & Invalidation (Decoupled from Performance Cost)")
    kernel_t4 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    broad_lambda = [ClosedInterval(0.90, 1.10)]
    _, miss_status = kernel_t4.synthesize_set_valued_invariant(H_safe, h_safe, action_space, broad_lambda, cost_func_name="CostA")
    assert miss_status == "CACHE_MISS_SYNTHESIZED"

    narrow_lambda = [ClosedInterval(0.95, 1.05)]
    _, hit_status = kernel_t4.synthesize_set_valued_invariant(H_safe, h_safe, action_space, narrow_lambda, cost_func_name="CostB")
    assert hit_status == "CACHE_HIT", "Cache key must be independent of cost function name!"
    print("  -> PASSED: Safety cache reuse verified independent of performance cost function changes.")

    # Test 5: Solver UNKNOWN Propagation (Feasibility, Containment, and Redundancy)
    print("\n[Test 5] Solver UNKNOWN Propagation & Fail-Closed (Feasibility, Containment, Redundancy)")
    for unk_mode in ["FORCE_UNKNOWN_FEASIBILITY", "FORCE_UNKNOWN_CONTAINMENT", "FORCE_UNKNOWN_REDUNDANCY"]:
        kernel_t5 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
        kernel_t5.mock_solver_status = unk_mode
        if unk_mode == "FORCE_UNKNOWN_REDUNDANCY":
            cert_un, status_un = kernel_t5.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t5.lambda_intervals)
            assert cert_un.status == STATUS_UNKNOWN, f"Test 5 failed for mode {unk_mode}"
        else:
            _, _, tel_unk = kernel_t5.synthesize_governance_optimal_policy(np.array([0.05 * lambda_true]), belief, action_space, H_safe, h_safe, demo_cost)
            assert "FAIL_CLOSED" in tel_unk["decision"], f"Test 5 failed for mode {unk_mode}"
    print("  -> PASSED: Solver UNKNOWN across feasibility, containment, and redundancy paths triggered Fail-Closed.")

    # Test 6: Pure Postcondition Authorization Testing (Positive ALLOW & Negative REJECT)
    print("\n[Test 6] Pure Postcondition Authorization Testing (Positive ALLOW & Negative REJECT)")
    kernel_t6 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    cert_t6, _ = kernel_t6.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t6.lambda_intervals)
    assert cert_t6.status == STATUS_CERTIFIED

    # Positive test: Certifiably safe candidate action
    safe_uc = np.array([0.01, 0.005])
    safe_uq = np.array([0.0, 0.0])
    allowed_pos, reason_pos = kernel_t6.authorize_candidate_action(cert_t6, belief, safe_uc, safe_uq, kernel_t6.lambda_intervals)
    assert allowed_pos == True and reason_pos == "CERTIFIED_ALLOW", "Test 6 Positive failed"

    # Negative test: Extreme unsafe candidate action
    extreme_unsafe_uc = np.array([30.0, 30.0])
    extreme_unsafe_uq = np.array([0.0, 0.0])
    allowed_neg, reason_neg = kernel_t6.authorize_candidate_action(cert_t6, belief, extreme_unsafe_uc, extreme_unsafe_uq, kernel_t6.lambda_intervals)
    assert allowed_neg == False and reason_neg == "FAIL_CLOSED_NO_CERTIFIED_ACTION", "Test 6 Negative failed"
    print("  -> PASSED: Pure candidate authorization successfully verified both positive ALLOW and negative REJECT.")

    # Test 7: Tamper-Evident Integrity Binding Verification
    print("\n[Test 7] Tamper-Evident Integrity Binding Verification")
    kernel_t7 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    cert_t7, _ = kernel_t7.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t7.lambda_intervals)
    assert cert_t7.verify_provenance() == True
    
    cert_t7.h_safe[0] += 0.01
    assert cert_t7.verify_provenance() == False, "Test 7 failed: Tampering not detected!"
    print("  -> PASSED: Tamper-evident integrity binding successfully triggered verification failure.")

    # Test 8: Decoupled Action-Space Binding Mismatch
    print("\n[Test 8] Decoupled Action-Space Binding Mismatch Rejection")
    kernel_t8 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    cert_t8, _ = kernel_t8.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t8.lambda_intervals)
    
    tampered_action_space = {
        "u_c": [np.array([0.0, 0.0]), np.array([0.05, 0.05])],
        "u_q": [np.array([0.0, 0.0])],
    }
    _, bind_status = kernel_t8.authorize_runtime_action(cert_t8, belief, tampered_action_space, kernel_t8.lambda_intervals)
    assert bind_status == "FAIL_CLOSED_ACTION_SPACE_BINDING", "Test 8 failed"
    print("  -> PASSED: Decoupled action-space binding mismatch correctly rejected.")

    # Test 9: Empty Parameter Set Rejection
    print("\n[Test 9] Empty Parameter Set Rejection ($\Lambda = \emptyset$) (Fail-Closed)")
    kernel_t9 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    empty_lambda = []
    cert_empty, status_empty = kernel_t9.synthesize_set_valued_invariant(H_safe, h_safe, action_space, empty_lambda)
    assert status_empty == "EMPTY_PARAMETER_SET", "Test 9 Synthesis failed"
    
    _, empty_auth_status = kernel_t9.authorize_runtime_action(cert_t8, belief, action_space, empty_lambda)
    assert empty_auth_status == "FAIL_CLOSED_EMPTY_PARAMETER_SET", "Test 9 Authorization failed"
    print("  -> PASSED: Empty parameter set successfully intercepted and failed-closed.")

    # Test 10: Belief Outside Certified Invariant Set
    print("\n[Test 10] Belief Outside Certified Invariant Set Rejection ($B_t \not\subseteq K_{\rm cert}$)")
    kernel_t10 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    cert_t10, _ = kernel_t10.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t10.lambda_intervals)
    
    outside_belief = (np.array([6.0, 0.0, 0.0, 0.0]), np.array([7.0, 0.0, 0.0, 0.0]))
    _, outside_auth_status = kernel_t10.authorize_runtime_action(cert_t10, outside_belief, action_space, kernel_t10.lambda_intervals)
    assert outside_auth_status == "FAIL_CLOSED_BELIEF_OUTSIDE_CERTIFIED_SET", "Test 10 failed"
    print("  -> PASSED: Belief outside certified invariant set correctly rejected (Fail-Closed).")

    # Test 11: Runtime Lambda Outside Certificate Lambda
    print("\n[Test 11] Runtime Lambda Outside Certificate Lambda Rejection ($\Lambda_t^+ \not\subseteq \Lambda_{\rm cert}$)")
    kernel_t11 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    cert_t11, _ = kernel_t11.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t11.lambda_intervals)
    
    out_of_bounds_lambda = [ClosedInterval(0.80, 1.25)]
    _, lambda_bind_status = kernel_t11.authorize_runtime_action(cert_t11, belief, action_space, out_of_bounds_lambda)
    assert lambda_bind_status == "FAIL_CLOSED_PARAMETER_SET_BINDING", "Test 11 failed"
    print("  -> PASSED: Runtime lambda set exceeding certificate bounds correctly rejected (Fail-Closed).")

    # Test 12: Candidate Action Outside Certificate Action Space Bounds
    print("\n[Test 12] Candidate Action Outside Certificate Action-Space Bounds (API Hardening)")
    kernel_t12 = IEMOSV5539Kernel(initial_lambda_interval=(np.array([0.90]), np.array([1.10])), dim_state=4, w_max=0.001, v_max=0.005)
    cert_t12, _ = kernel_t12.synthesize_set_valued_invariant(H_safe, h_safe, action_space, kernel_t12.lambda_intervals)
    
    unbound_uc = np.array([0.02, 0.02])
    unbound_uq = np.array([0.0, 0.0])
    allowed_ub, reason_ub = kernel_t12.authorize_candidate_action(cert_t12, belief, unbound_uc, unbound_uq, kernel_t12.lambda_intervals)
    assert allowed_ub == False and reason_ub == "FAIL_CLOSED_CANDIDATE_ACTION_OUT_OF_BOUNDS", "Test 12 failed"
    print("  -> PASSED: Candidate action outside certificate action-space bounds correctly rejected (Fail-Closed).")

    print("\n" + "=" * 72)
    print("SUCCESS: All V55.3.9 Definitive Reference Verification Tests Passed.")
    print("=" * 72)


if __name__ == "__main__":
    run_v5539_definitive_reference_suite()
