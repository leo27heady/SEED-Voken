"""Lightweight forward-path numerical health checks for NaN debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TensorCheck:
    name: str
    shape: tuple
    dtype: str
    finite: bool
    has_nan: bool
    has_inf: bool
    max_abs: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.finite


@dataclass
class NumericalReport:
    checks: List[TensorCheck] = field(default_factory=list)

    def record(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> TensorCheck:
        with torch.no_grad():
            finite_mask = torch.isfinite(tensor)
            finite = bool(finite_mask.all().item())
            has_nan = bool(torch.isnan(tensor).any().item())
            has_inf = bool(torch.isinf(tensor).any().item())
            if finite:
                max_abs = float(tensor.abs().max().item())
            else:
                safe = tensor[torch.isfinite(tensor)]
                max_abs = float(safe.abs().max().item()) if safe.numel() else float("nan")
        check = TensorCheck(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            finite=finite,
            has_nan=has_nan,
            has_inf=has_inf,
            max_abs=max_abs,
            extra=dict(extra or {}),
        )
        self.checks.append(check)
        return check

    def first_bad(self) -> Optional[TensorCheck]:
        for check in self.checks:
            if not check.ok:
                return check
        return None

    def as_log_dict(self, prefix: str = "debug") -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for check in self.checks:
            key = check.name.replace("/", "_")
            out[f"{prefix}/{key}_max"] = torch.tensor(check.max_abs)
            out[f"{prefix}/{key}_finite"] = torch.tensor(1.0 if check.ok else 0.0)
        return out

    def summary(self) -> str:
        lines = ["Numerical forward report:"]
        for check in self.checks:
            status = "ok" if check.ok else "BAD"
            extra = ""
            if check.extra:
                extra = " " + " ".join(f"{k}={v}" for k, v in check.extra.items())
            lines.append(
                f"  [{status}] {check.name} shape={check.shape} dtype={check.dtype} "
                f"max_abs={check.max_abs:.4g} nan={check.has_nan} inf={check.has_inf}{extra}"
            )
        bad = self.first_bad()
        if bad is not None:
            lines.append(f"First failure: {bad.name}")
        return "\n".join(lines)


_report: Optional[NumericalReport] = None
_enabled: bool = False
_fail_fast: bool = True


def set_numerical_debug(enabled: bool, *, fail_fast: bool = True) -> None:
    global _enabled, _fail_fast
    _enabled = bool(enabled)
    _fail_fast = bool(fail_fast)


def numerical_debug_enabled() -> bool:
    return _enabled


def reset_numerical_report() -> NumericalReport:
    global _report
    _report = NumericalReport()
    return _report


def get_numerical_report() -> Optional[NumericalReport]:
    return _report


def check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[TensorCheck]:
    if not _enabled or _report is None:
        return None
    check = _report.record(name, tensor, extra=extra)
    if _fail_fast and not check.ok:
        raise RuntimeError(
            f"Non-finite tensor in forward: {name}\n{_report.summary()}"
        )
    return check
