"""Tests for forward-path numerical debug helpers."""

import pytest
import torch

from src.Open_MAGVIT2.modules.numerical_debug import (
    check_tensor,
    get_numerical_report,
    reset_numerical_report,
    set_numerical_debug,
)


def test_numerical_debug_records_finite_tensor():
    set_numerical_debug(True, fail_fast=True)
    reset_numerical_report()
    check_tensor("x", torch.ones(2, 3))
    report = get_numerical_report()
    assert report is not None
    assert report.first_bad() is None
    assert report.checks[0].name == "x"
    set_numerical_debug(False)


def test_numerical_debug_fail_fast_on_nan():
    set_numerical_debug(True, fail_fast=True)
    reset_numerical_report()
    with pytest.raises(RuntimeError, match="sq/z_in|Non-finite"):
        check_tensor("bad", torch.tensor([float("nan")]))
    set_numerical_debug(False)
