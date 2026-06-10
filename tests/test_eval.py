"""Tests for the rank correlation helpers used in Section 3.1."""
from __future__ import annotations

import math

from scicustom.eval import kendall_tau_b, parse_letter, spearman


def test_spearman_perfect():
    a = [1, 2, 3, 4, 5]
    b = [2, 4, 6, 8, 10]
    assert spearman(a, b) == 1.0


def test_spearman_anti():
    a = [1, 2, 3, 4, 5]
    b = [5, 4, 3, 2, 1]
    assert spearman(a, b) == -1.0


def test_kendall_simple():
    a = [1, 2, 3, 4]
    b = [1, 3, 2, 4]
    tau = kendall_tau_b(a, b)
    # 5 concordant, 1 discordant -> tau = (5-1) / 6 = 0.6666...
    assert math.isclose(tau, 4 / 6, rel_tol=1e-6)


def test_parse_letter_clean():
    assert parse_letter("A") == "A"
    assert parse_letter("Answer: C\nBecause...") == "C"
    assert parse_letter("The correct option is D.") == "D"


def test_parse_letter_missing():
    assert parse_letter("") is None
    assert parse_letter("I am not sure.") is None
