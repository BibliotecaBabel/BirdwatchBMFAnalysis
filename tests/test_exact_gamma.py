"""The exact anchor of Theorem 3 used as the reference line in every figure."""
import pytest

from bmf.neurips_figures import exact_gamma


@pytest.mark.parametrize("mu_u, expected", [
    (0.25, 0.125), (0.5, 0.246), (1.0, 0.469), (1.5, 0.648), (2.0, 0.775),
])
def test_reference_values_at_unit_moments(mu_u, expected):
    assert exact_gamma(mu_u) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("kappa, expected", [
    (0.1, 1.82), (0.25, 1.57), (0.5, 1.23), (1, 0.77), (2, 0.42), (4, 0.22), (10, 0.09),
])
def test_reference_values_across_kappa(kappa, expected):
    assert exact_gamma(2.0, kappa=kappa) == pytest.approx(expected, abs=5e-3)


def test_zero_mean_gives_zero_anchor():
    assert exact_gamma(0.0) == 0.0
    assert exact_gamma(0.0, sigma_u2=2.0, sigma_v2=0.5, kappa=3.0) == 0.0


def test_anchor_is_odd_in_the_mean():
    for mu_u in (0.3, 1.0, 2.0):
        assert exact_gamma(-mu_u) == pytest.approx(-exact_gamma(mu_u))


@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
def test_slope_at_zero_is_one_over_one_plus_kappa(kappa):
    eps = 1e-4
    assert exact_gamma(eps, kappa=kappa) / eps == pytest.approx(1 / (1 + kappa), rel=1e-4)


def test_anchor_stays_between_zero_and_the_mean_and_falls_with_kappa():
    values = [exact_gamma(2.0, kappa=k) for k in (0.1, 0.5, 1, 2, 10)]
    assert all(0 < g < 2.0 for g in values)
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("mu_u, su2, sv2, kappa", [
    (1.0, 1.0, 1.0, 1.0), (2.0, 1.3, 0.8, 0.25), (-1.5, 0.7, 1.2, 4.0),
])
def test_anchor_solves_the_scalar_equation(mu_u, su2, sv2, kappa):
    gamma = exact_gamma(mu_u, su2, sv2, kappa)
    A, K = kappa * sv2, sv2
    t = gamma * A / (mu_u - gamma)          # invert gamma = mu_u t / (t + A)
    psi = t * t * su2 + mu_u ** 2 * A ** 2 * (t / (t + A)) ** 2 - K
    assert psi == pytest.approx(0.0, abs=1e-8)
