"""
core/physics_model.py  --  grey-box fuel model: Admiralty-law form + boosting residual.

WHY THIS EXISTS
---------------
Tree models are flat between split points. The optimizer minimises
rate(speed) / speed, and a flat rate divided by a rising speed is a falling
line -- so the voyage-fuel curve from a tree model is a SAWTOOTH, and the
"optimal speed" is whichever tooth happens to be lowest. This model fixes that
at the source: the speed response gets a smooth physical form, and boosting
learns only what the form cannot explain.

    rate = D^(2/3) * [ hotel + exp(k + n*ln(stw) + b*ln(hull)
                                     + a_w*waves + a_u*wind + a_l*laden) ]
           * exp(g(x))

  * White box (7 coefficients). A speed-independent hotel load plus a propulsion
    term that is a power law in speed-through-water (stw = speed - current).
    Fitted by least squares on ln(rate), because the noise is multiplicative.
    The exponent n is FITTED, not assumed: recovering n ~ 3 is the cubic law
    being read back out of noisy data.
  * g(x): gradient boosting on the log-residual, with early stopping so it stays
    near zero wherever the white box already explains the data.

HONESTY NOTE: the synthetic generator has this same "hotel + power law"
structure, so a near-perfect fit here shows correct CALIBRATION, not discovery.
The black-box models in train_model.py remain the test of learning the law from
scratch. With real voyage logs the residual booster absorbs what the form misses.

This module is import-only (never run as a script) so the pickled model can be
loaded from any entry point.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin, clone

COEF_NAMES = ["hotel", "prop_scale", "speed_exponent", "hull_exponent",
              "wave_coef", "wind_coef", "laden_coef"]


def _columns(X):
    """The few numeric arrays the physical form needs, from named columns."""
    def col(name):
        return X[name].to_numpy(dtype=float)

    stw = np.maximum(col("speed_knots") - col("current_speed_kn"), 1.0)  # log needs > 0
    return (np.log(col("displacement_tonnes")), np.log(stw),
            np.log(col("hull_coefficient")), col("wave_height_m"),
            col("wind_speed_kn"),
            (X["load_condition"].to_numpy() == "Laden").astype(float))


def _ln_rate(theta, cols):
    """ln(fuel rate) from the physical form."""
    ln_disp, ln_stw, ln_hull, waves, wind, laden = cols
    ln_hotel, k, n, b, a_w, a_u, a_l = theta
    propulsion = k + n * ln_stw + b * ln_hull + a_w * waves + a_u * wind + a_l * laden
    return (2.0 / 3.0) * ln_disp + np.logaddexp(ln_hotel, propulsion)


class PhysicsResidualRegressor(RegressorMixin, BaseEstimator):
    """rate = physical_form(x; theta) * exp(residual(x)).

    `residual` is any regressor taking the same feature DataFrame (None = white
    box only). Takes the DataFrame directly -- it reads named columns."""

    def __init__(self, residual=None):
        self.residual = residual

    def fit(self, X, y):
        cols, ln_y = _columns(X), np.log(np.asarray(y, dtype=float))
        # Start from the textbook cubic law at the data's own scale.
        level = np.median(ln_y - (2.0 / 3.0) * cols[0])    # typical ln(rate / D^(2/3))
        x0 = [level - 2.0, level - 3.0 * np.median(cols[1]), 3.0, 0.0, 0.0, 0.0, 0.0]
        self.theta_ = least_squares(lambda th: _ln_rate(th, cols) - ln_y, x0,
                                    x_scale="jac").x
        if self.residual is not None:
            self.residual_ = clone(self.residual).fit(
                X, ln_y - _ln_rate(self.theta_, cols))
        return self

    def predict(self, X):
        ln = _ln_rate(self.theta_, _columns(X))
        if self.residual is not None:
            ln = ln + self.residual_.predict(X)
        return np.exp(ln)

    def coefficients(self) -> dict:
        """Fitted white-box coefficients (hotel and prop_scale back in linear units)."""
        t = self.theta_
        return dict(zip(COEF_NAMES, [np.exp(t[0]), np.exp(t[1]), *t[2:]]))
