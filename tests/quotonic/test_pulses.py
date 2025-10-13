import numpy as np

import quotonic.pulses as pulses


def test_gaussian_t():
    t = np.linspace(-10.0, 10.0, 200)

    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_t(t, 0.0, 1.0)) ** 2, t))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_t(t, 0.0, 0.5)) ** 2, t))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_t(t, 0.0, 2.0)) ** 2, t))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_t(t, 3.0, 1.0)) ** 2, t))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_t(t, -3.0, 1.0)) ** 2, t))


def test_gaussian_w():
    w = np.linspace(-10.0, 10.0, 200)

    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_w(w, 0.0, 1.0)) ** 2, w))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_w(w, 0.0, 0.5)) ** 2, w))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_w(w, 0.0, 2.0)) ** 2, w))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_w(w, 3.0, 1.0)) ** 2, w))
    assert np.allclose(1.0, np.trapezoid(np.abs(pulses.gaussian_w(w, -3.0, 1.0)) ** 2, w))
