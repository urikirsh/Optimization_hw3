import math
import numpy as np
import unittest
import numpy.testing as npt


def target_function_f(x, y):
    return x * math.exp(x**2 - y**2)


def phi_f(x: np.ndarray):
    return np.tanh(x)


def phi_g(x: np.ndarray):
    g_i_i = 1 / (np.cosh(x)**2)
    return np. diag(np.ravel(g_i_i))


def dnn_forward(x: np.ndarray, params: dict, nargout=1):
    required_params = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']
    for par in required_params:
        assert par in params

    u1 = params['W1'].T @ x + params['b1']
    u2 = params['W2'].T @ phi_f(u1) + params['b2']
    u3 = params['W3'].T @ phi_f(u2) + params['b3']

    return u3


def error_f(out, y):
    return (out - y) ** 2


def dnn_error(x: np.ndarray, parameters: dict, nargout=1):
    assert 1 <= nargout <= 2
    if nargout == 1:
        out = dnn_forward(x, parameters)
        y = target_function_f(x[0], x[1])
        return error_f(out, y)


def analytic_calc_dir_grads_dnn_error(x: np.ndarray, parameters: dict, direction: str):
    assert direction in parameters

    y = target_function_f(x[0], x[1])
    out = dnn_forward(x, parameters)
    nabla_r_Psi = 2 * (out - y)
    if direction == 'b3':
        return nabla_r_Psi

    u1 = parameters['W1'].T @ x + parameters['b1']


def generate_bias(n: int, random=False):
    assert n > 0
    if random:
        return np.random.rand(n, 1)
    return np.zeros((n, 1))


def generate_weight(m: int, n: int):
    assert m > 0
    assert n > 0
    return np.random.rand(m, n)


def generate_params():
    params = dict()
    params['b1'] = generate_bias(4, random=True)
    params['b2'] = generate_bias(3, random=True)
    params['b3'] = generate_bias(1, random=True)
    params['W1'] = generate_weight(2, 4)
    params['W2'] = generate_weight(4, 3)
    params['W3'] = generate_weight(3, 1)
    return params


def numdiff_calc_dnn_error_grad(grad_of, x, params: dict, epsilon: float):
    assert epsilon > 0
    assert grad_of in params

    max_abs_val_of_x = max(x.min(), x.max(), key=abs)
    dim = len(x)
    epsilon = pow(epsilon, 1 / dim) * max_abs_val_of_x

    params[grad_of] += epsilon
    right_f = dnn_error(x, params)
    params[grad_of] -= 2*epsilon
    left_f = dnn_error(x, params)
    result = (right_f - left_f) / (2*epsilon)

    # cleanup
    params[grad_of] += epsilon

    return result


def main():
    raise NotImplementedError()


class task3_q_2 (unittest.TestCase):

    def test_target_function(self):
        npt.assert_almost_equal(target_function_f(0, 0), 0)
        npt.assert_almost_equal(target_function_f(0, 17), 0)
        npt.assert_almost_equal(target_function_f(1, 0), np.exp(1))


    def test_generate_params(self):
        params = generate_params()
        self.assertTrue(isinstance(params, dict))
        self.assertEqual(6, len(params))
        required_params = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']
        for par in required_params:
            self.assertTrue(par in params)


    def test_grad_numdiff(self):
        # from hw_1 import numdiff
        params = generate_params()
        x = 2 * np.random.rand(2, 1) - 1
        epsilon = pow(10, 0)

        anal = analytic_calc_dir_grads_dnn_error(x, params, 'b3')
        numeric = numdiff_calc_dnn_error_grad('b3', x, params, epsilon)
        npt.assert_almost_equal(numeric, anal)


if __name__ == "__main__":
    unittest.main()

