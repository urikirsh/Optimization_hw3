import math
import numpy as np
import unittest
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary despite Pycharm thinking otherwise
import BFGS


def true_function_y(x, y):
    '''
    The function out DNN tries to learn, the "labels"
    '''
    return x * math.exp(-x**2 - y**2)


def phi_f(x: np.ndarray):
    '''
    Returns vector phi at vector x.
    '''
    return np.tanh(x).reshape((len(x), 1))


def phi_g(x: np.ndarray):
    '''
    Returns gradient of phi at vector x. A diagonal matrix
    '''
    g_i_i = 1 / (np.cosh(x)**2)
    return np.diag(np.ravel(g_i_i))


def dnn_forward(x: np.ndarray, params: dict):
    '''
    returns F(x, W)
    '''
    required_params = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']
    for par in required_params:
        assert par in params

    u1 = params['W1'].T @ x + params['b1']
    u2 = params['W2'].T @ phi_f(u1) + params['b2']
    u3 = params['W3'].T @ phi_f(u2) + params['b3']

    return u3


def error_f(out, y):
    '''
    returns Psi(r_i)
    '''
    return (out - y) ** 2


def dnn_error(x: np.ndarray, parameters: dict):
    '''
    Returns the value of Psi(x), given weights and biases
    '''

    out = dnn_forward(x, parameters)
    y = true_function_y(x[0], x[1])
    return error_f(out, y)


def analytic_calc_dir_grads_dnn_error(x: np.ndarray, parameters: dict, direction: str):
    assert direction in parameters

    y = true_function_y(x[0], x[1])
    out = dnn_forward(x, parameters)
    nabla_r_Psi = 2 * (out - y)
    if direction == 'b3':
        return nabla_r_Psi

    u1 = parameters['W1'].T @ x + parameters['b1']
    u2 = parameters['W2'].T @ phi_f(u1) + parameters['b2']
    # u3 = parameters['W3'].T @ phi_f(u2) + parameters['b3']

    if direction == 'W3':
        return nabla_r_Psi @ phi_f(u2).T

    b2_dir_der = (phi_g(u2) @ parameters['W3']) * nabla_r_Psi
    if direction == 'b2':
        return b2_dir_der.T

    if direction == 'W2':
        return b2_dir_der @ phi_f(u1).T

    b1_dir_der = phi_g(u1) @ parameters['W2'] @ b2_dir_der
    if direction == 'b1':
        return b1_dir_der.T

    assert direction == 'W1'

    return b1_dir_der @ x.T


def generate_bias(n: int, random=False):
    assert n > 0
    if random:
        return np.random.rand(n, 1)
    return np.zeros((n, 1))


def generate_weight(m: int, n: int):
    assert m > 0
    assert n > 0
    return np.random.rand(m, n) / math. sqrt(n)


def generate_params(random=True):
    # If random is false, biases will be initialized as zeros
    params = dict()
    params['b1'] = generate_bias(4, random=random)
    params['b2'] = generate_bias(3, random=random)
    params['b3'] = generate_bias(1, random=random)
    params['W1'] = generate_weight(2, 4)
    params['W2'] = generate_weight(4, 3)
    params['W3'] = generate_weight(3, 1)
    return params


def numdiff_calc_dnn_error_grad(grad_of, x, params: dict, epsilon: float):
    assert epsilon > 0
    assert grad_of in params

    max_abs_val_of_x = abs(max(x.min(), x.max(), key=abs))
    x_dim = len(x)
    epsilon = pow(epsilon, 1 / x_dim) * max_abs_val_of_x
    assert epsilon > 0

    assert x.shape[1] == 1

    x_dim = params[grad_of].shape[0]
    y_dim = params[grad_of].shape[1]
    grad = np.zeros(params[grad_of].shape)
    for i in range(0, x_dim):
        for j in range (0,y_dim):
            params[grad_of][i][j] += epsilon
            right_f = dnn_error(x, params)
            params[grad_of][i][j] -= 2*epsilon
            left_f = dnn_error(x, params)
            diff = right_f - left_f
            assert diff.shape == (1, 1)
            diff = diff[0][0]
            grad[i][j] = diff / (2*epsilon)
            # cleanup
            params[grad_of][i][j] += epsilon

    return grad.T


def pack_params(params):
    # save the packing dimensions
    pack_params.shapes = [param.shape for param in params]
    return np.hstack(np.ravel(param) for param in params).reshape(-1, 1)


def unpack_params(packed: np.ndarray):
    shapes = pack_params.shapes
    sizes = map(lambda x: x[0] * x[1], shapes)
    indexes = list(sizes)
    for i in range(len(indexes) - 1):
        indexes[i+1] += indexes[i]
    arrays = np.split(packed, indexes)
    return (arr.reshape(shape) for arr, shape in zip(arrays, shapes))


def dnn_error_ang_grad(x: np.ndarray, y, parameters):
    x = x.reshape((-1, 1))
    W1, W2, W3, b1, b2, b3 = unpack_params(parameters)
    param_dict = {'W1': W1, 'W2': W2, 'W3': W3, 'b1': b1, 'b2': b2, 'b3': b3}
    out = dnn_forward(x, param_dict)
    error = error_f(out, y)
    grad_W1 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'W1').T
    grad_W2 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'W2').T
    grad_W3 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'W3').T
    grad_b1 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'b1').T
    grad_b2 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'b2').T
    grad_b3 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'b3').T
    return np.array((error,
                     pack_params((grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3))))


def target_function(X, Y, parameters):
    # W1, b1, W2, b2, W3, b3 = unpack_params(parameters)
    # dnn_error_dict = {'W1': W1, 'W2': W2, 'W3': W3, 'b1': b1, 'b2': b2, 'b3': b3}
    error_sum = sum(dnn_error_ang_grad(x, y, parameters)[0] for x, y in zip(X.T, Y))
    gradient = dnn_error_ang_grad(X.T[0], Y[0], parameters)[1]
    # return error_sum / X.shape[1], gradient
    return error_sum, gradient


def get_target_f_of_params(X, Y):
    return lambda p: target_function(X=X, Y=Y, parameters=p)


def main():
    # plot the target function
    line = np.arange(-2, 2, .2)
    X1, X2 = np.meshgrid(line, line)
    vectorized_target_function = np.vectorize(true_function_y)
    Y = vectorized_target_function(X1, X2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap=plt.cm.coolwarm, alpha=.6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    plt.title('$f(x_1, x_2) = x_1*exp(-x_1^2-x_2^2)$')

    plt.show(block=False)

    # generate train and test data
    # Ntrain = 500
    Ntrain = 4
    X_train = 4 * np.random.rand(2, Ntrain) - 2

    # Ntest = 200
    Ntest = 5
    X_test = 4 * np.random.rand(2, Ntest) - 2

    Y_train = np.zeros((Ntrain, 1))
    for i in range(0, Ntrain):
        Y_train[i] = vectorized_target_function(X_train[0][i], X_train[1][i])

    Y_test = np.zeros((Ntest, 1))
    for i in range(0, Ntest):
        Y_test[i] = vectorized_target_function(X_test[0][i], X_test[1][i])

    # Train the DNN
    params = generate_params(False)
    params = pack_params((params['W1'], params['W2'], params['W3'], params['b1'],
                         params['b2'], params['b3']))

    learned_params, f_history = BFGS.BFGS(get_target_f_of_params(X_train, Y_train), params)

    # Plotting the BFGS graph
    f_history = [f_history[i][0][0] for i in range(0, len(f_history))]
    plt.figure(figsize=(8, 7))
    plt.plot(f_history)
    plt.semilogy()
    plt.xlabel('Number of iterations')
    plt.ylabel('$|F(x, W_k)-f(x_1, x_2)|^2$')
    plt.grid()
    plt.title('BFGS of DNN trying to approximate $f(x_1, x_2) = x_1*exp(-x_1^2-x_2^2)$')
    plt.show(block=False)

    W1, W2, W3, b1, b2, b3 = unpack_params(learned_params)
    param_dict = {'W1': W1, 'W2': W2, 'W3': W3, 'b1': b1, 'b2': b2, 'b3': b3}
    reconstructed = np.array(list(dnn_forward(x.reshape(-1, 1), param_dict)[0][0] for x in X_test.T))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap=plt.cm.coolwarm, alpha=.6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$F(x, W)$')
    ax.scatter(X_test[:][0], X_test[:][1], reconstructed, c='g', alpha=.61)
    plt.title('$Predictions of trained DNN$')
    plt.show()

    print('success')


class task3_q_2 (unittest.TestCase):
    '''
    Unit test class, including the task 3 question 2 test
    '''

    def test_target_function(self):
        '''
        checks correctness of the target function, sanity check
        '''
        npt.assert_almost_equal(true_function_y(0, 0), 0)
        npt.assert_almost_equal(true_function_y(0, 17), 0)
        npt.assert_almost_equal(true_function_y(1, 0), np.exp(-1))


    def test_generate_params(self):
        '''
        Tests the generate params function
        '''
        params = generate_params()
        self.assertTrue(isinstance(params, dict))
        self.assertEqual(6, len(params))
        required_params = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']
        for par in required_params:
            self.assertTrue(par in params)


    def test_grad_numdiff(self):
        '''
        A singular test of correctness of our analytical gradients.
        '''
        params = generate_params()
        x = 2 * np.random.rand(2, 1) - 1
        epsilon = pow(2, -30)

        ready_tests = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        for test in ready_tests:
            anal = analytic_calc_dir_grads_dnn_error(x, params, test)
            numeric = numdiff_calc_dnn_error_grad(test, x, params, epsilon)
            npt.assert_almost_equal(numeric, anal)

    def test_stress_grad_numdiff(self):
        '''
        TASK 3 QUESTION 2 TEST
        '''
        for i in range(0, 100):
            self.test_grad_numdiff()

    def test_packing(self):
        '''
        check that packing and unpacking functions work
        '''
        a1 = np.array([[4, 5, 6], [41, 51, 63], [1, 2, 1]])
        a2 = np.array([[100]])
        a3 = np.array([[411, 225, 446, 55], [411, 225, 446, 55]])
        p = pack_params((a1, a2, a3))
        b1, b2, b3 = unpack_params(p)
        npt.assert_equal(a1, b1)
        npt.assert_equal(a2, b2)
        npt.assert_equal(a3, b3)


if __name__ == "__main__":
    main()
    # unittest.main()

