import numpy as np
import math
import matplotlib.pyplot as plt

ALPHA_0 = 1
SIGMA = 0.25
BETA = 0.5
EPSILON = pow(10, -5)


def armijo_line_search(x, fun, direction):
    '''
    Run an Armijo line search and return the step size
    :param x: The starting point, a vector
    :param fun: A function that returns two values, the value of f(x) and gradient of f(x)
    :param direction: A direction vector. Must be nonzero
    :return: Step size for the given direction from the given x
    '''

    assert isinstance(x, np.ndarray)
    assert isinstance(direction, np.ndarray)

    fun_x, grad_x = fun(x)
    alpha_k = ALPHA_0

    while True:
        left_side, dummy = fun(x + alpha_k * direction)
        right_side = fun_x + SIGMA * alpha_k * np.transpose(direction).dot(grad_x)
        if left_side <= right_side:
            # return min(alpha_k, pow(10, -5))
            return alpha_k

        alpha_k *= BETA


def BFGS(fun, x_0):
    '''
    Find the local minimum of fun via BFGS method, starting from point x_0
    :param fun: Target function to minimize. Should be of the form
                f(x), grad(f(x)) = fun(x)
                where x is a vector

    :param x_0:
    :return:
        x - local solution to the problem, a vector of the same size as x_0
        m - vector with values fun(x_k), for each iteration k
    '''
    assert isinstance(x_0, np.ndarray)

    x_len = len(x_0)

    m = []
    nI = np.identity(x_len)
    B_k = nI
    x_k = x_0

    while True:
        print("x_k =", x_k)
        f_val, f_grad = fun(x_k)
        m.append(f_val)
        if np.linalg.norm(f_grad) < EPSILON:
            break
        # 1. Compute approx Newton direction
        direction = -np.dot(B_k, f_grad)

        # Normalize the direction
        dir_size = np.linalg.norm(direction)
        if dir_size == 0:
            raise FloatingPointError()
        assert dir_size > 0
        direction = direction / dir_size
        assert abs(np.linalg.norm(direction) - 1) < 0.0000001  # Turns out not to be zero for rounding errors

        # 2. Inexact line search - Armijo method
        s_k = armijo_line_search(x_k, fun, direction) * direction
        x_k = x_k + s_k
        # 3. Compute next gradient
        dummy, g_next_x = fun(x_k)
        # 4. Update approximate inverse Hessian

        # Experiment
        old_dir_der = np.dot(f_grad, direction)
        dummy, new_f_grad = fun(x_k)
        new_dir_der = np.dot(direction, new_f_grad)

        if new_dir_der >= 0 or new_dir_der <= old_dir_der:
            y_k = g_next_x - f_grad
            s_k = s_k.reshape(len(s_k), 1)
            y_k = y_k.reshape(len(y_k), 1)
            sty = np.dot(np.transpose(s_k), y_k)  # S_k^T*Y_k, should be a scalar
            assert sty.shape == (1, 1)
            if sty == 0:
                continue
            ytby = np.dot(np.dot(np.transpose(y_k), B_k), y_k) # y_k^t * B_k * y_k, scalar
            assert ytby.shape == (1, 1)
            skskt = np.dot(s_k, np.transpose(s_k))  # s_k*s_k^t
            assert skskt.shape == (len(s_k), len(s_k))
            assert skskt.shape == B_k.shape

            second_additive = (sty + ytby) * skskt / (sty ** 2)
            assert second_additive.shape == B_k.shape

            byst = np.dot(np.dot(B_k, y_k), np.transpose(s_k))  # B_k*y_k*s_k^t
            assert byst.shape == B_k.shape
            sytb = np.dot(np.dot(s_k, np.transpose(y_k)), B_k)
            assert sytb.shape == B_k.shape
            third_additive = (byst + sytb) / sty
            assert third_additive.shape == B_k.shape

            new_B = B_k + second_additive - third_additive
            assert new_B.shape == B_k.shape

            B_k = new_B

    return x_k, m


def rosenbrock(x):
    '''
    :param x: A vector
    :return: (rosenbrock value of x, gradient of rosenbrock function at point x)
    '''
    assert isinstance(x, np.ndarray)

    scalar_ret_value = 0

    for i in range(0, len(x)-1):
        try:
            scalar_ret_value += (1-x[i])**2 + 100 * (x[i+1] - x[i]**2) ** 2
        except FloatingPointError:
            print("i =", i, "x=", x, "scalar =", scalar_ret_value)

    gradients = np.zeros(len(x))
    for i in range(0, len(x)-1):
        gradients[i] += -2*(1-x[i]) - 400 * (x[i+1]-x[i]**2)*x[i]
    for i in range(1, len(x)):
        gradients[i] += 200*(x[i] - x[i-1] ** 2)

    return scalar_ret_value, gradients


def main():
    # np.seterr(all='raise')
    x_opt, vals_of_x_k = rosenbrock(np.asarray([1, 1]))
    assert x_opt == 0
    for val in vals_of_x_k:
        assert val == 0

    x0 = np.zeros(10)
    x_opt, vals_of_x_k = BFGS(rosenbrock, x0)
    opt_val, grad_opt = rosenbrock(np.asarray(x_opt))
    grad_opt_size = np.linalg.norm(grad_opt)
    print("\n\nfinal x = ", x_opt, "\n\nfinal gradient=", grad_opt, "\n of size=", grad_opt_size)

    plt.figure()
    plt.plot(vals_of_x_k)
    plt.semilogy()
    plt.xlabel('Number of iterations')
    plt.ylabel('$f(x_k)-p^*$')
    plt.grid()
    plt.title('BFGS of Rosenbrock function')
    plt.show()
    print('success')


if __name__ == "__main__":
    main()

