import numpy as np
import math

ALPHA_0 = 1
SIGMA = 0.25
BETA = 0.5
EPSILON = pow(10, -5)


def armijo_line_search(x, fun, direction):
    fun_x, grad_x = fun(x)
    alpha_k = ALPHA_0

    while True:
        left_side, dummy = fun(x + alpha_k * direction)
        right_side = fun_x + SIGMA * alpha_k * np.transpose(direction).dot(grad_x)
        if left_side <= right_side:
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
    B_k = np.identity(x_len)
    x_k = x_0

    while True:
        f_val, f_grad = fun(x_k)
        m.append(f_val)
        # 1. Compute approx Newton direction
        direction = -np.dot(B_k, f_grad)
        # 2. Inexact line search - Armijo method
        s_k = armijo_line_search(x_k, fun, direction)*direction
        x_k = x_k + s_k
        # 3. Compute next gradient
        dummy, g_next_x = fun(x_k)
        if np.linalg.norm(g_next_x) < EPSILON:
            break
        # 4. Update approximate inverse Hessian

        y_k = g_next_x - f_grad
        new_B = B_k + (np.dot(y_k, y_k))/np.dot(y_k, s_k) + \
                np.dot(np.dot(np.dot(B_k, s_k), np.transpose(s_k)), np.transpose(B_k)) /\
                (np.dot(np.dot(np.transpose(s_k), B_k), s_k))

        new_dir_der = -np.dot(new_B, f_grad)
        assert new_dir_der < 0 #??
        if new_dir_der > -np.dot(B_k, f_grad):
            B_k = new_B

    return fun(x_k), m


def rosenbrock(x):
    scalar = 0
    for i in range(0, len(x)-1):
        scalar += (1-x[i])**2 + 100 * (x[i+1] - x[i]**2) ** 2
    gradients = np.zeros(len(x))
    for i in range(0, len(x)-1):
        gradients[i] += -2*(1-x[i]) - 400 * (x[i+1]-x[i]**2)*x[i]
    for i in range(1, len(x)):
        gradients[i] += 200*(x[i] - x[i-1] ** 2)

    return scalar, gradients


def main():

    scalar, grads = rosenbrock([1, 1])
    assert scalar == 0
    for val in grads:
        assert val == 0

    x0 = np.asarray([0, 0, 0])
    scalar, points = BFGS(rosenbrock, x0)
    print('success')


if __name__ == "__main__":
    main()

