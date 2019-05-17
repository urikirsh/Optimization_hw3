import numpy as np
import math

ALPHA_0 = 1
SIGMA = 0.25
BETA = 0.5
# EPSILON = pow(10, -5)
EPSILON = pow(10, -2)


def armijo_line_search(x, fun, direction):
    assert isinstance(direction, np.ndarray)
    dir_size = np.linalg.norm(direction)
    if dir_size == 0:
        raise FloatingPointError()
    assert dir_size > 0
    direction = direction/dir_size
    assert abs(np.linalg.norm(direction) - 1) < 0.0000001
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
        # 2. Inexact line search - Armijo method
        s_k = armijo_line_search(x_k, fun, direction) * direction
        x_k = x_k + s_k
        # 3. Compute next gradient
        dummy, g_next_x = fun(x_k)

        # 4. Update approximate inverse Hessian
        y_k = g_next_x - f_grad

        s_k = s_k.reshape(len(s_k), 1)
        y_k = y_k.reshape(len(y_k), 1)
        sty = np.dot(np.transpose(s_k), y_k) # S_k^T*Y_k, should be a scalar
        assert sty.shape == (1, 1)
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
        # old_directional_der = np.dot(f_grad, direction)
        # assert old_directional_der < 0
        # new_directional_der = np.dot(g_next_x, direction)
        # assert new_directional_der < 0
        # if old_directional_der < new_directional_der:
        #     y_k = g_next_x - f_grad
        #     new_B = B_k + (np.dot(y_k, y_k)) / np.dot(y_k, s_k) + \
        #             np.dot(np.dot(np.dot(B_k, s_k), np.transpose(s_k)), np.transpose(B_k)) / \
        #             (np.dot(np.dot(np.transpose(s_k), B_k), s_k))
        #     B_k = new_B
        # new_dir_der = -np.dot(new_B, f_grad)
        # assert new_dir_der < 0
        # if new_dir_der > -np.dot(B_k, f_grad):
        #     B_k = new_B

    return fun(x_k), m


def rosenbrock(x):
    scalar = 0

    for i in range(0, len(x)-1):
        try:
            scalar += (1-x[i])**2 + 100 * (x[i+1] - x[i]**2) ** 2
        except FloatingPointError:
            print("i =", i, "x=", x, "scalar =", scalar)

    gradients = np.zeros(len(x))
    for i in range(0, len(x)-1):
        gradients[i] += -2*(1-x[i]) - 400 * (x[i+1]-x[i]**2)*x[i]
    for i in range(1, len(x)):
        gradients[i] += 200*(x[i] - x[i-1] ** 2)

    return scalar, gradients


def main():
    np.seterr(all='raise')
    scalar, grads = rosenbrock([1, 1])
    assert scalar == 0
    for val in grads:
        assert val == 0

    x0 = np.asarray([0, 0, 0])
    scalar, points = BFGS(rosenbrock, x0)
    print('success')


if __name__ == "__main__":
    main()

