import numpy as np
import math
import matplotlib.pyplot as plt

ALPHA_0 = 1
SIGMA = 0.25
BETA = 0.5
EPSILON = pow(10, -5)


def sqr(vec):
    '''
    :param vec: column vector
    :return: vec * vec^T
    '''
    assert isinstance(vec, np.ndarray)
    assert vec.shape[1] == 1
    return vec.dot(vec.T)


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
    x = x.reshape(len(x), 1)

    fun_at_x, grad_x = fun(x)
    alpha_k = ALPHA_0
    df = np.dot(direction.T, grad_x)

    while fun(x + alpha_k * direction)[0] > fun_at_x + SIGMA * alpha_k * df:
        alpha_k *= BETA

    return alpha_k


def rosenbrock(x):
    '''
    :param x: A vector
    :return: (rosenbrock value of x, gradient of rosenbrock function at point x)
    '''
    assert isinstance(x, np.ndarray)
    assert x.shape[1] == 1

    scalar_ret_value = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)

    gradients = np.zeros(len(x))
    for i in range(0, len(x)-1):
        gradients[i] += -2*(1-x[i]) - 400 * (x[i+1]-x[i]**2)*x[i]
    for i in range(1, len(x)):
        gradients[i] += 200*(x[i] - x[i-1] ** 2)

    gradients = gradients.reshape((len(gradients), 1))

    return scalar_ret_value, gradients


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

    f_history = []
    B_k = np.identity(x_len)
    x_k = x_0

    while True:
        print("x_k =", x_k)
        f_x, g_x = fun(x_k)
        f_history.append(f_x)
        if np.linalg.norm(g_x) < EPSILON:
            break
        # 1. Compute approx Newton direction
        direction = -np.dot(B_k, g_x)

        # Normalize the direction
        dir_size = np.linalg.norm(direction)
        if dir_size == 0:
            raise FloatingPointError()
        assert dir_size > 0
        direction = direction / dir_size
        assert abs(np.linalg.norm(direction) - 1) < 0.0000001  # Turns out not to be zero for rounding errors

        # 2. Inexact line search - Armijo method
        s_k = armijo_line_search(x_k, fun, direction) * direction
        next_x = x_k + s_k

        # 3. Compute next gradient
        f_next_x, g_next_x = fun(next_x)

        # 4. Update approximate inverse Hessian
        p = next_x - x_k
        q = g_next_x - g_x
        s = B_k.dot(q)
        t = s.T.dot(q)

        m = p.T.dot(q)
        v = p / m - s / t
        next_B = B_k + sqr(p) / m - sqr(s) / t + t * sqr(v)

        B_k = next_B
        x_k = next_x

    return x_k, f_history


def main():
    x_opt, grad_at_x_opt = rosenbrock(np.asarray([1, 1]).reshape(2, 1))
    assert x_opt == 0
    for val in grad_at_x_opt:
        assert val == 0

    x0 = np.zeros(10).reshape(10, 1)
    x_opt, x_history = BFGS(rosenbrock, x0)
    print("solved after", len(x_history), "iterations")
    opt_val, grad_opt = rosenbrock(np.asarray(x_opt))
    grad_opt_size = np.linalg.norm(grad_opt)
    print("\n\nfinal x = ", x_opt, "\n\nfinal gradient=", grad_opt, "\n of size=", grad_opt_size)

    plt.figure()
    plt.plot(x_history)
    plt.semilogy()
    plt.xlabel('Number of iterations')
    plt.ylabel('$f(x_k)-p^*$')
    plt.grid()
    plt.title('BFGS of Rosenbrock function')
    plt.show()
    print('success')


if __name__ == "__main__":
    main()

