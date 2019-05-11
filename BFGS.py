ALPHA_0 = 1
SIGMA = 0.25
BETA = 0.5
EPSILON = pow(10, -5)


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
    raise NotImplementedError()


def main():
    print('main')


if __name__ == "__main__":
    main()

