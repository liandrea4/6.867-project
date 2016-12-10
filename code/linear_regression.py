from gradient_descent       import make_numeric_gradient_calculator
from sgd                    import plot_data
import matplotlib.pyplot    as plt
import numpy
import math
import sys

##### Maximum likelihood #####

def calculate_polynomial_phi(x_vector, M):
    phi = []
    for x in x_vector:
        phi_x = []
        for i in range(M+1):
            phi_x.append(x ** i)
        phi.append(phi_x)
    return phi

def calculate_cosine_phi(x_vector, M):
    if type(M) != int or M > 8:
        raise Exception("Only check first 8 cosines")

    phi = []
    for x in x_vector:
        phi_x = []
        for i in range(1, M+1):
            phi_x.append(math.cos(math.pi * x * i))
        phi.append(phi_x)
    return phi

def calculate_mle_weight(x_vector, y_vector, calculate_phi_fn, M):
    phi = calculate_phi_fn(x_vector, M)
    phi_transpose = numpy.transpose(phi)
    inversed = numpy.linalg.inv(numpy.dot(phi_transpose, phi))
    w_mle = numpy.dot(numpy.dot(inversed, phi_transpose), y_vector)
    return w_mle

##### Sum of square error #####
def make_sse_objective_fn(x, y, list_of_basis_functions):
    def sse_objective_fn(weight_vector):
        regression_fn = get_generic_regression_fn(list_of_basis_functions, weight_vector)

        sse = 0
        for x_i, y_i in zip(x, y):
            difference = y_i - regression_fn(x_i)
            sse += (difference ** 2)
        return sse

    return sse_objective_fn

def get_generic_regression_fn(list_of_basis_functions, w_mle):
    len_difference = len(list_of_basis_functions) - len(w_mle)
    if len_difference > 0:
        numpy.append(w_mle, [0] * len_difference)

    def generic_regression_fn(x):
        fn = 0
        for index in range(len(w_mle)):
            fn += w_mle[index] * list_of_basis_functions[index](x)
        return fn
    return generic_regression_fn

####Stochastic Gradient Descent###
def make_single_point_least_square_error(list_of_basis_functions):
    def single_point_least_square_error(x, y, theta):
        regression_fn = get_generic_regression_fn(list_of_basis_functions, theta)
        sse = y - regression_fn(x)
        sse = sse**2
        return sse
    return single_point_least_square_error

def calc_next_theta(old_theta, x, y, t, gradient):
    t0 = 100
    k = 0.6
    n = lambda t: (t0 + t)**(-k)
    print "n: ", n(t), "   gradient: ", gradient(old_theta)
    return old_theta - (n(t) * gradient(old_theta))

def sgd(x, y, theta, objective_f, threshold, gradient):
    number_of_samples = len(x)
    old_jthetas = [0] * number_of_samples
    differences = [False]*number_of_samples
    previous_values = []
    t = 0

    while not all(differences):
        i = t % number_of_samples
        # print "old_x: ", theta
        theta = calc_next_theta(theta, x[i], y[i], t, gradient)
        # print "new_x: ", theta
        new_jtheta = objective_f(x[i], y[i], theta)
        difference = new_jtheta - old_jthetas[i]

        if(abs(difference)<threshold):
            differences[i] = True

        previous_values.append((theta, new_jtheta))
        # print "old_jtheta: ", old_jthetas[i], "   new_jtheta: ", new_jtheta
        # print "difference: ", difference
        old_jthetas[i] = new_jtheta

        t += 1

    return previous_values


if __name__ == '__main__':
    M = 2

    # TODO: get_data function
    x, y = get_data()

    real_fn = lambda x: math.cos(math.pi * x) + 1.5 * math.cos(2 * math.pi * x)

    basis0 = lambda x: x**0
    basis1 = lambda x: x**1
    basis2 = lambda x: x**2
    basis3 = lambda x: x**3
    basis4 = lambda x: x**4
    basis5 = lambda x: x**5
    list_of_basis_functions = [basis0, basis1, basis2, basis3, basis4, basis5]

    ## Maximum likelihood calculations

    w_mle = calculate_mle_weight(x, y, calculate_polynomial_phi, M)

    ## Gradient descent
    weight_vector = numpy.array([100.0] * (M+1))
    step_size = 0.06
    threshold = 0.0001

    single_point_objective_f = make_single_point_least_square_error(list_of_basis_functions)

    objective_f = make_sse_objective_fn(x, y, list_of_basis_functions)
    numeric_gradient = make_numeric_gradient_calculator(objective_f, 0.00001)

    previous_values = sgd(x, y, weight_vector, single_point_objective_f, threshold, numeric_gradient)
    min_x, min_y = (previous_values[-1][0], previous_values[-1][1])

    print "min_x: ", min_x, "  min_y",  min_y
    print "number of steps: ", len(previous_values)
    print "w_mle: ", w_mle

    plot_data(previous_values, 0)
