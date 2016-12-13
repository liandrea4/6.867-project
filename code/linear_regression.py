from gradient_descent       import make_numeric_gradient_calculator
from sgd                    import plot_data
from regression             import get_data, get_spread, get_win_lose
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

def get_polynomial_regression_fn(w_mle):
    def regression_fn(x):
        fn = 0
        for index in range(len(w_mle)):
            fn += w_mle[index] * (x ** index)
        return fn
    return regression_fn


def get_cosine_regression_fn(w_mle):
    def regression_fn(x):
        fn = 0
        for index in range(len(w_mle)):
            fn += w_mle[index] * math.cos(x * (index + 1) * math.pi)
        return fn
    return regression_fn

def plot_regression(x, y, fns_to_plot, title):
    plt.figure()

    ## Plot data points
    plt.plot(x, y, 'o')

    x_fn_values = numpy.linspace(min(x), max(x), 1000)

    for fn in fns_to_plot.keys():
        fn_y_values = [ fns_to_plot[fn](x_i) for x_i in x_fn_values ]
        plt.plot(x_fn_values, fn_y_values, 'r-', linewidth=2, label=fn)

    plt.title(title)
    plt.xlabel("Win rate")
    plt.ylabel("Actual spread")
    plt.show()


def predict(x_i, w_mle):
    prediction = 0
    for index, w_i in zip(range(len(w_mle)), w_mle):
        prediction = x_i ** index * w_i

    return prediction


def get_average_error(x, y, w_mle):
    error_sum = 0.

    for x_i, y_i in zip(x, y):
        predicted_spread = predict(x_i, w_mle)

        error = abs(predicted_spread - y_i)
        error_sum += error

    return error_sum / len(x)


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


def validate(x_validate, y_validate, w_mle_dict):
    min_error = float('inf')

    for M in w_mle_dict:
        w_mle = w_mle_dict[M]
        error = get_average_error(x_validate, y_validate, w_mle)

        print "M: ", M, " error: ", error

        if error < min_error:
            min_error = error
            opt_M = M

    return opt_M, w_mle_dict[opt_M]


if __name__ == '__main__':
    M = 3
    M_list = range(11)

    filename = "data/2008-2009 Game Log.csv" # "data/game_log_06_07.csv"
    alphas = [ 0, 0.01, 0.1, 1, 10 ] # degree of regularizations

    away_indices = {
        "ELO": 0,
        "points scored": 1,
        "points allowed": 2,
        "games played": 3,
        "games won": 4,
        "win rate": 5,
        "home court points scored": 6,
        "home court points allowed": 7,
        "home court games played": 8,
        "home court win rate": 9,
        "possessions": 10,
        "defensive rebounds": 11,
        "offensive rebounds": 12
    }


    num_skip = 320 # stabilized by ~ each team's 20th game
    num_training = 900
    num_validate = 0
    num_testing = 0

    data_win_lose = get_data(filename, num_skip, num_training, num_validate, num_testing, get_win_lose)
    data_spreads = get_data(filename, num_skip, num_training, num_validate, num_testing, get_spread)

    away_index = away_indices["win rate"]
    home_index = away_index + 13
    x_training, y_training = [ x_i[away_index] - x_i[home_index] for x_i in data_spreads[0] ], data_spreads[1]
    x_validate, y_validate = [ x_i[away_index] - x_i[home_index] for x_i in data_spreads[2] ], data_spreads[3]
    x_testing, y_testing = [ x_i[away_index] - x_i[home_index] for x_i in data_spreads[4] ], data_spreads[5]

    ## Maximum likelihood calculations

    w_mle_dict = {}

    # for M in M_list:
    #     w_mle = calculate_mle_weight(x_training, y_training, calculate_polynomial_phi, M)
    #     w_mle_dict[M] = w_mle

    # opt_M, opt_w_mle = validate(x_validate, y_validate, w_mle_dict)

    # print "w_mle: ", opt_w_mle

    opt_M = 5
    opt_w_mle = calculate_mle_weight(x_training, y_training, calculate_polynomial_phi, opt_M)

    regression_fn = get_polynomial_regression_fn(opt_w_mle)
    fns_to_plot = {
        "Linear regression": regression_fn
    }

    print "training error: ", get_average_error(x_training, y_training, opt_w_mle)
    # print "testing error: ", get_average_error(x_testing, y_testing, opt_w_mle)
    plot_regression(x_training, y_training, fns_to_plot, "Regression (M=" + str(opt_M) + ")")

    # #Cosine Regression
    # w_mle = calculate_mle_weight(x_training, y_training, calculate_cosine_phi, M)
    # regression_fn = get_cosine_regression_fn(w_mle)
    # fns_to_plot = {
    #     "Actual": real_fn,
    #     "Cosine regression": regression_fn
    # }
    # print "w_mle: ", w_mle
    # plot_regression(x_training, y_training, fns_to_plot, "Cosine regression (M=" + str(M) + ")")




    # ## Gradient descent
    # weight_vector = numpy.array([100.0] * (M+1))
    # step_size = 0.06
    # threshold = 0.1

    # single_point_objective_f = make_single_point_least_square_error(list_of_basis_functions)

    # objective_f = make_sse_objective_fn(x_training, y_training, list_of_basis_functions)
    # numeric_gradient = make_numeric_gradient_calculator(objective_f, 0.00001)

    # previous_values = sgd(x_training, y_training, weight_vector, single_point_objective_f, threshold, numeric_gradient)
    # min_x, min_y = (previous_values[-1][0], previous_values[-1][1])

    # print x_training

    # print "min_x: ", min_x, "  min_y",  min_y
    # print "number of steps: ", len(previous_values)
    # print "w_mle: ", w_mle

    # plot_data(previous_values, 0)