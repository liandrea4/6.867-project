from gradient_descent       import gradient_descent, make_numeric_gradient_calculator
import matplotlib.pyplot    as plt
import numpy

####### Least squares error ########

def make_least_square_error(x_matrix, y_vector):
    def least_square_error(theta):
        length = len(y_vector)
        error_sum = 0
        for i in range(length):
            x_transposed = numpy.matrix.transpose(x_matrix[i])
            partial_sum = numpy.dot(x_transposed, theta) - y_vector[i]
            partial_sum_squared = partial_sum ** 2
            error_sum += partial_sum_squared
        return error_sum
    return least_square_error

def make_least_square_derivative(x, y):
    def least_square_derivative(theta):
        gradient_sum = numpy.dot(numpy.dot(numpy.matrix.transpose(x[0]), theta) - y[0], x[0])
        # least_square = []
        for i in range(1, len(y)):
            # least_square.append((numpy.dot(numpy.matrix.transpose(x[i]), theta) - y[i]))
            gradient_sum += (numpy.dot(numpy.dot(numpy.matrix.transpose(x[i]), theta) - y[i], x[i]))
        gradient_sum *= 2
        # print "least_square: ", least_square, "   sum: ", sum(least_square)
        return gradient_sum
    return least_square_derivative

####### SGD update ##########
def make_single_point_least_square_error():
    def single_point_least_square_error(x, y, theta):
        x_transposed = numpy.matrix.transpose(x)
        difference = numpy.dot(x_transposed, theta) - y
        return difference ** 2
    return single_point_least_square_error


def make_single_point_least_square_error_2(x, y):
    def single_point_least_square_error_2(theta):
        x_transposed = numpy.matrix.transpose(x)
        difference = numpy.dot(x_transposed, theta) - y
        return difference ** 2
    return single_point_least_square_error_2

def calc_next_theta(old_theta, x, y, t):
    t0 = 10**6
    k = 0.75
    n = lambda t: (t0 + t)**(-k)
    gradient = 2 * numpy.dot(numpy.dot(numpy.matrix.transpose(x), old_theta) - y, x)
    print "n: ", n(t), "   gradient: ", gradient
    return old_theta - (n(t) * gradient)

def sgd(x, y, theta, objective_f, threshold):
    number_of_samples = len(x)

    differences = [False]*number_of_samples
    old_jthetas = [0.0] * number_of_samples
    previous_values = []
    t = 0

    while not all(differences):
        i = t % number_of_samples
        theta = calc_next_theta(theta, x[i], y[i], t)
        new_jtheta = objective_f(x[i], y[i], theta)
        difference = new_jtheta - old_jthetas[i]

        if(abs(difference)<threshold):
            differences[i] = True

        previous_values.append((theta, new_jtheta))
        old_jthetas[i] = new_jtheta

        t += 1

    return previous_values

def plot_data(previous_values, x_channel):
    x = [ value[0][x_channel] for value in previous_values ]
    y = [ value[1] for value in previous_values ]
    # data_x = [ value[x_channel] for value in x_matrix ]
    plt.figure()
    plt.plot(x, y, 'ro')
    plt.title("Batch gradient descent", fontsize=20)
    # plt.plot(data_x, y_vector, 'bo')
    plt.show()

if __name__ == '__main__':

    step_size = 10**(-6)

    threshold = 1
    x_channel = 1

    fitting_data = getData()
    x_matrix = fitting_data[0]
    y_vector = fitting_data[1]
    theta = numpy.array([0.0] * 10)

    objective_f = make_least_square_error(x_matrix, y_vector)
    single_point_objective_f = make_single_point_least_square_error()
    gradient_f = make_numeric_gradient_calculator(objective_f, 0.001)
    gradient_f = make_least_square_derivative(x_matrix, y_vector)

    # calc_next_theta(theta, x_matrix[0], y_vector[0], 2.0)
    # print gradient_f(theta)

    # previous_values = gradient_descent(objective_f, gradient_f, theta, step_size, threshold)
    # min_x, min_y = (previous_values[-1][0], previous_values[-1][1])

    previous_values = sgd(x_matrix, y_vector, theta, single_point_objective_f, threshold)
    min_x, min_y = (previous_values[-1][0], previous_values[-1][1])

    print "min_x: ", min_x, "  min_y",  min_y
    print "number of steps: ", len(previous_values)

    plot_data(previous_values, x_channel)
    print len(y_vector)



