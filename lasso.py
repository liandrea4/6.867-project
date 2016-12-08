import sys
sys.path.append('../P3')
sys.path.append('../P2')
from sklearn                    import linear_model
from lasso_manual               import optimize, ridge_loss, create_features, predict
from ridge_regression           import make_sse_objective_fn, train_model, validate_models
from linear_regression          import get_generic_regression_fn, plot_regression, get_polynomial_regression_fn
import numpy                    as np
import matplotlib.pyplot        as plt
import matplotlib.pylab         as pylab
import scipy.optimize           as spo
import math
import lassoData


def sin_features(x_training, M):
    x_matrix = np.zeros((len(x_training),M))
    for i in range(len(x_training)):
        for j in range(M):
            if j == 0:
                x_matrix[i,j] = x_training[i]
            else:
                x_matrix[i,j] = math.sin(math.pi * 0.4 * x_training[i] * j)
    return x_matrix

def train(x_training, y_training, lambdas, M):
    lasso_coeffs = {}
    for lambda_val in lambdas:
        lasso_model = linear_model.Lasso(alpha=lambda_val)
        x_matrix = sin_features(x_training, M)
        lasso_model.fit(x_matrix, y_training)
        lasso_coeffs[lambda_val] = lasso_model.coef_

    return lasso_coeffs

def validate(lasso_coeffs, sse_fn):
    min_sse = float("inf")
    all_values = {}
    min_lambda_val = 0

    for lambda_val in lasso_coeffs.keys():
        sse = sse_fn(lasso_coeffs[lambda_val])
        all_values[lambda_val] = sse

        if sse < min_sse:
            min_sse = sse
            min_lambda_val = lambda_val

    return min_lambda_val, all_values

if __name__ == '__main__':
    x_validate, y_validate = lassoData.lassoValData()
    x_training, y_training = lassoData.lassoTrainData()
    x_testing, y_testing = lassoData.lassoTestData()
    w_true = lassoData.lassoW()

    lambdas = [0, .1, .2, .3, .4, .5, .6, .7 ,.8, .9, 1]
    M = len(w_true)
    sin_basis = [
        lambda x: x,
        lambda x: math.sin(0.4 * math.pi * x * 1),
        lambda x: math.sin(0.4 * math.pi * x * 2),
        lambda x: math.sin(0.4 * math.pi * x * 3),
        lambda x: math.sin(0.4 * math.pi * x * 4),
        lambda x: math.sin(0.4 * math.pi * x * 5),
        lambda x: math.sin(0.4 * math.pi * x * 6),
        lambda x: math.sin(0.4 * math.pi * x * 7),
        lambda x: math.sin(0.4 * math.pi * x * 8),
        lambda x: math.sin(0.4 * math.pi * x * 9),
        lambda x: math.sin(0.4 * math.pi * x * 10),
        lambda x: math.sin(0.4 * math.pi * x * 11),
        lambda x: math.sin(0.4 * math.pi * x * 12)
    ]

    ### Ridge regression
    # theta_guess = np.repeat(0,M).reshape(-1,1)
    # results = optimize(lambdas, M, theta_guess, x_training, x_testing, x_validate, y_training, y_testing, y_validate)
    # ridge_theta = results[1][2]
    # ridge_f = ridge_loss(x_train,y_train,0.2,5)
    # # ridge_theta = np.array(spo.fmin_bfgs(ridge_f,theta_guess,gtol=10**-6,disp=0)).reshape(-1,1)
    # X = create_features(x_train,M)

    # X_dom = np.linspace(-1,1,50)
    # ridge_predictions = predict(X,ridge_theta)
    # ridge_curve = predict(X_matrix,ridge_theta)
    # plt.plot(X_dom,ridge_curve,'g',X_dom,actual_values,'r')
    # plt.show()


    x_training, y_training = [ data[0] for data in x_training ], [ data[0] for data in y_training ]
    x_testing, y_testing = [ data[0] for data in x_testing], [ data[0] for data in y_testing ]
    x_validate, y_validate = [ data[0] for data in x_validate ], [ data[0] for data in y_validate ]

    polynomial_sse_fn = make_sse_objective_fn(x_validate, y_validate, sin_basis)
    polynomial_sse_fn_testing = make_sse_objective_fn(x_testing, y_testing, sin_basis)

    w_mles = train_model(x_training, y_training)
    best_M, best_lambda_val, best_w_mle, all_values = validate_models(w_mles, polynomial_sse_fn)
    ridge_regression_fn = get_polynomial_regression_fn(best_w_mle)
    print "best M: ", M, "   best_lambda_val: ", best_lambda_val
    print "best_value: ", all_values[(best_M, best_lambda_val)]
    print "best_w_mle: ", best_w_mle
    # print "best testing value: ", sse_fn_testing(best_w_mle)


    ### LASSO
    sin_sse_fn = make_sse_objective_fn(x_validate, y_validate, sin_basis)
    sin_sse_fn_testing = make_sse_objective_fn(x_testing, y_testing, sin_basis)

    lasso_coeffs = train(x_training, y_training, lambdas, M)
    best_lambda, all_values = validate(lasso_coeffs, sin_sse_fn)
    best_lasso_coeffs = lasso_coeffs[best_lambda]
    print "best_lasso_coeffs: ", best_lasso_coeffs

    lasso_regression_fn = get_generic_regression_fn(sin_basis, best_lasso_coeffs)

    ### Lambda = 0
    lambda0_coeffs = train(x_training, y_training, [0], M)[0]
    print "lambda0_coeffs: ", lambda0_coeffs
    lambda0_regression_fn = get_generic_regression_fn(sin_basis, lambda0_coeffs)

    actual_regression_fn = get_generic_regression_fn(sin_basis, w_true)

    fns_to_plot = {
        "Actual": actual_regression_fn,
        "LASSO": lasso_regression_fn,
        # "Ridge": ridge_regression_fn,
        "Lambda = 0": lambda0_regression_fn
    }
    plot_regression(x_training, y_training, fns_to_plot, "Training, lambda=" + str(best_lambda))
    plot_regression(x_validate, y_validate, fns_to_plot, "Validation, lambda=" + str(best_lambda))
    plot_regression(x_testing, y_testing, fns_to_plot, "Testing, lambda=" + str(best_lambda))



