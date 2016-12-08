from sklearn     import neural_network
from regression  import get_data, train_validate_test_classifier


layer_sizes = [ (1,), (5,), (10,), (100,), (1000,), (1,1), (5,5), (10,10), (100,100), (1000,1000) ]
alphas = [ 0, 0.01, 0.1, 1, 10 ] # degree of regularizations


def find_best_architecture(data):
  max_accuracy = 0
  opt_layer_size = ()
  opt_alpha = 0

  for layer_size in layer_sizes:
    for alpha_val in alphas:
      nn_classifier = neural_network.MLPClassifier(hidden_layer_sizes=layer_size, solver="sgd", alpha=alpha_val)

      testing_accuracy = train_validate_test_classifier(data, nn_classifier)

      if testing_accuracy > max_accuracy:
        opt_layer_size = layer_size
        opt_alpha = alpha_val
        max_accuracy = testing_accuracy

  return opt_layer_size, opt_alpha



data = get_data()
opt_layer_size, opt_alpha = find_best_architecture(data)

