from sklearn     import neural_network
from sklearn     import preprocessing
from regression  import get_data, train_validate_test_classifier
import csv
import pprint

layer_sizes = [ (1,), (5,), (10,), (100,), (1,1), (5,5), (10,10), (100,100), (10, 20, 10), (5, 10, 10, 5) ]
alphas = [ 0, 0.0001, 0.01, 0.1, 1, 10 ] # degree of regularizations

def get_win_lose(away_score, home_score):
  return home_score > away_score

def get_spread(away_score, home_score):
  return away_score - home_score

def get_data(filename, num_skip, num_training, num_validate, num_testing, y_fn):
  index = 0
  x_training, y_training, x_validate, y_validate, x_testing, y_testing = [], [], [], [], [], []

  with open(filename, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:

      if index == 0:
        index += 1
        continue

      x = [ float(value) for value in row[5:] ]
      away_score = float(row[3])
      home_score = float(row[4])
      y = y_fn(away_score, home_score)

      if index > num_skip and index <= num_skip + num_training:
        x_training.append(x)
        y_training.append(y)

      elif index > num_skip + num_training and index <= num_skip + num_training + num_validate:
        x_validate.append(x)
        y_validate.append(y)

      elif index > num_skip + num_training + num_validate and index <= num_skip + num_training + num_validate + num_testing:
        x_testing.append(x)
        y_testing.append(y)

      index += 1

  return [x_training, y_training, x_validate, y_validate, x_testing, y_testing]

def find_best_architecture(data, Classify_or_regress):
  max_accuracy = 0
  max_score = 20
  opt_layer_size = ()
  opt_alpha = 0
  various_architecture_performances = []

  for layer_size in layer_sizes:
    for alpha_val in alphas:
      if (Classify_or_regress):
        nn_predictor = neural_network.MLPClassifier(hidden_layer_sizes=layer_size, solver="lbfgs", alpha=alpha_val)
        validation_accuracy = train_validate_test_classifier(data, nn_predictor)

        various_architecture_performances.append((layer_size, alpha_val, validation_accuracy))

        if (validation_accuracy > max_accuracy):
          opt_layer_size = layer_size
          opt_alpha = alpha_val
          max_accuracy = validation_accuracy

      else:
        nn_predictor = neural_network.MLPRegressor(hidden_layer_sizes=layer_size, solver="lbfgs", alpha=alpha_val)  
        validation_score = calculate_validation_score_nn_regressor(data, nn_predictor)

        various_architecture_performances.append((layer_size, alpha_val, validation_score))

        if (validation_score < max_score):
            opt_layer_size = layer_size
            opt_alpha = alpha_val
            max_score = validation_score

  if(Classify_or_regress):
    return opt_layer_size, opt_alpha, max_accuracy, various_architecture_performances
  else:
    return opt_layer_size, opt_alpha, max_score, various_architecture_performances

def train_validate_test_classifier(data, classifier):
  x_training, y_training = data[0], data[1]
  x_validate, y_validate = data[2], data[3]
  classifier.fit(x_training, y_training)
  return classifier.score(x_validate, y_validate)

def get_testing_error_classifier(data, classifier):
  x_training, y_training = data[0], data[1]
  x_testing, y_testing = data[4], data[5]
  classifier.fit(x_training, y_training)
  return classifier.score(x_testing, y_testing)

def calculate_average_point_differential(data):
  x_testing, y_testing = data[4], data[5]

  total_error = 0
  for i in range(len(y_testing)):
    total_error += abs(y_testing[i])
  
  avg_error = float(total_error) / len(y_testing)

  return avg_error

def calculate_testing_score_nn_regressor(data, score_predictor):
  x_training, y_training = data[0], data[1]
  x_validate, y_validate = data[2], data[3]
  x_testing, y_testing = data[4], data[5]

  total_error = 0
  for i in range(len(x_testing)):
    total_error += abs(y_testing[i] - score_predictor.predict(x_testing[i]))
  
  avg_error = float(total_error) / len(x_testing)

  return avg_error

def calculate_validation_score_nn_regressor(data, score_predictor):
  x_training, y_training = data[0], data[1]
  x_validate, y_validate = data[2], data[3]
  x_testing, y_testing = data[4], data[5]

  x_training = preprocessing.normalize(x_training,  axis = 0)
  
  x_validate = preprocessing.normalize(x_validate,  axis = 0)
  


  score_predictor.fit(x_training, y_training)

  total_error = 0
  for i in range(len(x_validate)):
    total_error += abs(y_validate[i] - score_predictor.predict(x_validate[i]))
  
  avg_error = float(total_error) / len(x_validate)

  return avg_error

if __name__ == '__main__':
  filename = "data/2008-2009 Game Log.csv" # "data/game_log_06_07.csv"
  
  num_skip = 320 # stabilized by ~ each team's 20th game
  num_training = 300
  num_validate = 300
  num_testing = 300

  # Classification y in {0,1} indicates whether or not the home team won (1 = home team won)

  data_win_lose = get_data(filename, num_skip, num_training, num_validate, num_testing, get_win_lose)
  data_spreads = get_data(filename, num_skip, num_training, num_validate, num_testing, get_spread)

  #print calculate_average_point_differential(data_spreads)


  # clf = neural_network.MLPClassifier(solver = 'lbfgs', alpha = 0.1, hidden_layer_sizes = (10,))
  # clf.fit(data_win_lose[0], data_win_lose[1])
  # print clf.score(data_win_lose[4], data_win_lose[5])
  best_architecture = find_best_architecture(data_spreads, False)
  pprint.pprint(best_architecture[3])
  # best_classifier = neural_network.MLPRegressor(hidden_layer_sizes=best_architecture[0], solver="lbfgs", alpha=best_architecture[1])
  # print get_testing_error(data_win_lose, best_classifier)

  # score_predictor = neural_network.MLPRegressor( solver = 'lbfgs', alpha = best_architecture[1], hidden_layer_sizes = best_architecture[0])
  # score_predictor.fit(data_spreads[0], data_spreads[1])
  # print calculate_testing_score_nn_regressor(data_spreads, score_predictor)
