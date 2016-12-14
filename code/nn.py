from sklearn     import neural_network
from sklearn     import preprocessing
from regression  import get_data, train_validate_test_classifier
import csv
import pprint
import json


layer_sizes = [ (5,), (10,), (5,5), (10,10), (10, 20, 10), (5, 10, 10, 5) ]
alphas = [ 0, 0.0001, 0.01, 0.1, 1 ] # degree of regularizations
win_rate_indices = [ 10, 14, 23, 27 ]


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

def extract_from_file(filename, num_skip, y_fn):
  index = 0
  x_list, y_list = [], []

  with open(filename, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:

      if index == 0:
        index += 1
        continue

      x = [ float(row[i]) for i in win_rate_indices ]
      #x = [ float(value) for value in row[5:] ]
      away_score = float(row[3])
      home_score = float(row[4])
      y = y_fn(away_score, home_score)

      if index > num_skip:
        x_list.append(x)
        y_list.append(y)

      index += 1

  return x_list, y_list


def get_file_data(filename_training, filename_validation, filename_testing, num_skip, y_fn):
  index = 0

  x_training, y_training = extract_from_file(filename_training, num_skip, y_fn)

  x_validate, y_validate = extract_from_file(filename_validation, num_skip, y_fn)
  x_testing, y_testing = extract_from_file(filename_testing, num_skip, y_fn)

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


def would_beat(predicted_spread, other_spread, actual_spread):
  if predicted_spread < other_spread and actual_spread < other_spread:
    return 1

  if predicted_spread > other_spread and actual_spread > other_spread:
    return 1

  return 0

def get_entire_row(filename, num_skip, y_fn=get_spread):
  index = 0
  x_list, y_list = [], []

  with open(filename, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:

      if index == 0:
        index += 1
        continue

      away_score = float(row[3])
      home_score = float(row[4])
      y = y_fn(away_score, home_score)

      if index > num_skip:
        x_list.append(row)
        y_list.append(y)

      index += 1

  return x_list, y_list

def get_row_from_spreads(filename, date, away_team, home_team):
  with open(filename, 'r') as f:
    reader = csv.DictReader(f, fieldnames=spreads_fieldnames)

    for row in reader:
      if row['date'] == date and row['team1'] == away_team and row['team2'] == home_team:
        return row

  # print "date: ", date, " away_team: ", away_team, " home_team: ", home_team

def parse_spread(unparsed):
  try:
    return float(unparsed[1].split(" ")[0])
  except Exception as e:
    print e


def find_best_architecture_max_beating_spreads(data, x_validation_row, y_validation_row):
  max_accuracy = 0
  opt_layer_size = ()
  opt_alpha = 0
  classifiers = {}

  x_training, y_training = data[0], data[1]
  x_validate, y_validate = data[2], data[3]

  print "Training classifiers..."
  for layer_size in layer_sizes:
    for alpha_val in alphas:
      classifier = neural_network.MLPClassifier(hidden_layer_sizes=layer_size, solver="lbfgs", alpha=alpha_val)
      classifier.fit(x_training, y_training)
      classifiers[(layer_size, alpha_val)] = classifier

  print "Validating classifiers..."
  for key in classifiers.keys():
    layer_size = key[0]
    alpha_val = key[1]
    classifier = classifiers[key]

    print "layer_size: ", layer_size, " alpha_val: ", alpha_val

    num_win = 0
    num_total = 0

    for xi_validation_row, actual_spread in zip(x_validation_row, y_validation_row):
      input_data = [ float(val) for val in xi_validation_row[5:] ]

      predicted_spread = classifier.predict(input_data)

      date = xi_validation_row[0]
      away_team = xi_validation_row[1]
      home_team = xi_validation_row[2]
      away_score = xi_validation_row[3]
      home_score = xi_validation_row[4]

      spreads_row = get_row_from_spreads("data/spreads.csv", date, away_team, home_team)
      if spreads_row is None:
        continue

      spreads = json.loads(spreads_row['spreads_json'])
      if "238" in spreads.keys():
        unparsed_spread = spreads["238"]
      else:
        continue

      spread = parse_spread(unparsed_spread)
      if spread is None:
        continue

      would_win = would_beat(predicted_spread, spread, actual_spread)
      num_win += would_win
      num_total += 1

    ratio = float(num_win) / num_total

    if ratio > max_accuracy:
      max_accuracy = ratio
      opt_layer_size = layer_size
      opt_alpha = alpha_val

  return opt_layer_size, opt_alpha


if __name__ == '__main__':
  filename = "data/2008-2009 Game Log.csv" # "data/game_log_06_07.csv"

  filename_training = "data/2008-2009 Game Log.csv"
  filename_validation = "data/2009-2010 Game Log.csv"
  filename_testing = "data/2010-2011 Game Log.csv"

  num_skip = 320 # stabilized by ~ each team's 20th game
  num_training = 300
  num_validate = 300
  num_testing = 300

  # Classification y in {0,1} indicates whether or not the home team won (1 = home team won)

  # data_win_lose = get_data(filename, num_skip, num_training, num_validate, num_testing, get_win_lose)
  # data_spreads = get_data(filename, num_skip, num_training, num_validate, num_testing, get_spread)

  # data_win_lose_2 = get_file_data(filename_training, filename_validation, filename_testing, num_skip, get_win_lose)
  file_data_spreads = get_file_data(filename_training, filename_validation, filename_testing, num_skip, get_spread)

  x_validation_row, y_validation_row = get_entire_row(filename_validation, num_skip, get_spread)

  data_win_lose_2 = get_file_data(filename_training, filename_validation, filename_testing, num_skip, get_win_lose)
  data_spreads_2 = get_file_data(filename_training, filename_validation, filename_testing, num_skip, get_spread)
  #print calculate_average_point_differential(data_spreads)


  # clf = neural_network.MLPClassifier(solver = 'lbfgs', alpha = 0.1, hidden_layer_sizes = (10,))
  # clf.fit(data_win_lose[0], data_win_lose[1])
  # print clf.score(data_win_lose[4], data_win_lose[5])

  best_architecture = find_best_architecture(data_win_lose_2, True)
  pprint.pprint(best_architecture[3])
  best_predictor = neural_network.MLPClassifier(hidden_layer_sizes=best_architecture[0], solver="lbfgs", alpha=best_architecture[1])
  #best_predictor.fit(data_spreads_2[0], data_spreads_2[1])
  print get_testing_error_classifier(data_win_lose_2, best_predictor)



  # score_predictor = neural_network.MLPRegressor( solver = 'lbfgs', alpha = best_architecture[1], hidden_layer_sizes = best_architecture[0])
  # score_predictor.fit(data_spreads[0], data_spreads[1])
  # print calculate_testing_score_nn_regressor(data_spreads, score_predictor)


####
## architecture: (10, 20, 10)
## alpha: 0.0001

