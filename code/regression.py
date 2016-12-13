import matplotlib.pyplot as plt
from sklearn            import linear_model
import csv

win_rate_indices = [ 10, 14, 23, 27 ]

feature_names = [
    'Away Elo',
    'Points scored',
    'Points allowed',
    'Games played',
    'Games won',
    'Win rate',
    'Home court points scored',
    'Home court points allowed',
    'Home court games played',
    'Home court win rate',
    'Possessions',
    'Defensive rebounds',
    'Offensive rebounds',
    'Home Elo',
    'Points scored',
    'Points allowed',
    'Games played',
    'Games won',
    'Win rate',
    'Home court points scored',
    'Home court points allowed',
    'Home court games played',
    'Home court win rate',
    'Possessions',
    'Defensive rebounds',
    'Offensive rebounds'
  ]


def get_accuracy(x, y, classifier):
  num_correct = 0

  for x_i, y_i in zip(x, y):
    prediction = classifier.predict(x_i) > 0.5

    if prediction == y_i:
      num_correct += 1

  if len(x) == 0:
    return "No data samples"

  return float(num_correct) / len(x)


def get_average_error(x, y, classifier):
  error_sum = 0.

  for x_i, y_i in zip(x, y):
    predicted_spread = classifier.predict(x_i)

    error = abs(predicted_spread - y_i)
    error_sum += error

  if len(x) == 0:
    return "No data samples"

  return error_sum / len(x)


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
      # x = [ float(row[i]) for i in win_rate_indices ]
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

      # x = [ float(row[i]) for i in win_rate_indices ]
      x = [ float(value) for value in row[5:] ]
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


def train_validate_test_classifier(data, classifier, metric_fn):
  x_training, y_training = data[0], data[1]
  x_validate, y_validate = data[2], data[3]
  x_testing, y_testing = data[4], data[5]

  classifier.fit(x_training, y_training)

  # training = metric_fn(x_training, y_training, classifier)
  # print "training: ", training
  # validation = metric_fn(x_validate, y_validate, classifier)
  # print "validation: ", validation
  testing = metric_fn(x_testing, y_testing, classifier)
  print "testing: ", testing


if __name__ == '__main__':
  filename = "data/2008-2009 Game Log.csv" # "data/game_log_06_07.csv"

  filename_training = "data/2008-2009 Game Log.csv"
  filename_validation = "data/2009-2010 Game Log.csv"
  filename_testing = "data/2010-2011 Game Log.csv"

  alphas = [ 0, 0.01, 0.1, 1, 10 ] # degree of regularizations

  num_skip = 320 # stabilized by ~ each team's 20th game
  num_training = 300
  num_validate = 300
  num_testing = 300

  # Classification y in {0,1} indicates whether or not the home team won (1 = home team won)

  data_win_lose = get_data(filename, num_skip, num_training, num_validate, num_testing, get_win_lose)
  data_spreads = get_data(filename, num_skip, num_training, num_validate, num_testing, get_spread)

  # file_data_win_lose = get_file_data(filename_training, filename_validation, filename_testing, num_skip, get_win_lose)
  # file_data_spreads = get_file_data(filename_training, filename_validation, filename_testing, num_skip, get_spread)

  linear_classifier = linear_model.LinearRegression()
  logistic_classifier = linear_model.LogisticRegression()
  lasso_classifier = linear_model.Lasso()

  print "Linear classifier"
  train_validate_test_classifier(data_win_lose, linear_classifier, get_accuracy)
  # print "Logistic classifier"
  # train_validate_test_classifier(data_win_lose, logistic_classifier, get_accuracy)
  # print "Lasso classifier"
  # train_validate_test_classifier(data_win_lose, lasso_classifier, get_accuracy)


  # print "Linear classifier"
  # train_validate_test_classifier(file_data_win_lose, linear_classifier, get_accuracy)

  print "coeffs: ", linear_classifier.coef_

  # index = range(len(feature_names))
  # plt.figure()
  # plt.bar(index, linear_classifier.coef_)
  # plt.xticks(index, feature_names, rotation=90)
  # plt.title("3 seasons, win/lose prediction")
  # plt.ylabel("Feature weights")

  # fig = plt.gcf()
  # fig.subplots_adjust(bottom=0.4)
  # plt.show()


# Linear classifier with all basic features on 06-07 (win/lose)
# training: 0.643
# validation: 0.653
# testing: 0.633

# Logistic classifier with all basic features on 06-07 (win/lose)
# training: 0.643
# validation: 0.62
# testing: 0.633

# Linear classifier with all basic features on 06-07 (spreads)
# training: 9.31
# validation: 10.03
# testing: 10.07

# Linear classifier with more advanced features on 08-09 (win/lose)
# training: 0.723
# validation: 0.667
# testing: 0.669

# Linear classifier with more advanced features on 08-09 (spreads)
# training: 8.53
# validation: 10.722
# testing: 13.596

