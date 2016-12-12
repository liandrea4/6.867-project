from sklearn            import linear_model
import csv

def get_accuracy(x, y, classifier):
  num_correct = 0

  for x_i, y_i in zip(x, y):
    prediction = classifier.predict(x_i) > 0.5

    if prediction == y_i:
      num_correct += 1

  if len(x) == 0:
    return "No data samples"

  return float(num_correct) / len(x)


def get_data(filename, num_skip, num_training, num_validate, num_testing):
  index = 0
  x_training, y_training, x_validate, y_validate, x_testing, y_testing = [], [], [], [], [], []

  with open(filename, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:

      if index == 0:
        index += 1
        continue

      x = [ float(value) for value in row[5:] ]
      away_score = row[3]
      home_score = row[4]
      y = home_score > away_score

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


def train_validate_test_classifier(data, classifier):
  x_training, y_training = data[0], data[1]
  x_validate, y_validate = data[2], data[3]
  x_testing, y_testing = data[4], data[5]

  classifier.fit(x_training, y_training)
  print "coeffs: ", classifier.coef_

  validation_accuracy = get_accuracy(x_validate, y_validate, classifier)
  print "validation_accuracy: ", validation_accuracy
  # testing_accuracy = get_accuracy(x_testing, y_testing, classifier)
  # print "testing_accuracy: ", testing_accuracy


filename = "data/game_log_06_07.csv"
alphas = [ 0, 0.01, 0.1, 1, 10 ] # degree of regularizations

num_skip = 320 # stabilized by ~ each team's 20th game
num_training = 300
num_validate = 300
num_testing = 300

# Classification y in {0,1} indicates whether or not the home team won (1 = home team won)


data = get_data(filename, num_skip, num_training, num_validate, num_testing)

linear_classifier = linear_model.LinearRegression()
logistic_classifier = linear_model.LogisticRegression()
lasso_classifier = linear_model.Lasso()

print "Linear classifier"
train_validate_test_classifier(data, linear_classifier)
# print "Logistic classifier"
# train_validate_test_classifier(data, logistic_classifier)
# print "Lasso classifier"
# train_validate_test_classifier(data, lasso_classifier)

