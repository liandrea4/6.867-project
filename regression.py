from sklearn            import linear_model
import load_data


def get_accuracy(x, y, classifier):
  num_correct = 0

  for x_i, y_i in zip(x, y):
    prediction = classifier.predict(x_i)

    if prediction == y_i:
      num_correct += 1

  if len(x) == 0:
    return "No data samples"

  return float(num_correct) / len(x)


def get_data(filename, num_training, num_validate, num_testing):
  index = 0
  x_training, y_training, x_validate, y_validate, x_testing, y_testing = [], [], [], [], [], []

  with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      x = row[:-1]
      y = row[-1]

      if index > 0 and index <= num_training:
        x_training.append(x)
        y_training.append(y)

      elif index > num_training and index <= num_training + num_validate:
        x_validate.append(x)
        y_validate.append(y)

      elif index > num_training + num_validate and index <= num_training + num_validate + num_testing:
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
  testing_accuracy = get_accuracy(x_testing, y_testing, classifier)
  print "testing_accuracy: ", testing_accuracy
  return testing_accuracy


filename = ""
alphas = [ 0, 0.01, 0.1, 1, 10 ] # degree of regularizations

data = get_data(filename)

linear_classifier = linear_model.LinearRegression()
logistic_classifier = linear_model.LogisticRegression()
lasso_classifier = linear_model.Lasso()


print "Linear classifier"
train_validate_test_classifier(data, linear_classifier)
print "Logistic classifier"
train_validate_test_classifier(data, logistic_classifier)
print "Lasso classifier"
train_validate_test_classifier(data, lasso_classifier)

