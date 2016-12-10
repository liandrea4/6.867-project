from sklearn            import linear_model


def get_accuracy(x, y, classifier):
  num_correct = 0

  for x_i, y_i in zip(x, y):
    prediction = classifier.predict(x_i)

    if prediction == y_i:
      num_correct += 1

  return float(num_correct) / len(x)


# TODO: write get_data
def get_data():
  x_training, y_training = get_training_data()
  x_validate, y_validate = get_validate_data()
  x_testing, y_testing = get_testing_data()
  data = [x_training, y_training, x_validate, y_validate, x_testing, y_testing]

  return data

alphas = [ 0, 0.01, 0.1, 1, 10 ] # degree of regularizations


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


data = get_data()

linear_classifier = linear_model.LinearRegression()
logistic_classifier = linear_model.LogisticRegression()
lasso_classifier = linear_model.Lasso()


print "Linear classifier"
train_validate_test_classifier(data, linear_classifier)
print "Logistic classifier"
train_validate_test_classifier(data, logistic_classifier)
print "Lasso classifier"
train_validate_test_classifier(data, lasso_classifier)

