from regression      import get_spread, get_file_data
from sklearn         import linear_model
import csv
import json


win_rate_indices = [ 10, 14, 23, 27 ]
spreads_fieldnames = [ 'date', 'team1', 'team2', 'team1_score', 'team2_score', 'spreads_json' ]



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

def extract_from_file(filename, num_skip, y_fn=get_spread):
  index = 0
  x_list, y_list = [], []

  with open(filename, 'rU') as f:
    reader = csv.reader(f)
    for row in reader:

      if index == 0:
        index += 1
        continue

      x = [ float(row[i]) for i in win_rate_indices ]
      # x = [ float(value) for value in row[5:] ]
      away_score = float(row[3])
      home_score = float(row[4])
      y = y_fn(away_score, home_score)

      if index > num_skip:
        x_list.append(x)
        y_list.append(y)

      index += 1

  return x_list, y_list


def get_file_data(filename_training, filename_validation, filename_testing, num_skip, y_fn=get_spread):
  x_training, y_training = extract_from_file(filename_training, num_skip, y_fn)
  x_validate, y_validate = extract_from_file(filename_validation, num_skip, y_fn)
  x_testing, y_testing = extract_from_file(filename_testing, num_skip, y_fn)

  return [x_training, y_training, x_validate, y_validate, x_testing, y_testing]


def get_row_from_sreads(filename, date, away_team, home_team):
  with open(filename, 'r') as f:
    reader = csv.DictReader(f, fieldnames=spreads_fieldnames)

    for row in reader:
      if row['date'] == date and row['team1'] == away_team and row['team2'] == home_team:
        return row

  print "date: ", date, " away_team: ", away_team, " home_team: ", home_team

def parse_spread(unparsed):
  try:
    return float(unparsed[1].split(" ")[0])
  except Exception as e:
    print e


# win if -8 and other spread -5 and actual spread < -5
# win if +8 and other spread +5 and actual spread > +5
def get_spreads_winlose(predicted_spread, actual_spread, other_spread):
  # if predicted_spread < other_spread and actual_spread < other_spread:
  #   return 1

  # if predicted_spread > other_spread and actual_spread > other_spread:
  #   return 1

  # return 0

  if predicted_spread * other_spread > 0:
    return 1

  return 0


if __name__ == '__main__':
  filename = "data/2008-2009 Game Log.csv" # "data/game_log_06_07.csv"

  filename_training = "data/2008-2009 Game Log.csv"
  filename_validation = "data/2009-2010 Game Log.csv"
  filename_testing = "data/2010-2011 Game Log.csv"

  filename_spreads  = "data/spreads.csv"

  num_skip = 320 # stabilized by ~ each team's 20th game

  # Classification y in {0,1} indicates whether or not the home team won (1 = home team won)

  file_data_spreads = get_file_data(filename_training, filename_validation, filename_testing, num_skip)

  x_training, y_training = file_data_spreads[0], file_data_spreads[1]
  x_validate, y_validate = file_data_spreads[2], file_data_spreads[3]
  x_testing, y_testing = file_data_spreads[4], file_data_spreads[5]

  x_testing_row, y_testing_row = get_entire_row(filename_testing, num_skip, get_spread)

  linear_classifier = linear_model.LinearRegression()
  linear_classifier.fit(x_training, y_training)

  spreads_error_dict = {}
  spreads_winlose_dict = {}

  # Spread = away score - home score (always do from perspective of home team)
  for xi_testing_row, actual_spread in zip(x_testing_row, y_testing_row):
    input_data = [ float(xi_testing_row[i]) for i in win_rate_indices ]

    predicted_spread = linear_classifier.predict(input_data)

    date = xi_testing_row[0]
    away_team = xi_testing_row[1]
    home_team = xi_testing_row[2]
    away_score = xi_testing_row[3]
    home_score = xi_testing_row[4]

    spreads_row = get_row_from_sreads(filename_spreads, date, away_team, home_team)
    if spreads_row is None:
      continue

    spreads = json.loads(spreads_row['spreads_json'])

    for betting_authority in spreads:

      if betting_authority not in spreads_error_dict:
        spreads_error_dict[betting_authority] = (0, 0)

      if betting_authority not in spreads_winlose_dict:
        spreads_winlose_dict[betting_authority] = (0, 0)

      spread = parse_spread(spreads[betting_authority])
      if spread is None:
        continue

      spread_error = abs(spread - actual_spread)

      # sum_error = spreads_error_dict[betting_authority][0] + spread_error
      # num = spreads_error_dict[betting_authority][1] + 1
      # spreads_error_dict[betting_authority] = (sum_error, num)

      winlose = get_spreads_winlose(predicted_spread, actual_spread, spread)
      num_win = spreads_winlose_dict[betting_authority][0] + winlose
      num_total = spreads_winlose_dict[betting_authority][1] + 1
      spreads_winlose_dict[betting_authority] = (num_win, num_total)


  print spreads_winlose_dict
  # print spreads_error_dict

  winlose_accuracies = { betting_authority : float(spreads_winlose_dict[betting_authority][0]) / spreads_winlose_dict[betting_authority][1] for betting_authority in spreads_winlose_dict }
  print winlose_accuracies

  # spreads_errors = { betting_authority : float(spreads_error_dict[betting_authority][0]) / spreads_error_dict[betting_authority][1] for betting_authority in spreads_error_dict }
  # print spreads_errors






