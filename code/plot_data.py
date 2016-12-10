import matplotlib.pyplot     as plt
import csv

prefixes = [ "Away Team", "Home Team" ]

teams_elo = {}
teams_points_scored = {}
teams_points_allowed = {}
teams_games_played = {}
teams_games_won = {}
teams_win_rate = {}

def add_to_dicts(team, elo, points_scored, points_allowed, games_played, games_won, win_rate):
  if team not in teams_elo:
    teams_elo[team] = []
    teams_points_scored[team] = []
    teams_points_allowed[team] = []
    teams_games_played[team] = []
    teams_games_won[team] = []
    teams_win_rate[team] = []
  teams_elo[team].append(elo)
  teams_points_scored[team].append(points_scored)
  teams_points_allowed[team].append(points_allowed)
  teams_games_played[team].append(games_played)
  teams_games_won[team].append(games_won)
  teams_win_rate[team].append(win_rate)


def plot_data(team_dict):
  plt.figure()
  for team in team_dict:
    plt.plot(team_dict[team], label=team)

  plt.show()

def read_data(filename):
  with open(filename, 'r') as f:
    reader = csv.reader(f)
    fieldnames = next(reader)

    reader = csv.DictReader(f, fieldnames=fieldnames)
    for row in reader:

      for prefix in prefixes:
        team = row[prefix]
        elo = float(row[prefix + " ELO"])
        points_scored = float(row[prefix + " avg. points scored"])
        points_allowed = float(row[prefix + "avg. points allowed"])
        games_played = float(row[prefix + " games played"])
        games_won = float(row[prefix + " games won"])
        win_rate = float(row[prefix + " win rate"])

        add_to_dicts(team, elo, points_scored, points_allowed, games_played, games_won, win_rate)


if __name__ == '__main__':
  filename = "data/game_log_06_07.csv"

  read_data(filename)
  plot_data(teams_elo)


