import requests
import urllib2
import csv
import json
import sys

LINE_ID = "FINAL<strong" # xmlns:odds=\"http://odds.sbrforum.com\""

SCORE_START_STRING = "class=\"current-score\">"
PRE_TEAM_SCORE_STRING = "class=\"current-score\">"
POST_TEAM_SCORE_STRING = "</span>"

NAME_START_STRING = "<span class=\"team-name\""
PRE_TEAM_NAME_STRING = "/\">"
POST_TEAM_NAME_STRING = "</a>"

PREDICTOR_START_STRING = "\" rel=\""
PREDICTOR_END_STRING = "\""

SPREAD_START_STRING = "rel=\"\"><b>"
SPREAD_END_STRING = "</b></div>"


DAYS_IN_MONTH = {
  "01" : 31,
  "02" : 28,
  "03" : 31,
  "04" : 30,
  "05" : 31,
  "06" : 30,
  "07" : 31,
  "08" : 31,
  "09" : 30,
  "10" : 31,
  "11" : 30,
  "12" : 31
}

LEAP_YEAR_DAYS = "29"
LEAP_YEAR_FACTOR = 4

YEARS = [ "2006", "2007", "2008", "2009", "2010", "2011", "2012" ]
MONTHS = [ "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12" ]
PREPROCESSED_DAYS = [ "01", "02", "03", "04", "05", "06", "07", "08", "09"]

BASE_URL = "http://www.sportsbookreview.com/betting-odds/nba-basketball/?date="

def find_team_score(line):
  start_index = line.find(PRE_TEAM_SCORE_STRING)
  subline = line[start_index:]
  end_index = subline.find(POST_TEAM_SCORE_STRING)

  team_score = subline[len(PRE_TEAM_SCORE_STRING):end_index]

  line = subline[end_index + len(POST_TEAM_SCORE_STRING):]

  return team_score, line


def find_team_name(line):
  end_index = line.find(POST_TEAM_NAME_STRING)
  subline = line[:end_index]
  team_start_index = subline.rfind(PRE_TEAM_NAME_STRING)

  team_name = subline[team_start_index + len(PRE_TEAM_NAME_STRING):end_index]

  line = line[end_index + len(POST_TEAM_NAME_STRING):]

  return team_name, line


def find_predictor(line):
  subline = line[len(PREDICTOR_START_STRING):]
  end_index = subline.find(PREDICTOR_END_STRING)
  predictor = subline[:end_index]

  return predictor, subline


def find_spread(line):
  start_index = line.find(SPREAD_START_STRING)
  subline = line[start_index:]
  end_index = subline.find(SPREAD_END_STRING)

  spread = subline[len(SPREAD_START_STRING):end_index]
  line = subline[end_index:]

  if "\xc2\xbd\xc2\xa0" in spread:
    end_index = spread.index("\xc2")
    start_index = spread.index("\xa0") + 1
    adjusted_spread = spread[:end_index] + ".5 " + spread[start_index:]
  else:
    end_index = spread.index("\xc2")
    start_index = spread.index("\xa0") + 1

    adjusted_spread = spread[:end_index] + " " + spread[start_index:]

  return adjusted_spread, line


def scrape_line(filename, fieldnames, line, date):
  start_index = line.index(SCORE_START_STRING)
  line = line[start_index:]

  team1_score, line = find_team_score(line)
  team2_score, line = find_team_score(line)

  start_index = line.index(NAME_START_STRING)
  line = line[start_index:]

  team_name_1, line = find_team_name(line)
  team_name_2, line = find_team_name(line)

  spreads = {}

  start_index = line.find(PREDICTOR_START_STRING)
  line = line[start_index:]
  while start_index > 0:
    try:
      predictor, line = find_predictor(line)
      team_1_spread, line = find_spread(line)
      team_2_spread, line = find_spread(line)
    except ValueError:
      break

    spreads[predictor] = (team_1_spread, team_2_spread)

    start_index = line.find(PREDICTOR_START_STRING)
    line = line[start_index:]

  spreads_json = json.dumps(spreads)
  row_dict = {
    'date': date,
    'team1': team_name_1,
    'team2': team_name_2,
    'team1_score': team1_score,
    'team2_score': team2_score,
    'spreads_json': spreads_json
  }

  with open(filename, "a") as f:
    writer = csv.DictWriter(f, fieldnames, lineterminator='\n')
    writer.writerow(row_dict)


def get_relevant_url_lines(date):
  relevant_lines = set([])

  url = BASE_URL + date
  page = requests.get(url)
  page_content = page.content
  splitted = page_content.split("\n")
  i = 0

  for line in splitted:
    if LINE_ID in line:
      relevant_lines.add(line)

  return relevant_lines

def scrape_date(filename, fieldnames, date):
  relevant_lines = get_relevant_url_lines(date)

  for line in relevant_lines:
    scrape_line(filename, fieldnames, line, date)

def scrape_multiple_dates(filename, fieldnames, start_year, end_year):
  for year in YEARS:
    if int(year) < start_year or int(year) > end_year:
      continue

    for month in MONTHS:
      days = PREPROCESSED_DAYS + [ str(num) for num in range(10, DAYS_IN_MONTH[month] + 1)]
      if month == '02' and int(year) % LEAP_YEAR_FACTOR == 0:
        days += [LEAP_YEAR_DAYS]

      for day in days:
        date = year + month + day
        print "date: ", date

        scrape_date(filename, fieldnames, date)


if __name__ == '__main__':
  try:
    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
  except Exception:
    start_year = 2011
    end_year = 2012

  filename = "spreads.csv"
  fieldnames = [ 'date', 'team1', 'team2', 'team1_score', 'team2_score', 'spreads_json' ]
  with open(filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames, lineterminator='\n')
    writer.writeheader()

  scraped_dates = scrape_multiple_dates(filename, fieldnames, start_year, end_year)

  # date = "20161109"
  # scrape_date(filename, fieldnames, date)





