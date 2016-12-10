import math
import csv




def printplus(obj):
    """
    Pretty-prints the object passed in.

    """
    # Dict
    if isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            print u'{0}: {1}'.format(k, v)

    # List or tuple            
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for x in obj:
            print x

    # Other
    else:
        print obj


def generateNewELOs(Team1ELO, Team2ELO, result):
	newTeam1ELO = Team1ELO
	newTeam2ELO = Team2ELO
	if(result): #if Team 1 wins
		pointsTransferred = 5.0 + max(0.0, 0.1*(Team2ELO - Team1ELO))
		newTeam1ELO += pointsTransferred
		newTeam2ELO -= pointsTransferred
	else:
		pointsTransferred = 5.0 + max(0.0, 0.1*(Team1ELO - Team2ELO))
		newTeam1ELO -= pointsTransferred
		newTeam2ELO += pointsTransferred

	return (newTeam1ELO, newTeam2ELO)

def crawlSeason(matchupsData, writefile, numberOfMatches):

	with open(matchupsData, 'r') as f:
		reader = csv.reader(f, delimiter = "\t")
		reader.next() #this is to skip the headings
		count = 0

		with open(writefile, 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=',',)
	                           
			writer.writerow(['Date', 'Away Team', 'Home Team', 'Away Final Score', 'Home Final Score', 'Away Team ELO', 'Away Team avg. points scored', 'Away Team avg. points allowed',
							'Away Team games played', 'Away Team games won', 'Away Team win rate',
							'Away Team home court points scored', 'Away Team home court points allowed',
							'Away Team home court games played', 'Away Team home court win rate',
							'Home Team home court points scored', 'Home Team home court points allowed',
							'Home Team home court games played', 'Home Team home court win rate'
							'Home Team ELO', 'Home Team avg. points scored', 'Home Team avg. points allowed',
							'Home Team games played', 'Home Team games won', 'Home Team win rate'])
			while (count < numberOfMatches):

				try:
					currentLine = reader.next()
				except StopIteration:
					break
				
				currentGameID = currentLine[0]
				currentDate = currentGameID[0:8]
				currentHomeTeam = currentGameID[-3:len(currentGameID)]
				currentAwayTeam = currentGameID[-6:-3]
				

				
				while(True):
					try:
						nextLine = reader.next()
					except StopIteration:
						break
					
					if(currentGameID == nextLine[0]):
						currentLine = nextLine
					else:
						break
					

				# try:	
				# 	nextLine = reader.next()
				# except StopIteration:
				# 	break
				# 	endTime = nextLine[2]

				# if(nextLine[1] == '00:00:00'): # This section is to handle games that went into overtime
				# 	while(endTime != '-00:05:00'):
				# 		currentLine = reader.next()
				# 		endTime = currentLine[2]

				# 	try:	
				# 		nextNextLine = reader.next()
				# 	except StopIteration:
				# 		break
				# 	endTime =nextNextLine[2]

				# 	if(nextNextLine[1] == '-00:05:00'): # This section is to handle games that went into overtime
				# 		while(endTime != '-00:10:00'):
				# 			currentLine = reader.next()
				# 			endTime = currentLine[2]


				HomeTeamFinalScore = int(currentLine[27])
				AwayTeamFinalScore = int(currentLine[28])

				writer.writerow([currentDate, AbbreviationDictionary[currentAwayTeam], AbbreviationDictionary[currentHomeTeam],
								AwayTeamFinalScore, HomeTeamFinalScore,
								MasterTeamDictionary[currentAwayTeam][0], MasterTeamDictionary[currentAwayTeam][1],
								MasterTeamDictionary[currentAwayTeam][2], MasterTeamDictionary[currentAwayTeam][3],
								MasterTeamDictionary[currentAwayTeam][4], MasterTeamDictionary[currentAwayTeam][5],
								MasterTeamDictionary[currentAwayTeam][6], MasterTeamDictionary[currentAwayTeam][7],
								MasterTeamDictionary[currentAwayTeam][8], MasterTeamDictionary[currentAwayTeam][10],
								MasterTeamDictionary[currentHomeTeam][0], MasterTeamDictionary[currentHomeTeam][1],
								MasterTeamDictionary[currentHomeTeam][2], MasterTeamDictionary[currentHomeTeam][3],
								MasterTeamDictionary[currentHomeTeam][4], MasterTeamDictionary[currentHomeTeam][5],
								MasterTeamDictionary[currentHomeTeam][6], MasterTeamDictionary[currentHomeTeam][7],
								MasterTeamDictionary[currentHomeTeam][8], MasterTeamDictionary[currentHomeTeam][10]])
				result = False
				if(HomeTeamFinalScore > AwayTeamFinalScore):
					result = True
					MasterTeamDictionary[currentHomeTeam][4] += 1
					MasterTeamDictionary[currentHomeTeam][9] += 1
				else:
					MasterTeamDictionary[currentAwayTeam][4] += 1

				HomeTeamOldELO = MasterTeamDictionary[currentHomeTeam][0]
				AwayTeamOldELO = MasterTeamDictionary[currentAwayTeam][0]
				newELOs = generateNewELOs(HomeTeamOldELO, AwayTeamOldELO, result)

				
				
			

				MasterTeamDictionary[currentHomeTeam][0] = newELOs[0]
				MasterTeamDictionary[currentAwayTeam][0] = newELOs[1]

				newHomeTeamAvgPointsScored = (MasterTeamDictionary[currentHomeTeam][1] * MasterTeamDictionary[currentHomeTeam][3] + HomeTeamFinalScore)/(MasterTeamDictionary[currentHomeTeam][3]+1)
				newAwayTeamAvgPointsAllowed = (MasterTeamDictionary[currentAwayTeam][2] * MasterTeamDictionary[currentAwayTeam][3] + HomeTeamFinalScore)/(MasterTeamDictionary[currentAwayTeam][3]+1)

				newAwayTeamAvgPointsScored = (MasterTeamDictionary[currentAwayTeam][1] * MasterTeamDictionary[currentAwayTeam][3] + AwayTeamFinalScore)/(MasterTeamDictionary[currentAwayTeam][3]+1)
				newHomeTeamAvgPointsAllowed = (MasterTeamDictionary[currentHomeTeam][2] * MasterTeamDictionary[currentHomeTeam][3] + AwayTeamFinalScore)/(MasterTeamDictionary[currentHomeTeam][3]+1)

				MasterTeamDictionary[currentHomeTeam][1] = newHomeTeamAvgPointsScored
				MasterTeamDictionary[currentHomeTeam][2] = newHomeTeamAvgPointsAllowed

				MasterTeamDictionary[currentAwayTeam][1] = newAwayTeamAvgPointsScored
				MasterTeamDictionary[currentAwayTeam][2] = newAwayTeamAvgPointsAllowed

				MasterTeamDictionary[currentHomeTeam][6] = (MasterTeamDictionary[currentHomeTeam][6] * MasterTeamDictionary[currentHomeTeam][8] + HomeTeamFinalScore)/ (MasterTeamDictionary[currentHomeTeam][8]+1)
				MasterTeamDictionary[currentHomeTeam][7] = (MasterTeamDictionary[currentHomeTeam][7] * MasterTeamDictionary[currentHomeTeam][8] + AwayTeamFinalScore)/ (MasterTeamDictionary[currentHomeTeam][8]+1)
				MasterTeamDictionary[currentHomeTeam][8] +=1


				MasterTeamDictionary[currentHomeTeam][3] +=1
				MasterTeamDictionary[currentAwayTeam][3] +=1

				MasterTeamDictionary[currentHomeTeam][5] = float(MasterTeamDictionary[currentHomeTeam][4]) / float(MasterTeamDictionary[currentHomeTeam][3])
				MasterTeamDictionary[currentHomeTeam][10] = float(MasterTeamDictionary[currentHomeTeam][9]) / float(MasterTeamDictionary[currentHomeTeam][8])
				MasterTeamDictionary[currentAwayTeam][5] = float(MasterTeamDictionary[currentAwayTeam][4]) / float(MasterTeamDictionary[currentAwayTeam][3])
				count +=1
			
			
		printplus (MasterTeamDictionary)
	return

if __name__ == '__main__':
	StartingValues = [100.0,0.0,0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0] #(ELO, Avg. points scored, Avg. points allowed, games played, games won, win rate, avg. points scored in home court, avg. points allowed home court, home court games played, home court wins, home court win rate)
	MasterTeamDictionary = {
	'ATL' : list(StartingValues),
	'BOS' : list(StartingValues),
	'CHA' : list(StartingValues),
	'CHI' : list(StartingValues),
	'CLE' : list(StartingValues),
	'DAL' : list(StartingValues),
	'DEN' : list(StartingValues),
	'DET' : list(StartingValues),
	'GSW' : list(StartingValues),
	'HOU' : list(StartingValues),
	'IND' : list(StartingValues),
	'LAC' : list(StartingValues),
	'LAL' : list(StartingValues),
	'MEM' : list(StartingValues),
	'MIA' : list(StartingValues),
	'MIL' : list(StartingValues),
	'MIN' : list(StartingValues),
	'NJN' : list(StartingValues),
	'NOK' : list(StartingValues),
	'NOH' : list(StartingValues),
	'NYK' : list(StartingValues),
	'ORL' : list(StartingValues),
	'OKC' :	list(StartingValues),
	'PHI' : list(StartingValues),
	'PHX' : list(StartingValues),
	'POR' : list(StartingValues),
	'SAC' : list(StartingValues),
	'SAS' : list(StartingValues),
	'SEA' : list(StartingValues),
	'TOR' : list(StartingValues),
	'UTA' : list(StartingValues),
	'WAS' : list(StartingValues),
	}

	AbbreviationDictionary = {
	'ATL' : 'Atlanta',
	'BOS' : 'Boston',
	'CHA' : 'Charlotte',
	'CHI' : 'Chicago',
	'CLE' : 'Cleveland',
	'DAL' : 'Dallas',
	'DEN' : 'Denver',
	'DET' : 'Detroit',
	'GSW' : 'Golden State',
	'HOU' : 'Houston',
	'IND' : 'Indiana',
	'LAC' : 'L.A. Clippers',
	'LAL' : 'L.A. Lakers',
	'MEM' : 'Memphis',
	'MIA' : 'Miami',
	'MIL' : 'Milwaukee',
	'MIN' : 'Minnesota',
	'NJN' : 'New Jersey',
	'NOK' : 'New Orleans',
	'NOH' : 'New Orleans',
	'NYK' : 'New York',
	'OKC' : 'Oklahoma City',
	'ORL' : 'Orlando',
	'PHI' : 'Philadelphia',
	'PHX' : 'Phoenix',
	'POR' : 'Portland',
	'SAC' : 'Sacramento',
	'SAS' : 'San Antonio',
	'SEA' : 'Seattle',
	'TOR' : 'Toronto',
	'UTA' : 'Utah',
	'WAS' : 'Washington',
	}
	

	crawlSeason('data/matchups2007.txt', 'data/2006-2007 Game Log.csv', 1230)

