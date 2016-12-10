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

def crawlSeason(matchupsData, numberOfMatches):

	with open(matchupsData, 'r') as f:
		reader = csv.reader(f, delimiter = "\t")
		reader.next() #this is to skip the headings
		count = 0
		while (count < numberOfMatches):

			#try:
			currentLine = reader.next()
			#except StopIteration:
				#break
			
			currentGameID = currentLine[0]
			currentHomeTeam = currentGameID[-3:len(currentGameID)]
			currentAwayTeam = currentGameID[-6:-3]
		
			endTime = currentLine[2]

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
			result = False
			if(HomeTeamFinalScore > AwayTeamFinalScore):
				result = True
				MasterTeamDictionary[currentHomeTeam][4] += 1
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

			MasterTeamDictionary[currentHomeTeam][3] +=1
			MasterTeamDictionary[currentAwayTeam][3] +=1

			MasterTeamDictionary[currentHomeTeam][5] = float(MasterTeamDictionary[currentHomeTeam][4]) / float(MasterTeamDictionary[currentHomeTeam][3])
			MasterTeamDictionary[currentAwayTeam][5] = float(MasterTeamDictionary[currentAwayTeam][4]) / float(MasterTeamDictionary[currentAwayTeam][3])
			count +=1

			
		printplus (MasterTeamDictionary)
	return

if __name__ == '__main__':
	StartingValues = [100.0,0.0,0.0, 0, 0, 0.0] #(ELO, Avg. points scored, Avg. points allowed, games played, games won, win rate)
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
	'NYK' : list(StartingValues),
	'ORL' : list(StartingValues),
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

	crawlSeason('matchups2007.txt', 1230)

