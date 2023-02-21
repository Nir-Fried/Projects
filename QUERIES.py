import MySQLdb as mdb
import mysql.connector 

cnx = mysql.connector.connect(host = 'localhost', user = 'friednir' ,
         password='...',database= 'friednir', port = '3305')

#Query 1
print("Query 1:")
mycursor = cnx.cursor()
mycursor.execute("Select Country.continent,Count(League.league_id) From Country,League Where Country.country_id = League.country_id Group by Country.continent")

myresult = mycursor.fetchall()
for x in myresult:
    print(x)

mycursor.close()

#Query 2
print("Query 2:")
mycursor = cnx.cursor()
mycursor.execute("Select DISTINCT Country.country_name, Venue.venue_name From Country,Venue Where Country.country_id = Venue.country_id And Venue.capacity > 60000 And Country.continent = 'Europe'")

myresult = mycursor.fetchall()
for x in myresult:
    print(x)

mycursor.close()

#Query 3
print("Query 3:")
mycursor = cnx.cursor()
mycursor.execute('''Select table1.season_name,table1.league_name, table1.AwayWins, table2.HomeWins
From (
Select Season.season_name,Count(*) as AwayWins, League.league_name
From Season,Matches,League
Where Matches.home_score < Matches.away_score
And Season.season_id = Matches.season_id
And Season.league_id = League.league_id
Group by Season.season_name,League.league_name) as table1,
(Select Season.season_name,Count(*) as HomeWins, League.league_name
From Season,Matches,League
Where Matches.home_score > Matches.away_score
And Season.season_id = Matches.season_id
And Season.league_id = League.league_id
Group by Season.season_name,league_name) as table2
Where table1.season_name = table2.season_name
And table1.league_name = table2.league_name''')

myresult = mycursor.fetchall()
for x in myresult:
    print(x)

mycursor.close()


#Query 4
print("Query 4:")
mycursor = cnx.cursor()
mycursor.execute('''Select Distinct Team.team_name,Country.country_name
From Team,Country,
(Select Matches.home_team_id as home,Matches.away_team_id as away
From Matches
Where Matches.final_score = '0-0' 
And Matches.match_date < '2010-01-01'
) as zz
Where (Team.team_id = zz.home 
		OR Team.team_id = zz.away)

And Team.country_id = Country.country_id
''')

myresult = mycursor.fetchall()
for x in myresult:
    print(x)

mycursor.close()

#Query 5
print("Query 5:")
mycursor = cnx.cursor()
mycursor.execute('''
Select Venue.venue_name, Count(*) as count
From Matches,Venue,Season
Where Venue.venue_id = Matches.venue_id
And Matches.season_id = Season.season_id
And Season.is_current = 1
And MATCH(Venue.venue_name)
AGAINST('STADIUM')
Group by Venue.venue_name
Order by count DESC
''')

myresult = mycursor.fetchall()
for x in myresult:
    print(x)

mycursor.close()

#Query 6
print("Query 6:")
mycursor = cnx.cursor()
mycursor.execute('''
Select Team.team_name,Count(*) as countWins
From Team,Matches,Country
Where Team.team_id = Matches.home_team_id
And Matches.home_score > Matches.away_score
And Team.country_id = Country.country_id
And Country.country_name = 'Germany'
Group by Team.team_name
Order by countWins DESC
''') 

myresult = mycursor.fetchall()
for x in myresult:
    print(x)

mycursor.close()
