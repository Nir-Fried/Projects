
import requests
import pandas as pd
import os
path = "insert path here..."

headers = { 
  "apikey": "insert api key here"}


'''
############################################################################################################
Retrievals from API
############################################################################################################
'''

#Country
params = (
    
)

Country = pd.DataFrame(columns=['country_id','country_name','continent'])
response = requests.get('https://app.sportdataapi.com/api/v1/soccer/countries', headers=headers, params=params)
json = response.json()
for row in json['data']:
    line = [row['country_id'],row['name'],row['continent']]
    if line[1] != line[2]:
        Country = pd.concat([Country,pd.DataFrame([line], columns=['country_id','country_name','continent'])], ignore_index=True)

print(Country)
Country.to_csv(os.path.join(path,r'Country.csv'),index = False) 

#Teams
Teams = pd.DataFrame(columns=['team_id', 'name', 'country_id'])
counter = 1
for c in Country['country_id']:
    params = (('country_id', c),)
    response = requests.get('https://app.sportdataapi.com/api/v1/soccer/teams/', headers=headers, params=params)
    if response.status_code == 200: #OK
        json = response.json()
        if json['data'] == []:
            continue
        
        for row in json['data']:
            line = [row['team_id'],row['name'],row['country']['country_id']]
            if line[0] == 'NaN':
                print("here!")
                line[0] = str("a" + counter)
                counter += 1
            Teams = pd.concat([Teams,pd.DataFrame([line], columns=['team_id', 'name', 'country_id'])], ignore_index=True)

print(Teams)
Teams.to_csv(os.path.join(path,r'Teams.csv'), index = False)


#League


Country = pd.read_csv(os.path.join(path,r'Country.csv'))

League = pd.DataFrame(columns=['league_id','league_name','country_id'])


for c in Country['country_id']:
    params = (
        ("country_id",c),
    )
    response = requests.get('https://app.sportdataapi.com/api/v1/soccer/leagues', headers=headers, params=params)
    json = response.json()
    for row in json['data']:
        current = json['data'][row]
        line = [current['league_id'],current['name'],current['country_id']]
        League = pd.concat([League,pd.DataFrame([line], columns=['league_id','league_name','country_id'])], ignore_index=True)

print(League)
League.to_csv(os.path.join(path,r'League.csv'), index = False) 


#Season

League = pd.read_csv(os.path.join(path,r'League.csv'))
Season = pd.DataFrame(columns=['season_id','season_name','is_current','league_id'])

for l in League['league_id']:
    # if l <= 400:
    #     continue
    params = (
    ("league_id",l),
    )
    response = requests.get('https://app.sportdataapi.com/api/v1/soccer/seasons', headers=headers, params=params)
    json = response.json()
    for row in json['data']:
        print(row)
        line = [row['season_id'],row['name'],row['is_current'],row['league_id']]
        Season = pd.concat([Season,pd.DataFrame([line], columns=['season_id','season_name','is_current','league_id'])], ignore_index=True)

print(Season)
Season.to_csv(os.path.join(path,r'Season.csv'), index = False) 


#Stage

Match = pd.DataFrame(columns=['match_id','start_date','season_id','round_id','referee_id','home_team_id','away_team_id','home_score','away_score','final_score','overtime_score','penalties_score','venue_id'])
cSeason = pd.read_csv(os.path.join(path,r'cSeasons.csv'))

for s in cSeason['season_id']:
    params = (
        ("season_id",s),
    )
    response = requests.get('https://app.sportdataapi.com/api/v1/soccer/matches', headers=headers, params=params)
    json = response.json()
    for row in json['data']:
        print(row)
        if row['venue'] == None:
            line = [row['match_id'],row['match_start'],row['season_id'], row['round']['round_id'], row['referee_id'], row['home_team']['team_id'], row['away_team']['team_id'], row['stats']['home_score'], row['stats']['away_score'],row['stats']['ft_score'],row['stats']['et_score'],row['stats']['ps_score'],"-1"]
        else:
            line = [row['match_id'],row['match_start'],row['season_id'], row['round']['round_id'], row['referee_id'], row['home_team']['team_id'], row['away_team']['team_id'], row['stats']['home_score'], row['stats']['away_score'],row['stats']['ft_score'],row['stats']['et_score'],row['stats']['ps_score'],row['venue']['venue_id']]
        Match = pd.concat([Match,pd.DataFrame([line], columns=['match_id','start_date','season_id','round_id','referee_id','home_team_id','away_team_id','home_score','away_score','final_score','overtime_score','penalties_score','venue_id'])], ignore_index=True)
        #print(line)
print()
print(Match)

Match.to_csv(os.path.join(path,r'Match.csv'), index = False) 


Country = pd.read_csv(os.path.join(path,r'Country.csv'))

Arena = pd.DataFrame(columns=['venue_id','venue_name','capacity','country_id'])

for c in Country['country_id']:
    params = (
        ("country_id",c),
    )
    response = requests.get('https://app.sportdataapi.com/api/v1/soccer/venues', headers=headers, params=params)
    json = response.json()
    for row in json['data']:
        line = [row['venue_id'],row['name'],row['capacity'],row['country_id']]
        Arena = pd.concat([Arena,pd.DataFrame([line], columns=['venue_id','venue_name','capacity','country_id'])], ignore_index=True)

print(Arena)
Arena.to_csv(os.path.join(path,r'Arena.csv'), index = False)

params = (
   ("country_id","48"),
)

response = requests.get('https://app.sportdataapi.com/api/v1/soccer/players', headers=headers, params=params)
json = response.json()
print(json['data'])


'''
############################################################################################################
Data Insertion
############################################################################################################
'''
import mysql.connector 

cnx = mysql.connector.connect(host = 'localhost', user = 'friednir' ,
         password='...',database= 'friednir', port = '3305')

cursor = cnx.cursor()

add_country = ("INSERT INTO Country "
                "(country_id, country_name, continent) "
                "VALUES (%s, %s, %s)") 

add_team = ("INSERT INTO Team "
                "(team_id, team_name, country_id) "
                "VALUES (%s, %s, %s)")

add_league = ("INSERT INTO League "
                "(league_id, league_name, country_id) "
                "VALUES (%s, %s, %s)")

add_season = ("INSERT INTO Season "
                "(season_id, season_name, is_current, league_id) "
                "VALUES (%s, %s, %s, %s)")

add_venue = ("INSERT INTO Venue "
                "(venue_id, venue_name, capacity, country_id) "
                "VALUES (%s, %s, %s, %s)") 

add_match = ("INSERT INTO Matches "
                "(match_id, match_date, season_id, home_team_id, away_team_id, home_score, away_score, final_score, venue_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")


Country = pd.read_csv(os.path.join(path,r'Country.csv'))
Team = pd.read_csv(os.path.join(path,r'Teams.csv'))
League = pd.read_csv(os.path.join(path,r'League.csv'))
Season = pd.read_csv(os.path.join(path,r'Season.csv'))
Venue = pd.read_csv(os.path.join(path,r'Arena.csv'))
Matches = pd.read_csv(os.path.join(path,r'Match.csv'))

for row in Country.itertuples():
    data_country = (row.country_id, row.country_name, row.continent)
    cursor.execute(add_country, data_country)
    cnx.commit()

print("inserted country successfully")

for row in Team.itertuples():
    data_team = (row.team_id, row.name, row.country_id)
    cursor.execute(add_team, data_team)
    cnx.commit()

print("inserted team successfully")

for row in League.itertuples():
    data_league = (row.league_id, row.league_name, row.country_id)
    cursor.execute(add_league, data_league)
    cnx.commit()

print("inserted league successfully")

for row in Season.itertuples():
    data_season = (row.season_id, row.season_name, row.is_current, row.league_id)
    cursor.execute(add_season, data_season)
    cnx.commit()

print("inserted season successfully")

for row in Venue.itertuples():
    data_venue = (row.venue_id, row.venue_name, row.capacity, row.country_id)
    cursor.execute(add_venue, data_venue)
    cnx.commit()

print("inserted venue successfully") 

for row in Matches.itertuples():
    data_match = (row.match_id, row.match_date, row.season_id, row.home_team_id, row.away_team_id, row.home_score, row.away_score, row.final_score, row.venue_id)
    print(data_match)
    cursor.execute(add_match, data_match)
    cnx.commit()

print("done")

cursor.close()
cnx.close()