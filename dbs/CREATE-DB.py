import mysql.connector 
import pandas as pd
import os

# Creating the tables


TABLES = {}
'''
TABLES['Country'] = (
    "CREATE TABLE `Country` ("
    "  `country_id` smallint(4) NOT NULL,"
    "  `country_name` varchar(50) NOT NULL,"
    " `continent` varchar(30) NOT NULL,"
    " Primary Key (`country_id`)"
    ") ENGINE=InnoDB")

TABLES['Team'] = (
    "CREATE TABLE `Team` ("
    "  `team_id` smallint(5) NOT NULL,"
    "  `team_name` varchar(50) NOT NULL,"
    "  `country_id` smallint(4) NOT NULL,"
    "  PRIMARY KEY (`team_id`),"
    "  FOREIGN KEY (`country_id`) REFERENCES Country(`country_id`)"
    ") ENGINE=InnoDB"
)

TABLES['League'] = (
    "CREATE TABLE `League` ("
    "  `league_id` smallint(4) NOT NULL,"
    "  `league_name` varchar(50) NOT NULL,"
    "  `country_id` smallint(4) NOT NULL,"
    "  PRIMARY KEY (`league_id`),"
    "  FOREIGN KEY (`country_id`) REFERENCES Country(`country_id`)"
    ") ENGINE=InnoDB"
)

TABLES['Season'] = (
    "CREATE TABLE `Season` ("
    "  `season_id` smallint(5) NOT NULL,"
    "  `season_name` varchar(10) NOT NULL,"
    "  `is_current` tinyint(1) NOT NULL,"
    "  `league_id` smallint(4) NOT NULL,"
    "  PRIMARY KEY (`season_id`),"
    "  FOREIGN KEY (`league_id`) REFERENCES League(`league_id`)"
    ") ENGINE=InnoDB"
)

TABLES['Venue'] = (
    "CREATE TABLE `Venue` ("
    "  `venue_id` smallint(5) NOT NULL,"
    "  `venue_name` varchar(100) NOT NULL,"
    "  `capacity` mediumint(6) NOT NULL,"
    "  `country_id` smallint(4) NOT NULL,"
    "  PRIMARY KEY (`venue_id`),"
    "  FOREIGN KEY (`country_id`) REFERENCES Country(`country_id`)"
    ") ENGINE=InnoDB"
) 
'''

TABLES['Matches'] = (
    "CREATE TABLE `Matches` ("
    "  `match_id` mediumint(7) NOT NULL,"
    "  `match_date` date NOT NULL,"
    "  `season_id` smallint(5) NOT NULL,"
    "  `home_team_id` smallint(5) NOT NULL,"
    "  `away_team_id` smallint(5) NOT NULL,"
    "  `home_score` tinyint(2) NOT NULL,"
    "  `away_score` tinyint(2) NOT NULL,"
    "  `final_score` varchar(5) NOT NULL,"
    "  `venue_id` smallint(5) NOT NULL,"
    "  PRIMARY KEY (`match_id`),"
    "  FOREIGN KEY (`season_id`) REFERENCES Season(`season_id`),"
    "  FOREIGN KEY (`home_team_id`) REFERENCES Team(`team_id`),"
    "  FOREIGN KEY (`away_team_id`) REFERENCES Team(`team_id`),"
    "  FOREIGN KEY (`venue_id`) REFERENCES Venue(`venue_id`)"
    ") ENGINE=InnoDB"
)

cnx = mysql.connector.connect(host = 'localhost', user = 'friednir' ,
         password='...',database= 'friednir', port = '3305')

cursor = cnx.cursor()

for table_name in TABLES:
    table_description = TABLES[table_name]
    print("Creating table {}: ".format(table_name), end='')
    cursor.execute(table_description)

cursor.close()
cnx.close()


# Create a full text index on the venue_name column of the Venue table
cnx = mysql.connector.connect(host = 'localhost', user = 'friednir' ,
         password='...',database= 'friednir', port = '3305')

cursor = cnx.cursor()

create_index = "CREATE FULLTEXT INDEX `vidx` ON `Venue` (`venue_name`)"
cursor.execute(create_index)
cnx.commit()
cnx.close()