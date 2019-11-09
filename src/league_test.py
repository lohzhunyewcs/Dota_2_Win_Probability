# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:20:02 2019

@author: ZY
"""

import requests
import allModelFile as rft
import pickle


key = 'B7303CCF8E7D626136174C243D689CEA'
ti9_id = 10749

def getLiveGames():
    result = requests.get(f'http://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/?key={key}')
    print(f'status_code: {result.status_code}')
    #print(f'result Json:\n{result.json()}')
    
    print(f"amount of matches: {len(result.json()['result']['games'])}")
    
    players = result.json()['result']['games'][0]['players']
    
    print(f'len players: {len(players)}')
    
    for player in players:
        print(player['hero_id'], player['team'])
        
def getLeagues():
    result = requests.get(f'http://api.steampowered.com/IDOTA2AutomatedTourney_570/GetActiveTournamentList/v1/?key={key}')
    print(f'status_code: {result.status_code}')
    #print(f'result Json:\n{result.json()}')
    if result.status_code != 403:
        print(f'amount of leagues: {len(result["leagues"])}')
    
def tiOnly(match):
    return match['league_id'] == ti9_id
        
    
def getTIGames():
    result = requests.get(f'http://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/?key={key}')
    matches = result.json()['result']['games']
    matches = filter(tiOnly, matches)
    rHeros = []
    dHeros = []
    matchIDS = []
    teamNames = []
    for match in matches:
        players = match['players']
        rHero = []
        dHero = []
        for player in players:
            hero_id = player['hero_id']
            team_id = player['team_id']
            if team_id == 0:
                rHero.append(hero_id)
            elif team_id == 1:
                dHero.append(hero_id)
        matchID = match['match_id']
        matchIDS.append(matchID)
        rHeros.append(rHero)
        dHeros.append(dHero)
        matchData = requests.get(f'https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/V001/?match_id={matchID}&key={key}')
        r_team_id = matchData['radiant_team_id']
        d_team_id = matchData['dire_team_id']
        r_team = requests.get(f'http://api.steampowered.com/IDOTA2Match_570/GetTeamInfo/v1/?team_id={r_team_id}&key={key}')
        d_team = requests.get(f'http://api.steampowered.com/IDOTA2Match_570/GetTeamInfo/v1/?team_id={d_team_id}&key={key}')        
        teamNames.append([r_team['name'], d_team['name']])
    botPost(matchIDS, rHeros, dHeros, teamNames)

def generateModels():
    matchData = rft.readData("testMatches_noBool.csv")
    cleanedData = rft.cleanData(matchData)
    rforestmodel = rft.scikitRForest(cleanedData)
    nnmlpmodel = rft.scikitMLP(cleanedData)
    xgbforestmodel = rft.xgBoost(cleanedData)
    testmodel = rft.scikitTest(cleanedData)

    rfilename = 'scikitrforest_model.sav'
    nnmlpfilename = 'nnmlp_model.sav'
    xgbfilename = 'xgboost_model.sav'
    testfilename = 'test_model.sav'
    pickle.dump(rforestmodel, open(rfilename, 'wb'))
    pickle.dump(nnmlpmodel, open(nnmlpfilename, 'wb'))
    pickle.dump(xgbforestmodel, open(xgbfilename, 'wb'))
    pickle.dump(testmodel, open(testfilename, 'wb'))


def predict(radiant, dire):
    rforestrate = rft.predictResult(radiant, dire, rforest) * 100
    xgbrate = rft.predictResult(radiant, dire, xgbforest) * 100
    avgrate = ((rforestrate + xgbrate) / 2)
    return avgrate

def botPost(matchIDS, rHeros, dHeros, teamNames, post, template):
    final = ''
    for index in range(len(matchIDS)):
        rHero = rHeros[index]
        dHero = dHeros[index]
        percentage = predict(rHero, dHero)
        team = int(percentage >= 50)
        final += template.formamt()
        
    
    
if __name__ == "__main__":
    getLiveGames()
    getLeagues()
    generateModels()
    rforest = pickle.load(open('scikitrforest_model.sav', 'rb'))
    nnmlp = pickle.load(open('nnmlp_model.sav', 'rb'))
    xgbforest = pickle.load(open('xgboost_model.sav', 'rb'))
    testmo = pickle.load(open('test_model.sav', 'rb'))
    print('predictor complete')
    