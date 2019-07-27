import os
import requests
import json

from flask import Flask, session, request, render_template, jsonify
from flask_session import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
#from predictor import predict, createClassifier

import randomForesttest as rft

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

matchData = rft.readData("testMatches_noBool.csv")
cleanedData = rft.cleanData(matchData)
rforest = rft.scikitRForest(cleanedData)
tforest = rft.tensorFlowRForest(cleanedData)
xgbforest = rft.xgBoost(cleanedData)

print('predictor complete')

class Hero:
    def __init__(self, ID, name):
        self.id = ID
        self.name = name
    
    def __str__(self):
        return f'ID: {self.id} Name: {self.name}'

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name

class predictHero:
    def __init__(self, name, rate):
        self.name = name
        self.rate = rate

    def __lt__(self, other):
        return self.rate < other.rate

    def __le__(self, other):
        return self.rate <= other.rate

    def __eq__(self, other):
        return self.rate == other.rate

    def __str__(self):
        return f'rate: {self.rate} Name: {self.name}'

def getHeroList():
    heroList = []
    with open('heroes.json') as f:
        heroes = json.load(f)
        for hero in heroes['heroes']:
            newHero = Hero(hero['id'], hero['localized_name'])
            # print(f'newHero: {newHero}')
            heroList.append(newHero)
    heroList.sort()
    return heroList

heroList = getHeroList()

@app.route("/")
def index():
    return render_template('index.html', heroes=heroList)

# @app.route("/hello", methods=["POST"])
# def hello():
#     name = request.form.get("name")
#     return render_template("hello.html", name=name)

@app.route("/predict", methods=["POST"])
def predictor():
    print('starting prediction')
    radiant = request.form.get('rHero')
    radiant = radiant.split(',')
    print(f'Radiant Heroes: {radiant}, {type(radiant)}')
    dire = request.form.get('dHero')
    print(f'Dire Heroes: {dire}')
    dire = dire.split(',')
    rforestrate = rft.predictResult(radiant, dire, rforest) * 100
    tforestrate = rft.predictResult(radiant, dire, tforest, True) * 100
    xgbrate = rft.predictResult(radiant, dire, xgbforest) * 100
    avgrate = ((rforestrate + tforestrate + xgbrate) / 3)
    print('prediction done')
    # TODO: Change rate to actual predictor
    return jsonify({'rforestrate': rforestrate, 'tforestrate': tforestrate, 'xgbrate': xgbrate, 'avgrate': avgrate})


@app.route("/<string:team>/last_pick")
def last_pick_page(team):
    return render_template("last_pick_template.html", team=team, heroes=heroList)

def predict(radiant, dire):
    rforestrate = rft.predictResult(radiant, dire, rforest) * 100
    tforestrate = rft.predictResult(radiant, dire, tforest, True) * 100
    xgbrate = rft.predictResult(radiant, dire, xgbforest) * 100
    return ((rforestrate + tforestrate + xgbrate) / 3)

@app.route("/last_pick", methods=["POST"])
def last_pick():
    radiant = request.form.get('rHero')
    radiant = radiant.split(',')
    print(f'Radiant Heroes: {radiant}, {type(radiant)}')
    dire = request.form.get('dHero')
    print(f'Dire Heroes: {dire}')
    dire = dire.split(',')
    
    remainingHeroes = heroList[:]
    # index_to_pop = []
    # for index, hero in enumerate(heroList):
    #     if hero.id in radiant or hero.id in dire:
    #         index_to_pop.append(index)

    # for index, popIndex in enumerate(index_to_pop):
    #     remainingHeroes.pop(popIndex - index) 

    # for hero_id in radiant:
    #     for index, hero in enumerate(remainingHeroes):
    #         if hero.id == hero_id:
    #             break
    #     remainingHeroes.pop(index)

    # for hero_id in dire:
    #     for index, hero in enumerate(remainingHeroes):
    #         if hero.id == hero_id:
    #             break
    #     remainingHeroes.pop(index)
    print('starting last pick prediction')
    rateList = []
    # TODO Remove oriDire/oriRadiant
    # if dire last pick
    if len(dire) == 4:
        oriDire = dire
        currentID = 0
        currentRate = 0
        for hero in remainingHeroes:
            if hero.id not in radiant and hero.id not in oriDire:
                dire = oriDire + [hero.id]
                rate = predict(radiant, dire)
                heroRate = predictHero(hero.name, rate) 
                rateList.append(heroRate)
                rateList.sort()
                rateList.reverse()
                if len(rateList) > 5:
                    rateList.pop()
                
                

    # else if radiant last pick
    else:
        oriRadiant = radiant
        currentID = 0
        currentRate = 0
        for hero in remainingHeroes:
            if hero.id not in oriRadiant and hero.id not in dire:
                radiant = oriRadiant + [hero.id]
                rate = predict(radiant, dire) 
                heroRate = predictHero(hero.name, rate) 
                rateList.append(heroRate)
                rateList.sort()
                rateList.reverse()
                if len(rateList) > 5:
                    rateList.pop()

    print(f'currentID: {currentID}')
    print('last pick prediction complete')
    # # rate = predict(radiant, dire, classifier) * 100
    # for hero in remainingHeroes:
    #     if hero.id == currentID:
    #         hero_name = hero.name
    #         break
    print(f'rateList: {rateList}')
    currentRate = [hero.rate for hero in rateList]
    hero_name = [hero.name for hero in rateList]
    print(f'currentRate: \n{currentRate}')
    print(f'hero_namem: \n{hero_name}')
    return jsonify({"rate": currentRate, 'hero': hero_name})    

