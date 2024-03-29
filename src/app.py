import os
import requests
import json

from flask import Flask, session, request, render_template, jsonify
from flask_session import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
#from predictor import predict, createClassifier

import numpy as np

import allModelFile as rft
import pickle
import deepNN as nn

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

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

# generateModels()
rforest = pickle.load(open('scikitrforest_model.sav', 'rb'))
nnmlp = pickle.load(open('nnmlp_model.sav', 'rb'))
xgbforest = pickle.load(open('xgboost_model.sav', 'rb'))
testmo = pickle.load(open('test_model.sav', 'rb'))
try:
    nn_model = nn.load_model()
except Exception:
    nn.train_model()
    nn_model = nn.load_model()
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
    nnmlprate = rft.predictResult(radiant, dire, nnmlp) * 100
    xgbrate = rft.predictResult(radiant, dire, xgbforest) * 100
    testrate = rft.predictResult(radiant, dire, testmo) * 100

    uniqueColList = [24, 115, 116, 117, 118, 122, 123, 124, 125, 126, 127, 128]
    data = np.zeros(258, dtype=int)
    for i in radiant:
        data[int(i)-1] = 1
    for i in dire:
        data[int(i) + 129 - 1] = 1

    data = np.array([data]).astype(np.float32)

    nn_rate = nn_model.predict(data)[0][0] * 100
    print(f'nn_rate: {nn_rate}')
    avgrate = ((rforestrate + xgbrate) / 2)
    print('prediction done')
    # TODO: Change rate to actual predictor
    return jsonify({'rforestrate': rforestrate, 'nnmlprate': nnmlprate, 'xgbrate': xgbrate, 'testrate': testrate, 'nn_rate': nn_rate, 'avgrate': avgrate})


@app.route("/<string:team>/last_pick")
def last_pick_page(team):
    return render_template("last_pick_template.html", team=team, heroes=heroList)

# def predict(radiant, dire):
#     rforestrate = rft.predictResult(radiant, dire, rforest) * 100
#     nnmlprate = rft.predictResult(radiant, dire, nnmlp, True) * 100
#     xgbrate = rft.predictResult(radiant, dire, xgbforest) * 100
#     return ((rforestrate + nnmlprate + xgbrate) / 3)

@app.route("/last_pick", methods=["POST"])
def last_pick():
    radiant = request.form.get('rHero')
    radiant = radiant.split(',')
    print(f'Radiant Heroes: {radiant}, {type(radiant)}')
    dire = request.form.get('dHero')
    print(f'Dire Heroes: {dire}')
    dire = dire.split(',')
    
    # remainingHeroes = heroList[:]
    # # index_to_pop = []
    # # for index, hero in enumerate(heroList):
    # #     if hero.id in radiant or hero.id in dire:
    # #         index_to_pop.append(index)
    #
    # # for index, popIndex in enumerate(index_to_pop):
    # #     remainingHeroes.pop(popIndex - index)
    #
    # # for hero_id in radiant:
    # #     for index, hero in enumerate(remainingHeroes):
    # #         if hero.id == hero_id:
    # #             break
    # #     remainingHeroes.pop(index)
    #
    # # for hero_id in dire:
    # #     for index, hero in enumerate(remainingHeroes):
    # #         if hero.id == hero_id:
    # #             break
    # #     remainingHeroes.pop(index)
    # print('starting last pick prediction')
    # rateList = []
    # # TODO Remove oriDire/oriRadiant
    # # if dire last pick
    # if len(dire) == 4:
    #     oriDire = dire
    #     currentID = 0
    #     currentRate = 0
    #     for hero in remainingHeroes:
    #         if hero.id not in radiant and hero.id not in oriDire:
    #             dire = oriDire + [hero.id]
    #             rate = predict(radiant, dire)
    #             heroRate = predictHero(hero.name, rate)
    #             rateList.append(heroRate)
    #             rateList.sort()
    #             rateList.reverse()
    #             if len(rateList) > 5:
    #                 rateList.pop()
    #
    #
    #
    # # else if radiant last pick
    # else:
    #     oriRadiant = radiant
    #     currentID = 0
    #     currentRate = 0
    #     for hero in remainingHeroes:
    #         if hero.id not in oriRadiant and hero.id not in dire:
    #             radiant = oriRadiant + [hero.id]
    #             rate = predict(radiant, dire)
    #             heroRate = predictHero(hero.name, rate)
    #             rateList.append(heroRate)
    #             rateList.sort()
    #             rateList.reverse()
    #             if len(rateList) > 5:
    #                 rateList.pop()
    top5picks = rft.predictLastPick(radiant, dire, [rforest, nnmlp, xgbforest, testmo])
    heroName = []
    allRate = []
    for i in range(5):
        for j in range(len(heroList)):
            if heroList[j].id == top5picks[i][0]:
                heroName.append(heroList[j].name)
                allRate.append(top5picks[i][1])

    # print(f'currentID: {currentID}')
    # print('last pick prediction complete')
    # # # rate = predict(radiant, dire, classifier) * 100
    # # for hero in remainingHeroes:
    # #     if hero.id == currentID:
    # #         hero_name = hero.name
    # #         break
    # print(f'rateList: {rateList}')
    # # currentRate = [hero.rate for hero in rateList]
    # # hero_name = [hero.name for hero in rateList]
    # print(f'currentRate: \n{currentRate}')
    # print(f'hero_namem: \n{hero_name}')
    return jsonify({"rate": allRate, 'hero': heroName})

