import os
import requests
import json

from flask import Flask, session, request, render_template, jsonify
from flask_session import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from predictor import predict, createClassifier
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

classifier = createClassifier()

class Hero:
    def __init__(self, ID, name):
        self.id = ID
        self.name = name
    
    def __str__(self):
        return f'ID: {self.id} Name: {self.name}'

@app.route("/")
def index():
    heroList = []
    with open('heroes.json') as f:
        heroes = json.load(f)
        for hero in heroes['heroes']:
            newHero = Hero(hero['id'], hero['localized_name'])
            # print(f'newHero: {newHero}')
            heroList.append(newHero)
    return render_template('index.html', heroes=heroList)

# @app.route("/hello", methods=["POST"])
# def hello():
#     name = request.form.get("name")
#     return render_template("hello.html", name=name)

@app.route("/predict", methods=["POST"])
def predictor():
    radiant = request.form.get('rHero')
    radiant = radiant.split(',')
    print(f'Radiant Heroes: {radiant}, {type(radiant)}')
    dire = request.form.get('dHero')
    print(f'Dire Heroes: {dire}')
    dire = dire.split(',')
    rate = predict(radiant, dire, classifier) * 100
    # TODO: Change rate to actual predictor
    return jsonify({"rate": rate})