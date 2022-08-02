import os
import cs50

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, url_for, request, session
from flask_session import Session

import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Configure application
app = Flask(__name__)


'''
Flask boiler plate from CS50
'''


# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = cs50.SQL("sqlite:///test.db")


# Load own model
clickbait_model = open("clickbait_model.pkl", "rb")
clf = joblib.load(clickbait_model)

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

'''
End boiler plate 
'''

@app.route("/")
def home():
    return "Clickbait time!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
	message = request.form['message']
	data = [message]
	vect = cv.transform(data).toarray()
	my_prediction = clf.predict(vect)
    
return render_template('result.html', prediction = my_prediction)