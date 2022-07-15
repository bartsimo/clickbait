# To start in virtual environment: View --> Command Palette --> Terminal: Create New Terminal

# https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
# https://code.visualstudio.com/docs/python/tutorial-flask#_prerequisites


from flask import Flask, render_template
from datetime import datetime
import re

app = Flask(__name__)

# Flask's app.route decorator maps the URL route / to a function 
# that returns content:


@app.route("/")
def home():
    return "Hello, Flask!"

#several routes can lead to the "same page"
@app.route("/hello/")
#eine Art wildcard kann scheinbar mit <> gesetzt werden.
@app.route("/hello/<name>")
def hello_there(name = None):
    #in der args list können python funktionen benutzt werden
    return render_template("hello_there.html", name=name, date=datetime.now())

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")
# run app from terminal with: python -m flask run
# or: python -m flask run --host=127.0.0.1 --port=5050

