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

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template("hello_there.html", name=name, date=datetime.now())

# run app from terminal with: python -m flask run
# or: python -m flask run --host=127.0.0.1 --port=5050

