# To start in virtual environment: View --> Command Palette --> Terminal: Create New Terminal

# https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
# https://code.visualstudio.com/docs/python/tutorial-flask#_prerequisites


from flask import Flask
from datetime import datetime
import re

app = Flask(__name__)

# Flask's app.route decorator maps the URL route / to a function 
# that returns content:


@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content

# run app from terminal with: python -m flask run
# or: python -m flask run --host=127.0.0.1 --port=5050

