from flask import Flask
app = Flask(__name__)

# Flask's app.route decorator maps the URL route / to a function 
# that returns content:
@app.route("/")
def home():
    return "Hello, Flask!"