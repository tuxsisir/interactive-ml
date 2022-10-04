from flask import Flask
from flask import render_template


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/dashboard/home")
def dashboard():
    return render_template('dashboard/home.html')


@app.route("/dashboard/upload")
def dashboard_upload():
    return render_template('dashboard/upload.html')


@app.route("/dashboard/history")
def dashboard_history():
    return render_template('dashboard/history.html')


@app.route("/pipeline")
def pipeline():
    return render_template('pipeline/workflow.html')
