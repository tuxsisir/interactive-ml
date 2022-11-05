import sys,os
sys.path.append(os.getcwd())

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import Blueprint, render_template, redirect, request, url_for

from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine

from flask_humanize import Humanize

from models import db
from views.dashboard import dashboard_blueprint
from views.user_profile import user_profile_blueprint
from views.pipeline import pipeline_blueprint


def create_app():
    app = Flask(__name__)
    # db_uri = 'postgresql://postgres:postgres@localhost:5432/mlflask'
    db_uri = 'sqlite:///db.sqlite3'
    app.config.update({
        'SQLALCHEMY_DATABASE_URI': db_uri,
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    })
    db.init_app(app)
    migrate = Migrate(app, db)
    humanize = Humanize(app)
    app.jinja_env.filters['basename'] = humanize
    return app

app = create_app()
app.register_blueprint(dashboard_blueprint)
app.register_blueprint(user_profile_blueprint, url_prefix='/user')
app.register_blueprint(pipeline_blueprint, url_prefix='/pipeline')

