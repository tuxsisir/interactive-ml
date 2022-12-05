import sys,os
sys.path.append(os.getcwd())

from flask import Flask, render_template, redirect, url_for, flash
from flask_migrate import Migrate

from flask_humanize import Humanize

from models import db, User, OAuth
from views.dashboard import dashboard_blueprint
from views.user_profile import user_profile_blueprint
from views.pipeline import pipeline_blueprint

from flask_dance.contrib.twitter import make_twitter_blueprint, twitter
from flask_login import current_user, LoginManager, login_required, login_user, logout_user
from flask_dance.consumer.storage.sqla import SQLAlchemyStorage
from flask_dance.consumer import oauth_authorized
from sqlalchemy.orm.exc import NoResultFound

# config = dotenv_values(".env")

login_manager = LoginManager()

# root of the project
basedir = os.path.dirname(os.path.abspath(__file__))

def page_not_found(_):
  return render_template('error-pages/404.html'), 404

def unauthorized(_):
  return render_template('error-pages/401.html'), 403

def server_error(_):
  return render_template('error-pages/500.html'), 500

def create_app():
    app = Flask(__name__)
    db_uri = os.getenv('DB_URI')
    app.config.update({
        'SQLALCHEMY_DATABASE_URI': db_uri,
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'SQLALCHEMY_ENGINE_OPTIONS': {
            "max_overflow": 15,
            "pool_pre_ping": True,
            "pool_recycle": 60 * 60,
            "pool_size": 30,
            }
        })
    db.init_app(app)
    migrate = Migrate(app, db)
    humanize = Humanize(app)
    app.jinja_env.filters['basename'] = humanize
    login_manager.init_app(app)
    app.register_error_handler(404, page_not_found)
    app.register_error_handler(401, unauthorized)
    app.register_error_handler(500, server_error)
    return app

app = create_app()
app.secret_key = os.getenv('APP_SECRET')
twitter_blueprint = make_twitter_blueprint(
    api_key=os.getenv('TWITTER_CLIENT_ID'),
    api_secret=os.getenv('TWITTER_CLIENT_SECRET'),
)
app.register_blueprint(twitter_blueprint, url_prefix="/login")
app.register_blueprint(dashboard_blueprint)
app.register_blueprint(user_profile_blueprint, url_prefix='/user')
app.register_blueprint(pipeline_blueprint, url_prefix='/pipeline')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

twitter_blueprint.backend = SQLAlchemyStorage(OAuth, db.session, user=current_user)

@app.route("/twitter")
def index():
    if not twitter.authorized:
        print('not twitter authorized')
        return redirect(url_for("twitter.login"))
    account_info = twitter.get("account/settings.json")
    account_info_json = account_info.json()
    return '<h1>Your Twitter name is @{}'.format(account_info_json['screen_name'])


@oauth_authorized.connect_via(twitter_blueprint)
def twitter_logged_in(blueprint, token):

    account_info = blueprint.session.get('account/settings.json')

    if account_info.ok:
        account_info_json = account_info.json()
        username = account_info_json['screen_name']

        query = User.query.filter_by(username=username)

        try:
            user = query.one()
        except NoResultFound:
            user = User(
                    username=username,
                    first_name='',
                    last_name='',
                    phone='',
                    email='',
                    designation='',
                    twitter_handle=username
                    )
            db.session.add(user)
            db.session.commit()

        login_user(user)


@app.route('/logout')
@login_required
def logout():
    logout_user() # Delete Flask-Login's session cookie
    del twitter_blueprint.token # Delete OAuth token from storage
    flash('Logged out! See you again!', 'success')
    return redirect('/')
