import uuid
from datetime import datetime

from flask import Blueprint, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_required
from flask import session

from models import User

user_profile_blueprint = Blueprint('user_profile', __name__)

db = SQLAlchemy()

@user_profile_blueprint.route("/edit-profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    user_id = session['_user_id']
    user = db.session.query(User).get(int(user_id))
    if request.method == "POST":
        user.first_name = request.form.get('first_name', None)
        user.last_name = request.form.get('last_name', None)
        user.phone = request.form.get('phone', None)
        user.email = request.form.get('email', None)
        user.twitter_handle = uuid.uuid4().hex
        user.designation = request.form.get('designation', None)
        db.session.commit()
        return redirect('/user/edit-profile')
    return render_template(
            'user/edit-profile.html',
            active_dashboard='edit-profile',
            user=user)
