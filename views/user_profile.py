import uuid
from datetime import datetime

from flask import Blueprint, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy

from models import User

user_profile_blueprint = Blueprint('user_profile', __name__)

db = SQLAlchemy()

@user_profile_blueprint.route("/edit-profile", methods=["GET", "POST"])
def edit_profile():
    if request.method == "POST":
        user = db.session.query(User).get(1)
        user.first_name = request.form.get('first_name', None)
        user.last_name = request.form.get('last_name', None)
        user.phone = request.form.get('phone', None)
        user.email = request.form.get('email', None)
        user.twitter_handle = uuid.uuid4().hex
        user.designation = request.form.get('designation', None)
        db.session.commit()
        return redirect('/user/edit-profile')
    user = db.session.query(User).get(1)
    if not user:
        user = User(
                id=1,
                created_at=datetime.now(),
                first_name="Matthew",
                last_name="Andersen",
                phone="+6043692971",
                email="matt@imlp.com",
                designation="Senior Data Analyst",
                twitter_handle=uuid.uuid4().hex)
        db.session.add(user)
        db.session.commit()
    return render_template(
            'user/edit-profile.html',
            active_dashboard='edit-profile',
            user=user)
