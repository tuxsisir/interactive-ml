import os
import uuid

from datetime import datetime

from pathlib import Path
from flask import Blueprint, render_template, redirect, request, session, flash
from flask_login import login_required
from models import MLProject, MLProjectConfig
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc

from werkzeug.utils import secure_filename

basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_id():
    return uuid.uuid4().hex


dashboard_blueprint = Blueprint('dashboard', __name__)

db = SQLAlchemy()


@dashboard_blueprint.route("/")
def index():
    """
    Home page
    """
    if session.get('_user_id', None):
        flash('Welcome to Interactive ML Pipeline.', 'success')
        return redirect('/dashboard/home')
    return render_template('index.html', page_index=True)


@dashboard_blueprint.route("/dashboard/explore")
@login_required
def explore():
    """
    Explore page
    """
    return render_template('dashboard/explore.html', page_index=True)


@dashboard_blueprint.route("/dashboard/home")
@login_required
def dashboard():
    res = db.session.query(MLProject).filter_by(created_by=session['_user_id']).order_by(
        desc(MLProject.created_at)).all()
    counts = {
        "draft": db.session.query(MLProject).filter_by(created_by=session['_user_id'], status='Draft').count(),
        "completed": db.session.query(MLProject).filter_by(created_by=session['_user_id'], status='Completed').count(),
        "in_progress": db.session.query(MLProject).filter_by(created_by=session['_user_id'], status='In Progress').count(),
    }
    return render_template(
        'dashboard/home.html',
        active_dashboard="home",
        counts=counts,
        projects=res)


@dashboard_blueprint.route('/<int:id>/edit-project', methods=['GET', 'POST'])
@login_required
def dashboard_edit_project(id):
    project = db.session.query(MLProject).get(id)
    if request.method == 'POST':
        project.title = request.form.get('title', None)
        project.description = request.form.get('description', None)
        project.status = request.form.get('btnradio', None)
        db.session.commit()
        db.session.close()
        flash('Successfully edited your project.', 'success')
        return redirect('/dashboard/home')
    return render_template(
        'dashboard/upload.html',
        active_dashboard="upload",
        edit=True,
        project=project)


@dashboard_blueprint.route("/dashboard/upload", methods=["POST", "GET"])
@login_required
def dashboard_upload():
    """
    Dashboard upload dataset
    """
    if request.method == 'POST':
        title = request.form.get('title', None)
        description = request.form.get('description', None)
        status = request.form.get('btnradio', None)
        file = request.files['file']
        if file and file.mimetype != 'text/csv':
            flash('Please upload csv datasets only.', 'danger')
            return render_template('dashboard/upload.html', active_dashboard="upload")
        if not all([title, description, file]):
            flash('Please complete all fields including dataset.', 'danger')
            return render_template('dashboard/upload.html', active_dashboard="upload")

        secure_file_name = secure_filename(file.filename)  # to make the directory
        original_file_name = secure_file_name.replace('.', '_original.')

        current_user = session['twitter_oauth_token']['screen_name']
        dataset_url = f"static/uploads/datasets/{current_user}/{secure_file_name.split('.')[0]}"

        Path(basedir, dataset_url).mkdir(parents=True, exist_ok=True)

        file.save(os.path.join(basedir, dataset_url, original_file_name))
        ml_project = MLProject(
            created_at=datetime.now(),
            created_by=session['_user_id'],
            title=title,
            description=description,
            filename=original_file_name,
            filename_preferred=secure_file_name.split('.')[0], # directory name
            status=status)
        db.session.add(ml_project)
        db.session.commit()
        ml_project_config = MLProjectConfig(
            created_at=datetime.now(),
            ml_project=ml_project.id,
            config={},
            description='')
        db.session.add(ml_project_config)
        db.session.commit()
        db.session.close()
        flash('Successfully created your project.', 'success')
        return redirect('/dashboard/home')
    return render_template('dashboard/upload.html', active_dashboard="upload")


@dashboard_blueprint.route("/dashboard/history")
@login_required
def dashboard_history():
    res = db.session.query(MLProject).order_by(
        desc(MLProject.created_at)).all()
    return render_template(
        'dashboard/history.html',
        history=res,
        active_dashboard="history")
