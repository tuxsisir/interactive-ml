import os
import uuid

from datetime import datetime

from flask import Blueprint, render_template, redirect, request, url_for
from models import MLProject
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
    return render_template('index.html', page_index=True)


@dashboard_blueprint.route("/dashboard/explore")
def explore():
    """
    Explore page
    """
    return render_template('dashboard/explore.html', page_index=True)


@dashboard_blueprint.route("/dashboard/home")
def dashboard():
    res = db.session.query(MLProject).order_by(desc(MLProject.created_at)).all()
    counts = {
            "draft": db.session.query(MLProject).filter_by(status = 'Draft').count(),
            "completed": db.session.query(MLProject).filter_by(status = 'Completed').count(),
            "in_progress": db.session.query(MLProject).filter_by(status = 'In Progress').count(),
        }
    return render_template(
            'dashboard/home.html',
            active_dashboard="home",
            counts=counts,
            projects=res)


@dashboard_blueprint.route('/<int:id>/edit-project',methods = ['GET', 'POST'])
def dashboard_edit_project(id):
    project = db.session.query(MLProject).get(id)
    if request.method == 'POST':
        project.title = request.form.get('title', None)
        project.description = request.form.get('description', None)
        project.status = request.form.get('btnradio', None)
        db.session.commit()
        return redirect('/dashboard/home')
    return render_template(
            'dashboard/upload.html',
            active_dashboard="upload",
            edit=True,
            project=project)


@dashboard_blueprint.route("/dashboard/upload", methods=["POST", "GET"])
def dashboard_upload():
    """
    Dashboard upload dataset
    """
    if request.method == 'POST':
        form_errors = {}
        title = request.form.get('title', None)
        description = request.form.get('description', None)
        status = request.form.get('btnradio', None)
        file = request.files['file']
        if not all([title, description, file]):
            form_errors['error'] = 'All fields are required.'
            return render_template('dashboard/upload.html', form_errors=form_errors,
                    active_dashboard="upload")
        secure_file_name = secure_filename(file.filename)

        # file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        file.save(os.path.join(basedir, 'uploads/datasets', secure_file_name))
        ml_project = MLProject(
                created_at=datetime.now(),
                created_by=1,
                title=title,
                description=description,
                filename=secure_file_name,
                filename_preferred='',
                status=status)
        db.session.add(ml_project)
        db.session.commit()
        return redirect('/dashboard/home')
    return render_template('dashboard/upload.html', active_dashboard="upload")


@dashboard_blueprint.route("/dashboard/history")
def dashboard_history():
    res = db.session.query(MLProject).order_by(desc(MLProject.created_at)).all()
    return render_template(
            'dashboard/history.html',
            history=res,
            active_dashboard="history")

