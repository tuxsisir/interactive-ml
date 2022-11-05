import pandas as pd
from flask import Blueprint, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy

from ..models import MLProject

pipeline_blueprint = Blueprint('pipeline', __name__)

db = SQLAlchemy()

@pipeline_blueprint.route('/',methods = ['GET'])
def pipeline_demo():
    return render_template('pipeline/workflow.html')


@pipeline_blueprint.route('<int:id>/cleaning',methods = ['GET'])
def pipeline_detail_cleaning(id):
    project = db.session.query(MLProject).get(id)
    df = pd.read_csv(f'uploads/datasets/{project.filename}')
    head = df.head()
    summary = df.describe()
    return render_template(
            'pipeline/data-cleaning.html',
            project=project,
            dataset=head.to_html(classes=('table table-hover'), table_id='dataset-table'),
            columns=head.columns.to_list(),
            progress=16.66,
            active_pipeline='cleaning',
            shape=df.shape)


@pipeline_blueprint.route('<int:id>/eda',methods = ['GET'])
def pipeline_detail_eda(id):
    """
    Exploratory Data Analysis
    """
    project = db.session.query(MLProject).get(id)
    df = pd.read_csv(f'uploads/datasets/{project.filename}')
    head = df.head()
    summary = df.describe()
    # df.hist()
    return render_template(
            'pipeline/eda.html',
            project=project,
            dataset=head.to_html(classes=('table')),
            columns=head.columns.to_list(),
            active_pipeline='eda',
            summary=summary.to_html(classes=('table table-hover'), table_id='dataset-table'),
            progress=33.32,
            shape=df.shape)


@pipeline_blueprint.route('<int:id>/scaling',methods = ['GET'])
def pipeline_detail_scaling(id):
    project = db.session.query(MLProject).get(id)
    df = pd.read_csv(f'uploads/datasets/{project.filename}')
    head = df.head()
    try:
        df.drop(columns='dteday', inplace=True)
    except:
        pass
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    mm = MinMaxScaler()
    minmax = mm.fit_transform(df)

    feature_mm = pd.DataFrame(minmax, index=df.index, columns=df.columns)
    feature_mm = feature_mm.reset_index(drop=True)
    feature_mm = feature_mm.head()
    return render_template(
            'pipeline/data-scaling.html',
            active_pipeline='scaling',
            progress=49.98,
            scaled_dataset=feature_mm.to_html(classes=('table table-hover'), table_id='dataset-table'),
            project=project)


@pipeline_blueprint.route('<int:id>/features',methods = ['GET'])
def pipeline_detail_features(id):
    project = db.session.query(MLProject).get(id)
    return render_template(
            'pipeline/feature-engineering.html',
            active_pipeline='features',
            progress=66.64,
            project=project)


@pipeline_blueprint.route('<int:id>/train-models',methods = ['GET'])
def pipeline_detail_train(id):
    project = db.session.query(MLProject).get(id)
    return render_template(
            'pipeline/train-models.html',
            active_pipeline='train',
            progress=83.30,
            project=project)


@pipeline_blueprint.route('<int:id>/predict',methods = ['GET'])
def pipeline_detail_predict(id):
    project = db.session.query(MLProject).get(id)
    return render_template(
            'pipeline/predict.html',
            active_pipeline='predict',
            progress=100,
            project=project)

