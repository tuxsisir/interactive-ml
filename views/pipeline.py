import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from flask import Blueprint, render_template, redirect, request, session, abort, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_required

from models import MLProject

# ML models
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
import matplotlib
matplotlib.use('Agg')

pipeline_blueprint = Blueprint('pipeline', __name__)

db = SQLAlchemy()
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pipeline_blueprint.route('/', methods=['GET'])
def pipeline_demo():
    return render_template('pipeline/workflow.html')

def get_project_user_df(id):
    """
    GET logged in user from session, project from url id and the original dataframe
    """
    current_user = session['twitter_oauth_token']['screen_name']
    project = db.session.query(MLProject).get(id)
    db.session.close()

    # READ original
    df = pd.read_csv(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename))
    return current_user, project, df



@pipeline_blueprint.route('<int:id>/cleaning', methods=['GET'])
@login_required
def pipeline_detail_cleaning(id):
    """
    DETAIL CLEANING PAGE
    """
    current_user, project, df = get_project_user_df(id)

    # READ cleaned
    df_cleaned = None
    df_cleaned_head = None
    df_cleaned_shape = None
    try:
        cleaned_url = os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
        df_cleaned = pd.read_csv(cleaned_url)
        df_cleaned_head = df_cleaned.head().to_html(classes=('table table-hover dataset-table'))
        df_cleaned_shape = df_cleaned.shape
    except Exception as e:
        print(e)

    df_original_head = df.head()

    # report null values
    null_values = df.isnull().sum().to_dict()
    null_values_cleaned = df_cleaned.isnull().sum().to_dict() if df_cleaned is not None else {}

    return render_template(
        'pipeline/data-cleaning.html',
        project=project,
        dataset=df_original_head.to_html(classes=('table table-hover dataset-table')),
        columns=df_original_head.columns.to_list(),
        progress=16.66,
        active_pipeline='cleaning',
        all_df=df.to_html(classes=('table table-hover dataset-table')),
        shape=df.shape,
        null_values=null_values,
        df_cleaned_head=df_cleaned_head,
        df_cleaned_shape=df_cleaned_shape,
        cleaned_null_values=null_values_cleaned)


@pipeline_blueprint.route('<int:id>/cleaning/perform-cleanup', methods=['POST'])
@login_required
def pipeline_perform_cleaning(id):
    """
    PROCESS CLEANING
    """
    project = db.session.query(MLProject).get(id)
    current_user = session['twitter_oauth_token']['screen_name']
    # READ original
    df = pd.read_csv(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}',
        project.filename))
    filtered_columns = request.form.getlist('filtered_columns')
    null_values = request.form.get('null_values', None, type=int)

    if null_values == 3:
        print('cleaned remove null')
        df.dropna(inplace=True)
    df.drop(columns=filtered_columns, inplace=True)

    df.to_csv(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}',
        project.filename.replace('_original', '_cleaned')), index=False)

    return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))


@pipeline_blueprint.route('<int:id>/eda', methods=['GET', 'POST'])
@login_required
def pipeline_detail_eda(id):
    """
    Exploratory Data Analysis
    """
    current_user, project, all_df = get_project_user_df(id)
    dataset_summary = True
    hist_plot = False
    correlation = False
    hist_col = ''
    plot_url = ''
    if request.method == "POST":
        active_vis = request.form.get('visualization', None)
        dataset_summary = True if active_vis == 'summary' else False
        hist_plot = True if active_vis == 'histogram' else False
        correlation = True if active_vis == 'correlation' else False
        if hist_plot and not request.form.get('hist_col'):
            flash('Please select columns for histogram distribution.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_eda', id=id))
        hist_col = request.form.get('hist_col')

    if not project:
        abort(404)

    try:
        cleaned_url = os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
        df = pd.read_csv(cleaned_url)
    except FileNotFoundError:
        return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    head = df.head()
    summary = df.describe()

    if hist_plot:
        sns.displot(df[hist_col])
        plt.title(f'{hist_col} Distribution')
        plot_url = f"static/uploads/plots/{current_user}"
        Path(basedir, plot_url).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(basedir, plot_url, 'hist.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.3)
    else:
        hist_col = None

    if correlation:
        plt.figure(figsize=(6,6))
        sns.heatmap(df.corr(), annot=True, cmap="Blues")
        plot_url = f"static/uploads/plots/{current_user}"
        plt.savefig(os.path.join(basedir, plot_url, 'correlation.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.2)


    return render_template(
        'pipeline/eda.html',
        project=project,
        dataset=head.to_html(classes=('table table-hover dataset-table')),
        columns=head.columns.to_list(),
        active_pipeline='eda',
        summary=summary.to_html(classes=('table table-hover dataset-table')),
        progress=33.32,
        dataset_summary=dataset_summary,
        hist_summary=hist_plot,
        correlation=correlation,
        plot_url=f"{plot_url[7:]}/{'hist.png' if hist_plot else 'correlation.png'}",
        hist_col=hist_col,
        shape=df.shape,
        all_df=all_df.to_html(classes=('table table-hover dataset-table')),
        )


@pipeline_blueprint.route('<int:id>/scaling', methods=['GET', 'POST'])
@login_required
def pipeline_detail_scaling(id):
    """
    APPLY different scaler to the cleaned dataset
    """
    current_user, project, all_df = get_project_user_df(id)

    try:
        cleaned_url = os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
        df = pd.read_csv(cleaned_url)
    except Exception as e:
        flash('Please apply cleaning data first -- .', 'success')
        return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    head = df.head()
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    mm = MinMaxScaler()
    rs = RobustScaler()
    ss = StandardScaler()

    required_scaler = None
    revert_scaler = None
    if request.method == "POST":
        required_scaler = request.form.get('scaler', None)
        revert_scaler = bool(request.form.get('reset_scaling', None))
        print(revert_scaler)

    if revert_scaler:
        required_scaler = None
        print(cleaned_url.replace('_cleaned', '_scaled'))
        os.remove(cleaned_url.replace('_cleaned', '_scaled'))

    if required_scaler and required_scaler == 'robust':
        robust = rs.fit_transform(df)
        feature_scaled = pd.DataFrame(robust, index=df.index, columns=df.columns)
    elif required_scaler and required_scaler == 'standard':
        robust = ss.fit_transform(df)
        feature_scaled = pd.DataFrame(robust, index=df.index, columns=df.columns)
    elif required_scaler and required_scaler == 'minmax':
        minmax = mm.fit_transform(df)
        feature_scaled = pd.DataFrame(minmax, index=df.index, columns=df.columns)
    else:
        feature_scaled = head

    feature_scaled = feature_scaled.reset_index(drop=True)

    if required_scaler:
        feature_scaled.to_csv(os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}',
            project.filename.replace('_original', '_scaled')), index=False)


    feature_scaled = feature_scaled.head()

    return render_template(
        'pipeline/data-scaling.html',
        active_pipeline='scaling',
        progress=49.98,
        scaled_dataset=feature_scaled.to_html(classes=('table table-hover dataset-table')),
        all_df=all_df.to_html(classes=('table table-hover dataset-table')),
        required_scaler=required_scaler,
        project=project)


@pipeline_blueprint.route('<int:id>/train-test', methods=['GET', 'POST'])
@login_required
def pipeline_detail_train_test(id):
    """
    Train Test SPLIT
    """
    current_user, project, all_df = get_project_user_df(id)

    try:
        scaled_url = os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_scaled'))
        df = pd.read_csv(scaled_url)
    except Exception as e:
        # GET CLEANED IF NOT scaled dataset found
        print(e)
        try:
            cleaned_url = os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except FileNotFoundError:
            flash('Please apply cleaning data first.', 'success')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    head = df.head()

    return render_template(
        'pipeline/train-test.html',
        active_pipeline='train-test',
        progress=59.64,
        project=project,
        active_dataset=head.to_html(classes=('table table-hover dataset-table')),
        columns=head.columns.to_list(),
        shape=all_df.shape,
        all_df=all_df.to_html(classes=('table table-hover dataset-table')))


@pipeline_blueprint.route('<int:id>/features', methods=['GET', 'POST'])
@login_required
def pipeline_detail_features(id):
    """
    Features selection
    """
    current_user, project, all_df = get_project_user_df(id)
    supervised_learning_task = ''
    predictor = ''
    regressor_features = {'random_forest_regressor': {}, 'decision_tree_regressor': {}}
    if request.method == "GET":
        supervised_learning_task = request.args.get('learning')

    try:
        scaled_url = os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_scaled'))
        df = pd.read_csv(scaled_url)
    except Exception as e:
        # GET CLEANED IF NOT scaled dataset found
        print(e)
        try:
            cleaned_url = os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except FileNotFoundError:
            flash('Please apply cleaning data first.', 'success')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    head = df.head()

    if request.method == "POST":
        predictor = request.form.get('predictor', None)
        supervised_learning_task = request.form.get('learning', None)
        if predictor:
            X = df.drop(columns=[predictor])
            y = df[predictor]
        else:
            flash('Please select predictor value.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_features', id=id))

        if supervised_learning_task == 'regression':
            model = RandomForestRegressor()
            model.fit(X, y)
            score = round(model.score(X, y) * 100, 2)
            regressor_features['random_forest_regressor']['score'] = score
            regressor_features['random_forest_regressor']['features'] = dict(zip(X.columns, model.feature_importances_))

            model = DecisionTreeRegressor()
            model.fit(X, y)
            score = round(model.score(X, y) * 100, 2)
            regressor_features['decision_tree_regressor']['score'] = score
            regressor_features['decision_tree_regressor']['features'] = dict(zip(X.columns, model.feature_importances_))

        elif supervised_learning_task == 'classification':
            if len(y.unique() > 2):
                flash('Unable to continue further with classification. Classification requires predictor to be in binary.', 'danger')
                return redirect(url_for('pipeline.pipeline_detail_features', id=id))
            model_logregression = SelectFromModel(estimator=LogisticRegression(max_iter=5000)).fit(X, y)
            # coefficients = model_logregression.estimator_.coef_

        try:
            pass
        except Exception as e:
            flash('Please remove any NaN values.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    return render_template(
        'pipeline/feature-engineering.html',
        active_pipeline='features',
        progress=66.64,
        project=project,
        active_dataset=head.to_html(classes=('table table-hover dataset-table')),
        columns=head.columns.to_list(),
        shape=df.shape,
        all_df=all_df.to_html(classes=('table table-hover dataset-table')),
        supervised_learning_task=supervised_learning_task,
        regressor_features=regressor_features,
        predictor=predictor)


@pipeline_blueprint.route('<int:id>/train-models', methods=['GET'])
@login_required
def pipeline_detail_train(id):
    current_user = session['twitter_oauth_token']['screen_name']
    project = db.session.query(MLProject).get(id)
    classifiers = ["KNN", "Naive Bayes", "Decision Tree",
                   "Random Forest", "Ada BOOST", "XGBOOST"]
    return render_template(
        'pipeline/train-models.html',
        active_pipeline='train',
        progress=83.30,
        classifiers=classifiers,
        project=project)


@pipeline_blueprint.route('<int:id>/predict', methods=['GET'])
@login_required
def pipeline_detail_predict(id):
    project = db.session.query(MLProject).get(id)
    return render_template(
        'pipeline/predict.html',
        active_pipeline='predict',
        progress=100,
        project=project)


@pipeline_blueprint.route('delete', methods=['POST'])
@login_required
def pipeline_delete():
    id = request.form.get('id', None)
    current_user, project, _ = get_project_user_df(id)
    path = Path(os.path.join(basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}'))
    shutil.rmtree(path)
    db.session.query(MLProject).filter(MLProject.id == id).delete()
    db.session.commit()
    db.session.close()
    return redirect('/dashboard/home')
