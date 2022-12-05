import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
from flask import Blueprint, render_template, redirect, request, session, abort, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_required
from sqlalchemy.orm.attributes import flag_modified

from models import MLProject, MLProjectConfig

# ML models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sqlalchemy.exc import NoResultFound

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
    try:
        current_user = session['twitter_oauth_token']['screen_name']
        project = db.session.query(MLProject).get(id)
        config = db.session.query(MLProjectConfig).filter(
            MLProjectConfig.ml_project == id).one()
    except NoResultFound:
        abort(404)
    db.session.close()
    # READ original
    df = pd.read_csv(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename))
    return current_user, project, config, df


@pipeline_blueprint.route('<int:id>/cleaning', methods=['GET'])
@login_required
def pipeline_detail_cleaning(id):
    """
    DETAIL CLEANING PAGE
    """
    current_user, project, config, df = get_project_user_df(id)

    # READ cleaned
    df_cleaned = None
    df_cleaned_head = None
    df_cleaned_shape = None
    try:
        cleaned_url = os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
        df_cleaned = pd.read_csv(cleaned_url)
        df_cleaned_head = df_cleaned.head().to_html(
            classes=('table table-hover dataset-table'))
        df_cleaned_shape = df_cleaned.shape
    except Exception as e:
        print(e)

    df_original_head = df.head()

    # report null values
    null_values = df.isnull().sum().to_dict()
    null_values_cleaned = df_cleaned.isnull().sum(
    ).to_dict() if df_cleaned is not None else {}

    return render_template(
        'pipeline/data-cleaning.html',
        project=project,
        dataset=df_original_head.to_html(
            classes=('table table-hover dataset-table')),
        columns=df_original_head.columns.to_list(),
        progress=16.66,
        active_pipeline='cleaning',
        all_df=df.to_html(classes=('table table-hover dataset-table')),
        shape=df.shape,
        null_values=null_values,
        df_cleaned_head=df_cleaned_head,
        df_cleaned_shape=df_cleaned_shape,
        cleaned_null_values=null_values_cleaned,
        project_config=config)


@pipeline_blueprint.route('<int:id>/cleaning/perform-cleanup', methods=['POST'])
@login_required
def pipeline_perform_cleaning(id):
    """
    PROCESS CLEANING
    """
    project = db.session.query(MLProject).get(id)
    current_user = session['twitter_oauth_token']['screen_name']
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()
    # READ original
    df = pd.read_csv(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}',
        project.filename))
    filtered_columns = request.form.getlist('filtered_columns')
    original_columns_len = len(df.columns.to_list())
    if original_columns_len - len(filtered_columns) <= 1:
        db.session.commit()
        flash('Please keep at least two columns in the dataset.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))
    null_values = request.form.get('null_values', None, type=int)
    cleaning_config = {'cleaning': {
        'fields': filtered_columns, 'nan': null_values}}
    project_config.config = cleaning_config
    db.session.commit()

    if null_values == 1:
        df.fillna(df.mean(), inplace=True)
    elif null_values == 2:
        df.fillna(df.median(), inplace=True)
    elif null_values == 3:
        df.dropna(inplace=True)
    df.drop(columns=filtered_columns, inplace=True)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()

    df.to_csv(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}',
        project.filename.replace('_original', '_cleaned')), index=False)

    db.session.close()
    flash('Successfully cleaned data.', 'success')
    return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))


@pipeline_blueprint.route('<int:id>/eda', methods=['GET', 'POST'])
@login_required
def pipeline_detail_eda(id):
    """
    Exploratory Data Analysis
    """
    current_user, project, config, all_df = get_project_user_df(id)
    dataset_summary = True
    hist_plot = False
    correlation = False
    pairplot = False
    boxplot = False
    hist_col = ''
    plot_url = ''
    if request.method == "POST":
        active_vis = request.form.get('visualization', None)
        dataset_summary = True if active_vis == 'summary' else False
        hist_plot = True if active_vis == 'histogram' else False
        correlation = True if active_vis == 'correlation' else False
        pairplot = True if active_vis == 'pairplot' else False
        boxplot = True if active_vis == 'boxplot' else False
        if hist_plot and not request.form.get('hist_col'):
            db.session.close()
            flash('Please select columns for histogram distribution.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_eda', id=id))
        hist_col = request.form.get('hist_col')

    if not project:
        abort(404)

    try:
        cleaned_url = os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
        df = pd.read_csv(cleaned_url)
    except FileNotFoundError:
        db.session.close()
        flash('Please clean the dataset to continue.', 'danger')
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
        plot_url = f"uploads/plots/{current_user}/hist.png"
        flash('Successfully generated histogram.', 'success')
    else:
        hist_col = None

    if correlation:
        plt.figure(figsize=(10, 10))
        sns.heatmap(df.corr(), annot=True, cmap="Blues")
        plot_url = f"static/uploads/plots/{current_user}"
        Path(basedir, plot_url).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(basedir, plot_url, 'correlation.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.2)
        plot_url = f"uploads/plots/{current_user}/correlation.png"
        flash('Successfully generated correlation.', 'success')

    if pairplot:
        sns.pairplot(df.iloc[:,0:4], height=2.5)
        plt.tight_layout()
        plot_url = f"static/uploads/plots/{current_user}"
        Path(basedir, plot_url).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(basedir, plot_url, 'pairplot.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.2)
        plot_url = f"uploads/plots/{current_user}/pairplot.png"
        flash('Successfully generated pairplot.', 'success')

    if boxplot:
        plt.figure().clear()
        sns.boxplot(data=df.iloc[:, 0:4])
        plot_url = f"static/uploads/plots/{current_user}"
        Path(basedir, plot_url).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(basedir, plot_url, 'boxplot.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.2)
        plot_url = f"uploads/plots/{current_user}/boxplot.png"
        flash('Successfully generated boxplot.', 'success')

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
        pairplot=pairplot,
        boxplot=boxplot,
        plot_url=plot_url,
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
    current_user, project, config, all_df = get_project_user_df(id)
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()

    # READ original
    all_df = pd.read_csv(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}',
        project.filename))

    try:
        cleaned_url = os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
        df = pd.read_csv(cleaned_url)
    except Exception as e:
        db.session.commit()
        flash('Please clean the dataset to continue.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    if not all(df.applymap(np.isreal).all(1)):
        flash('Please drop any values that is not numeric.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    head = df.head()
    mm = MinMaxScaler()
    rs = RobustScaler()
    ss = StandardScaler()

    required_scaler = project_config.config.get('scaler', None)
    revert_scaler = None
    if request.method == "POST":
        required_scaler = request.form.get('scaler', None)
        revert_scaler = bool(request.form.get('reset_scaling', None))

    if revert_scaler:
        required_scaler = None
        try:
            os.remove(cleaned_url.replace('_cleaned', '_scaled'))
        except FileNotFoundError:
            db.session.commit()
            flash('File not found.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_scaling', id=id))

    if required_scaler and required_scaler == 'robust':
        robust = rs.fit_transform(df)
        feature_scaled = pd.DataFrame(
            robust, index=df.index, columns=df.columns)
    elif required_scaler and required_scaler == 'standard':
        robust = ss.fit_transform(df)
        feature_scaled = pd.DataFrame(
            robust, index=df.index, columns=df.columns)
    elif required_scaler and required_scaler == 'minmax':
        minmax = mm.fit_transform(df)
        feature_scaled = pd.DataFrame(
            minmax, index=df.index, columns=df.columns)
    else:
        feature_scaled = head

    feature_scaled = feature_scaled.reset_index(drop=True)

    if required_scaler:
        flash('Successfully generated scaled dataset.', 'success')
        feature_scaled.to_csv(os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}',
            project.filename.replace('_original', '_scaled')), index=False)

    feature_scaled = feature_scaled.head()

    existing_config = project_config.config
    existing_config['scaler'] = required_scaler
    project_config.config = existing_config
    flag_modified(project_config, "config")
    db.session.commit()
    db.session.close()
    return render_template(
        'pipeline/data-scaling.html',
        active_pipeline='scaling',
        progress=49.98,
        scaled_dataset=feature_scaled.to_html(
            classes=('table table-hover dataset-table')),
        all_df=all_df.to_html(classes=('table table-hover dataset-table')),
        required_scaler=required_scaler,
        project=project)


@pipeline_blueprint.route('<int:id>/train-test', methods=['GET', 'POST'])
@login_required
def pipeline_detail_train_test(id):
    """
    Train Test SPLIT
    """
    current_user, project, config, all_df = get_project_user_df(id)
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()

    try:
        scaled_url = os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_scaled'))
        df = pd.read_csv(scaled_url)
    except Exception as e:
        # GET CLEANED IF NOT scaled dataset found
        print(e)
        try:
            cleaned_url = os.path.join(
                basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except FileNotFoundError:
            db.session.close()
            flash('Please clean the dataset to continue.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    if not all(df.applymap(np.isreal).all(1)):
        db.session.close()
        flash('Please drop any values that is not numeric.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    head = df.head()

    test_size = project_config.config.get(
        'train_test', {'test_size': 0.20})['test_size']
    random_state = project_config.config.get(
        'train_test', {'random_state': 42})['random_state']
    predictor = project_config.config.get('predictor', None)

    if request.method == "POST":
        print(request.form)
        test_size = request.form.get('test_size', 0.13, type=float)
        random_state = request.form.get('random_state', 42, type=int)
        predictor = request.form.get('predictor', None)
        if not predictor:
            db.session.close()
            flash('Please select predictor value.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_train_test', id=id))
        if test_size > 1 or test_size < 0:
            db.session.close()
            flash('Please select test_size between 0 to 1.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_train_test', id=id))
        existing_config = project_config.config
        existing_config['train_test'] = {
            'test_size': test_size, 'random_state': random_state}
        existing_config['predictor'] = predictor
        print(predictor)
        project_config.config = existing_config
        flag_modified(project_config, "config")
        db.session.commit()
        db.session.close()
        flash('Successfully generated train/test split with predictor.', 'success')

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    db.session.close()
    return render_template(
        'pipeline/train-test.html',
        active_pipeline='train-test',
        progress=59.64,
        project=project,
        test_size=test_size,
        random_state=random_state,
        predictor=predictor,
        active_dataset=head.to_html(
            classes=('table table-hover dataset-table')),
        columns=head.columns.to_list(),
        shape=all_df.shape,
        all_df=all_df.to_html(classes=('table table-hover dataset-table')))


@pipeline_blueprint.route('<int:id>/features', methods=['GET', 'POST'])
@login_required
def pipeline_detail_features(id):
    """
    Features selection
    """
    current_user, project, config, all_df = get_project_user_df(id)
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()

    train_test_config = project_config.config.get('train_test', None)
    predictor = project_config.config.get('predictor', '')
    if not train_test_config or not predictor:
        db.session.close()
        flash('Please configure train-test and predictor to continue.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_train_test', id=id))

    supervised_learning_task = project_config.config.get(
        'supervised_learning_task', '')
    regressor_features = []
    classifier_features = []
    selected_features = project_config.config.get('selected_features', [])
    if request.method == "GET":
        supervised_learning_task = request.args.get('learning')

    try:
        scaled_url = os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_scaled'))
        df = pd.read_csv(scaled_url)
    except Exception as e:
        # GET CLEANED IF NOT scaled dataset found
        try:
            cleaned_url = os.path.join(
                basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except Exception as e:
            db.session.close()
            flash('Please clean the dataset to continue.', 'success')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    print(supervised_learning_task)
    if supervised_learning_task == 'classification':
        # REPLACE DF WITH cleaned one rather than scaled, last minute bug
        try:
            cleaned_url = os.path.join(
                basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except Exception as e:
            db.session.close()
            flash('Please clean the dataset to continue.', 'success')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    head = df.head()

    if request.method == "POST":
        supervised_learning_task = request.form.get('learning', None)
        selected_features = request.form.getlist('selected_features', None)

        # save to db
        existing_config = project_config.config
        existing_config['supervised_learning_task'] = supervised_learning_task
        existing_config['selected_features'] = selected_features
        project_config.config = existing_config
        flag_modified(project_config, "config")
        db.session.commit()
        flash('Successfully saved features for training model.', 'success')

    X = df.drop(columns=[predictor])
    y = df[predictor]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_config['test_size'],
        random_state=train_test_config['random_state'])

    if supervised_learning_task == 'regression':
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        score = round(model.score(X_train, y_train) * 100, 2)
        regressor_features.append(
            ['Algorithm'] + X_train.columns.to_list() + ['Score'])
        regressor_features.append(['RandomForestRegressor'] + list(
            map(lambda n: round(n * 100, 2), model.feature_importances_)) + [score])

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        score = round(model.score(X_train, y_train) * 100, 2)
        regressor_features.append(['DecisionTreeRegressor'] + list(
            map(lambda n: round(n * 100, 2), model.feature_importances_)) + [score])

    elif supervised_learning_task == 'classification':
        print(len(y.unique()))
        if len(y.unique()) > 2:
            db.session.close()
            flash(
                'Unable to continue further with classification. Classification requires predictor to be in binary.', 'danger')
            return redirect(url_for('pipeline.pipeline_detail_features', id=id))

        model = RandomForestClassifier(n_estimators=340)
        model.fit(X_train, y_train)
        score = round(model.score(X_train, y_train) * 100, 2)
        classifier_features.append(
            ['Algorithm'] + X_train.columns.to_list() + ['Score'])
        classifier_features.append(['RandomForestClassifier'] + list(
            map(lambda n: round(n * 100, 2), model.feature_importances_)) + [score])

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        score = round(model.score(X_train, y_train) * 100, 2)
        classifier_features.append(['DecisionTreeClassifier'] + list(
            map(lambda n: round(n * 100, 2), model.feature_importances_)) + [score])

    db.session.close()
    return render_template(
        'pipeline/feature-engineering.html',
        active_pipeline='features',
        progress=66.64,
        project=project,
        active_dataset=head.to_html(
            classes=('table table-hover dataset-table')),
        columns=head.columns.to_list(),
        shape=df.shape,
        all_df=all_df.to_html(classes=('table table-hover dataset-table')),
        supervised_learning_task=supervised_learning_task,
        regressor_features=regressor_features,
        classifier_features=classifier_features,
        selected_features=selected_features,
        predictor=predictor)


@pipeline_blueprint.route('<int:id>/train-models', methods=['GET', 'POST'])
@login_required
def pipeline_detail_train(id):
    current_user, project, config, all_df = get_project_user_df(id)
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()

    selected_features = project_config.config.get('selected_features', None)

    if not selected_features:
        db.session.close()
        flash('Please finalize features to train model.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_features', id=id))

    learning = project_config.config.get('supervised_learning_task', '')
    test_size = project_config.config.get('train_test')['test_size']
    random_state = project_config.config.get('train_test')['random_state']
    predictor = project_config.config.get('predictor')
    default_regression_model_values = {
        'LinearRegression': None,
        'Ridge': 0.1,
        'Lasso': 0.5,
        'DecisionTreeRegressor': 42,
        'BayesianRidge': None,
        'finalized_model': '',
        'history': {'features': [], 'predicted': []}
    }
    default_classifier_model_values = {
        'DecisionTreeClassifier': 5,
        'RandomForestClassifier': {
            'n_estimators': 500,
            'max_depth': 5,
            'max_leaf_nodes': 16
        },
        'XGBClassifier': {
            'base_score': 0.5,
            'max_depth': 5,
            'validate_parameters': 1,
            },
        'finalized_model': '',
        'history': {'features': [], 'predicted': []}
    }
    regression_model_values = project_config.config.get(
        'regression_model_values', default_regression_model_values)
    classifier_model_values = project_config.config.get(
        'classifier_model_values', default_classifier_model_values)

    classifiers = ["KNN", "Naive Bayes", "Decision Tree",
                   "Random Forest", "Ada BOOST", "XGBOOST"]

    try:
        scaled_url = os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_scaled'))
        df = pd.read_csv(scaled_url)
    except Exception as e:
        # GET CLEANED IF NOT scaled dataset found
        print(e)
        try:
            cleaned_url = os.path.join(
                basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except FileNotFoundError:
            db.session.commit()
            flash('Please clean the dataset to continue.', 'success')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    if learning == 'classification':
        # REPLACE DF WITH cleaned one rather than scaled, last minute bug
        try:
            cleaned_url = os.path.join(
                basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except Exception as e:
            db.session.close()
            flash('Please clean the dataset to continue.', 'success')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    if request.method == "POST":
        if learning == 'regression':
            regression_model_values['Ridge'] = request.form.get(
                'ridge', 0.5, type=float)
            regression_model_values['Lasso'] = request.form.get(
                'lasso', 0.1, type=float)
            regression_model_values['DecisionTreeRegressor'] = request.form.get(
                'dtr', 42, type=int)
            existing_config = project_config.config
            existing_config['regression_model_values'] = regression_model_values
            project_config.config = existing_config
            flag_modified(project_config, "config")
            db.session.commit()
            flash('Successfully tuned models on Regression.', 'success')
        elif learning == 'classification':
            classifier_model_values['DecisionTreeClassifier'] = request.form.get(
                'dtr_max_depth', 5, type=int)
            # RandomForestClassifier
            classifier_model_values['RandomForestClassifier']['n_estimators'] = request.form.get(
                'rfc_n_estimators', 500, type=int)
            classifier_model_values['RandomForestClassifier']['max_depth'] = request.form.get(
                'rfc_max_depth', 5, type=int)
            classifier_model_values['RandomForestClassifier']['max_leaf_nodes'] = request.form.get(
                'rfc_max_leaf_nodes', 16, type=int)
            # XGB
            classifier_model_values['XGBClassifier']['base_score'] = request.form.get(
                'xgb_base_score', 0.5, type=float)
            classifier_model_values['XGBClassifier']['max_depth'] = request.form.get(
                'xgb_max_depth', 5, type=int)
            classifier_model_values['XGBClassifier']['validate_parameters'] = request.form.get(
                'xgb_validate_params', 1, type=int)
            existing_config = project_config.config
            existing_config['classifier_model_values'] = classifier_model_values
            project_config.config = existing_config
            flag_modified(project_config, "config")
            db.session.commit()
            flash('Successfully tuned models on Classification.', 'success')

    X = df[selected_features]
    y = df[predictor]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    regression_results = {}
    classifier_results = {}
    if learning == 'regression':
        def linear_model_metrics(model, X_test, y_test, decimals=5, X_train=X_train, y_train=y_train):
            start = datetime.now()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = np.round(mean_squared_error(y_test, y_pred), decimals)
            r2 = np.round(r2_score(y_test, y_pred) * 100, decimals)
            return {'mean_squared_error': mse, 'r_squared': r2, 'time': (datetime.now() - start).seconds}

        lr_model = LinearRegression()
        ridge_model = Ridge(alpha=regression_model_values['Ridge'])
        lasso_model = Lasso(alpha=regression_model_values['Lasso'])
        dtr = DecisionTreeRegressor(
            random_state=regression_model_values['DecisionTreeRegressor'])
        bayridge_model = BayesianRidge()

        regression_results = {
            'LinearRegression': linear_model_metrics(lr_model, X_test, y_test),
            'Ridge': linear_model_metrics(ridge_model, X_test, y_test),
            'Lasso': linear_model_metrics(lasso_model, X_test, y_test),
            'DecisionTreeRegressor': linear_model_metrics(dtr, X_test, y_test),
            'BayesianRidge': linear_model_metrics(bayridge_model, X_test, y_test)
        }
        regression_results = dict(sorted(regression_results.items(
        ), key=lambda item: item[1]['r_squared'], reverse=True))
    elif learning == 'classification':
        classifiers = [
            GaussianNB(),
            KNeighborsClassifier(),
            DecisionTreeClassifier(max_depth=classifier_model_values['DecisionTreeClassifier']),
            RandomForestClassifier(
                n_estimators=classifier_model_values['RandomForestClassifier']['n_estimators'],
                max_depth=classifier_model_values['RandomForestClassifier']['max_depth'],
                max_leaf_nodes=classifier_model_values['RandomForestClassifier']['max_leaf_nodes']),
            AdaBoostClassifier(),
            XGBClassifier(
                base_score=classifier_model_values['XGBClassifier']['base_score'],
                max_depth=classifier_model_values['XGBClassifier']['max_depth'],
                validate_parameters=classifier_model_values['XGBClassifier']['validate_parameters'])
        ]
        for clf in classifiers:
            print(X_train)
            print(y_train)
            clf_model = clf.fit(X_train, y_train)
            y_pred = clf_model.predict(X_test)
            score = round(accuracy_score(y_test, y_pred) * 100, 2)
            classifier_results[clf_model.__str__().split("(")[0]] = score
        classifier_results = dict(sorted(classifier_results.items(), key=lambda item: item[1], reverse=True))

    head = X.head()

    db.session.close()
    return render_template(
        'pipeline/train-models.html',
        active_pipeline='train',
        progress=83.30,
        active_dataset=head.to_html(
            classes=('table table-hover dataset-table')),
        columns=head.columns.to_list(),
        shape=df.shape,
        predictor_dataset=y.to_frame().head().to_html(
            classes=('table table-hover dataset-table')),
        classifiers=classifier_results,
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        project=project,
        learning=learning,
        regression_model_values=regression_model_values,
        classifier_model_values=classifier_model_values,
        all_df=all_df.to_html(classes=('table table-hover dataset-table')),
        predictor=predictor,
        regression_results=regression_results)


@pipeline_blueprint.route('<int:id>/regression/model/finalize', methods=['POST'])
@login_required
def pipeline_finalize_regression_model(id):
    """
    FINALIZE REGRESSION MODEL
    """
    project = db.session.query(MLProject).get(id)
    current_user = session['twitter_oauth_token']['screen_name']
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()

    regression_model_values = project_config.config.get(
        'regression_model_values', None)

    if not regression_model_values:
        db.session.close()
        flash('Please tune the regression model values first.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_train', id=id))

    if request.method == "POST":
        finalized_model = request.form.get('finalized_model', '')
        existing_config = project_config.config
        existing_config['regression_model_values']['finalized_model'] = finalized_model
        project_config.config = existing_config
        flag_modified(project_config, "config")
        db.session.commit()

    db.session.close()
    flash('Successfully selected regression model!', 'success')
    return redirect(url_for('pipeline.pipeline_detail_train', id=id))


@pipeline_blueprint.route('<int:id>/classifier/model/finalize', methods=['POST'])
@login_required
def pipeline_finalize_classifier_model(id):
    project = db.session.query(MLProject).get(id)
    current_user = session['twitter_oauth_token']['screen_name']
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()

    classifier_model_values = project_config.config.get(
        'classifier_model_values', None)

    if not classifier_model_values:
        db.session.close()
        flash('Please tune the classifier model values first.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_train', id=id))

    if request.method == "POST":
        finalized_model = request.form.get('finalized_model', '')
        existing_config = project_config.config
        existing_config['classifier_model_values']['finalized_model'] = finalized_model
        project_config.config = existing_config
        flag_modified(project_config, "config")
        db.session.commit()

    db.session.close()
    flash('Successfully selected classifier model!', 'success')
    return redirect(url_for('pipeline.pipeline_detail_train', id=id))


@pipeline_blueprint.route('<int:id>/predict', methods=['GET', 'POST'])
@login_required
def pipeline_detail_predict(id):
    """
    PREDICTIONS
    """
    current_user, project, config, all_df = get_project_user_df(id)
    project_config = db.session.query(MLProjectConfig).filter(
        MLProjectConfig.ml_project == id).one()
    plot_url = f"static/uploads/plots/{current_user}"

    # fetch cleaned data for predictions
    try:
        cleaned_url = os.path.join(
            basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
        df = pd.read_csv(cleaned_url)
    except FileNotFoundError:
        db.session.close()
        flash('Please clean the dataset to continue.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    selected_features = project_config.config.get('selected_features', None)

    if not selected_features:
        db.session.close()
        flash('Please finalize features to train model.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_features', id=id))

    supervised_learning_task = project_config.config.get(
        'supervised_learning_task', None)
    predictor = project_config.config.get('predictor', None)
    regression_model_values = project_config.config.get(
        'regression_model_values', None)
    classifier_model_values = project_config.config.get(
        'classifier_model_values', None)

    test_size = project_config.config.get('train_test')['test_size']
    random_state = project_config.config.get('train_test')['random_state']

    if not any([regression_model_values, classifier_model_values]):
        db.session.close()
        flash('Please finalize model for prediction.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_train', id=id))

    if supervised_learning_task == 'regression' and not regression_model_values.get('finalized_model', None):
        db.session.close()
        flash('Please finalize model for prediction.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_train', id=id))
    elif supervised_learning_task == 'classification' and not classifier_model_values.get('finalized_model', None):
        db.session.close()
        flash('Please finalize model for prediction.', 'danger')
        return redirect(url_for('pipeline.pipeline_detail_train', id=id))


    if supervised_learning_task == 'classification':
        # REPLACE DF WITH cleaned one rather than scaled, last minute bug
        try:
            cleaned_url = os.path.join(
                basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}', project.filename.replace('_original', '_cleaned'))
            df = pd.read_csv(cleaned_url)
        except Exception as e:
            db.session.close()
            flash('Please clean the dataset to continue.', 'success')
            return redirect(url_for('pipeline.pipeline_detail_cleaning', id=id))

    to_predict_features = []
    predicted = []
    selected_model = ''
    if supervised_learning_task == 'regression':
        selected_model = regression_model_values['finalized_model']
        if request.method == "POST":
            for feature in selected_features:
                feature_val = request.form.get(feature, None)
                if not feature_val:
                    flash('Please enter all the required values to predict.', 'danger')
                    return redirect(url_for('pipeline.pipeline_detail_predict', id=id))
                to_predict_features.append(request.form.get(feature, 0, type=int))
            lr_model = LinearRegression()
            ridge_model = Ridge(alpha=regression_model_values['Ridge'])
            lasso_model = Lasso(alpha=regression_model_values['Lasso'])
            dtr = DecisionTreeRegressor(
                random_state=regression_model_values['DecisionTreeRegressor'])
            bayridge_model = BayesianRidge()

            model_mapper = {
                'LinearRegression': lr_model,
                'Ridge': ridge_model,
                'Lasso': lasso_model,
                'DecisionTreeRegressor': dtr,
                'BayesianRidge': bayridge_model
            }

            model = model_mapper[selected_model]
            X = df[selected_features]
            y = df[predictor]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predicted = model.predict([to_predict_features])

            # generate linear prediction actual vs predicted
            x_ax = range(len(predictions))
            plt.figure().clear()
            plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
            plt.plot(x_ax, predictions, lw=0.8, color="red", label="predicted")
            plt.legend()

            Path(basedir, plot_url).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(basedir, plot_url, 'linear-actual-predicted.png'),
                        dpi=300, bbox_inches='tight', pad_inches=0.2)
            # SAVE HISTORY
            existing_config = project_config.config
            prediction_history = existing_config['regression_model_values']['history']
            feature_text = ''
            prediction_zip = dict(zip(selected_features, to_predict_features))
            for key, val in  prediction_zip.items():
                feature_text += f"{key}: {val}, "
            prediction_history['features'].insert(0, feature_text)
            prediction_history['predicted'].insert(0, str(predicted[0]))
            existing_config['regression_model_values']['history'] = prediction_history
            flag_modified(project_config, "config")
            db.session.commit()
    elif supervised_learning_task == 'classification':
        selected_model = classifier_model_values['finalized_model']
        gaussian_nb = GaussianNB()
        knn_classifier = KNeighborsClassifier()
        dtr_classifier = DecisionTreeClassifier(max_depth=classifier_model_values['DecisionTreeClassifier']),
        random_forest_classifier = RandomForestClassifier(
                n_estimators=classifier_model_values['RandomForestClassifier']['n_estimators'],
                max_depth=classifier_model_values['RandomForestClassifier']['max_depth'],
                max_leaf_nodes=classifier_model_values['RandomForestClassifier']['max_leaf_nodes'])
        adaboost_classifier = AdaBoostClassifier()
        xgb_classifier = XGBClassifier(
                base_score=classifier_model_values['XGBClassifier']['base_score'],
                max_depth=classifier_model_values['XGBClassifier']['max_depth'],
                validate_parameters=classifier_model_values['XGBClassifier']['validate_parameters'])
        model_mapper = {
                'GaussianNB': gaussian_nb,
                'KNeighborsClassifier': knn_classifier,
                'DecisionTreeClassifier': dtr_classifier,
                'RandomForestClassifier': random_forest_classifier,
                'AdaBoostClassifier': adaboost_classifier,
                'XGBClassifier': xgb_classifier
                }
        to_predict_features = {}

        if request.method == "POST":
            for feature in selected_features:
                feature_val = request.form.get(feature, None)
                if not feature_val:
                    flash('Please enter all the required values to predict.', 'danger')
                    return redirect(url_for('pipeline.pipeline_detail_predict', id=id))
                to_predict_features[feature] = [request.form.get(feature, 0, type=int)]
            model = model_mapper[selected_model]
            X = df[selected_features]
            y = df[predictor]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            data_cm = confusion_matrix(y_test, y_pred)

            # out of sample
            out_of_sample = pd.DataFrame(data=to_predict_features)
            predicted = model.predict(out_of_sample)

            # GENERATE CONFUSION MATRIX

            df_cm = pd.DataFrame(data_cm, columns=np.unique(y_test), index=np.unique(y_test))
            df_cm.index.name = "Actual"
            df_cm.columns.name = "Predicted"
            plt.figure(figsize=(6,5))
            sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")

            Path(basedir, plot_url).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(basedir, plot_url, 'confusion_matrix.png'),
                        dpi=300, bbox_inches='tight', pad_inches=0.2)

            # SAVE HISTORY
            existing_config = project_config.config
            prediction_history = existing_config['classifier_model_values']['history']
            feature_text = ''
            for feat in  to_predict_features:
                feature_text += f"{feat}: {to_predict_features[feat][0]}, "
            prediction_history['features'].insert(0, feature_text)
            prediction_history['predicted'].insert(0, str(predicted[0]))
            existing_config['classifier_model_values']['history'] = prediction_history
            flag_modified(project_config, "config")
            db.session.commit()

    render_prediction_vis = f"{plot_url[7:]}/confusion_matrix.png" if supervised_learning_task == 'classification' else f"{plot_url[7:]}/linear-actual-predicted.png"
    db.session.close()
    return render_template(
        'pipeline/predict.html',
        active_pipeline='predict',
        progress=100,
        selected_features=selected_features,
        project=project,
        predicted=predicted,
        predict_features=to_predict_features,
        predictor=predictor,
        learning=supervised_learning_task,
        selected_model=selected_model,
        regression_model_values=regression_model_values,
        classifier_model_values=classifier_model_values,
        plot_url=render_prediction_vis,
        all_df=all_df.to_html(classes=('table table-hover dataset-table')))


@pipeline_blueprint.route('delete', methods=['POST'])
@login_required
def pipeline_delete():
    id = request.form.get('id', None)
    current_user, project, config, _ = get_project_user_df(id)
    path = Path(os.path.join(
        basedir, f'static/uploads/datasets/{current_user}/{project.filename_preferred}'))
    # shutil.rmtree(path)
    db.session.query(MLProject).filter(MLProject.id == id).delete()
    db.session.commit()
    db.session.close()
    flash('Successfully deleted project.', 'success')
    return redirect('/dashboard/home')
