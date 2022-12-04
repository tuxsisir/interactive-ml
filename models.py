from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_dance.consumer.storage.sqla import OAuthConsumerMixin
from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import JSONB

db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    designation = db.Column(db.String(255), nullable=False)
    twitter_handle = db.Column(db.String(255), nullable=False)
    projects = db.relationship('MLProject', backref='projects', lazy=True)

    def __repr__(self):
        return '<User %r> %d' % self.username, self.id


class OAuth(OAuthConsumerMixin, db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey(User.id))
    user = db.relationship(User)


class MLProject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(
        db.Integer,
        db.ForeignKey(User.id),
        nullable=False)
    title = db.Column(db.String(50))
    description = db.Column(db.String(500))
    filename = db.Column(db.String(255))
    filename_preferred = db.Column(db.String(255))
    status = db.Column(db.String(255))

    def __repr__(self):
        return f'<MLProject {self.title} - {self.id}>'


class MLProjectConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ml_project = db.Column(
        db.Integer,
        db.ForeignKey(MLProject.id, ondelete='CASCADE'),
        nullable=False)
    config = db.Column(JSONB)
    description = db.Column(db.String(500))

    def __repr__(self):
        return f'<MLProjectConfig {self.id}>'
