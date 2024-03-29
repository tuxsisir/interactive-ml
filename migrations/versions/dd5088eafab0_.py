"""empty message

Revision ID: dd5088eafab0
Revises: 99e2700632af
Create Date: 2022-11-24 23:44:29.495341

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'dd5088eafab0'
down_revision = '99e2700632af'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('flask_dance_oauth',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('provider', sa.String(length=50), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('token', sa.JSON(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.add_column('user', sa.Column('username', sa.String(length=255), nullable=False))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('user', 'username')
    op.drop_table('flask_dance_oauth')
    # ### end Alembic commands ###
