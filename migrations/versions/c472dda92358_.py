"""empty message

Revision ID: c472dda92358
Revises: 31d7945d14b1
Create Date: 2022-12-03 14:35:24.030676

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c472dda92358'
down_revision = '31d7945d14b1'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('ml_project_config_ml_project_fkey', 'ml_project_config', type_='foreignkey')
    op.create_foreign_key(None, 'ml_project_config', 'ml_project', ['ml_project'], ['id'], ondelete='CASCADE')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'ml_project_config', type_='foreignkey')
    op.create_foreign_key('ml_project_config_ml_project_fkey', 'ml_project_config', 'ml_project', ['ml_project'], ['id'])
    # ### end Alembic commands ###
