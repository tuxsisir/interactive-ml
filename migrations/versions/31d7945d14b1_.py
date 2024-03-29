"""empty message

Revision ID: 31d7945d14b1
Revises: dd5088eafab0
Create Date: 2022-12-03 13:48:01.177716

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '31d7945d14b1'
down_revision = 'dd5088eafab0'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('ml_project_config', sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('ml_project_config', 'config')
    # ### end Alembic commands ###
