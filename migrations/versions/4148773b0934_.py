"""empty message

Revision ID: 4148773b0934
Revises: 25349a154e93
Create Date: 2022-10-20 21:11:54.018494

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4148773b0934'
down_revision = '25349a154e93'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('ml_project', 'data')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('ml_project', sa.Column('data', sa.BLOB(), nullable=True))
    # ### end Alembic commands ###