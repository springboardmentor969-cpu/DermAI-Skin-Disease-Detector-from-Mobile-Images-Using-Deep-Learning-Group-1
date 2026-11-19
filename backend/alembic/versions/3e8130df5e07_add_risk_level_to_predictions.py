"""add_risk_level_to_predictions

Revision ID: 3e8130df5e07
Revises: 94502164cedd
Create Date: 2026-11-15 21:15:29.691532

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3e8130df5e07'
down_revision: Union[str, Sequence[str], None] = '94502164cedd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('risk_level', sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.drop_column('risk_level')
