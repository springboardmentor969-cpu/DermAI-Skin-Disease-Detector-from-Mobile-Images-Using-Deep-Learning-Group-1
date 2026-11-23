"""add_clinical_reviews

Revision ID: b57a1b2c3d4e
Revises: 6fd691f333bd
Create Date: 2026-11-23 20:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b57a1b2c3d4e'
down_revision: Union[str, Sequence[str], None] = '6fd691f333bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'clinical_reviews',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('clinician_id', sa.Integer(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('confirmed_class', sa.String(length=10), nullable=False),
        sa.Column('overridden', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['clinician_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('clinical_reviews', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_clinical_reviews_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_clinical_reviews_image_id'), ['image_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_clinical_reviews_clinician_id'), ['clinician_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('clinical_reviews', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_clinical_reviews_clinician_id'))
        batch_op.drop_index(batch_op.f('ix_clinical_reviews_image_id'))
        batch_op.drop_index(batch_op.f('ix_clinical_reviews_id'))

    op.drop_table('clinical_reviews')
