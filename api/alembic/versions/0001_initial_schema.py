"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-10-05

DataLens tables: datasets, samples, qc_results, exports, embeddings (pgvector).

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('datasets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('hf_repo_id', sa.String(length=255), nullable=False),
    sa.Column('config', sa.String(length=100), nullable=False),
    sa.Column('split', sa.String(length=50), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('revision', sa.String(length=100), nullable=True),
    sa.Column('fields', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('num_samples', sa.Integer(), nullable=False),
    sa.Column('total_samples', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('hf_repo_id', 'config', 'split')
    )
    op.create_table('exports',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('filter', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('val_ratio', sa.Float(), nullable=False),
    sa.Column('format', sa.String(length=20), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('num_samples', sa.Integer(), nullable=False),
    sa.Column('s3_key', sa.String(length=512), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_exports_dataset_id'), 'exports', ['dataset_id'], unique=False)
    op.create_table('samples',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('sample_index', sa.Integer(), nullable=False),
    sa.Column('prompt', sa.Text(), nullable=False),
    sa.Column('context', sa.Text(), nullable=True),
    sa.Column('response', sa.Text(), nullable=False),
    sa.Column('category', sa.String(length=100), nullable=True),
    sa.Column('prompt_chars', sa.Integer(), nullable=False),
    sa.Column('response_chars', sa.Integer(), nullable=False),
    sa.Column('tokens_est', sa.Integer(), nullable=False),
    sa.Column('lang', sa.String(length=8), nullable=True),
    sa.Column('content_hash', sa.String(length=64), nullable=True),
    sa.Column('qc_status', sa.String(length=20), nullable=False),
    sa.Column('qc_score', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id', 'sample_index')
    )
    op.create_index(op.f('ix_samples_category'), 'samples', ['category'], unique=False)
    op.create_index(op.f('ix_samples_content_hash'), 'samples', ['content_hash'], unique=False)
    op.create_index(op.f('ix_samples_dataset_id'), 'samples', ['dataset_id'], unique=False)
    op.create_index(op.f('ix_samples_qc_status'), 'samples', ['qc_status'], unique=False)
    op.create_table('embeddings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sample_id', sa.Integer(), nullable=False),
    sa.Column('model', sa.String(length=100), nullable=False),
    sa.Column('embedding', Vector(384), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['sample_id'], ['samples.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('sample_id')
    )
    op.create_table('qc_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sample_id', sa.Integer(), nullable=False),
    sa.Column('check_name', sa.String(length=64), nullable=False),
    sa.Column('passed', sa.Boolean(), nullable=False),
    sa.Column('severity', sa.String(length=20), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('details', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['sample_id'], ['samples.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_qc_results_check_name'), 'qc_results', ['check_name'], unique=False)
    op.create_index(op.f('ix_qc_results_sample_id'), 'qc_results', ['sample_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_qc_results_sample_id'), table_name='qc_results')
    op.drop_index(op.f('ix_qc_results_check_name'), table_name='qc_results')
    op.drop_table('qc_results')
    op.drop_table('embeddings')
    op.drop_index(op.f('ix_samples_qc_status'), table_name='samples')
    op.drop_index(op.f('ix_samples_dataset_id'), table_name='samples')
    op.drop_index(op.f('ix_samples_content_hash'), table_name='samples')
    op.drop_index(op.f('ix_samples_category'), table_name='samples')
    op.drop_table('samples')
    op.drop_index(op.f('ix_exports_dataset_id'), table_name='exports')
    op.drop_table('exports')
    op.drop_table('datasets')
