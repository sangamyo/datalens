"""HNSW cosine index on embeddings

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-08

Approximate nearest-neighbour index for semantic search: `ORDER BY embedding <=> :query LIMIT k`
(pgvector cosine distance). m=16 / ef_construction=64 are pgvector's defaults, written out.

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        'ix_embeddings_embedding_hnsw',
        'embeddings',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_embeddings_embedding_hnsw', table_name='embeddings', postgresql_using='hnsw')
