# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.

"""Invalidate legacy unhashed password strings

Revision ID: b7d89ea341ef
Revises: afe47ca427cd
Create Date: 2026-07-26 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'b7d89ea341ef'
down_revision: Union[str, Sequence[str], None] = 'afe47ca427cd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE users SET hashed_password = 'INVALIDATED_LEGACY_HASH' WHERE hashed_password LIKE 'hash_%'"
    )


def downgrade() -> None:
    pass
