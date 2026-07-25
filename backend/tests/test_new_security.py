# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.

import pytest
import jwt
from fastapi import HTTPException
from unittest.mock import AsyncMock, patch
from backend.security.auth import get_current_user, require_role, SECRET_KEY, ALGORITHM
from backend.security.n8n_notifier import trigger_n8n_webhook
from backend.database.models import User

def test_jwt_role_enforcement():
    # Helper role requirement dependency test
    class MockUser:
        def __init__(self, role):
            self.role = role

    user = MockUser("farmer")
    dep = require_role(["admin", "agronomist"])
    
    with pytest.raises(HTTPException) as exc_info:
        dep(user)
    assert exc_info.value.status_code == 403

    # Success case
    admin_user = MockUser("admin")
    assert dep(admin_user) == admin_user

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_trigger_n8n_webhook_success(mock_post):
    mock_post.return_value = AsyncMock(status_code=200)
    
    # Temporarily set N8N_WEBHOOK_URL to verify post call
    with patch("backend.security.n8n_notifier.N8N_WEBHOOK_URL", "http://mock-n8n/webhook"):
        await trigger_n8n_webhook("TEST_EVENT", {"foo": "bar"})
        
    mock_post.assert_called_once()
