# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

from fastapi import FastAPI
from backend.api.router import api_router
from backend.db.database import engine, Base

app = FastAPI(title='AgriSense API', version='0.1.0')
app.include_router(api_router)

@app.on_event('startup')
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
