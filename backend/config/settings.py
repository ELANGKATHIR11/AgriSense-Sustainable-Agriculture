# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.

# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if (
        "pytest" in sys.modules
        or (len(sys.argv) > 0 and "pytest" in sys.argv[0])
        or os.getenv("PYTEST_CURRENT_TEST")
    ):
        SECRET_KEY = "test_agrisense_secret_key_for_testing"
    else:
        raise ValueError(
            "SECRET_KEY environment variable is not set. Please set SECRET_KEY in environment or .env file."
        )

ALGORITHM = os.getenv("ALGORITHM", "HS256")

