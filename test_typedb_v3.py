from __future__ import annotations

import os

import pytest
from typedb.driver import Credentials, DriverOptions, DriverTlsConfig, TypeDB


def test_typedb_driver_connection_smoke() -> None:
    """Smoke-test the installed TypeDB 3.x driver against a configured address.

    If the current runtime cannot reach the configured TypeDB instance, skip the
    test instead of failing during import or collection.
    """

    addr = os.getenv("TYPEDB_ADDR", "typedb:1729")
    creds = Credentials(
        os.getenv("TYPEDB_USERNAME", "admin"),
        os.getenv("TYPEDB_PASSWORD", "password"),
    )
    opts = DriverOptions(
        DriverTlsConfig.disabled(),
        request_timeout_millis=15000,
    )

    try:
        driver = TypeDB.driver(addr, creds, opts)
    except Exception as exc:
        pytest.skip(f"TypeDB smoke test skipped: {exc}")

    driver.close()
