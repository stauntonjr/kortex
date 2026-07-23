from __future__ import annotations

import os

import pytest
from typedb.driver import Credentials, DriverOptions, DriverTlsConfig, TypeDB


def test_typedb_connection_smoke() -> None:
    """Smoke-test a live TypeDB connection using the installed 3.x API.

    If the current runtime cannot reach the configured TypeDB instance, skip the
    test instead of failing during import or collection.
    """

    typedb_address = os.getenv("TYPEDB_ADDR", "typedb:1729")
    database_name = os.getenv("TYPEDB_DATABASE", "kortex")
    creds = Credentials(
        os.getenv("TYPEDB_USERNAME", "admin"),
        os.getenv("TYPEDB_PASSWORD", "password"),
    )
    opts = DriverOptions(
        DriverTlsConfig.disabled(),
        request_timeout_millis=15000,
    )

    try:
        driver = TypeDB.driver(typedb_address, creds, opts)
    except Exception as exc:
        pytest.skip(f"TypeDB smoke test skipped: {exc}")

    try:
        _ = driver.databases.contains(database_name)
    finally:
        driver.close()
