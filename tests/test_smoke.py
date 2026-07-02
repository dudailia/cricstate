import cricstate


def test_package_constants() -> None:
    assert cricstate.SCHEMA_VERSION == "1.0.0"
    assert cricstate.PINNED_DATA_VERSION == "1.1.0"
