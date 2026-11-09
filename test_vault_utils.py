"""Unit tests for Vault settings and clients without a live Vault server."""

import os
from types import SimpleNamespace
from unittest.mock import Mock, call

import pydantic
import pytest
import requests
from hvac import exceptions as vault_exceptions

import vault_utils


@pytest.fixture(autouse=True)
def clean_vault_environment(monkeypatch):
    """Keep settings tests independent of the developer's credentials."""
    settings_names = {
        "approle_path", "url", "role_id", "role-id", "secret_id", "secret-id",
        "verify", "env", "mount_point",
    }
    for name in list(os.environ):
        if name.lower() in settings_names or name.lower().startswith("vault_"):
            monkeypatch.delenv(name)


@pytest.fixture
def auth():
    return vault_utils.VaultAuthConfig(
        url="https://vault.example.test", role_id="test-role", secret_id="test-secret",
    )


@pytest.fixture
def vault_backend(monkeypatch):
    backend = Mock()
    backend.is_authenticated.return_value = True
    backend.secrets.kv.v2.read_secret.return_value = {"data": {"data": {"password": "value"}}}
    constructor = Mock(return_value=backend)
    sleep = Mock()
    jitter = Mock(return_value=0.02)
    monkeypatch.setattr(vault_utils.hvac, "Client", constructor)
    monkeypatch.setattr(vault_utils.time, "sleep", sleep)
    monkeypatch.setattr(vault_utils.random, "uniform", jitter)
    return SimpleNamespace(client=backend, constructor=constructor, sleep=sleep, jitter=jitter)


@pytest.mark.parametrize(
    "credentials",
    [
        {"role_id": "test-role", "secret_id": "test-secret"},
        {"role-id": "test-role", "secret-id": "test-secret"},
    ],
)
def test_auth_config_accepts_field_names_and_aliases(credentials):
    config = vault_utils.VaultAuthConfig(url="https://vault.example.test", **credentials)

    assert config.role_id == "test-role"
    assert config.secret_id.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(config)
    assert config.verify is True


@pytest.mark.parametrize("value, expected", [("false", False), ("true", True), ("/certs/ca.pem", "/certs/ca.pem")])
def test_auth_config_parses_tls_verification(value, expected):
    config = vault_utils.VaultAuthConfig(
        url="https://vault.example.test", role_id="role", secret_id="secret", verify=value,
    )

    assert config.verify == expected


def test_auth_config_requires_credentials():
    with pytest.raises(pydantic.ValidationError) as exc_info:
        vault_utils.VaultAuthConfig()

    assert {error["loc"][0] for error in exc_info.value.errors()} == {"url", "role-id", "secret-id"}


def test_config_is_frozen_and_rejects_extra_fields(auth):
    config = vault_utils.SimpleVaultConfig(meta=vault_utils.VaultMetaConfig(), auth=auth)

    with pytest.raises(pydantic.ValidationError, match="frozen"):
        config.auth = auth
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        config.auth.url = "https://other.example.test"
    with pytest.raises(pydantic.ValidationError, match="Extra inputs"):
        vault_utils.SimpleVaultConfig(meta=config.meta, auth=auth, unknown="value")


def test_meta_config_requires_an_existing_approle_directory(tmp_path):
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("content")

    for invalid_path in [file_path, tmp_path / "missing"]:
        with pytest.raises(pydantic.ValidationError):
            vault_utils.VaultMetaConfig(approle_path=invalid_path)


def test_airflow_config_from_env_file(tmp_path):
    env_file = tmp_path / "vault.env"
    env_file.write_text(
        "URL=https://vault.example.test\nROLE-ID=file-role\nSECRET-ID=file-secret\n"
        "VERIFY=false\nENV=stage\nMOUNT_POINT=airflow-secrets\nUNRELATED=ignored\n"
    )

    config = vault_utils.VaultAirflowConfig.from_env_file(env_file)

    assert isinstance(config.meta, vault_utils.VaultAirflowMetaConfig)
    assert config.meta.env == "stage"
    assert config.meta.mount_point == "airflow-secrets"
    assert config.meta.approle_path is None
    assert config.auth.url == "https://vault.example.test"
    assert config.auth.role_id == "file-role"
    assert config.auth.secret_id.get_secret_value() == "file-secret"
    assert config.auth.verify is False


def test_config_from_env_file_reads_approle_secret_files(tmp_path):
    secrets_dir = tmp_path / "approle"
    secrets_dir.mkdir()
    (secrets_dir / "role-id").write_text("file-role\n")
    (secrets_dir / "secret-id").write_text("file-secret\n")
    env_file = tmp_path / "vault.env"
    env_file.write_text(f"URL=https://vault.example.test\nAPPROLE_PATH={secrets_dir}\n")

    config = vault_utils.SimpleVaultConfig.from_env_file(str(env_file))

    assert config.meta.approle_path == secrets_dir
    assert config.auth.role_id == "file-role"
    assert config.auth.secret_id.get_secret_value() == "file-secret"


def test_config_from_env_uses_case_insensitive_prefixed_names(monkeypatch):
    for name, value in {
        "vault_url": "https://vault.example.test", "VAULT_ROLE_ID": "env-role",
        "Vault_Secret_Id": "env-secret", "VAULT_VERIFY": "false",
        "VAULT_ENV": "prod", "VAULT_MOUNT_POINT": "airflow-secrets",
    }.items():
        monkeypatch.setenv(name, value)

    config = vault_utils.VaultAirflowConfig.from_env()

    assert config.meta.env == "prod"
    assert config.meta.mount_point == "airflow-secrets"
    assert config.auth.url == "https://vault.example.test"
    assert config.auth.role_id == "env-role"
    assert config.auth.secret_id.get_secret_value() == "env-secret"
    assert config.auth.verify is False


def test_config_from_env_reads_unprefixed_secret_filenames(tmp_path, monkeypatch):
    (tmp_path / "role-id").write_text("file-role")
    (tmp_path / "secret-id").write_text("file-secret")
    monkeypatch.setenv("VAULT_URL", "https://vault.example.test")
    monkeypatch.setenv("VAULT_APPROLE_PATH", str(tmp_path))

    config = vault_utils.SimpleVaultConfig.from_env()

    assert config.auth.role_id == "file-role"
    assert config.auth.secret_id.get_secret_value() == "file-secret"


def test_airflow_meta_config_rejects_unknown_environment():
    with pytest.raises(pydantic.ValidationError, match="env"):
        vault_utils.VaultAirflowMetaConfig(env="qa", mount_point="airflow-secrets")


@pytest.mark.parametrize("verify", [False, True, "/certs/ca.pem"])
def test_client_copies_auth_and_passes_tls_settings(auth, vault_backend, verify):
    auth = auth.model_copy(update={"verify": verify})

    client = vault_utils.VaultClient(auth)

    assert client.auth == auth
    assert client.auth is not auth
    assert client.auth.secret_id is not auth.secret_id
    vault_backend.constructor.assert_called_once_with(url=auth.url, verify=verify)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"initial_delay": -0.1}, "initial_delay"),
        ({"delay_jitter": -0.1}, "delay_jitter"),
        ({"attempts": 0}, "attempts"),
        ({"attempts": -1}, "attempts"),
        pytest.param(
            {"attempts": 1.5}, "attempts",
            marks=pytest.mark.xfail(
                strict=True, reason="VaultClient accepts fractional attempts despite its positive-integer contract",
            ),
        ),
    ],
)
def test_client_rejects_invalid_retry_options(auth, vault_backend, kwargs, message):
    with pytest.raises(ValueError, match=message):
        vault_utils.VaultClient(auth, **kwargs)

    vault_backend.constructor.assert_not_called()


@pytest.mark.parametrize("authenticated", [True, False])
@pytest.mark.parametrize("secret", [{}, {"username": "user", "password": "value"}])
def test_read_secret_authenticates_only_when_needed(auth, vault_backend, authenticated, secret):
    vault_backend.client.is_authenticated.return_value = authenticated
    vault_backend.client.secrets.kv.v2.read_secret.return_value = {"data": {"data": secret}}
    client = vault_utils.VaultClient(auth)

    assert client.read_secret(path="app/database", mount_point="secrets") == secret

    vault_backend.client.secrets.kv.v2.read_secret.assert_called_once_with(
        path="app/database", mount_point="secrets",
    )
    if authenticated:
        vault_backend.client.auth.approle.login.assert_not_called()
    else:
        vault_backend.client.auth.approle.login.assert_called_once_with(
            role_id="test-role", secret_id="test-secret",
        )
    vault_backend.sleep.assert_not_called()


@pytest.mark.parametrize("response", [None, {}, {"data": None}, {"data": {}}, {"data": []}])
def test_read_secret_rejects_malformed_responses_without_retry(auth, vault_backend, response):
    vault_backend.client.secrets.kv.v2.read_secret.return_value = response
    client = vault_utils.VaultClient(auth)

    with pytest.raises(vault_utils.VaultQueryError, match="Malformed data") as exc_info:
        client.read_secret(path="app/database", mount_point="secrets")

    assert isinstance(exc_info.value.__cause__, (KeyError, TypeError))
    assert vault_backend.client.secrets.kv.v2.read_secret.call_count == 1
    vault_backend.sleep.assert_not_called()


@pytest.mark.parametrize("secret", [None, [], "text", {"password": 42}, {1: "value"}, {"nested": {}}])
def test_read_secret_rejects_invalid_values_without_retry(auth, vault_backend, secret):
    vault_backend.client.secrets.kv.v2.read_secret.return_value = {"data": {"data": secret}}
    client = vault_utils.VaultClient(auth)

    with pytest.raises(vault_utils.VaultValueError) as exc_info:
        client.read_secret(path="app/database", mount_point="secrets")

    assert isinstance(exc_info.value.__cause__, pydantic.ValidationError)
    assert vault_backend.client.secrets.kv.v2.read_secret.call_count == 1
    vault_backend.sleep.assert_not_called()


@pytest.mark.parametrize(
    "error_type, expected_type",
    [
        (vault_exceptions.Forbidden, vault_utils.VaultPermissionDeniedError),
        (vault_exceptions.InvalidPath, vault_utils.VaultInvalidPathError),
    ],
)
@pytest.mark.parametrize("operation", ["login", "read"])
def test_read_secret_translates_permanent_errors_without_retry(
    auth, vault_backend, error_type, expected_type, operation,
):
    error = error_type("Vault rejected the request")
    if operation == "login":
        vault_backend.client.is_authenticated.return_value = False
        failing_call = vault_backend.client.auth.approle.login
    else:
        failing_call = vault_backend.client.secrets.kv.v2.read_secret
    failing_call.side_effect = error
    client = vault_utils.VaultClient(auth)

    with pytest.raises(expected_type) as exc_info:
        client.read_secret(path="app/database", mount_point="secrets")

    assert exc_info.value.__cause__ is error
    assert failing_call.call_count == 1
    vault_backend.sleep.assert_not_called()


@pytest.mark.parametrize("error_type", [requests.ConnectionError, vault_exceptions.VaultError])
def test_read_secret_retries_transient_failures(auth, vault_backend, error_type):
    vault_backend.client.secrets.kv.v2.read_secret.side_effect = [
        error_type("temporary failure"), {"data": {"data": {"password": "value"}}},
    ]
    client = vault_utils.VaultClient(auth, initial_delay=0.1, delay_jitter=0.05)

    assert client.read_secret(path="app/database", mount_point="secrets") == {"password": "value"}

    assert vault_backend.client.secrets.kv.v2.read_secret.call_args_list == [
        call(path="app/database", mount_point="secrets"),
        call(path="app/database", mount_point="secrets"),
    ]
    vault_backend.jitter.assert_called_once_with(-0.05, 0.05)
    vault_backend.sleep.assert_called_once_with(pytest.approx(0.12))


def test_read_secret_retries_login_and_clamps_negative_sleep(auth, vault_backend):
    vault_backend.client.is_authenticated.return_value = False
    vault_backend.client.auth.approle.login.side_effect = [requests.ConnectionError("offline"), None]
    vault_backend.jitter.return_value = -0.5
    client = vault_utils.VaultClient(auth, initial_delay=0.1, delay_jitter=0.5)

    assert client.read_secret(path="app/database", mount_point="secrets") == {"password": "value"}

    assert vault_backend.client.auth.approle.login.call_count == 2
    assert vault_backend.client.secrets.kv.v2.read_secret.call_count == 1
    vault_backend.sleep.assert_called_once_with(0)


@pytest.mark.parametrize("error_type", [requests.ConnectionError, vault_exceptions.VaultError])
@pytest.mark.parametrize("attempts", [1, 3])
def test_read_secret_stops_at_attempt_limit_and_chains_last_error(auth, vault_backend, error_type, attempts):
    errors = [error_type(f"failure {index}") for index in range(attempts)]
    vault_backend.client.secrets.kv.v2.read_secret.side_effect = errors
    client = vault_utils.VaultClient(auth, attempts=attempts)

    with pytest.raises(vault_utils.VaultQueryError, match=f"after {attempts} attempts") as exc_info:
        client.read_secret(path="app/database", mount_point="secrets")

    assert exc_info.value.__cause__ is errors[-1]
    assert vault_backend.client.secrets.kv.v2.read_secret.call_count == attempts


@pytest.fixture
def airflow_client(auth, monkeypatch):
    config = vault_utils.VaultAirflowConfig(
        meta=vault_utils.VaultAirflowMetaConfig(env="stage", mount_point="airflow-secrets"),
        auth=auth,
    )
    backend = Mock()
    constructor = Mock(return_value=backend)
    monkeypatch.setattr(vault_utils, "VaultClient", constructor)
    client = vault_utils.VaultAirflowClient(config)
    return SimpleNamespace(client=client, config=config, backend=backend, constructor=constructor)


def test_airflow_client_copies_config(airflow_client):
    copied = airflow_client.client.config

    assert copied == airflow_client.config
    assert copied is not airflow_client.config
    assert copied.meta is not airflow_client.config.meta
    assert copied.auth is not airflow_client.config.auth
    airflow_client.constructor.assert_called_once_with(copied.auth)


def test_airflow_client_reads_connection_from_environment_directory(airflow_client):
    airflow_client.backend.read_secret.return_value = {"host": "database.example.test"}

    result = airflow_client.client.get_connection("warehouse")

    assert result == {"host": "database.example.test"}
    airflow_client.backend.read_secret.assert_called_once_with(
        mount_point="airflow-secrets", path="stage/airflow/connections/warehouse",
    )


@pytest.mark.parametrize(
    "conn_id",
    [
        "", ".", "../warehouse", "/warehouse", "nested/warehouse",
        pytest.param(
            "..",
            marks=pytest.mark.xfail(
                strict=True, reason="VaultAirflowClient accepts '..' as a connection filename",
            ),
        ),
    ],
)
def test_airflow_client_rejects_connection_paths(airflow_client, conn_id):
    with pytest.raises(ValueError, match="conn_id.*filename"):
        airflow_client.client.get_connection(conn_id)

    airflow_client.backend.read_secret.assert_not_called()


def test_airflow_client_preserves_vault_errors(airflow_client):
    error = vault_utils.VaultPermissionDeniedError("access denied")
    airflow_client.backend.read_secret.side_effect = error

    with pytest.raises(vault_utils.VaultPermissionDeniedError) as exc_info:
        airflow_client.client.get_connection("warehouse")

    assert exc_info.value is error
