"""Unit tests for Vault settings and clients without a live Vault server."""

import os
import random
import time
from pathlib import Path
from typing import NamedTuple, TypedDict
from unittest.mock import Mock, call

import hvac
import pydantic
import pytest
import requests
from hvac import exceptions as vault_exceptions

import vault_utils

VALID_FILENAMES = ["warehouse", ".hidden", "name..", "name with spaces", "данные", r"name\with\backslashes"]
INVALID_FILENAMES = [
    "", ".", "..", "/", "../warehouse", "/warehouse", "nested/warehouse",
    "./warehouse", "warehouse/", "ware\0house",
]


class VaultBackend(NamedTuple):
    client: Mock
    constructor: Mock
    sleep: Mock
    jitter: Mock


class AirflowClientFixture(NamedTuple):
    client: vault_utils.VaultAirflowClient
    config: vault_utils.VaultAirflowConfig
    backend: Mock
    constructor: Mock


class RetryOptions(TypedDict, total=False):
    initial_delay: float
    delay_jitter: float
    attempts: int


AuthCredentials = TypedDict(
    "AuthCredentials",
    {"role_id": str, "secret_id": str, "role-id": str, "secret-id": str},
    total=False,
)


@pytest.mark.parametrize("filename", VALID_FILENAMES)
def test_validate_filename_preserves_valid_posix_filenames(filename: str) -> None:
    """Verify that valid POSIX filenames are returned unchanged.

    Args:
        filename: A valid filename, including characters allowed on POSIX systems.

    Raises:
        AssertionError: If validation changes a valid filename.
    """
    assert vault_utils.validate_filename(filename) == filename


@pytest.mark.parametrize("filename", INVALID_FILENAMES)
def test_validate_filename_rejects_paths_and_invalid_filenames(filename: str) -> None:
    """Verify that invalid filenames raise the dedicated validation error.

    Args:
        filename: An invalid filename or a path that must be rejected.

    Raises:
        pytest.fail.Exception: If the invalid filename is accepted.
    """
    with pytest.raises(vault_utils.InvalidFilenameError, match="Invalid POSIX filename"):
        vault_utils.validate_filename(filename)


@pytest.fixture(autouse=True)
def clean_vault_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep settings tests independent of the developer's credentials.

    Args:
        monkeypatch: Pytest fixture that restores removed environment variables.
    """
    settings_names = {
        "approle_path", "url", "role_id", "role-id", "secret_id", "secret-id",
        "verify", "env", "mount_point",
    }
    for name in list(os.environ):
        if name.lower() in settings_names or name.lower().startswith("vault_"):
            monkeypatch.delenv(name)


@pytest.fixture
def auth() -> vault_utils.VaultAuthConfig:
    """Build authentication settings with synthetic Vault credentials.

    Returns:
        Valid settings containing the test server URL and AppRole credentials.
    """
    return vault_utils.VaultAuthConfig(
        url="https://vault.example.test", role_id="test-role", secret_id="test-secret",
    )


@pytest.fixture
def vault_backend(monkeypatch: pytest.MonkeyPatch) -> VaultBackend:
    """Replace Vault access and retry timing with controllable mocks.

    Args:
        monkeypatch: Pytest fixture that installs and restores mocked dependencies.

    Returns:
        Mocks for the Vault client, its constructor, sleep, and random jitter.
    """
    backend = Mock()
    backend.is_authenticated.return_value = True
    backend.secrets.kv.v2.read_secret.return_value = {"data": {"data": {"password": "value"}}}
    constructor = Mock(return_value=backend)
    sleep = Mock()
    jitter = Mock(return_value=0.02)
    monkeypatch.setattr(hvac, "Client", constructor)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(random, "uniform", jitter)
    return VaultBackend(client=backend, constructor=constructor, sleep=sleep, jitter=jitter)


@pytest.mark.parametrize(
    "credentials",
    [
        {"role_id": "test-role", "secret_id": "test-secret"},
        {"role-id": "test-role", "secret-id": "test-secret"},
    ],
)
def test_auth_config_accepts_field_names_and_aliases(credentials: AuthCredentials) -> None:
    """Verify credential aliases, secret masking, and the TLS default.

    Args:
        credentials: AppRole credentials keyed by field names or their aliases.

    Raises:
        AssertionError: If credentials are parsed incorrectly, exposed, or TLS is disabled.
    """
    config = vault_utils.VaultAuthConfig(url="https://vault.example.test", **credentials)

    assert config.role_id == "test-role"
    assert config.secret_id.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(config)
    assert config.verify is True


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("true", True), ("/certs/ca.pem", "/certs/ca.pem")])
def test_auth_config_parses_tls_verification(value: str, expected: bool | str) -> None:
    """Verify that TLS settings accept boolean strings and certificate paths.

    Args:
        value: Raw TLS verification setting supplied to the configuration.
        expected: Expected boolean or certificate path after validation.

    Raises:
        AssertionError: If the parsed TLS setting differs from the expected value.
    """
    config = vault_utils.VaultAuthConfig(
        url="https://vault.example.test", role_id="role", secret_id="secret", verify=value,
    )

    assert config.verify == expected


def test_auth_config_requires_credentials() -> None:
    """Verify that authentication validation identifies every missing credential.

    Raises:
        AssertionError: If the validation errors identify the wrong missing fields.
        pytest.fail.Exception: If authentication settings accept missing credentials.
    """
    with pytest.raises(pydantic.ValidationError) as exc_info:
        vault_utils.VaultAuthConfig()

    assert {error["loc"][0] for error in exc_info.value.errors()} == {"url", "role-id", "secret-id"}


def test_config_is_frozen_and_rejects_extra_fields(auth: vault_utils.VaultAuthConfig) -> None:
    """Verify that configuration models reject mutation and unknown fields.

    Args:
        auth: Valid authentication settings used to build the combined configuration.

    Raises:
        pytest.fail.Exception: If a frozen field can be changed or an extra field is accepted.
    """
    config = vault_utils.SimpleVaultConfig(meta=vault_utils.VaultMetaConfig(), auth=auth)

    frozen_updates: list[tuple[pydantic.BaseModel, str, object]] = [
        (config, "auth", auth),
        (config.auth, "url", "https://other.example.test"),
    ]
    for model, field_name, value in frozen_updates:
        with pytest.raises(pydantic.ValidationError, match="frozen"):
            setattr(model, field_name, value)
    with pytest.raises(pydantic.ValidationError, match="Extra inputs"):
        vault_utils.SimpleVaultConfig.model_validate({"meta": config.meta, "auth": auth, "unknown": "value"})


def test_meta_config_requires_an_existing_approle_directory(tmp_path: Path) -> None:
    """Verify that the AppRole location must be an existing directory.

    Args:
        tmp_path: Temporary directory for a regular file and a missing path.

    Raises:
        pytest.fail.Exception: If a regular file or missing path is accepted as a directory.
    """
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("content")

    for invalid_path in [file_path, tmp_path / "missing"]:
        with pytest.raises(pydantic.ValidationError):
            vault_utils.VaultMetaConfig(approle_path=invalid_path)


def test_airflow_config_from_env_file(tmp_path: Path) -> None:
    """Verify that an env file supplies Airflow metadata and authentication settings.

    Args:
        tmp_path: Temporary directory in which to create the settings file.

    Raises:
        AssertionError: If the loaded configuration does not match the file contents.
    """
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


def test_config_from_env_file_reads_approle_secret_files(tmp_path: Path) -> None:
    """Verify that an env file can locate AppRole credentials in separate secret files.

    Args:
        tmp_path: Temporary directory for the env file and AppRole secret directory.

    Raises:
        AssertionError: If the AppRole directory or loaded credentials are incorrect.
    """
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


def test_config_from_env_uses_case_insensitive_prefixed_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that Vault environment variable names are prefixed and case insensitive.

    Args:
        monkeypatch: Pytest fixture that supplies temporary Vault environment variables.

    Raises:
        AssertionError: If the loaded metadata or authentication settings are incorrect.
    """
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


def test_config_from_env_reads_unprefixed_secret_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that prefixed settings still load unprefixed AppRole secret filenames.

    Args:
        tmp_path: Temporary directory containing the AppRole secret files.
        monkeypatch: Pytest fixture that supplies the server URL and secret directory.

    Raises:
        AssertionError: If the credentials do not match the unprefixed secret files.
    """
    (tmp_path / "role-id").write_text("file-role")
    (tmp_path / "secret-id").write_text("file-secret")
    monkeypatch.setenv("VAULT_URL", "https://vault.example.test")
    monkeypatch.setenv("VAULT_APPROLE_PATH", str(tmp_path))

    config = vault_utils.SimpleVaultConfig.from_env()

    assert config.auth.role_id == "file-role"
    assert config.auth.secret_id.get_secret_value() == "file-secret"


def test_airflow_meta_config_rejects_unknown_environment() -> None:
    """Verify that Airflow metadata rejects an unsupported environment name.

    Raises:
        pytest.fail.Exception: If an environment outside dev, stage, and prod is accepted.
    """
    with pytest.raises(pydantic.ValidationError, match="env"):
        vault_utils.VaultAirflowMetaConfig(env="qa", mount_point="airflow-secrets")


@pytest.mark.parametrize("verify", [False, True, "/certs/ca.pem"])
def test_client_copies_auth_and_passes_tls_settings(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, verify: bool | str,
) -> None:
    """Verify that the client copies credentials and forwards the TLS setting.

    Args:
        auth: Original authentication settings whose identity must be preserved.
        vault_backend: Mocked Vault constructor and supporting dependencies.
        verify: Boolean TLS setting or certificate path to pass to the backend.

    Raises:
        AssertionError: If credentials are shared instead of copied or TLS settings are lost.
    """
    auth = auth.model_copy(update={"verify": verify})

    client = vault_utils.VaultClient(auth)

    assert client.auth == auth
    assert client.auth is not auth
    assert client.auth.secret_id is not auth.secret_id
    vault_backend.constructor.assert_called_once_with(url=auth.url, verify=verify)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_delay": -0.1}, "initial_delay"),
        ({"delay_jitter": -0.1}, "delay_jitter"),
        ({"attempts": 0}, "attempts"),
        ({"attempts": -1}, "attempts"),
    ],
)
def test_client_rejects_invalid_retry_options(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, kwargs: RetryOptions, message: str,
) -> None:
    """Verify that invalid retry settings are rejected before creating a backend.

    Args:
        auth: Valid authentication settings for constructing the client.
        vault_backend: Mocked Vault constructor and supporting dependencies.
        kwargs: Retry setting containing a negative delay or non-positive attempt count.
        message: Parameter name expected in the validation error.

    Raises:
        AssertionError: If a backend is created despite invalid retry settings.
        pytest.fail.Exception: If the invalid retry setting is accepted.
    """
    with pytest.raises(ValueError, match=message):
        vault_utils.VaultClient(auth, **kwargs)

    vault_backend.constructor.assert_not_called()


@pytest.mark.parametrize("authenticated", [True, False])
@pytest.mark.parametrize("secret", [{}, {"username": "user", "password": "value"}])
def test_read_secret_authenticates_only_when_needed(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, authenticated: bool, secret: dict[str, str],
) -> None:
    """Verify successful secret reads and conditional AppRole authentication.

    Args:
        auth: AppRole credentials available to the client.
        vault_backend: Mocked Vault client and retry timing dependencies.
        authenticated: Whether the backend reports an existing authenticated session.
        secret: Valid secret dictionary returned by the backend.

    Raises:
        AssertionError: If the secret, login behavior, read arguments, or retry timing is wrong.
    """
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
def test_read_secret_rejects_malformed_responses_without_retry(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, response: object,
) -> None:
    """Verify that malformed response envelopes fail immediately and preserve their cause.

    Args:
        auth: Valid authentication settings for the Vault client.
        vault_backend: Mocked Vault client and retry timing dependencies.
        response: Response without the required nested secret data structure.

    Raises:
        AssertionError: If the underlying error is lost or the malformed response is retried.
        pytest.fail.Exception: If the malformed response is accepted.
    """
    vault_backend.client.secrets.kv.v2.read_secret.return_value = response
    client = vault_utils.VaultClient(auth)

    with pytest.raises(vault_utils.VaultQueryError, match="Malformed data") as exc_info:
        client.read_secret(path="app/database", mount_point="secrets")

    assert isinstance(exc_info.value.__cause__, (KeyError, TypeError))
    assert vault_backend.client.secrets.kv.v2.read_secret.call_count == 1
    vault_backend.sleep.assert_not_called()


@pytest.mark.parametrize("secret", [None, [], "text", {"password": 42}, {1: "value"}, {"nested": {}}])
def test_read_secret_rejects_invalid_values_without_retry(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, secret: object,
) -> None:
    """Verify that invalid secret values fail validation without retries.

    Args:
        auth: Valid authentication settings for the Vault client.
        vault_backend: Mocked Vault client and retry timing dependencies.
        secret: Secret payload that is not a dictionary mapping strings to strings.

    Raises:
        AssertionError: If the validation cause is lost or invalid data triggers a retry.
        pytest.fail.Exception: If the invalid secret payload is accepted.
    """
    vault_backend.client.secrets.kv.v2.read_secret.return_value = {"data": {"data": secret}}
    client = vault_utils.VaultClient(auth)

    with pytest.raises(vault_utils.VaultValueError) as exc_info:
        client.read_secret(path="app/database", mount_point="secrets")

    assert isinstance(exc_info.value.__cause__, pydantic.ValidationError)
    assert vault_backend.client.secrets.kv.v2.read_secret.call_count == 1
    vault_backend.sleep.assert_not_called()


@pytest.mark.parametrize(
    ("error_type", "expected_type"),
    [
        (vault_exceptions.Forbidden, vault_utils.VaultPermissionDeniedError),
        (vault_exceptions.InvalidPath, vault_utils.VaultInvalidPathError),
    ],
)
@pytest.mark.parametrize("operation", ["login", "read"])
def test_read_secret_translates_permanent_errors_without_retry(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, error_type: type[Exception],
    expected_type: type[vault_utils.VaultQueryError], operation: str,
) -> None:
    """Verify that permanent Vault failures are translated once with their cause preserved.

    Args:
        auth: Valid authentication settings for the Vault client.
        vault_backend: Mocked Vault client and retry timing dependencies.
        error_type: Backend exception type to raise during the selected operation.
        expected_type: Application exception type expected after translation.
        operation: Backend operation to fail, either login or secret reading.

    Raises:
        AssertionError: If the cause is lost or a permanent error triggers a retry.
        pytest.fail.Exception: If the permanent error is swallowed.
    """
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
def test_read_secret_retries_transient_failures(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, error_type: type[Exception],
) -> None:
    """Verify that transient read failures are retried after the configured jittered delay.

    Args:
        auth: Valid authentication settings for the Vault client.
        vault_backend: Mocked Vault client and retry timing dependencies.
        error_type: Transient exception type raised before a successful response.

    Raises:
        AssertionError: If the retry result, read arguments, or delay is incorrect.
    """
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


def test_read_secret_retries_login_and_clamps_negative_sleep(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend,
) -> None:
    """Verify that login failures are retried and negative retry delays become zero.

    Args:
        auth: AppRole credentials used when retrying authentication.
        vault_backend: Mocked Vault client and retry timing dependencies.

    Raises:
        AssertionError: If login is not retried, the secret is wrong, or the delay is negative.
    """
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
def test_read_secret_stops_at_attempt_limit_and_chains_last_error(
    auth: vault_utils.VaultAuthConfig, vault_backend: VaultBackend, error_type: type[Exception], attempts: int,
) -> None:
    """Verify that exhausted retries report the final backend failure as their cause.

    Args:
        auth: Valid authentication settings for the Vault client.
        vault_backend: Mocked Vault client and retry timing dependencies.
        error_type: Transient exception type raised by every secret read.
        attempts: Maximum number of read attempts allowed by the client.

    Raises:
        AssertionError: If the attempt limit is ignored or the final cause is incorrect.
        pytest.fail.Exception: If exhausted retries do not raise a query error.
    """
    errors = [error_type(f"failure {index}") for index in range(attempts)]
    vault_backend.client.secrets.kv.v2.read_secret.side_effect = errors
    client = vault_utils.VaultClient(auth, attempts=attempts)

    with pytest.raises(vault_utils.VaultQueryError, match=f"after {attempts} attempts") as exc_info:
        client.read_secret(path="app/database", mount_point="secrets")

    assert exc_info.value.__cause__ is errors[-1]
    assert vault_backend.client.secrets.kv.v2.read_secret.call_count == attempts


@pytest.fixture
def airflow_client(auth: vault_utils.VaultAuthConfig, monkeypatch: pytest.MonkeyPatch) -> AirflowClientFixture:
    """Build an Airflow client backed by a mocked Vault client.

    Args:
        auth: Valid authentication settings for the Airflow configuration.
        monkeypatch: Pytest fixture that replaces the underlying Vault client constructor.

    Returns:
        The Airflow client, its original configuration, and the mocked Vault dependencies.
    """
    config = vault_utils.VaultAirflowConfig(
        meta=vault_utils.VaultAirflowMetaConfig(env="stage", mount_point="airflow-secrets"),
        auth=auth,
    )
    backend = Mock()
    constructor = Mock(return_value=backend)
    monkeypatch.setattr(vault_utils, "VaultClient", constructor)
    client = vault_utils.VaultAirflowClient(config)
    return AirflowClientFixture(client=client, config=config, backend=backend, constructor=constructor)


def test_airflow_client_copies_config(airflow_client: AirflowClientFixture) -> None:
    """Verify that the Airflow client deeply copies its configuration.

    Args:
        airflow_client: Airflow client, original configuration, and mocked Vault dependencies.

    Raises:
        AssertionError: If configuration objects are shared or the copied auth is not forwarded.
    """
    copied = airflow_client.client.config

    assert copied == airflow_client.config
    assert copied is not airflow_client.config
    assert copied.meta is not airflow_client.config.meta
    assert copied.auth is not airflow_client.config.auth
    airflow_client.constructor.assert_called_once_with(copied.auth)


@pytest.mark.parametrize("conn_id", VALID_FILENAMES)
def test_airflow_client_reads_connection_from_environment_directory(
    airflow_client: AirflowClientFixture, conn_id: str,
) -> None:
    """Verify that connection IDs are read beneath the configured environment directory.

    Args:
        airflow_client: Airflow client and mocked Vault dependencies.
        conn_id: Valid connection filename to append to the environment-specific directory.

    Raises:
        AssertionError: If the connection result, secret path, or mount point is incorrect.
    """
    airflow_client.backend.read_secret.return_value = {"host": "database.example.test"}

    result = airflow_client.client.get_connection(conn_id)

    assert result == {"host": "database.example.test"}
    airflow_client.backend.read_secret.assert_called_once_with(
        mount_point="airflow-secrets", path=f"stage/airflow/connections/{conn_id}",
    )


@pytest.mark.parametrize("conn_id", INVALID_FILENAMES)
def test_airflow_client_rejects_connection_paths(airflow_client: AirflowClientFixture, conn_id: str) -> None:
    """Verify that invalid connection IDs are rejected before Vault is queried.

    Args:
        airflow_client: Airflow client and mocked Vault dependencies.
        conn_id: Invalid connection filename or path that must be rejected.

    Raises:
        AssertionError: If the validation cause is lost or Vault is queried for an invalid ID.
        pytest.fail.Exception: If the invalid connection ID is accepted.
    """
    with pytest.raises(vault_utils.InvalidFilenameError, match=r"conn_id.*filename") as exc_info:
        airflow_client.client.get_connection(conn_id)

    assert isinstance(exc_info.value.__cause__, vault_utils.InvalidFilenameError)
    airflow_client.backend.read_secret.assert_not_called()


def test_airflow_client_preserves_vault_errors(airflow_client: AirflowClientFixture) -> None:
    """Verify that Airflow connection reads propagate the original Vault error.

    Args:
        airflow_client: Airflow client and mocked Vault dependencies.

    Raises:
        AssertionError: If the propagated error is not the original Vault exception.
        pytest.fail.Exception: If the Vault error is swallowed.
    """
    error = vault_utils.VaultPermissionDeniedError("access denied")
    airflow_client.backend.read_secret.side_effect = error

    with pytest.raises(vault_utils.VaultPermissionDeniedError) as exc_info:
        airflow_client.client.get_connection("warehouse")

    assert exc_info.value is error
