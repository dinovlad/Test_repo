"""Provides a client for interacting with Vault, a secrets management tool.

Based on
https://scm.x5.ru/mlops-public/libraries/x5-transport/-/blob/4923447c152c025b03e9daa2891c64f4ac253df0/x5_transport/vault.py
"""

import abc
import random
import threading
import time
from pathlib import PurePosixPath
from typing import ClassVar, Generic, Literal, Self, TypeVar

import hvac
import pydantic
import pydantic_settings
import requests
from hvac import exceptions as vault_exceptions


class InvalidFilenameError(Exception):
    """The value is not a valid POSIX filename."""


def validate_filename(filename: str) -> str:
    """Validate a single POSIX filename without accessing the filesystem.

    Args:
        filename:
            The filename to validate. Backslashes and Unicode characters are allowed.

    Returns:
        The original filename, unchanged.

    Raises:
        InvalidFilenameError:
            If the filename is empty, is '.' or '..', or contains '/' or a null byte.
    """
    if filename in {"", ".", ".."} or "/" in filename or "\0" in filename:
        raise InvalidFilenameError(f"Invalid POSIX filename: {filename!r}.")
    return filename


class VaultValueError(Exception):
    """Secret values stored in Vault are invalid."""


class VaultQueryError(Exception):
    """Vault query failed."""


class VaultPermissionDeniedError(VaultQueryError):
    """Vault token does not have sufficient permissions to access the requested secret."""


class VaultInvalidPathError(VaultQueryError):
    """The specified path in Vault does not exist or is not a valid path for secrets."""


class VaultMetaConfig(pydantic_settings.BaseSettings, extra="ignore", frozen=True, case_sensitive=False):
    """Vault additional settings loaded from an env file or environment variables.

    Attributes:
        approle_path:
            The path to the secrets directory containing AppRole credentials files.
    """

    approle_path: pydantic.DirectoryPath | None = None


class VaultAuthConfig(
    pydantic_settings.BaseSettings,
    extra="ignore",
    frozen=True,
    validate_by_name=True,
    case_sensitive=False,
):
    """Vault AppRole authentication settings loaded from an env file or secrets directory.

    Attributes:
        url:
            The base URL of the Vault server.
        role_id:
            The AppRole role ID.
        secret_id:
            The AppRole secret ID.
        verify:
            Either a boolean controlling TLS certificate verification,
            or a path to a CA bundle file or directory with certificates
            of trusted CAs. Passed to `hvac.Client` as the `verify` parameter.
    """

    url: str
    role_id: str = pydantic.Field(validation_alias="role-id")
    secret_id: pydantic.SecretStr = pydantic.Field(validation_alias="secret-id")
    verify: bool | str = pydantic.Field(default=True, union_mode="left_to_right")


VaultMetaConfigType = TypeVar("VaultMetaConfigType", bound=VaultMetaConfig)


class ABCVaultConfig(
    pydantic.BaseModel,
    Generic[VaultMetaConfigType],
    extra="forbid",
    frozen=True,
    metaclass=abc.ABCMeta,
):
    """Vault settings.

    Subclass this config and set `meta_cls` to a subclass of `VaultMetaConfig`.

    Attributes:
        meta_cls:
            The Vault additional settings class.
        meta:
            The Vault additional settings.
        auth:
            The Vault AppRole authentication settings.
    """

    meta_cls: ClassVar[type[VaultMetaConfigType]]
    meta: VaultMetaConfigType
    auth: VaultAuthConfig

    @classmethod
    @pydantic.validate_call
    def from_env_file(cls, env_file: str | pydantic.FilePath) -> Self:
        """Build the whole Vault configuration from an env file.

        Args:
            env_file:
                The path to the env file containing Vault settings.
                The file must define all the `meta_cls` parameters.
                Optionally, it can contain other parameters from `VaultAuthConfig`,
                which otherwise would be loaded from the secrets directory
                specified by `meta_cls.approle_path`.

        Returns:
            An initialized config instance.
        """
        meta = cls.meta_cls(_env_file=env_file)
        auth = VaultAuthConfig(_env_file=env_file, _secrets_dir=meta.approle_path)
        return cls(meta=meta, auth=auth)

    @classmethod
    def from_env(cls) -> Self:
        """Build the whole Vault configuration from environment variables.

        Similar to `from_env_file()` but reads the environment variables
        instead of an env file.

        The environment variable names must correspond to the names
        of the respective fields of `cls.meta_cls` and `VaultAuthConfig`
        with the prefix 'VAULT_'. The names are case-insensitive.

        Returns:
            An initialized config instance.
        """
        meta = cls.meta_cls(
            _env_prefix="VAULT_",
            _env_prefix_target="all",
        )
        auth = VaultAuthConfig(
            _secrets_dir=meta.approle_path,
            # The prefix will be used for reading fields by name
            # but not by alias, e.g., `url` will be read from 'VAULT_URL' whereas
            # `role_id` will be read from file 'role-id' (alias without the prefix).
            # Still, `role_id` can be read from 'VAULT_ROLE_ID'
            # if such environment variable is provided.
            _env_prefix="VAULT_",
            _env_prefix_target="variable",
        )
        return cls(meta=meta, auth=auth)


class SimpleVaultConfig(ABCVaultConfig[VaultMetaConfig]):
    """Vault settings.

    Attributes:
        meta:
            The Vault additional settings.
        auth:
            The Vault AppRole authentication settings.
    """

    meta_cls: ClassVar[type[VaultMetaConfig]] = VaultMetaConfig


_VAULT_VALUE_TYPE_ADAPTER = pydantic.TypeAdapter(dict[str, str])


class VaultClient:
    """Client for interaction with Vault."""

    def __init__(
        self,
        auth: VaultAuthConfig,
        *,
        initial_delay: float = 0.1,
        delay_jitter: float = 0.05,
        attempts: int = 3,
    ) -> None:
        """Initialize the client.

        Args:
            auth:
                The Vault authentication configuration.
            initial_delay:
                The initial delay in seconds before a retry attempt. Must be non-negative.
            delay_jitter:
                The maximum random jitter in seconds added to or subtracted from `initial_delay`.
                Must be non-negative.
            attempts:
                The number of attempts to connect and read a secret from Vault. Transient failures
                `requests.ConnectionError` and generic `vault_exceptions.VaultError` are retried.
                Must be a positive integer.

        Raises:
            ValueError:
                If `initial_delay` or `delay_jitter` is negative,
                or if `attempts` is non-positive.
        """
        if initial_delay < 0:
            raise ValueError(f"`initial_delay` must be non-negative, got {initial_delay}.")
        if delay_jitter < 0:
            raise ValueError(f"`delay_jitter` must be non-negative, got {delay_jitter}.")
        if attempts <= 0:
            raise ValueError(f"`attempts` must be a positive integer, got {attempts}.")

        self.auth = auth.model_copy(deep=True)
        self._client = hvac.Client(url=self.auth.url, verify=self.auth.verify)
        self._lock = threading.Lock()
        self.initial_delay = initial_delay
        self.delay_jitter = delay_jitter
        self.attempts = attempts

    def read_secret(self, *, path: str, mount_point: str) -> dict[str, str]:
        """Extract a secret from Vault by path and mount point.

        Args:
            path:
                The path that the secret will be extracted from.
            mount_point:
                The Vault mount point.

        Returns:
            A dictionary containing the secret data.

        Raises:
            VaultPermissionDeniedError:
                If the Vault token does not have sufficient permissions to access the secret.
            VaultInvalidPathError:
                If the specified path or mount point does not exist in Vault.
            VaultQueryError:
                If the secret cannot be read from Vault for any other reason, or if
                a connection error occurs on every attempt (retried up to `attempts` times).
            VaultValueError:
                If the fetched secret is not `dict[str, str]`.
        """
        with self._lock:
            last_exc: Exception | None = None
            for _ in range(self.attempts):
                try:
                    if not self._client.is_authenticated():
                        # This is `hvac.api.auth_methods.AppRole.login()`
                        self._client.auth.approle.login(
                            role_id=self.auth.role_id,
                            secret_id=self.auth.secret_id.get_secret_value(),
                        )

                    response = self._client.secrets.kv.v2.read_secret(mount_point=mount_point, path=path)

                    try:
                        secret_creds = response["data"]["data"]
                        try:
                            _VAULT_VALUE_TYPE_ADAPTER.validate_python(secret_creds)
                        except pydantic.ValidationError as ve:
                            raise VaultValueError from ve
                        return secret_creds
                    except (KeyError, TypeError) as vde:
                        raise VaultQueryError(f"Malformed data received from Vault. Error: {vde}") from vde

                except vault_exceptions.Forbidden as fe:
                    raise VaultPermissionDeniedError(f"Permission denied. Check token status. Error: {fe}") from fe
                except vault_exceptions.InvalidPath as ipe:
                    raise VaultInvalidPathError(
                        f"Invalid path: {path} on mount point {mount_point}. Error: {ipe}",
                    ) from ipe
                except (requests.ConnectionError, vault_exceptions.VaultError) as e:
                    last_exc = e
                    sleep_time = max(self.initial_delay + random.uniform(-self.delay_jitter, self.delay_jitter), 0)
                    time.sleep(sleep_time)

            raise VaultQueryError(f"Failed to read the secret after {self.attempts} attempts.") from last_exc


class VaultAirflowMetaConfig(VaultMetaConfig):
    """Airflow-related Vault settings loaded from an env file.

    Attributes:
        env:
            The environment mode.
        mount_point:
            The Vault mount point where Airflow secrets are stored.
    """

    env: Literal["dev", "stage", "prod"]
    mount_point: str


class VaultAirflowConfig(ABCVaultConfig[VaultAirflowMetaConfig]):
    """Configuration for reading Airflow connection credentials from Vault.

    Attributes:
        meta_cls:
            The Vault Airflow additional settings class.
        meta:
            The Vault Airflow additional settings.
        auth:
            The Vault AppRole authentication settings.
    """

    meta_cls: ClassVar[type[VaultAirflowMetaConfig]] = VaultAirflowMetaConfig


class VaultAirflowClient:
    """Client for reading Airflow connection credentials from Vault."""

    def __init__(
        self,
        config: VaultAirflowConfig,
    ) -> None:
        """Initialize the client.

        Args:
            config:
                The Vault Airflow configuration containing the secrets paths
                and the authentication settings.
        """
        self.config = config.model_copy(deep=True)
        self._vc = VaultClient(self.config.auth)

    def get_connection(self, conn_id: str) -> dict[str, str]:
        """Read Airflow connection credentials from Vault.

        Args:
            conn_id:
                The Airflow connection ID. Must be a filename, not a path.

        Returns:
            A dictionary containing the Airflow connection credentials.

        Raises:
            InvalidFilenameError:
                If `conn_id` is not a valid POSIX filename.
            Errors raised by `VaultClient.read_secret()`.
        """
        try:
            conn_id = validate_filename(conn_id)
        except InvalidFilenameError as exc:
            raise InvalidFilenameError(f"`conn_id` must be a valid filename, got {conn_id!r}.") from exc

        connections_dir = PurePosixPath(self.config.meta.env) / "airflow" / "connections"
        conn_secrets_path = connections_dir / conn_id

        return self._vc.read_secret(mount_point=self.config.meta.mount_point, path=str(conn_secrets_path))
