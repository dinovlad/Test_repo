"""This module contains classes and functions for interacting with the MLflow Tracking Server.
To use this module, the required environment variables must be set, either directly
or via an env file passed to `setup_mlflow_env()`.
"""

import contextlib
import io
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TypeVar

import mlflow
import mlflow.artifacts
import pydantic
import pydantic_settings


logger = logging.getLogger(__name__)


class MlflowConfig(
    pydantic_settings.BaseSettings,
    extra="ignore",
    frozen=True,
    case_sensitive=True,
):
    """Configuration settings for the MLflow Tracking Server.

    Loaded from environment variables by default.

    Attributes:
        MLFLOW_TRACKING_URI:
            The URI of the MLflow Tracking Server.
        MLFLOW_TRACKING_USERNAME:
            The username for authenticating with the Tracking Server.
        MLFLOW_TRACKING_PASSWORD:
            The password for authenticating with the Tracking Server.
        MLFLOW_TRACKING_SERVER_CERT_PATH:
            The path to a certificate file for verifying the server's TLS identity.
    """

    MLFLOW_TRACKING_URI: str = pydantic.Field(min_length=1)
    MLFLOW_TRACKING_USERNAME: str = pydantic.Field(min_length=1)
    MLFLOW_TRACKING_PASSWORD: pydantic.SecretStr = pydantic.Field(min_length=1)
    MLFLOW_TRACKING_SERVER_CERT_PATH: pydantic.FilePath

    @pydantic.field_serializer("*")
    def dump_as_str(self, value: str | pydantic.SecretStr | pydantic.FilePath) -> str:
        """Dump each field as a string.

        Args:
            value:
                The field value to serialize.

        Returns:
            The string representation of the value, with secret strings revealed.
        """
        if isinstance(value, pydantic.SecretStr):
            return value.get_secret_value()
        return str(value)


@pydantic.validate_call
def setup_mlflow_env(
    *,
    env_file: pydantic.FilePath | None = None,
    env_override: bool = False,
) -> None:
    """Set up MLflow by adding/validating the required environment variables.

    Args:
        env_file:
            The path to a .env file to load into the environment variables.
            If None, the existing environment variables are checked and validated.
        env_override:
            Whether variables from `env_file` should override existing env vars.

    Raises:
        pydantic.ValidationError:
            - If `env_file` is provided but does not exist.
            - If env variables required for MLflow are missing or empty.
    """
    config = MlflowConfig(_env_file=env_file)
    if env_override:
        os.environ.update(config.model_dump())
    else:
        os.environ.update(config.model_dump() | os.environ)


class NoActiveRunError(Exception):
    """No active MLflow run."""


M = TypeVar("M", bound=pydantic.BaseModel)


def log_pydantic(data: M, artifact_path: str) -> None:
    """Serialize a pydantic model to JSON and log it as an MLflow text artifact.

    Args:
        data:
            The pydantic model instance to serialize and log.
        artifact_path:
            The path of the artifact within the active run.

    Raises:
        NoActiveRunError:
            If there is no active MLflow run.
    """
    if mlflow.active_run() is None:
        raise NoActiveRunError
    text = data.model_dump_json()
    mlflow.log_text(text, artifact_path)


def load_pydantic(model_cls: type[M], artifact_path: str, *, run_id: str) -> M:
    """Load a JSON text artifact and validate it into a pydantic model.

    Args:
        model_cls:
            The pydantic model class to validate the artifact into.
        artifact_path:
            The path of the artifact within the run.
        run_id:
            The ID of the run containing the artifact.

    Returns:
        The pydantic model instance built from the artifact contents.
    """
    text = mlflow.artifacts.load_text(f"runs:/{run_id}/{artifact_path}")
    return model_cls.model_validate_json(text)


def log_binary(file: BinaryIO, artifact_path: str) -> None:
    """Read a binary file object and log its contents as an MLflow artifact.

    Args:
        file:
            The binary file object to read and log.
        artifact_path:
            The relative path of the artifact within the active run.
            Must not be absolute or contain '..' components.

    Raises:
        NoActiveRunError:
            If there is no active MLflow run.
        ValueError:
            If `artifact_path` is absolute or contains '..' components.
    """
    if mlflow.active_run() is None:
        raise NoActiveRunError
    artifact_posix_path = PurePosixPath(artifact_path)
    if artifact_posix_path.is_absolute() or ".." in artifact_posix_path.parts:
        raise ValueError(f"`artifact_path` must be a relative path without '..' components, got {artifact_path!r}.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, *artifact_posix_path.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as dst:
            shutil.copyfileobj(file, dst)
        if len(artifact_posix_path.parents) > 1:
            artifact_dir: str | None = str(artifact_posix_path.parent)
        else:
            artifact_dir = None
        mlflow.log_artifact(local_path=str(path), artifact_path=artifact_dir)


def log_file(local_path: Path, artifact_path: str) -> None:
    """Open a local file in binary mode and log it as an MLflow artifact.

    Args:
        local_path:
            The path of the local file to log.
        artifact_path:
            The relative path of the artifact within the active run.
            Must not be absolute or contain '..' components.

    Raises:
        NoActiveRunError:
            If there is no active MLflow run.
        ValueError:
            If `artifact_path` is absolute or contains '..' components.
    """
    with open(local_path, "rb") as file:
        log_binary(file, artifact_path)


@contextlib.contextmanager
def log_buffer(artifact_path: str) -> Iterator[BinaryIO]:
    """Provide an in-memory buffer and log its contents as an artifact on exit.

    The contents are logged only if the context exits without an error.

    Args:
        artifact_path:
            The relative path of the artifact within the active run.
            Must not be absolute or contain '..' components.

    Yields:
        A binary buffer; its contents are logged to MLflow when the
        context manager exits without an error.

    Raises:
        NoActiveRunError:
            If there is no active MLflow run.
        ValueError:
            If `artifact_path` is absolute or contains '..' components.
    """
    buffer = io.BytesIO()
    yield buffer
    buffer.seek(0)
    log_binary(buffer, artifact_path)


@contextlib.contextmanager
def load_binary(artifact_path: str, *, run_id: str) -> Iterator[BinaryIO]:
    """Download an artifact into a temporary directory and open it for reading.

    Args:
        artifact_path:
            The path of the artifact within the run.
        run_id:
            The ID of the run containing the artifact.

    Yields:
        A binary file object with the downloaded artifact contents.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=tmp_dir,
        )
        with open(path, "rb") as file:
            yield file


@dataclass(frozen=True, slots=True)
class LogResult:
    """Result metadata for a completed MLflow logging operation.

    Attributes:
        experiment_id:
            The MLflow experiment ID.
        run_id:
            The MLflow run ID.
        model_name:
            The name of the logged model artifact.
        model_uri:
            The URI of the logged model artifact.
    """

    experiment_id: str
    run_id: str
    model_name: str
    model_uri: str


class MlflowClient:
    """Wrapper around MLflow client for tracking operations.

    Attributes:
        _client: The MLflow client instance.
    """

    def __init__(
        self,
        *,
        dotenv_path: Path | str,
        dotenv_override: bool = False,
    ) -> None:
        """Initialize the MLflow client.

        Args:
            dotenv_path:
                Path to a .env file to load for MLflow configuration.
            dotenv_override:
                Whether variables from `dotenv_path` should override existing env vars.

        Raises:
            pydantic.ValidationError:
                - If `dotenv_path` does not exist.
                - If env variables required for MLflow are missing or empty.
        """
        setup_mlflow_env(env_file=Path(dotenv_path).expanduser(), env_override=dotenv_override)
        self._client = mlflow.MlflowClient()

    def log_pyfunc_model(
        self,
        *,
        python_model: mlflow.pyfunc.PythonModel,  # type: ignore[name-defined]
        experiment_name: str,
        model_name: str = "model",
        run_name: str | None = None,
        params: dict[str, object] | None = None,
        metrics: dict[str, float | int] | None = None,
        tags: dict[str, str] | None = None,
        input_example: Any | None = None,
        run_artifacts: list[str] | None = None,
        model_artifacts: dict[str, str] | None = None,
    ) -> LogResult:
        """Execute a run, log parameters/metrics, and save the PyFunc model artifact.

        Args:
            python_model:
                The PythonModel instance to log.
            experiment_name:
                The name of the MLflow experiment.
            model_name:
                The name for the model artifact. Defaults to 'model'.
            run_name:
                The name for the MLflow run.
            params:
                The dictionary of parameters to log.
            metrics:
                The dictionary of metrics to log.
            tags:
                The dictionary of tags to apply to the run.
            input_example:
                The input example for the model.
            run_artifacts:
                The list of artifact paths to log.
            model_artifacts:
                The dictionary (name, path) of model artifacts to log.
                Later these artifacts will be available as context.artifacts in the model.

        Returns:
            LogResult object containing `experiment_id`, `run_id`, `model_name`, and `model_uri`.

        Raises:
            ValueError:
                If `experiment_name` or `model_name` is empty.
            RuntimeError:
                If MLflow logging fails.
        """
        if not experiment_name.strip():
            raise ValueError("`experiment_name` must be a non-empty string.")
        if not model_name.strip():
            raise ValueError("`model_name` must be a non-empty string.")

        run_id: str | None = None
        try:
            exp_id = self._get_or_create_experiment_id(experiment_name)

            run = self._client.create_run(
                experiment_id=exp_id,
                tags=dict(tags or {}),
                run_name=run_name,
            )
            run_id = str(run.info.run_id)

            params_payload = [mlflow.entities.Param(str(k), str(v)) for k, v in (params or {}).items()]
            metrics_payload = [
                mlflow.entities.Metric(str(k), float(v), timestamp=int(time.time() * 1000), step=0)
                for k, v in (metrics or {}).items()
            ]
            self._client.log_batch(run_id=run_id, metrics=metrics_payload, params=params_payload)

            for path in run_artifacts or []:
                local_path = Path(path).expanduser()
                if not local_path.exists():
                    logger.warning("artifact path does not exist (skipping): %s", local_path)
                    continue
                self._client.log_artifact(run_id=run_id, local_path=local_path)

            # Ensure that the run is active to allow fluent API usage for model logging
            with mlflow.start_run(run_id=run_id):
                model_info = mlflow.pyfunc.log_model(
                    name=model_name,
                    python_model=python_model,
                    input_example=input_example,
                    artifacts=model_artifacts,
                )

            return LogResult(
                experiment_id=exp_id,
                run_id=run_id,
                model_name=model_name,
                model_uri=model_info.model_uri,
            )
        except Exception as exc:
            if run_id is not None:
                try:
                    self._client.set_terminated(run_id, status="FAILED")
                except mlflow.exceptions.MlflowException:
                    logger.exception("Failed to set MLflow run status=FAILED (run_id=%s).", run_id)
            raise RuntimeError(f"MLflow logging failed: {exc}") from exc

    def register_model(
        self,
        *,
        model_uri: str,
        registered_model_name: str,
        alias: str | None = None,
    ) -> None:
        """Register a model artifact from an existing run into the Model Registry.

        Args:
            model_uri:
                The URI of the logged model artifact.
            registered_model_name:
                The name in the Model Registry.
            alias:
                The alias to apply to the model version.

        Raises:
            ValueError:
                If `model_uri` or `registered_model_name` is empty.
            RuntimeError:
                If model registration fails.
        """
        if not model_uri.strip():
            raise ValueError("`model_uri` must be a non-empty string.")
        if not registered_model_name.strip():
            raise ValueError("`registered_model_name` must be a non-empty string.")

        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            if alias:
                self._client.set_registered_model_alias(
                    name=registered_model_name,
                    alias=alias,
                    version=result.version,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to register model '{registered_model_name}' from model URI '{model_uri}': {exc}",
            ) from exc

    def load_model(self, model_uri: str) -> mlflow.pyfunc.PyFuncModel:  # type: ignore[name-defined]
        """Load a model from a URI.

        Args:
            model_uri:
                The URI of the model.

        Returns:
            The loaded model.

        Raises:
            RuntimeError:
                If the model cannot be loaded.
        """
        try:
            return mlflow.pyfunc.load_model(model_uri=model_uri)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from '{model_uri}': {exc}") from exc

    def _get_or_create_experiment_id(self, experiment_name: str) -> str:
        """Get an existing experiment by name or create a new one.

        If an experiment with the given name exists but is deleted, it will be
        restored automatically.

        Args:
            experiment_name:
                The name of the MLflow experiment.

        Returns:
            The experiment ID.

        Raises:
            mlflow.exceptions.RestException:
                If the experiment cannot be found or created.
        """
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return self._restore_if_deleted(experiment_name, experiment)

        try:
            return self._client.create_experiment(experiment_name)
        # Check for race condition
        except mlflow.exceptions.RestException as exc:
            if exc.error_code == "RESOURCE_ALREADY_EXISTS":
                experiment = self._client.get_experiment_by_name(experiment_name)
                if experiment is not None:
                    return self._restore_if_deleted(experiment_name, experiment)
            raise exc

    def _restore_if_deleted(self, experiment_name: str, experiment: mlflow.entities.Experiment) -> str:
        """Restore an experiment if deleted and return its ID.

        Args:
            experiment_name:
                The name of the experiment.
            experiment:
                The MLflow experiment entity.

        Returns:
            The ID of the experiment.
        """
        experiment_id = str(experiment.experiment_id)
        if experiment.lifecycle_stage == "deleted":
            logger.info("Restoring deleted experiment '%s' (ID=%s).", experiment_name, experiment_id)
            self._client.restore_experiment(experiment_id)
        return experiment_id
