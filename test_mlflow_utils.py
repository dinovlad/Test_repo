"""Unit tests for MLflow helpers; all tracking-server calls are mocked."""

import io
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast
from unittest.mock import MagicMock, Mock, patch

import mlflow
import pydantic
import pytest

import mlflow_utils

# MLflow's RestException initializer has no annotations.
make_rest_exception = cast(
    Callable[[dict[str, str]], mlflow.exceptions.RestException],
    mlflow.exceptions.RestException,
)


class ExampleModel(pydantic.BaseModel):
    name: str
    count: int


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    """Restore variables written directly by setup_mlflow_env after each test."""
    with patch.dict(os.environ):
        for name in mlflow_utils.MlflowConfig.model_fields:
            os.environ.pop(name, None)
            os.environ.pop(name.lower(), None)
        yield


@pytest.fixture
def config_values(tmp_path: Path) -> dict[str, str]:
    certificate = tmp_path / "ca.pem"
    certificate.write_text("test certificate", encoding="utf-8")
    return {
        "MLFLOW_TRACKING_URI": "https://mlflow.example.test",
        "MLFLOW_TRACKING_USERNAME": "test-user",
        "MLFLOW_TRACKING_PASSWORD": "test-password",
        "MLFLOW_TRACKING_SERVER_CERT_PATH": str(certificate),
    }


@pytest.fixture
def env_file(tmp_path: Path, config_values: dict[str, str]) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "".join(f"{name}={value}\n" for name, value in config_values.items()),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[mlflow_utils.MlflowClient, Mock]:
    sdk_client = Mock(spec=mlflow.MlflowClient)
    sdk_client.get_experiment_by_name.return_value = mlflow.entities.Experiment(
        experiment_id="experiment-1",
        name="test-experiment",
        artifact_location="file:///unused",
        lifecycle_stage="active",
    )
    sdk_client.create_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1"),
    )
    monkeypatch.setattr(mlflow_utils, "setup_mlflow_env", Mock())
    monkeypatch.setattr(mlflow, "MlflowClient", Mock(return_value=sdk_client))
    return mlflow_utils.MlflowClient(dotenv_path="unused.env"), sdk_client


@pytest.fixture
def model_logging(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, Mock]:
    start_run = MagicMock()
    log_model = Mock(return_value=SimpleNamespace(model_uri="models:/model-1"))
    monkeypatch.setattr(mlflow, "start_run", start_run)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    return start_run, log_model


def test_config_serializes_secrets_and_paths_as_strings(config_values: dict[str, str]) -> None:
    config = mlflow_utils.MlflowConfig.model_validate(config_values)

    assert config.model_dump() == config_values
    assert isinstance(config.MLFLOW_TRACKING_PASSWORD, pydantic.SecretStr)
    assert isinstance(config.MLFLOW_TRACKING_SERVER_CERT_PATH, Path)
    assert config_values["MLFLOW_TRACKING_PASSWORD"] not in repr(config)
    with pytest.raises(pydantic.ValidationError):
        setattr(config, "MLFLOW_TRACKING_URI", "https://other.example.test")


@pytest.mark.parametrize("field", list(mlflow_utils.MlflowConfig.model_fields))
@pytest.mark.parametrize("invalid", ["", None])
def test_config_rejects_empty_or_missing_fields(
    config_values: dict[str, str],
    field: str,
    invalid: str | None,
) -> None:
    values = dict(config_values)
    if invalid is None:
        values.pop(field)
    else:
        values[field] = invalid

    with pytest.raises(pydantic.ValidationError) as error:
        mlflow_utils.MlflowConfig.model_validate(values)

    assert field in {entry["loc"][0] for entry in error.value.errors()}


def test_config_requires_existing_certificate_file(config_values: dict[str, str], tmp_path: Path) -> None:
    config_values["MLFLOW_TRACKING_SERVER_CERT_PATH"] = str(tmp_path / "missing.pem")

    with pytest.raises(pydantic.ValidationError):
        mlflow_utils.MlflowConfig.model_validate(config_values)


@pytest.mark.parametrize("env_override", [False, True])
def test_setup_environment_loads_dotenv(
    env_file: Path,
    config_values: dict[str, str],
    env_override: bool,
) -> None:
    mlflow_utils.setup_mlflow_env(env_file=env_file, env_override=env_override)

    assert {name: os.environ[name] for name in config_values} == config_values


def test_setup_environment_preserves_existing_values(
    env_file: Path,
    config_values: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "existing-user")
    monkeypatch.setenv("UNRELATED_SETTING", "keep-me")

    mlflow_utils.setup_mlflow_env(env_file=env_file)

    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "existing-user"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == config_values["MLFLOW_TRACKING_PASSWORD"]
    assert os.environ["UNRELATED_SETTING"] == "keep-me"


def test_setup_environment_can_override_existing_values(
    env_file: Path,
    config_values: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in config_values:
        monkeypatch.setenv(name, "existing-value")
    monkeypatch.setenv("UNRELATED_SETTING", "keep-me")

    mlflow_utils.setup_mlflow_env(env_file=env_file, env_override=True)

    assert {name: os.environ[name] for name in config_values} == config_values
    assert os.environ["UNRELATED_SETTING"] == "keep-me"


@pytest.mark.parametrize("field", list(mlflow_utils.MlflowConfig.model_fields))
@pytest.mark.parametrize("invalid", [None, ""])
def test_setup_environment_override_requires_valid_dotenv_values(
    env_file: Path,
    config_values: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: str | None,
) -> None:
    for name, value in config_values.items():
        monkeypatch.setenv(name, value)
    file_values = dict(config_values)
    if invalid is None:
        file_values.pop(field)
    else:
        file_values[field] = invalid
    env_file.write_text(
        "".join(f"{name}={value}\n" for name, value in file_values.items()),
        encoding="utf-8",
    )
    before = dict(os.environ)

    with pytest.raises(pydantic.ValidationError) as error:
        mlflow_utils.setup_mlflow_env(env_file=env_file, env_override=True)

    assert field in {entry["loc"][0] for entry in error.value.errors()}
    assert dict(os.environ) == before


def test_setup_environment_override_requires_env_file(
    config_values: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in config_values.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)

    with pytest.raises(pydantic.ValidationError) as error:
        mlflow_utils.setup_mlflow_env(env_override=True)

    assert {entry["loc"][0] for entry in error.value.errors()} == set(config_values)
    assert dict(os.environ) == before


def test_setup_environment_validates_existing_variables(
    config_values: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in config_values.items():
        monkeypatch.setenv(name, value)

    mlflow_utils.setup_mlflow_env()

    assert {name: os.environ[name] for name in config_values} == config_values


def test_setup_environment_rejects_missing_file_without_mutating_environment(tmp_path: Path) -> None:
    before = dict(os.environ)

    with pytest.raises(pydantic.ValidationError):
        mlflow_utils.setup_mlflow_env(env_file=tmp_path / "missing.env")

    assert dict(os.environ) == before


def test_setup_environment_rejects_lowercase_variables(
    config_values: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in config_values.items():
        monkeypatch.setenv(name.lower(), value)

    with pytest.raises(pydantic.ValidationError):
        mlflow_utils.setup_mlflow_env()


def test_log_pydantic_serializes_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ExampleModel(name="sample", count=2)
    log_text = Mock()
    monkeypatch.setattr(mlflow, "active_run", Mock(return_value=object()))
    monkeypatch.setattr(mlflow, "log_text", log_text)

    mlflow_utils.log_pydantic(model, "config/model.json")

    log_text.assert_called_once_with(model.model_dump_json(), "config/model.json")


@pytest.mark.parametrize("kind", ["pydantic", "binary"])
def test_logging_requires_active_run(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    monkeypatch.setattr(mlflow, "active_run", Mock(return_value=None))
    log_text = Mock()
    log_artifact = Mock()
    monkeypatch.setattr(mlflow, "log_text", log_text)
    monkeypatch.setattr(mlflow, "log_artifact", log_artifact)

    def log_without_run() -> None:
        if kind == "pydantic":
            mlflow_utils.log_pydantic(ExampleModel(name="sample", count=2), "model.json")
        else:
            mlflow_utils.log_binary(io.BytesIO(b"data"), "data.bin")

    with pytest.raises(mlflow_utils.NoActiveRunError):
        log_without_run()

    log_text.assert_not_called()
    log_artifact.assert_not_called()


def test_load_pydantic_uses_run_uri_and_validates_json(monkeypatch: pytest.MonkeyPatch) -> None:
    load_text = Mock(return_value='{"name":"sample","count":2}')
    monkeypatch.setattr(mlflow.artifacts, "load_text", load_text)

    result = mlflow_utils.load_pydantic(ExampleModel, "config/model.json", run_id="run-1")

    assert result == ExampleModel(name="sample", count=2)
    load_text.assert_called_once_with("runs:/run-1/config/model.json")


@pytest.mark.parametrize("text", ["not json", '{"name":"sample","count":"invalid"}'])
def test_load_pydantic_propagates_validation_errors(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(mlflow.artifacts, "load_text", Mock(return_value=text))

    with pytest.raises(pydantic.ValidationError):
        mlflow_utils.load_pydantic(ExampleModel, "model.json", run_id="run-1")


@pytest.mark.parametrize(
    ("artifact_path", "artifact_dir"),
    [("data.bin", None), ("nested/more/data.bin", "nested/more")],
)
def test_log_binary_uploads_contents_and_cleans_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    artifact_path: str,
    artifact_dir: str | None,
) -> None:
    uploaded: list[Path] = []

    def upload(*, local_path: str, artifact_path: str | None) -> None:
        path = Path(local_path)
        assert path.name == "data.bin"
        assert path.read_bytes() == b"\x00\xffpayload"
        assert artifact_path == artifact_dir
        uploaded.append(path)

    monkeypatch.setattr(mlflow, "active_run", Mock(return_value=object()))
    log_artifact = Mock(side_effect=upload)
    monkeypatch.setattr(mlflow, "log_artifact", log_artifact)

    mlflow_utils.log_binary(io.BytesIO(b"\x00\xffpayload"), artifact_path)

    log_artifact.assert_called_once()
    assert not uploaded[0].exists()


@pytest.mark.parametrize("artifact_path", ["/absolute.bin", "../escape.bin", "safe/../escape.bin"])
def test_log_binary_rejects_unsafe_paths(monkeypatch: pytest.MonkeyPatch, artifact_path: str) -> None:
    monkeypatch.setattr(mlflow, "active_run", Mock(return_value=object()))
    log_artifact = Mock()
    monkeypatch.setattr(mlflow, "log_artifact", log_artifact)
    source = io.BytesIO(b"payload")

    with pytest.raises(ValueError, match="relative path"):
        mlflow_utils.log_binary(source, artifact_path)

    log_artifact.assert_not_called()
    assert source.tell() == 0


def test_log_binary_cleans_temporary_directory_on_upload_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    uploaded: list[Path] = []
    failure = RuntimeError("upload failed")

    def upload(*, local_path: str, artifact_path: str | None) -> None:
        uploaded.append(Path(local_path))
        raise failure

    monkeypatch.setattr(mlflow, "active_run", Mock(return_value=object()))
    monkeypatch.setattr(mlflow, "log_artifact", upload)

    with pytest.raises(RuntimeError) as error:
        mlflow_utils.log_binary(io.BytesIO(b"payload"), "data.bin")

    assert error.value is failure
    assert not uploaded[0].parent.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_log_file_opens_binary_stream_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail: bool,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"\x00\xffpayload")
    streams: list[BinaryIO] = []

    def log_binary(stream: BinaryIO, artifact_path: str) -> None:
        streams.append(stream)
        assert stream.read() == b"\x00\xffpayload"
        assert artifact_path == "renamed.bin"
        if fail:
            raise RuntimeError("upload failed")

    monkeypatch.setattr(mlflow_utils, "log_binary", log_binary)

    if fail:
        with pytest.raises(RuntimeError, match="upload failed"):
            mlflow_utils.log_file(source, "renamed.bin")
    else:
        mlflow_utils.log_file(source, "renamed.bin")

    assert streams[0].closed


def test_log_buffer_rewinds_before_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[bytes, str]] = []

    def capture_binary(stream: BinaryIO, path: str) -> None:
        logged.append((stream.read(), path))

    monkeypatch.setattr(mlflow_utils, "log_binary", capture_binary)

    with mlflow_utils.log_buffer("data.bin") as buffer:
        buffer.write(b"complete payload")
        assert logged == []

    assert logged == [(b"complete payload", "data.bin")]


def test_log_buffer_skips_logging_when_context_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    log_binary = Mock()
    monkeypatch.setattr(mlflow_utils, "log_binary", log_binary)

    def write_incomplete_buffer() -> None:
        with mlflow_utils.log_buffer("data.bin") as buffer:
            buffer.write(b"incomplete")
            raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        write_incomplete_buffer()

    log_binary.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_load_binary_downloads_and_cleans_up(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    directories: list[Path] = []

    def download(*, run_id: str, artifact_path: str, dst_path: str) -> str:
        assert run_id == "run-1"
        assert artifact_path == "nested/data.bin"
        directory = Path(dst_path)
        directories.append(directory)
        path = directory / "data.bin"
        path.write_bytes(b"\x00\xffpayload")
        return str(path)

    download_artifacts = Mock(side_effect=download)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download_artifacts)
    streams: list[BinaryIO] = []

    def consume() -> None:
        with mlflow_utils.load_binary("nested/data.bin", run_id="run-1") as stream:
            streams.append(stream)
            assert stream.read() == b"\x00\xffpayload"
            if fail:
                raise RuntimeError("consumer failed")

    if fail:
        with pytest.raises(RuntimeError, match="consumer failed"):
            consume()
    else:
        consume()

    download_artifacts.assert_called_once()
    assert streams[0].closed
    assert not directories[0].exists()


def test_client_initializes_environment_before_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_env_file = Path.home() / "mlflow.env"
    calls = Mock()
    sdk_client = Mock()
    calls.create_client.return_value = sdk_client
    monkeypatch.setattr(mlflow_utils, "setup_mlflow_env", calls.setup_env)
    monkeypatch.setattr(mlflow, "MlflowClient", calls.create_client)

    result = mlflow_utils.MlflowClient(dotenv_path="~/mlflow.env", dotenv_override=True)

    calls.setup_env.assert_called_once_with(env_file=expected_env_file, env_override=True)
    calls.create_client.assert_called_once_with()
    assert [call[0] for call in calls.mock_calls] == ["setup_env", "create_client"]
    assert result._client is sdk_client


@pytest.mark.parametrize("lifecycle_stage", ["active", "deleted"])
def test_client_reuses_and_restores_existing_experiment(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    lifecycle_stage: str,
) -> None:
    wrapper, sdk = client
    sdk.get_experiment_by_name.return_value = mlflow.entities.Experiment(
        "experiment-1", "test-experiment", "file:///unused", lifecycle_stage,
    )

    assert wrapper._get_or_create_experiment_id("test-experiment") == "experiment-1"

    sdk.get_experiment_by_name.assert_called_once_with("test-experiment")
    sdk.create_experiment.assert_not_called()
    if lifecycle_stage == "deleted":
        sdk.restore_experiment.assert_called_once_with("experiment-1")
    else:
        sdk.restore_experiment.assert_not_called()


def test_client_creates_missing_experiment(client: tuple[mlflow_utils.MlflowClient, Mock]) -> None:
    wrapper, sdk = client
    sdk.get_experiment_by_name.return_value = None
    sdk.create_experiment.return_value = "new-experiment"

    assert wrapper._get_or_create_experiment_id("test-experiment") == "new-experiment"

    sdk.create_experiment.assert_called_once_with("test-experiment")


@pytest.mark.parametrize("lifecycle_stage", ["active", "deleted"])
def test_client_recovers_from_experiment_creation_race(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    lifecycle_stage: str,
) -> None:
    wrapper, sdk = client
    experiment = mlflow.entities.Experiment(
        "race-winner", "test-experiment", "file:///unused", lifecycle_stage,
    )
    sdk.get_experiment_by_name.side_effect = [None, experiment]
    sdk.create_experiment.side_effect = make_rest_exception(
        {"error_code": "RESOURCE_ALREADY_EXISTS", "message": "already exists"},
    )

    assert wrapper._get_or_create_experiment_id("test-experiment") == "race-winner"

    assert sdk.get_experiment_by_name.call_count == 2
    if lifecycle_stage == "deleted":
        sdk.restore_experiment.assert_called_once_with("race-winner")
    else:
        sdk.restore_experiment.assert_not_called()


@pytest.mark.parametrize("error_code", ["PERMISSION_DENIED", "RESOURCE_ALREADY_EXISTS"])
def test_client_propagates_unrecoverable_experiment_creation_errors(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    error_code: str,
) -> None:
    wrapper, sdk = client
    sdk.get_experiment_by_name.return_value = None
    failure = make_rest_exception({"error_code": error_code, "message": "cannot create"})
    sdk.create_experiment.side_effect = failure

    with pytest.raises(mlflow.exceptions.RestException) as error:
        wrapper._get_or_create_experiment_id("test-experiment")

    assert error.value is failure
    assert sdk.get_experiment_by_name.call_count == (2 if error_code == "RESOURCE_ALREADY_EXISTS" else 1)


def test_log_pyfunc_model_logs_payload_artifacts_and_result(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    model_logging: tuple[MagicMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrapper, sdk = client
    start_run, log_model = model_logging
    monkeypatch.setattr(time, "time", lambda: 1234.5)
    artifact = tmp_path / "notes.txt"
    artifact.write_text("notes", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    python_model = Mock(spec=mlflow.pyfunc.PythonModel)
    tags = {"team": "test"}
    example = {"values": [1, 2]}
    model_artifacts = {"weights": "weights.bin"}

    result = wrapper.log_pyfunc_model(
        python_model=python_model,
        experiment_name="test-experiment",
        model_name="classifier",
        run_name="training",
        params={"depth": 3, "enabled": True},
        metrics={"accuracy": 0.9, "count": 2},
        tags=tags,
        input_example=example,
        run_artifacts=[str(artifact), str(missing)],
        model_artifacts=model_artifacts,
    )

    sdk.create_run.assert_called_once_with(experiment_id="experiment-1", tags=tags, run_name="training")
    payload = sdk.log_batch.call_args.kwargs
    assert payload["run_id"] == "run-1"
    assert [(param.key, param.value) for param in payload["params"]] == [("depth", "3"), ("enabled", "True")]
    assert [(metric.key, metric.value, metric.timestamp, metric.step) for metric in payload["metrics"]] == [
        ("accuracy", 0.9, 1234500, 0), ("count", 2.0, 1234500, 0),
    ]
    sdk.log_artifact.assert_called_once_with(run_id="run-1", local_path=artifact)
    assert "artifact path does not exist (skipping)" in caplog.text
    assert str(missing) in caplog.text
    start_run.assert_called_once_with(run_id="run-1")
    start_run.return_value.__enter__.assert_called_once()
    start_run.return_value.__exit__.assert_called_once_with(None, None, None)
    log_model.assert_called_once_with(
        name="classifier", python_model=python_model, input_example=example, artifacts=model_artifacts,
    )
    assert result == mlflow_utils.LogResult("experiment-1", "run-1", "classifier", "models:/model-1")
    sdk.set_terminated.assert_not_called()


def test_log_pyfunc_model_supports_defaults(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    model_logging: tuple[MagicMock, Mock],
) -> None:
    wrapper, sdk = client
    _, log_model = model_logging
    python_model = Mock(spec=mlflow.pyfunc.PythonModel)

    result = wrapper.log_pyfunc_model(python_model=python_model, experiment_name="test-experiment")

    sdk.create_run.assert_called_once_with(experiment_id="experiment-1", tags={}, run_name=None)
    sdk.log_batch.assert_called_once_with(run_id="run-1", params=[], metrics=[])
    sdk.log_artifact.assert_not_called()
    log_model.assert_called_once_with(name="model", python_model=python_model, input_example=None, artifacts=None)
    assert result.model_name == "model"


@pytest.mark.parametrize("field", ["experiment_name", "model_name"])
@pytest.mark.parametrize("value", ["", " \t"])
def test_log_pyfunc_model_rejects_blank_names(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    model_logging: tuple[MagicMock, Mock],
    field: str,
    value: str,
) -> None:
    wrapper, sdk = client
    start_run, log_model = model_logging
    arguments = {"experiment_name": "test-experiment", "model_name": "model", field: value}

    with pytest.raises(ValueError, match=field):
        wrapper.log_pyfunc_model(
            python_model=Mock(),
            experiment_name=arguments["experiment_name"],
            model_name=arguments["model_name"],
        )

    sdk.get_experiment_by_name.assert_not_called()
    sdk.create_run.assert_not_called()
    start_run.assert_not_called()
    log_model.assert_not_called()


@pytest.mark.parametrize("stage", ["get_experiment_by_name", "create_run", "log_batch", "log_model"])
def test_log_pyfunc_model_wraps_failures_and_marks_created_runs_failed(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    model_logging: tuple[MagicMock, Mock],
    stage: str,
) -> None:
    wrapper, sdk = client
    _, log_model = model_logging
    failure = RuntimeError("tracking unavailable")
    failing_call = log_model if stage == "log_model" else getattr(sdk, stage)
    failing_call.side_effect = failure

    with pytest.raises(RuntimeError, match="MLflow logging failed") as error:
        wrapper.log_pyfunc_model(python_model=Mock(), experiment_name="test-experiment")

    assert error.value.__cause__ is failure
    if stage in {"log_batch", "log_model"}:
        sdk.set_terminated.assert_called_once_with("run-1", status="FAILED")
    else:
        sdk.set_terminated.assert_not_called()


def test_log_pyfunc_model_preserves_original_failure_when_status_update_fails(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    model_logging: tuple[MagicMock, Mock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrapper, sdk = client
    failure = RuntimeError("logging failed")
    sdk.log_batch.side_effect = failure
    sdk.set_terminated.side_effect = mlflow.exceptions.MlflowException("status update failed")

    with pytest.raises(RuntimeError) as error:
        wrapper.log_pyfunc_model(python_model=Mock(), experiment_name="test-experiment")

    assert error.value.__cause__ is failure
    assert "Failed to set MLflow run status=FAILED" in caplog.text


@pytest.mark.parametrize("alias", [None, "", "champion"])
def test_register_model_optionally_sets_alias(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    monkeypatch: pytest.MonkeyPatch,
    alias: str | None,
) -> None:
    wrapper, sdk = client
    register_model = Mock(return_value=SimpleNamespace(version="7"))
    monkeypatch.setattr(mlflow, "register_model", register_model)

    wrapper.register_model(model_uri="models:/model-1", registered_model_name="classifier", alias=alias)

    register_model.assert_called_once_with(model_uri="models:/model-1", name="classifier")
    if alias:
        sdk.set_registered_model_alias.assert_called_once_with(name="classifier", alias=alias, version="7")
    else:
        sdk.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("field", ["model_uri", "registered_model_name"])
@pytest.mark.parametrize("value", ["", " \t"])
def test_register_model_rejects_blank_names(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    wrapper, _ = client
    register_model = Mock()
    monkeypatch.setattr(mlflow, "register_model", register_model)
    arguments = {"model_uri": "models:/model-1", "registered_model_name": "classifier", field: value}

    with pytest.raises(ValueError, match=field):
        wrapper.register_model(**arguments)

    register_model.assert_not_called()


@pytest.mark.parametrize("fail_alias", [False, True])
def test_register_model_wraps_registration_and_alias_errors(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    monkeypatch: pytest.MonkeyPatch,
    fail_alias: bool,
) -> None:
    wrapper, sdk = client
    failure = RuntimeError("registry unavailable")
    register_model = Mock(return_value=SimpleNamespace(version="7"))
    monkeypatch.setattr(mlflow, "register_model", register_model)
    if fail_alias:
        sdk.set_registered_model_alias.side_effect = failure
    else:
        register_model.side_effect = failure

    with pytest.raises(RuntimeError, match="Failed to register model 'classifier'") as error:
        wrapper.register_model(model_uri="models:/model-1", registered_model_name="classifier", alias="champion")

    assert error.value.__cause__ is failure
    if not fail_alias:
        sdk.set_registered_model_alias.assert_not_called()


def test_load_model_returns_pyfunc_model(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _ = client
    model = object()
    load_model = Mock(return_value=model)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", load_model)

    assert wrapper.load_model("models:/classifier@champion") is model

    load_model.assert_called_once_with(model_uri="models:/classifier@champion")


def test_load_model_wraps_errors_with_uri(
    client: tuple[mlflow_utils.MlflowClient, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _ = client
    failure = RuntimeError("model not found")
    monkeypatch.setattr(mlflow.pyfunc, "load_model", Mock(side_effect=failure))

    with pytest.raises(RuntimeError, match="Failed to load model from 'models:/missing'") as error:
        wrapper.load_model("models:/missing")

    assert error.value.__cause__ is failure
