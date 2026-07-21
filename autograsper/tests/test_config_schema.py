"""Tests for autograsper.config_schema (design 02 §4).

Uses the checked-in template `autograsper/granular-config.yaml` as the golden "valid" document,
then mutates copies to exercise the aggregated-error-reporting path.
"""

import os

import pytest
import yaml

from autograsper.config_schema import Config, ConfigError, load_config

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "granular-config.yaml"
)


def _load_template_doc() -> dict:
    with open(_TEMPLATE_PATH) as f:
        return yaml.safe_load(f)


def _write(tmp_path, doc) -> str:
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(doc, f)
    return str(path)


def test_template_config_loads_successfully():
    config = load_config(_TEMPLATE_PATH)
    assert isinstance(config, Config)
    assert config.robot.idx == "robot23"
    assert config.camera.fps == 2.5
    assert config.experiment.n_pushes == 10
    assert config.workspace.grasp_height == 0.34
    assert config.workspace.sweep_height == 0.57
    assert config.workspace.clearance_height == 0.8
    assert config.workspace.tool_length_robot == 0.36
    assert config.workspace.tool_width_robot == 0.017
    assert config.workspace.fence_center == (0.5, 0.49)
    assert config.workspace.fence_size == (0.94, 0.955)
    assert config.perception.crop.center_px == (275, 200)
    assert config.perception.crop.size == (360, 360)
    assert len(config.camera.m) == 3 and all(len(row) == 3 for row in config.camera.m)
    assert len(config.camera.H) == 3 and all(len(row) == 3 for row in config.camera.H)
    assert config.storage.tool_size_px_for_transitions == (8, 120)


def test_missing_keys_are_all_reported_together(tmp_path):
    doc = _load_template_doc()
    del doc["camera"]["fps"]
    del doc["robot"]["idx"]
    del doc["workspace"]["grasp_height"]
    del doc["experiment"]["n_pushes"]
    path = _write(tmp_path, doc)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    errors = exc_info.value.errors
    joined = "\n".join(errors)
    assert "camera.fps" in joined
    assert "robot.idx" in joined
    assert "workspace.grasp_height" in joined
    assert "experiment.n_pushes" in joined
    # All four problems must be reported in one shot, not just the first one encountered.
    assert len(errors) >= 4


def test_missing_section_reported(tmp_path):
    doc = _load_template_doc()
    del doc["tool_check"]
    path = _write(tmp_path, doc)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    assert any("tool_check" in e for e in exc_info.value.errors)


def test_missing_required_calibration_matrix_reported(tmp_path):
    doc = _load_template_doc()
    del doc["camera"]["H"]
    del doc["workspace"]["homography_npz_path"]
    path = _write(tmp_path, doc)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    joined = "\n".join(exc_info.value.errors)
    assert "camera.H" in joined
    assert "workspace.homography_npz_path" in joined


def test_type_errors_are_reported(tmp_path):
    doc = _load_template_doc()
    doc["camera"]["fps"] = "fast"
    doc["robot"]["rotation_bias"] = "zero"
    doc["experiment"]["n_pushes"] = 3.5  # int required, float given
    path = _write(tmp_path, doc)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    errors = exc_info.value.errors
    joined = "\n".join(errors)
    assert "camera.fps" in joined
    assert "robot.rotation_bias" in joined
    assert "experiment.n_pushes" in joined
    assert len(errors) >= 3


def test_bad_matrix_shape_reported(tmp_path):
    doc = _load_template_doc()
    doc["camera"]["m"] = [[1.0, 2.0], [3.0, 4.0]]  # 2x2, not 3x3
    path = _write(tmp_path, doc)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    assert any("camera.m" in e for e in exc_info.value.errors)


def test_invalid_perception_provider_reported(tmp_path):
    doc = _load_template_doc()
    doc["perception"]["provider"] = "clairvoyance"
    path = _write(tmp_path, doc)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    assert any("perception.provider" in e for e in exc_info.value.errors)


def test_defaults_applied_when_optional_keys_absent(tmp_path):
    doc = _load_template_doc()
    del doc["robot"]["token_env_var"]
    del doc["camera"]["save_bottom_raw"]
    del doc["storage"]
    del doc["ui"]
    path = _write(tmp_path, doc)

    config = load_config(path)
    assert config.robot.token_env_var == "CLOUDGRIPPER_TOKEN"
    assert config.camera.save_bottom_raw is False
    assert config.storage.base_dir == "autograsper/recorded_data"
    assert config.ui.port == 3000
