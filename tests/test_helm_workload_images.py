from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "charts/video-commerce"


def _values():
    return yaml.safe_load((CHART / "values.yaml").read_text())


def test_content_worker_uses_image_with_paddle_ocr_runtime():
    values = _values()
    worker = values["backend"]["workloads"]["contentWorker"]
    image_ref = worker["imageRef"]

    assert image_ref == "contentWorker"
    assert values["images"][image_ref]["repository"] != values["images"]["backend"][
        "repository"
    ]
    # The PaddleOCR venv only exists in the Dockerfile content-worker target.
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "AS content-worker" in dockerfile
    assert values["appConfig"]["ocrBackend"] == "paddle"


def test_every_workload_image_ref_resolves():
    values = _values()
    for name, workload in values["backend"]["workloads"].items():
        image_ref = workload.get("imageRef")
        if image_ref:
            assert image_ref in values["images"], name


def test_backend_deployments_render_per_workload_image():
    template = (CHART / "templates/backend-deployments.yaml").read_text()
    configmap = (CHART / "templates/backend-configmap.yaml").read_text()

    assert 'include "video-commerce.workloadImage"' in template
    assert 'include "video-commerce.backendImage"' not in template
    assert "MODEL_OCR_BACKEND: {{ .Values.appConfig.ocrBackend | quote }}" in configmap


def test_ci_builds_content_worker_image():
    ci_path = ROOT / ".github/workflows/ci.yml"
    if not ci_path.exists():  # .github is excluded from the Docker test image
        pytest.skip("CI workflow not available in this checkout")
    ci = ci_path.read_text()
    assert "--target content-worker -t video-commerce-content-worker:ci" in ci
