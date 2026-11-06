import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from app_services.inference_service import MedicalInferenceEngine
from app_utils.imaging import compute_mask_metrics, generate_benchmark_slice


def test_mock_is_explicit_and_reference_mask_is_not_prediction():
    engine = MedicalInferenceEngine("mock")
    image = Image.new("L", (32, 24), 0)
    result = engine.predict_segmentation(image, preset_mask=Image.new("L", image.size, 255))
    assert result["metrics"]["tumor_pixels"] == 0
    assert "MÔ PHỎNG" in result["execution_mode"]
    assert not engine.predict_classification(image)["is_positive"]


def test_real_mode_never_falls_back(monkeypatch):
    engine = MedicalInferenceEngine("mock")
    engine.mode = "real"
    engine.error = "missing weights"
    with pytest.raises(RuntimeError, match="missing weights"):
        engine.predict_classification(Image.new("L", (10, 10)))


def test_screening_does_not_execute_full_pipeline():
    class Classifier:
        def predict_slice(self, *args, **kwargs):
            return {"tumor_probability": 0.8}
    class Pipeline:
        classifier = Classifier()
        cls_size = 224
        def predict(self, *args, **kwargs):
            raise AssertionError("Screening must not run segmentation")
    engine = MedicalInferenceEngine("mock")
    engine.is_real_loaded = True
    engine.pipeline = Pipeline()
    assert engine.predict_classification(Image.new("L", (10, 10)))["is_positive"]


def test_metrics_and_samples():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    mask[8, 18] = 255
    metrics = compute_mask_metrics(mask)
    assert metrics["region_count"] == 2
    assert metrics["tumor_pixels"] == 5
    assert metrics["bbox"] == (2, 1, 18, 8)
    assert metrics["tumor_ratio_percent"] == 2.5
    a, _ = generate_benchmark_slice("glioma")
    b, _ = generate_benchmark_slice("glioma")
    assert np.array_equal(a, b)


def navigate(app, page):
    next(b for b in app.sidebar.button if b.label == page).click().run()


def test_app_gating_and_reruns():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert "Phân vùng" not in [b.label for b in app.sidebar.button]
    navigate(app, "Phân tích MRI")
    app.radio[0].set_value("Ảnh mô phỏng").run()
    next(b for b in app.button if b.label == "Nạp ảnh mẫu").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Thực hiện sàng lọc").click().run()
    assert app.session_state["screening_result"]["is_positive"]
    next(b for b in app.button if b.label == "Thực hiện phân vùng").click().run()
    assert not app.exception
    before = app.session_state["segmentation_result"]["metrics"].copy()
    app.slider[0].set_value(0.75).run()
    assert app.session_state["segmentation_result"]["metrics"] == before
    navigate(app, "Tổng quan")
    navigate(app, "Phân tích MRI")
    assert not app.exception
    assert app.session_state["segmentation_result"]["metrics"] == before
    for page in ["Thông tin mô hình", "Giới thiệu", "Tổng quan"]:
        navigate(app, page)
        assert not app.exception


def test_replacing_input_clears_results_and_negative_cannot_segment():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    navigate(app, "Phân tích MRI")
    app.radio[0].set_value("Ảnh mô phỏng").run()
    next(b for b in app.button if b.label == "Nạp ảnh mẫu").click().run()
    next(b for b in app.button if b.label == "Thực hiện sàng lọc").click().run()
    next(b for b in app.button if b.label == "Thực hiện phân vùng").click().run()
    app.selectbox[0].set_value("normal").run()
    next(b for b in app.button if b.label == "Nạp ảnh mẫu").click().run()
    assert app.session_state["screening_result"] is None
    assert app.session_state["segmentation_result"] is None
    next(b for b in app.button if b.label == "Thực hiện sàng lọc").click().run()
    assert not app.session_state["screening_result"]["is_positive"]
    assert not any(b.label == "Thực hiện phân vùng" for b in app.button)
    navigate(app, "Tổng quan")
    navigate(app, "Phân tích MRI")
    assert app.session_state["segmentation_result"] is None
    assert not app.exception
