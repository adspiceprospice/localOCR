import io

from PIL import Image

import app
from core.models import FieldEvidence, Result
from core.settings import Settings
from ui.components.sidebar import SidebarState


class _Progress:
    def __init__(self):
        self.values = []

    def progress(self, value):
        self.values.append(value)


class _Status:
    def __init__(self):
        self.messages = []

    def text(self, value):
        self.messages.append(value)


class _FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.errors = []
        self.warnings = []
        self.infos = []

    def progress(self, _value):
        return _Progress()

    def empty(self):
        return _Status()

    def warning(self, value):
        self.warnings.append(value)

    def error(self, value):
        self.errors.append(value)

    def info(self, value):
        self.infos.append(value)


class _Upload:
    name = "scan.png"

    def __init__(self):
        image = Image.new("RGB", (4, 4), color="white")
        self._buffer = io.BytesIO()
        image.save(self._buffer, format="PNG")
        self._buffer.seek(0)

    def read(self, size=-1):
        return self._buffer.read(size)

    def seek(self, position):
        return self._buffer.seek(position)

    def tell(self):
        return self._buffer.tell()


def test_logo_data_uri_loads_public_svg():
    assert app._logo_data_uri().startswith("data:image/svg+xml;base64,")


def test_brand_bar_style_targets_streamlit_header():
    style = app._brand_bar_style("data:image/svg+xml;base64,abc")

    assert '[data-testid="stHeader"]::before' in style
    assert 'background-image: url("data:image/svg+xml;base64,abc")' in style
    assert "left: 3.55rem;" in style
    assert '[data-testid="stHeader"]::after' not in style
    assert 'content: "Curiosity AI"' not in style


def test_display_entry_from_result_preserves_render_metadata():
    evidence = {"total": FieldEvidence(value="42", evidence_text="Total 42")}
    entry = app._display_entry_from_result(
        Result(
            source="scan.png",
            mode="extract",
            text='{"total": "42"}',
            fields={"total": "42"},
            latency_ms=1234,
            dimensions=(640, 480),
            encoded_bytes=2048,
            preview_image_bytes=b"preview",
            engine="hybrid",
            profile_id="invoice",
            backend_note="auto fallback",
            field_evidence=evidence,
        )
    )

    assert entry == {
        "filename": "scan.png",
        "content": '{"total": "42"}',
        "duration_sec": 1.234,
        "dimensions": (640, 480),
        "encoded_bytes": 2048,
        "structured_data": {"filename": "scan.png", "total": "42"},
        "image_bytes": b"preview",
        "page_note": None,
        "status": "done",
        "error": None,
        "engine": "hybrid",
        "profile_id": "invoice",
        "backend_note": "auto fallback",
        "field_evidence": evidence,
    }


def test_run_processing_skips_ollama_preflight_for_docling_and_routes_config(monkeypatch):
    fake_st = _FakeStreamlit()
    captured = {}

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("docling should not preflight Ollama models")

    def fake_run_batch(jobs, cfg):
        captured["jobs"] = list(jobs)
        captured["cfg"] = cfg
        yield Result(
            source="scan.png",
            mode="describe",
            text="OCR text",
            engine="docling",
            profile_id="invoice",
            backend_note="docling only",
            field_evidence={
                "total": FieldEvidence(value="42", evidence_text="Total 42")
            },
        )

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "_resolve_model_name", fail_resolve)
    monkeypatch.setattr(app, "_run_batch", fake_run_batch)
    monkeypatch.setattr(app, "render_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_session_key", lambda suffix: suffix)

    state = SidebarState(
        uploaded_files=[_Upload()],
        selected_model="gemma3:12b",
        settings=Settings(),
        profile_id="invoice",
        ocr_backend="docling",
        preprocess="document-clean",
    )

    app.run_processing(state, object())

    cfg = captured["cfg"]
    assert cfg.ocr_backend == "docling"
    assert cfg.profile_id == "invoice"
    assert cfg.preprocess == "document-clean"
    assert cfg.fields is None
    assert cfg.use_profile_fields is False
    assert cfg.model == "gemma3:12b"
    assert len(captured["jobs"]) == 1
    assert fake_st.errors == []


def test_run_processing_preflights_when_backend_may_need_ollama(monkeypatch):
    fake_st = _FakeStreamlit()
    captured = {}

    def fake_resolve(model, *, settings):
        captured["model"] = model
        captured["settings"] = settings
        return True, "gemma3:12b", "using local model"

    def fake_run_batch(jobs, cfg):
        captured["cfg"] = cfg
        yield Result(source="scan.png", mode="describe", text="OCR text")

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "_resolve_model_name", fake_resolve)
    monkeypatch.setattr(app, "_run_batch", fake_run_batch)
    monkeypatch.setattr(app, "render_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_session_key", lambda suffix: suffix)

    state = SidebarState(
        uploaded_files=[_Upload()],
        selected_model="gemma3:12b",
        settings=Settings(),
        fields=["total"],
        profile_id="receipt",
        ocr_backend="hybrid",
        preprocess="high-accuracy-scan",
        extraction_mode="Custom field extraction",
    )

    app.run_processing(state, object())

    assert captured["model"] == "gemma3:12b"
    assert fake_st.warnings == ["using local model"]
    assert captured["cfg"].ocr_backend == "hybrid"
    assert captured["cfg"].profile_id == "receipt"
    assert captured["cfg"].preprocess == "high-accuracy-scan"
    assert captured["cfg"].use_profile_fields is True


def test_run_processing_hybrid_description_does_not_preflight_ollama(monkeypatch):
    fake_st = _FakeStreamlit()
    captured = {}

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("hybrid description should not preflight Ollama")

    def fake_run_batch(jobs, cfg):
        captured["cfg"] = cfg
        yield Result(
            source="scan.png",
            mode="describe",
            text="Docling description",
            engine="hybrid",
            profile_id="invoice",
        )

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "_resolve_model_name", fail_resolve)
    monkeypatch.setattr(app, "_run_batch", fake_run_batch)
    monkeypatch.setattr(app, "render_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_session_key", lambda suffix: suffix)

    state = SidebarState(
        uploaded_files=[_Upload()],
        selected_model="gemma3:12b",
        settings=Settings(),
        profile_id="invoice",
        ocr_backend="hybrid",
        extraction_mode="General description",
    )

    app.run_processing(state, object())

    assert captured["cfg"].fields is None
    assert captured["cfg"].use_profile_fields is False
    assert fake_st.errors == []


def test_run_processing_auto_pdf_text_route_does_not_preflight_ollama(monkeypatch):
    fake_st = _FakeStreamlit()
    captured = {}

    class PdfUpload(_Upload):
        name = "scan.pdf"

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("auto PDF text route should not preflight Ollama")

    def fake_run_batch(jobs, cfg):
        captured["jobs"] = list(jobs)
        captured["cfg"] = cfg
        yield Result(
            source="scan.pdf",
            mode="describe",
            text="OCR text",
            engine="docling",
            profile_id="generic",
        )

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "PDF_SUPPORT", True)
    monkeypatch.setattr(app, "get_pdf_page_count", lambda _bytes: 1)
    monkeypatch.setattr(app, "_resolve_model_name", fail_resolve)
    monkeypatch.setattr(app, "_run_batch", fake_run_batch)
    monkeypatch.setattr(app, "render_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_session_key", lambda suffix: suffix)

    state = SidebarState(
        uploaded_files=[PdfUpload()],
        selected_model="gemma3:12b",
        settings=Settings(),
        profile_id="generic",
        ocr_backend="auto",
        preprocess="none",
    )

    app.run_processing(state, object())

    assert captured["cfg"].ocr_backend == "auto"
    assert captured["cfg"].profile_id == "generic"
    assert fake_st.errors == []


def test_run_processing_auto_pdf_explicit_empty_fields_does_not_preflight_ollama(monkeypatch):
    fake_st = _FakeStreamlit()
    captured = {}

    class PdfUpload(_Upload):
        name = "invoice.pdf"

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("explicit empty fields should keep auto PDF on docling text route")

    def fake_run_batch(jobs, cfg):
        captured["jobs"] = list(jobs)
        captured["cfg"] = cfg
        yield Result(
            source="invoice.pdf",
            mode="describe",
            text="OCR text",
            engine="docling",
            profile_id="invoice",
        )

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "PDF_SUPPORT", True)
    monkeypatch.setattr(app, "get_pdf_page_count", lambda _bytes: 1)
    monkeypatch.setattr(app, "_resolve_model_name", fail_resolve)
    monkeypatch.setattr(app, "_run_batch", fake_run_batch)
    monkeypatch.setattr(app, "render_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_session_key", lambda suffix: suffix)

    state = SidebarState(
        uploaded_files=[PdfUpload()],
        selected_model="gemma3:12b",
        settings=Settings(),
        fields=[],
        profile_id="invoice",
        ocr_backend="auto",
        preprocess="none",
    )

    app.run_processing(state, object())

    assert captured["cfg"].ocr_backend == "auto"
    assert captured["cfg"].fields == []
    assert captured["cfg"].profile_id == "invoice"
    assert fake_st.errors == []
