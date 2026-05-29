from __future__ import annotations

import os
import time
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from PIL import Image

from core.errors import PDFError
from core.extraction import (
    build_extraction_schema,
    build_field_evidence,
    clean_result_fields,
)
from core.image_utils import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_IMAGE_SIZE,
    image_to_base64,
    image_to_png_bytes,
    resize_image,
)
from core.json_extract import extract_structured_data
from core.logging import get_logger
from core.models import FieldEvidence, Mode, Result
from core.ocr_backends import (
    BACKEND_AUTO,
    BACKEND_DOCLING,
    BACKEND_HYBRID,
    BACKEND_OLLAMA,
    BackendSelection,
    resolve_backend,
)
from core.pdf_utils import PDFNotSupportedError, ensure_pdf_support, iter_pdf_pages
from core.preprocessing import (
    expected_preprocess_steps,
    normalize_preprocess_preset,
    preprocess_image,
)
from core.profiles import get_profile
from core.prompts import PromptConfig
from core.settings import Settings

JSONDict = Dict[str, object]
PDFPageResult = Tuple[
    Optional[int],
    Optional[int],
    Optional[Image.Image],
    str,
    str,
    Optional[JSONDict],
    Optional[float],
    Optional[Tuple[int, int]],
    Optional[int],
]

_log = get_logger("pipeline")


def query_ollama(prompt: str, image_base64: str, model: str, **kwargs: Any) -> str:
    """Lazy wrapper so importing pipeline does not import Ollama bindings."""
    from adapters.ollama_adapter import query_ollama as _query_ollama

    return _query_ollama(prompt, image_base64, model, **kwargs)


def query_ollama_text(prompt: str, model: str, **kwargs: Any) -> str:
    """Lazy wrapper so hybrid routing imports Ollama only when it is used."""
    from adapters.ollama_adapter import query_ollama_text as _query_ollama_text

    return _query_ollama_text(prompt, model, **kwargs)


def _accepts_kwarg(func: Callable[..., str], name: str) -> bool:
    try:
        sig = signature(func)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind is Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _call_inference(
    func: Callable[..., str],
    prompt: str,
    image_base64: str,
    model: str,
    **kwargs: Any,
) -> str:
    supported_kwargs = {
        name: value for name, value in kwargs.items() if _accepts_kwarg(func, name)
    }
    return func(prompt, image_base64, model, **supported_kwargs)


def _resolve_prompts(prompts: Optional[PromptConfig]) -> PromptConfig:
    if prompts is not None:
        return prompts
    # Honour legacy module-global templates for back-compat.
    from core.templates import current_prompt_config

    return current_prompt_config()


def _number_as_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_value(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    return None


def _dimensions_from_result(result: JSONDict) -> Optional[Tuple[int, int]]:
    width = result.get("input_width")
    height = result.get("input_height")
    if isinstance(width, int) and isinstance(height, int):
        return (width, height)
    return None


def process_image(
    image: Image.Image,
    filename: str,
    fields: Optional[List[str]] = None,
    *,
    model: str,
    system_prompt: Optional[str],
    options: Optional[JSONDict],
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    inference: Optional[Callable[[str, str, str], str]] = None,
    prompts: Optional[PromptConfig] = None,
    settings: Optional[Settings] = None,
    preprocess: str = "none",
) -> Tuple[JSONDict, str, Optional[JSONDict]]:
    """Process a single image with optional field extraction.

    Returns: (result_dict, raw_content, structured_dict_or_none)
    """
    pcfg = _resolve_prompts(prompts)
    t0 = time.perf_counter()
    preprocessed = preprocess_image(image, preprocess)
    prepared = resize_image(preprocessed.image, max_image_size)
    img_base64, encoded_size = image_to_base64(prepared, jpeg_quality)

    run_infer = inference
    if run_infer is None:

        def run_infer(prompt: str, img_b64: str, _model: str = model, **kwargs) -> str:  # type: ignore
            return query_ollama(
                prompt,
                img_b64,
                _model,
                options=options or {},
                system_prompt=system_prompt,
                **kwargs,
            )

    if not fields:
        prompt = pcfg.build_description()
        content = _call_inference(
            run_infer, prompt, img_base64, model, settings=settings
        )
        elapsed = time.perf_counter() - t0
        return (
            {
                "filename": filename,
                "description": content,
                "duration_sec": round(elapsed, 3),
                "input_width": prepared.size[0],
                "input_height": prepared.size[1],
                "encoded_bytes": int(encoded_size),
                "preprocess_steps": list(preprocessed.steps),
            },
            content,
            None,
        )
    else:
        prompt = pcfg.build_extraction(fields)
        schema = build_extraction_schema(fields)
        content = _call_inference(
            run_infer, prompt, img_base64, model, format=schema, settings=settings
        )
        structured_data: JSONDict = {"filename": filename}
        parsed = extract_structured_data(content, fields)
        structured_data.update(parsed)
        elapsed = time.perf_counter() - t0
        return (
            {
                "filename": filename,
                "extraction": content,
                "duration_sec": round(elapsed, 3),
                "input_width": prepared.size[0],
                "input_height": prepared.size[1],
                "encoded_bytes": int(encoded_size),
                "preprocess_steps": list(preprocessed.steps),
            },
            content,
            structured_data,
        )


def process_pdf(
    file_bytes: bytes,
    filename: str,
    fields: Optional[List[str]] = None,
    process_pages_separately: bool = True,
    *,
    model: str,
    system_prompt: Optional[str],
    options: Optional[JSONDict],
    max_image_size: int,
    jpeg_quality: int,
    pdf_scale: float,
    inference: Optional[Callable[[str, str, str], str]] = None,
    prompts: Optional[PromptConfig] = None,
    settings: Optional[Settings] = None,
    preprocess: str = "none",
) -> Generator[PDFPageResult, None, None]:
    """Process a PDF file using PyMuPDF, yielding page-level results."""

    def _process_page(
        page_num: int,
        page_count: int,
        image: Image.Image,
        page_filename: str,
    ) -> PDFPageResult:
        result, content, structured_data = process_image(
            image,
            page_filename,
            fields,
            model=model,
            system_prompt=system_prompt,
            options=options,
            max_image_size=max_image_size,
            jpeg_quality=jpeg_quality,
            inference=inference,
            prompts=prompts,
            settings=settings,
            preprocess=preprocess,
        )
        return (
            page_num,
            page_count,
            image,
            page_filename,
            content,
            structured_data,
            _number_as_float(result.get("duration_sec")),
            _dimensions_from_result(result),
            _int_value(result.get("encoded_bytes")),
        )

    try:
        ensure_pdf_support()
        if process_pages_separately:
            for page_num, page_count, img in iter_pdf_pages(
                file_bytes, scale=pdf_scale
            ):
                page_filename = f"{filename} (Page {page_num + 1})"
                yield _process_page(page_num, page_count, img, page_filename)
        else:
            first = next(iter_pdf_pages(file_bytes, scale=pdf_scale), None)
            if first is None:
                raise PDFError("PDF contains no renderable pages.")
            _page_num, page_count, img = first
            yield _process_page(0, page_count, img, filename)
    except PDFNotSupportedError as e:
        yield None, None, None, filename, str(e), None, None, None, None
    except Exception as e:
        _log.exception("pdf_processing_failed", extra={"source": filename})
        yield (
            None,
            None,
            None,
            filename,
            f"Error processing PDF: {str(e)}",
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


@dataclass
class BatchJob:
    """One unit of work for :func:`run_batch`."""

    source: str  # display name / path
    data: Union[bytes, Image.Image]
    kind: str  # "image" or "pdf"


@dataclass
class BatchConfig:
    model: str = ""
    fields: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    settings: Settings = None  # type: ignore[assignment]
    prompts: PromptConfig = None  # type: ignore[assignment]
    pdf_pages_separately: bool = True
    inference: Optional[Callable[[str, str, str], str]] = None
    ocr_backend: str = BACKEND_OLLAMA
    profile_id: str = "generic"
    preprocess: Optional[str] = None
    use_profile_fields: bool = True

    def __post_init__(self) -> None:
        if self.settings is None:
            self.settings = Settings.from_env()
        if not self.model:
            self.model = self.settings.default_model
        if self.prompts is None:
            self.prompts = PromptConfig()
        if self.options is None:
            self.options = {}
        profile = get_profile(self.profile_id)
        if self.use_profile_fields and self.fields is None and profile.fields:
            self.fields = list(profile.fields)
        selected_preprocess = (
            profile.default_preprocess if self.preprocess is None else self.preprocess
        )
        selection = resolve_backend(
            self.ocr_backend,
            profile_id=profile.id,
            preprocess=normalize_preprocess_preset(selected_preprocess),
        )
        self.ocr_backend = selection.backend
        self.profile_id = selection.profile_id
        self.preprocess = selection.preprocess


def _result_metadata(selection: BackendSelection) -> Dict[str, Any]:
    return {
        "engine": selection.backend,
        "profile_id": selection.profile_id,
    }


def _preprocess_steps(result: JSONDict) -> List[str]:
    steps = result.get("preprocess_steps")
    if isinstance(steps, list) and all(isinstance(step, str) for step in steps):
        return list(steps)
    return []


def _fields_and_evidence(
    *,
    requested_fields: Optional[List[str]],
    structured: Optional[JSONDict],
    raw_content: str,
    engine: str,
    page: Optional[int] = None,
) -> tuple[JSONDict, dict[str, FieldEvidence]]:
    clean_fields: JSONDict = {}
    if structured:
        parsed_fields = {k: v for k, v in structured.items() if k != "filename"}
        clean_fields = clean_result_fields(
            requested_fields=requested_fields,
            parsed_fields=parsed_fields,
        )

    field_evidence: dict[str, FieldEvidence] = {}
    if requested_fields:
        field_evidence = build_field_evidence(
            requested_fields=requested_fields,
            parsed_fields=clean_fields,
            raw_content=raw_content,
            engine=engine,
            page=page,
        )
    return clean_fields, field_evidence


def _expected_preprocess_steps(preprocess: str) -> List[str]:
    return expected_preprocess_steps(preprocess)


def _docling_adapter() -> Any:
    from adapters import docling_adapter

    return docling_adapter


def _convert_with_docling(job: BatchJob) -> Result:
    adapter = _docling_adapter()
    if job.kind == "pdf":
        assert isinstance(job.data, (bytes, bytearray))
        return adapter.convert_document_bytes(job.source, bytes(job.data), "pdf")
    assert isinstance(job.data, Image.Image)
    return adapter.convert_image(job.source, job.data)


def _docling_error_result(
    *,
    source: str,
    mode: Mode,
    error: Exception,
    selection: BackendSelection,
    engine: str,
) -> Result:
    return Result(
        source=source,
        mode=mode,
        text="",
        error=str(error),
        engine=engine,
        profile_id=selection.profile_id,
        preprocess_steps=[],
    )


def _docling_result(
    *,
    source_result: Result,
    mode: Mode,
    selection: BackendSelection,
    engine: str,
    preview_image_bytes: Optional[bytes] = None,
    preprocess_steps: Optional[List[str]] = None,
) -> Result:
    text = source_result.ocr_text or source_result.text or source_result.raw
    return Result(
        source=source_result.source,
        mode=mode,
        text=text,
        raw=source_result.raw or text,
        ocr_text=text,
        ocr_blocks=list(source_result.ocr_blocks),
        engine=engine,
        profile_id=selection.profile_id,
        dimensions=source_result.dimensions,
        preview_image_bytes=preview_image_bytes,
        preprocess_steps=list(preprocess_steps or []),
    )


def _hybrid_prompt(fields: List[str], ocr_text: str, prompts: PromptConfig) -> str:
    base = prompts.build_extraction(fields)
    return (
        f"{base}\n\n"
        "Use the OCR/layout text below as the only source. Return JSON with the "
        "requested field names.\n\n"
        f"OCR text:\n{ocr_text}"
    )


def _run_text_extraction(
    *,
    prompt: str,
    fields: List[str],
    cfg: BatchConfig,
) -> str:
    schema = build_extraction_schema(fields)
    return query_ollama_text(
        prompt,
        cfg.model,
        options=cfg.options,
        system_prompt=cfg.system_prompt,
        format=schema,
        settings=cfg.settings,
    )


def _preview_bytes_for_job(job: BatchJob) -> Optional[bytes]:
    if isinstance(job.data, Image.Image):
        return image_to_png_bytes(job.data)
    return None


def _preprocessed_docling_job(
    job: BatchJob,
    selection: BackendSelection,
) -> tuple[BatchJob, List[str]]:
    if not isinstance(job.data, Image.Image):
        return job, []
    preprocessed = preprocess_image(job.data, selection.preprocess)
    return (
        BatchJob(source=job.source, data=preprocessed.image, kind=job.kind),
        list(preprocessed.steps),
    )


def _job_from_path(path: str) -> BatchJob:
    name = os.path.basename(path)
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            return BatchJob(source=name, data=f.read(), kind="pdf")
    img = Image.open(path)
    img.load()
    return BatchJob(source=name, data=img, kind="image")


def _iter_docling_input_jobs(
    job: BatchJob,
    cfg: BatchConfig,
) -> Iterator[tuple[BatchJob, Optional[int], Optional[int]]]:
    if job.kind != "pdf" or not cfg.pdf_pages_separately:
        yield job, None, None
        return

    assert isinstance(job.data, (bytes, bytearray))
    ensure_pdf_support()
    for page_num, page_count, img in iter_pdf_pages(
        bytes(job.data),
        scale=cfg.settings.pdf_scale,
    ):
        page_source = f"{job.source} (Page {page_num + 1})"
        yield BatchJob(source=page_source, data=img, kind="image"), page_num, page_count


def _with_pdf_page_metadata(
    result: Result,
    *,
    page_num: Optional[int],
    page_count: Optional[int],
) -> Result:
    if page_num is not None:
        result.page = page_num
        result.page_count = page_count
    return result


def _run_ollama_job(
    *,
    job: BatchJob,
    cfg: BatchConfig,
    mode: Mode,
    selection: BackendSelection,
) -> Iterator[Result]:
    if job.kind == "pdf":
        assert isinstance(job.data, (bytes, bytearray))
        for page_info in process_pdf(
            bytes(job.data),
            job.source,
            cfg.fields,
            cfg.pdf_pages_separately,
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            options=cfg.options,
            max_image_size=cfg.settings.max_image_size,
            jpeg_quality=cfg.settings.jpeg_quality,
            pdf_scale=cfg.settings.pdf_scale,
            inference=cfg.inference,
            prompts=cfg.prompts,
            settings=cfg.settings,
            preprocess=selection.preprocess,
        ):
            (
                page_num,
                page_count,
                image,
                page_filename,
                content,
                structured,
                elapsed,
                dims,
                size_bytes,
            ) = page_info
            if page_num is None:
                yield Result(
                    source=job.source,
                    mode=mode,
                    text="",
                    error=content,
                    **_result_metadata(selection),
                )
                continue
            clean_fields, field_evidence = _fields_and_evidence(
                requested_fields=cfg.fields,
                structured=structured,
                raw_content=content,
                engine=selection.backend,
                page=page_num,
            )
            yield Result(
                source=page_filename,
                mode=mode,
                text=content,
                raw=content,
                fields=clean_fields,
                field_evidence=field_evidence,
                page=page_num,
                page_count=page_count,
                latency_ms=int((elapsed or 0.0) * 1000),
                dimensions=dims,
                encoded_bytes=size_bytes,
                preview_image_bytes=image_to_png_bytes(image)
                if image is not None
                else None,
                preprocess_steps=_expected_preprocess_steps(selection.preprocess),
                **_result_metadata(selection),
            )
    else:
        assert isinstance(job.data, Image.Image)
        result, content, structured = process_image(
            job.data,
            job.source,
            cfg.fields,
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            options=cfg.options,
            max_image_size=cfg.settings.max_image_size,
            jpeg_quality=cfg.settings.jpeg_quality,
            inference=cfg.inference,
            prompts=cfg.prompts,
            settings=cfg.settings,
            preprocess=selection.preprocess,
        )
        clean_fields, field_evidence = _fields_and_evidence(
            requested_fields=cfg.fields,
            structured=structured,
            raw_content=content,
            engine=selection.backend,
        )
        elapsed_sec = _number_as_float(result.get("duration_sec")) or 0.0
        dims_t = _dimensions_from_result(result)
        encoded_bytes = _int_value(result.get("encoded_bytes"))
        yield Result(
            source=job.source,
            mode=mode,
            text=content,
            raw=content,
            fields=clean_fields,
            field_evidence=field_evidence,
            latency_ms=int(elapsed_sec * 1000),
            dimensions=dims_t,
            encoded_bytes=encoded_bytes,
            preview_image_bytes=image_to_png_bytes(job.data),
            preprocess_steps=_preprocess_steps(result),
            **_result_metadata(selection),
        )


def _run_docling_job(
    *,
    job: BatchJob,
    selection: BackendSelection,
    mode: Mode = "describe",
    engine: str = BACKEND_DOCLING,
) -> Result:
    prepared_job, preprocess_steps = _preprocessed_docling_job(job, selection)
    converted = _convert_with_docling(prepared_job)
    return _docling_result(
        source_result=converted,
        mode=mode,
        selection=selection,
        engine=engine,
        preview_image_bytes=_preview_bytes_for_job(prepared_job),
        preprocess_steps=preprocess_steps,
    )


def _run_hybrid_job(
    *,
    job: BatchJob,
    cfg: BatchConfig,
    mode: Mode,
    selection: BackendSelection,
) -> Result:
    prepared_job, preprocess_steps = _preprocessed_docling_job(job, selection)
    converted = _convert_with_docling(prepared_job)
    docling_result = _docling_result(
        source_result=converted,
        mode=mode,
        selection=selection,
        engine=BACKEND_HYBRID,
        preview_image_bytes=_preview_bytes_for_job(prepared_job),
        preprocess_steps=preprocess_steps,
    )
    if not cfg.fields:
        docling_result.mode = "describe"
        return docling_result

    fields = list(cfg.fields)
    ocr_text = docling_result.ocr_text or docling_result.text
    prompt = _hybrid_prompt(fields, ocr_text, cfg.prompts)
    content = _run_text_extraction(prompt=prompt, fields=fields, cfg=cfg)
    parsed = extract_structured_data(content, fields)
    clean_fields = clean_result_fields(
        requested_fields=fields,
        parsed_fields=parsed,
    )
    field_evidence = build_field_evidence(
        requested_fields=fields,
        parsed_fields=clean_fields,
        raw_content=content,
        engine=BACKEND_HYBRID,
    )
    docling_result.mode = "extract"
    docling_result.text = content
    docling_result.raw = content
    docling_result.fields = clean_fields
    docling_result.field_evidence = field_evidence
    return docling_result


def _run_docling_job_results(
    *,
    job: BatchJob,
    cfg: BatchConfig,
    selection: BackendSelection,
    mode: Mode = "describe",
    engine: str = BACKEND_DOCLING,
) -> Iterator[Result]:
    for page_job, page_num, page_count in _iter_docling_input_jobs(job, cfg):
        yield _with_pdf_page_metadata(
            _run_docling_job(
                job=page_job,
                selection=selection,
                mode=mode,
                engine=engine,
            ),
            page_num=page_num,
            page_count=page_count,
        )


def _run_hybrid_job_results(
    *,
    job: BatchJob,
    cfg: BatchConfig,
    mode: Mode,
    selection: BackendSelection,
) -> Iterator[Result]:
    for page_job, page_num, page_count in _iter_docling_input_jobs(job, cfg):
        yield _with_pdf_page_metadata(
            _run_hybrid_job(
                job=page_job,
                cfg=cfg,
                mode=mode,
                selection=selection,
            ),
            page_num=page_num,
            page_count=page_count,
        )


def _run_auto_job(
    *,
    job: BatchJob,
    cfg: BatchConfig,
    mode: Mode,
    selection: BackendSelection,
) -> Iterator[Result]:
    fallback_note: Optional[str] = None
    if job.kind == "pdf":
        try:
            if cfg.fields:
                docling_results = list(
                    _run_hybrid_job_results(
                        job=job,
                        cfg=cfg,
                        mode=mode,
                        selection=selection,
                    )
                )
            else:
                docling_results = list(
                    _run_docling_job_results(
                        job=job,
                        cfg=cfg,
                        selection=selection,
                        engine=BACKEND_DOCLING,
                    )
                )
            yield from docling_results
            return
        except Exception as docling_error:
            fallback_note = f"auto fallback from docling: {docling_error}"
            _log.info(
                "auto_docling_fallback",
                extra={"source": job.source, "err": str(docling_error)},
            )

    ollama_selection = BackendSelection(
        backend=BACKEND_OLLAMA,
        profile_id=selection.profile_id,
        preprocess=selection.preprocess,
    )
    for result in _run_ollama_job(
        job=job,
        cfg=cfg,
        mode=mode,
        selection=ollama_selection,
    ):
        if fallback_note is not None:
            result.backend_note = fallback_note
        yield result


def run_batch(
    jobs: Iterable[Union[BatchJob, str]],
    cfg: BatchConfig,
) -> Iterator[Result]:
    """Yield :class:`Result` instances for each job.

    ``jobs`` accepts either :class:`BatchJob` instances or file path strings.
    This is the canonical entry point used by both the CLI and the Streamlit
    UI so they share a single execution path.
    """
    mode: Mode = "extract" if cfg.fields else "describe"
    preprocess = cfg.preprocess if cfg.preprocess is not None else "none"
    selection = resolve_backend(
        cfg.ocr_backend,
        profile_id=cfg.profile_id,
        preprocess=preprocess,
    )
    for raw_job in jobs:
        job = _job_from_path(raw_job) if isinstance(raw_job, str) else raw_job
        try:
            if selection.backend == BACKEND_DOCLING:
                if cfg.fields:
                    yield _docling_error_result(
                        source=job.source,
                        mode=mode,
                        error=ValueError(
                            "Docling backend only provides OCR text. Use "
                            "ocr_backend='hybrid' for field extraction."
                        ),
                        selection=selection,
                        engine=BACKEND_DOCLING,
                    )
                    continue
                yield from _run_docling_job_results(
                    job=job,
                    cfg=cfg,
                    selection=selection,
                )
                continue

            if selection.backend == BACKEND_HYBRID:
                yield from _run_hybrid_job_results(
                    job=job,
                    cfg=cfg,
                    mode=mode,
                    selection=selection,
                )
                continue

            if selection.backend == BACKEND_AUTO:
                yield from _run_auto_job(
                    job=job,
                    cfg=cfg,
                    mode=mode,
                    selection=selection,
                )
                continue

            yield from _run_ollama_job(
                job=job,
                cfg=cfg,
                mode=mode,
                selection=selection,
            )
        except Exception as e:
            _log.exception("job_failed", extra={"source": job.source})
            engine = selection.backend
            if selection.backend == BACKEND_DOCLING:
                engine = BACKEND_DOCLING
            elif selection.backend == BACKEND_HYBRID:
                engine = BACKEND_HYBRID
            yield Result(
                source=job.source,
                mode=mode,
                text="",
                error=str(e),
                engine=engine,
                profile_id=selection.profile_id,
                preprocess_steps=(
                    []
                    if engine in (BACKEND_DOCLING, BACKEND_HYBRID)
                    else _expected_preprocess_steps(selection.preprocess)
                ),
            )
