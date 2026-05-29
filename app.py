from __future__ import annotations

import base64
from pathlib import Path
import time
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

from core.models import Result
from core.ocr_backends import BACKEND_AUTO, BACKEND_HYBRID, BACKEND_OLLAMA
from core.pdf_utils import get_pdf_page_count, is_pdf_supported
from core.profiles import get_profile
from core.settings import Settings
from ui.components import (
    SidebarState,
    render_downloads,
    render_results,
    render_sidebar,
)
from ui.components.setup_status import PDF_PER_PAGE_MODE
from ui.theme import render_app_theme

APP_TITLE = "Curiosity AI Scans"
LOGO_PATH = Path(__file__).parent / "public" / "curiosity-ai-logo (2026).svg"
SETTINGS = Settings.from_env()
PDF_SUPPORT = is_pdf_supported()
CUSTOM_EXTRACTION_MODE = "Custom field extraction"


def _session_key(suffix: str) -> str:
    """Scope session state keys to the current Streamlit session id.

    Prevents results bleeding between users on a shared deployment.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        sid = getattr(ctx, "session_id", "local") if ctx else "local"
    except Exception:
        sid = "local"
    return f"{sid}:{suffix}"


def _get_state(key: str, default):
    k = _session_key(key)
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]


def _set_state(key: str, value) -> None:
    st.session_state[_session_key(key)] = value


def _logo_data_uri() -> str:
    try:
        logo_bytes = LOGO_PATH.read_bytes()
    except OSError:
        return ""
    encoded_logo = base64.b64encode(logo_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded_logo}"


def _brand_bar_style(logo_src: str) -> str:
    if not logo_src:
        return ""
    return f"""
    <style>
        [data-testid="stHeader"]::before {{
            content: "";
            position: absolute;
            left: 3.55rem;
            top: 50%;
            width: 26px;
            height: 26px;
            transform: translateY(-50%);
            border-radius: 8px;
            background-image: url("{logo_src}");
            background-position: center;
            background-repeat: no-repeat;
            background-size: contain;
            pointer-events: none;
        }}

    </style>
    """


def _resolve_model_name(model: str, *, settings: Settings):
    from adapters.ollama_adapter import resolve_model_name

    return resolve_model_name(model, settings=settings)


def _effective_fields(state: SidebarState) -> List[str]:
    if state.fields is not None:
        return list(state.fields)
    if state.extraction_mode != CUSTOM_EXTRACTION_MODE:
        return []
    try:
        return list(get_profile(state.profile_id).fields)
    except ValueError:
        return []


def _auto_workload_needs_ollama(state: SidebarState) -> bool:
    has_images = any(
        not str(getattr(upload, "name", "")).lower().endswith(".pdf")
        for upload in state.uploaded_files
    )
    if has_images:
        return True
    return bool(_effective_fields(state))


def _should_preflight_ollama(state: SidebarState) -> bool:
    if state.ocr_backend == BACKEND_OLLAMA:
        return True
    if state.ocr_backend == BACKEND_HYBRID:
        return bool(_effective_fields(state))
    if state.ocr_backend == BACKEND_AUTO:
        return _auto_workload_needs_ollama(state)
    return False


def _run_batch(jobs, cfg):
    from core.pipeline import run_batch

    return run_batch(jobs, cfg)


def _count_items(uploaded_files, pdf_process_mode: str) -> int:
    total = 0
    for uf in uploaded_files:
        file_bytes = uf.read()
        uf.seek(0)
        if uf.name.lower().endswith(".pdf") and PDF_SUPPORT:
            if pdf_process_mode == PDF_PER_PAGE_MODE:
                try:
                    total += get_pdf_page_count(file_bytes)
                except Exception:
                    total += 1
            else:
                total += 1
        else:
            total += 1
    return total


def _display_entry_from_result(result: Result) -> Dict[str, Any]:
    return {
        "filename": result.source,
        "content": result.text if not result.error else "",
        "duration_sec": (
            round(result.latency_ms / 1000.0, 3) if result.latency_ms else None
        ),
        "dimensions": result.dimensions,
        "encoded_bytes": result.encoded_bytes,
        "structured_data": (
            {"filename": result.source, **result.fields} if result.fields else None
        ),
        "image_bytes": result.preview_image_bytes,
        "page_note": None,
        "status": "error" if result.error else "done",
        "error": result.error,
        "engine": result.engine,
        "profile_id": result.profile_id,
        "backend_note": result.backend_note,
        "field_evidence": result.field_evidence,
    }


def run_processing(state: SidebarState, results_placeholder) -> None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    _set_state("results", [])
    _set_state("display_entries", [])

    resolved_model = state.selected_model
    if _should_preflight_ollama(state):
        available, resolved_model, note = _resolve_model_name(
            state.selected_model,
            settings=state.settings,
        )
        if note:
            st.warning(note)
        if not available:
            st.error(
                f"Model '{state.selected_model}' is not available locally. "
                f"Run: ollama pull {state.selected_model}"
            )
            return

    total_items = _count_items(state.uploaded_files, state.pdf_process_mode)
    processed = 0
    durations: List[float] = []
    batch_start = time.perf_counter()

    from core.pipeline import BatchConfig, BatchJob

    cfg = BatchConfig(
        model=resolved_model,
        fields=state.fields,
        system_prompt=state.system_prompt,
        options=state.options,
        settings=state.settings,
        prompts=state.prompts,
        pdf_pages_separately=(state.pdf_process_mode == PDF_PER_PAGE_MODE),
        ocr_backend=state.ocr_backend,
        profile_id=state.profile_id,
        preprocess=state.preprocess,
        use_profile_fields=(state.extraction_mode == CUSTOM_EXTRACTION_MODE),
    )

    results_list: List[Result] = _get_state("results", [])
    entries: List[Dict[str, Any]] = _get_state("display_entries", [])

    for uf in state.uploaded_files:
        file_bytes = uf.read()
        uf.seek(0)
        if uf.name.lower().endswith(".pdf"):
            if not PDF_SUPPORT:
                st.error(f"Cannot process PDF {uf.name}. Install PyMuPDF.")
                processed += 1
                progress_bar.progress(min(processed / max(total_items, 1), 1.0))
                continue
            job = BatchJob(source=uf.name, data=file_bytes, kind="pdf")
        else:
            with Image.open(uf) as img:
                img.load()
                job = BatchJob(source=uf.name, data=img.copy(), kind="image")

        for result in _run_batch([job], cfg):
            status_text.text(f"Processing {result.source}")
            results_list.append(result)
            if result.latency_ms:
                durations.append(result.latency_ms / 1000.0)

            entries.append(_display_entry_from_result(result))
            render_results(results_placeholder, entries, state.show_images, state.compact_view)

            processed += 1
            progress_bar.progress(min(processed / max(total_items, 1), 1.0))

    _set_state("results", results_list)
    _set_state("display_entries", entries)

    batch_elapsed = time.perf_counter() - batch_start
    status_text.text("Processing complete!")
    if durations:
        avg = sum(durations) / max(len(durations), 1)
        st.info(f"Total processing time: {batch_elapsed:.2f} s  |  Avg per item: {avg:.2f} s")


def main() -> None:
    st.set_page_config(
        page_title="Curiosity AI Scans",
        page_icon="assets/gemma3.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_app_theme()
    logo_src = _logo_data_uri()
    st.markdown(_brand_bar_style(logo_src), unsafe_allow_html=True)
    logo_markup = (
        f'<img class="ocr-brand-logo" src="{logo_src}" alt="Curiosity AI logo" />'
        if logo_src
        else ""
    )

    st.markdown(
        f"""
        <section class="ocr-app-header" aria-label="{APP_TITLE}">
            <div class="ocr-brand-row">
                {logo_markup}
                <div class="ocr-brand-copy">
                    <p class="ocr-eyebrow">Local document OCR</p>
                    <h1>{APP_TITLE}</h1>
                    <p class="ocr-app-description">Private scans, structured extraction, and export-ready results from local vision models.</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not PDF_SUPPORT:
        st.warning("PDF support requires PyMuPDF. Install it with: pip install pymupdf")

    _get_state("results", [])
    _get_state("display_entries", [])

    state = render_sidebar(SETTINGS, pdf_supported=PDF_SUPPORT)

    results_placeholder = st.empty()

    if state.uploaded_files and state.process_button:
        run_processing(state, results_placeholder)
    else:
        render_results(
            results_placeholder,
            _get_state("display_entries", []),
            state.show_images,
            state.compact_view,
        )

    results_list: List[Result] = _get_state("results", [])
    if results_list:
        render_downloads(results_list)

    if not state.uploaded_files and not results_list:
        st.markdown(
            """
            <div class="ocr-empty-state">
                <strong>Add files to start</strong>
                Upload images or PDFs in the sidebar, choose a local model, then run a scan.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="ocr-footer">
            Curiosity AI Scans by Adrian | <a href="https://curiosityai.nl" target="_blank">curiosityai.nl</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
