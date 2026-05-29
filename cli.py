from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from core.errors import OCRError
from core.export import write_evidence, write_results
from core.logging import get_logger
from core.models import Mode, Result
from core.ocr_backends import BACKEND_AUTO, BACKEND_HYBRID, BACKEND_OLLAMA
from core.pdf_utils import PDFNotSupportedError, ensure_pdf_support, iter_pdf_pages
from core.pipeline import BatchConfig, process_image, run_batch
from core.prompts import PromptConfig
from core.settings import Settings
from core.templates import load_templates_file  # noqa: F401 (back-compat export)

_log = get_logger("cli")

OLLAMA_BACKENDS = {BACKEND_OLLAMA, BACKEND_HYBRID, BACKEND_AUTO}


def query_ollama(prompt: str, image_b64: str, model: str, **kwargs) -> str:
    """Lazy wrapper so importing the CLI does not import Ollama bindings."""
    from adapters.ollama_adapter import query_ollama as _query_ollama

    return _query_ollama(prompt, image_b64, model, **kwargs)


def resolve_model_name(model: str, **kwargs):
    """Lazy wrapper preserving the existing CLI monkeypatch surface."""
    from adapters.ollama_adapter import resolve_model_name as _resolve_model_name

    return _resolve_model_name(model, **kwargs)


class RateLimiter:
    def __init__(self, per_second: Optional[float] = None):
        self.per_second = per_second
        self._last = 0.0
        import threading
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.per_second or self.per_second <= 0:
            return
        with self._lock:
            now = time.monotonic()
            min_interval = 1.0 / float(self.per_second)
            delta = now - self._last
            if delta < min_interval:
                time.sleep(min_interval - delta)
            self._last = time.monotonic()


def _make_inference(
    options: Dict,
    system_prompt: Optional[str],
    limiter: RateLimiter,
) -> Callable:
    def _inner(prompt: str, image_b64: str, model: str, **kwargs) -> str:
        limiter.wait()
        return query_ollama(prompt, image_b64, model, options=options, system_prompt=system_prompt, **kwargs)
    return _inner


def _should_preflight_ollama(backend: str, fields: Optional[List[str]]) -> bool:
    normalized = backend.strip().lower()
    if normalized == BACKEND_HYBRID:
        return bool(fields)
    return normalized in OLLAMA_BACKENDS


def _effective_fields_for_preflight(
    *,
    fields: Optional[List[str]],
    profile_id: str,
    use_profile_fields: bool,
) -> Optional[List[str]]:
    if fields:
        return fields
    if fields is not None and not use_profile_fields:
        return fields
    if not use_profile_fields:
        return None
    try:
        from core.profiles import get_profile

        profile_fields = get_profile(profile_id).fields
    except ValueError:
        return None
    return list(profile_fields) if profile_fields else None


# ---------------------------------------------------------------------------
# Back-compat shim: a few external scripts (e.g. evaluate.py) import
# ``_process_file`` directly. Keep the signature + return shape intact.
# ---------------------------------------------------------------------------


def _process_file(
    path: str,
    *,
    fields: Optional[List[str]],
    model: str,
    system_prompt: Optional[str],
    options: Dict,
    max_image_size: int,
    jpeg_quality: int,
    pdf_scale: float,
    pdf_pages: bool,
    inference: Callable[[str, str, str], str],
    prompts: Optional[PromptConfig] = None,
) -> Tuple[List[Dict], List[Dict]]:
    results: List[Dict] = []
    structured: List[Dict] = []
    filename = os.path.basename(path)
    try:
        if path.lower().endswith(".pdf"):
            ensure_pdf_support()
            with open(path, "rb") as f:
                file_bytes = f.read()

            def _handle_page(page_img: Image.Image, page_name: str) -> None:
                result, content, structured_data = process_image(
                    page_img,
                    page_name,
                    fields,
                    model=model,
                    system_prompt=system_prompt,
                    options=options,
                    max_image_size=max_image_size,
                    jpeg_quality=jpeg_quality,
                    inference=inference,
                    prompts=prompts,
                )
                entry: Dict = {
                    "filename": page_name,
                    "description": content if fields is None else result.get("extraction", content),
                }
                duration_sec = result.get("duration_sec")
                if isinstance(duration_sec, (int, float)):
                    entry["duration_sec"] = float(duration_sec)
                results.append(entry)
                if structured_data and len(structured_data) > 1:
                    structured.append(structured_data)

            if pdf_pages:
                for page_index, _, img in iter_pdf_pages(file_bytes, scale=pdf_scale):
                    page_name = f"{filename} (Page {page_index + 1})"
                    _handle_page(img, page_name)
            else:
                first_entry = next(iter_pdf_pages(file_bytes, scale=pdf_scale), None)
                if first_entry is None:
                    raise RuntimeError("PDF contains no renderable pages.")
                _, _, img = first_entry
                _handle_page(img, filename)
        else:
            with Image.open(path) as image:
                image.load()
                result, content, structured_data = process_image(
                    image,
                    filename,
                    fields,
                    model=model,
                    system_prompt=system_prompt,
                    options=options,
                    max_image_size=max_image_size,
                    jpeg_quality=jpeg_quality,
                    inference=inference,
                    prompts=prompts,
                )
            results.append(result)
            if structured_data and len(structured_data) > 1:
                structured.append(structured_data)
    except PDFNotSupportedError as e:
        results.append({"filename": filename, "description": f"Error: {e}"})
    except OCRError as e:
        results.append({"filename": filename, "description": f"Error: {e}"})
    except Exception as e:
        _log.exception("cli_file_failed", extra={"path": path})
        results.append({"filename": filename, "description": f"Error: {e}"})
    return results, structured


def _json_summary_payload(
    results: List[Result], processed_files: int
) -> Dict[str, object]:
    return {
        "ok": not any(r.error for r in results),
        "processed_files": processed_files,
        "total_results": len(results),
        "results": [
            {
                "source": r.source,
                "mode": r.mode,
                "text": r.text,
                "fields": r.fields,
                "error": r.error,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    settings = Settings.from_env()
    p = argparse.ArgumentParser(description="Headless OCR/Vision batch scans via Ollama")
    p.add_argument("files", nargs="+", help="Image or PDF files to process")
    p.add_argument("--model", default=settings.default_model, help="Model name")
    p.add_argument("--mode", choices=["description", "extract"], default="description")
    p.add_argument("--fields", default="", help="Comma-separated fields for extract mode")
    p.add_argument("--system-prompt", default="", help="Optional system prompt")
    p.add_argument("--templates", default="", help="Path to JSON templates with 'description' and 'extraction'")
    p.add_argument("--schema", default="", help="Path to JSON file with {\"fields\": [..]} for extract mode")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--max-image-size", type=int, default=settings.max_image_size)
    p.add_argument("--jpeg-quality", type=int, default=settings.jpeg_quality)
    p.add_argument("--pdf-scale", type=float, default=settings.pdf_scale)
    p.add_argument("--pdf-pages", action="store_true", help="Process each PDF page separately")
    p.add_argument("--ocr-backend", choices=["ollama", "docling", "hybrid", "auto"], default="ollama")
    p.add_argument("--profile", choices=["generic", "invoice", "receipt", "table"], default="generic")
    p.add_argument(
        "--preprocess",
        choices=["none", "document-clean", "high-accuracy-scan"],
        default=None,
    )
    p.add_argument("--out-results", default="results.csv")
    p.add_argument("--out-structured", default="structured.csv")
    p.add_argument("--out-jsonl", default="", help="Optional path to write results as JSONL")
    p.add_argument("--out-evidence", default="", help="Optional path to write detailed OCR evidence JSON")
    p.add_argument("--max-concurrency", type=int, default=settings.max_concurrency)
    p.add_argument("--rate-limit", type=float, default=0.0, help="Requests per second (0 = unlimited)")
    p.add_argument("--json", action="store_true", help="Print one machine-readable JSON summary to stdout")
    p.add_argument("--quiet", action="store_true", help="Suppress human progress output")

    args = p.parse_args(argv)

    run_settings = settings.merged(
        max_image_size=args.max_image_size,
        jpeg_quality=args.jpeg_quality,
        pdf_scale=args.pdf_scale,
    )

    options = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "num_predict": int(args.num_predict),
        "num_ctx": int(args.num_ctx),
    }

    prompts = PromptConfig()
    if args.templates:
        try:
            prompts = PromptConfig.from_file(args.templates)
        except Exception as e:
            print(f"[!] Failed to load templates: {e}", file=sys.stderr)

    fields = None
    if args.mode == "extract":
        if args.schema:
            try:
                with open(args.schema, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fields = [str(x) for x in data.get("fields", []) if str(x).strip()]
            except Exception as e:
                print(f"[!] Failed to load schema: {e}", file=sys.stderr)
        if not fields:
            fields = [s.strip() for s in args.fields.split(",") if s.strip()]
        if not fields:
            fields = None
    system_prompt = args.system_prompt or None
    use_profile_fields = args.mode == "extract"
    mode: Mode = "extract" if args.mode == "extract" else "describe"
    preflight_fields = _effective_fields_for_preflight(
        fields=fields,
        profile_id=args.profile,
        use_profile_fields=use_profile_fields,
    )

    ordered_results: List[List[Result]] = [[] for _ in args.files]
    valid_inputs: List[Tuple[int, str]] = []
    for index, path in enumerate(args.files):
        if os.path.exists(path):
            valid_inputs.append((index, path))
        else:
            ordered_results[index] = [
                Result(
                    source=os.path.basename(path),
                    mode=mode,
                    text="",
                    error=f"Input file not found: {path}",
                )
            ]

    start = time.perf_counter()
    if valid_inputs:
        resolved_model = args.model
        if _should_preflight_ollama(args.ocr_backend, preflight_fields):
            available, resolved_model, note = resolve_model_name(args.model, settings=run_settings)
            if note and not args.quiet and not args.json:
                print(f"[!] {note}")
            if not available and not args.quiet and not args.json:
                print(f"[i] Could not confirm model '{args.model}'. Attempting anyway.")

        limiter = RateLimiter(args.rate_limit if args.rate_limit > 0 else None)
        inference = _make_inference(options, system_prompt, limiter)

        cfg = BatchConfig(
            model=resolved_model,
            fields=fields,
            system_prompt=system_prompt,
            options=options,
            settings=run_settings,
            prompts=prompts,
            pdf_pages_separately=args.pdf_pages,
            inference=inference,
            ocr_backend=args.ocr_backend,
            profile_id=args.profile,
            preprocess=args.preprocess,
            use_profile_fields=use_profile_fields,
        )

        if args.max_concurrency <= 1:
            for index, path in valid_inputs:
                ordered_results[index] = list(run_batch([path], cfg))
        else:
            # Preserve per-file ordering semantics by submitting each file as one job.
            def _run_one(fpath: str) -> List[Result]:
                return list(run_batch([fpath], cfg))

            with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
                futs = {
                    ex.submit(_run_one, fpath): index
                    for index, fpath in valid_inputs
                }
                for fut in as_completed(futs):
                    ordered_results[futs[fut]] = fut.result()

    all_results = [
        result
        for result_group in ordered_results
        for result in result_group
    ]

    elapsed = time.perf_counter() - start
    if not args.quiet and not args.json:
        print(f"Processed {len(args.files)} file(s) in {elapsed:.2f}s")

    write_results(all_results, args.out_results, format="csv")
    structured_any = any(r.fields for r in all_results)
    wrote = [args.out_results]
    if structured_any:
        write_results(all_results, args.out_structured, format="structured_csv")
        wrote.append(args.out_structured)
    if args.out_jsonl:
        write_results(all_results, args.out_jsonl, format="jsonl")
        wrote.append(args.out_jsonl)
    if args.out_evidence:
        write_evidence(all_results, args.out_evidence)
        wrote.append(args.out_evidence)
    if args.json:
        print(
            json.dumps(
                _json_summary_payload(all_results, processed_files=len(args.files)),
                ensure_ascii=False,
            )
        )
    elif not args.quiet:
        print("Wrote: " + ", ".join(wrote))

    has_errors = any(r.error for r in all_results)
    if not valid_inputs and has_errors:
        return 1
    if args.json and has_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
