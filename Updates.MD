## 2026-05-29

- Simplified `core.pipeline` PDF page handling, Ollama field/evidence construction, and auto-backend fallback routing with small internal helpers while preserving the existing PDF tuple and fallback semantics.
- Consolidated Ollama chat message/kwargs/content handling across image, text, and stream requests, with a stream regression covering shared system/image message construction and timeout usage.
- Moved Streamlit display-entry shaping into a pure helper, closed loaded image handles in app/legacy CLI paths, and kept the CLI JSON summary schema behind a deterministic helper.
- Verified the refactors with focused pipeline, Ollama adapter, app, CLI, and lint checks before the full regression run.

## 2026-05-04

- Kept the GitHub Actions evaluation job as a live Ollama-backed gate, with a readiness loop, a longer request timeout, and a job timeout so hosted-runner startup and inference delays are handled explicitly.
- Added evaluator `--fail-on-errors` so CI exits non-zero when live OCR requests time out or otherwise return per-file errors after artifacts are written.
- Added CI and evaluator regressions to prevent replacing the live gate with mock mode and to keep timeout/error results from being reported as successful evaluations.
- Updated README CI notes and evaluator flag documentation for the live Ollama gate.

## 2026-05-03

- Tightened the branded app header into a compact logo/title lockup with the description aligned under the title, smaller responsive type, and a measured logo size.
- Moved the top-menu logo away from the sidebar expand control so the collapsed sidebar button and logo no longer overlap.

## 2026-05-03

- Simplified the Streamlit top menu branding to show only the Curiosity AI logo mark, removing the extra text label from the menu chrome.

## 2026-05-03

- Added the public Curiosity AI logo to the Streamlit top menu and the app header as an embedded SVG data URI so the UI has visible branding without relying on static file routing.
- Refined the header layout around the brand mark with responsive logo sizing, aligned title copy, and theme regressions for the branded header styles.

## 2026-05-03

- Made the default DeepSeek OCR model tag explicit as `deepseek-ocr:latest` across runtime settings, sidebar options, docs, evaluation commands, and regressions.

## 2026-05-03

- Changed the standard OCR model from `gemma4:latest` to `deepseek-ocr:latest` in runtime settings, environment fallback defaults, and the sidebar fallback model order.
- Pinned the sidebar model selector to show the configured default first even when Ollama returns a different local model order.
- Updated README and Makefile examples so first-run setup and evaluation commands pull/use `deepseek-ocr:latest` by default.
- Added regressions covering the default settings model, sidebar default ordering, and `BatchConfig()` model resolution.

## 2026-05-03

- Added a final high-specificity button foreground pass so Streamlit's nested markdown labels inside Upload and primary scan buttons stay white before hover, not only during hover.
- Changed primary button hover treatment to a dark navy accent while preserving white text and icon foregrounds.
- Added theme regression assertions for `-webkit-text-fill-color` and the primary hover accent contract.

## 2026-05-03

- Tightened uploaded-file chip styling with higher-specificity remove/cancel button selectors so Streamlit's blue icon-button theme no longer leaks into file rows.
- Made uploaded-file rows more compact with smaller document icons, full-width wrappers, neutral remove buttons, and a smaller add-files button.
- Forced the upload dropzone button label, icon, spans, and SVG paths to share the same white foreground color.

## 2026-05-03

- Polished the uploaded-file list by targeting Streamlit's `stFileChips` rows directly with compact full-width file chips, readable filename/size text, and bounded scrolling.
- Narrowed primary uploader button styling to the dropzone upload button so file remove/add controls no longer inherit oversized blue primary-button treatment.
- Added theme regressions for file-chip rows, delete buttons, and the add-files control.

## 2026-05-03

- Expanded the Docling setup docs with concrete `docling-tools` commands for both standard Docling pipeline artifacts and RapidOCR artifacts.
- Documented that `DOCLING_ARTIFACTS_PATH` must point at the shared artifacts root and added checks for `model.safetensors` and the RapidOCR detection weights.
- Added troubleshooting guidance for incomplete Docling artifact downloads and partial-cache recovery with `--force`.

## 2026-05-03

- Reworked the Advanced Model Options expander from cramped sidebar sliders into compact numeric inputs so values, labels, and bounds no longer overlap.
- Added light, tabular number-input styling for the advanced controls, including restrained stepper buttons that do not inherit primary button treatment.
- Added focused regressions for advanced option defaults/routing and the number-input theme contract.

## 2026-05-03

- Fixed nested Streamlit button labels and icons so upload, run scan, and download buttons keep white text/icons on blue primary backgrounds across normal, hover, and active states.
- Added a theme regression assertion covering nested button descendants, including uploader button content.

## 2026-05-02

- Added a Streamlit theme config to force light-mode widgets at the framework level instead of relying only on post-render CSS overrides.
- Deepened the UI CSS overrides for BaseWeb select menus, listbox options, text areas, expander headers, checkbox/radio controls, and sidebar alert surfaces so dark widget defaults do not leak into the light app.
- Extended theme tests to cover the light Streamlit config plus nested listbox, text-area, and expander styling contracts.

## 2026-05-02

- Refined the Streamlit shell into a cohesive modern-minimal OCR workspace with lighter app chrome, calmer header/card surfaces, and consistent document-gray/scan-blue tokens.
- Fixed out-of-place dark sidebar controls by forcing select boxes, uploader buttons, labels, popovers, and setup/help surfaces onto the shared light theme.
- Added a focused theme regression assertion for the Streamlit header and select-control styling contract.

## 2026-05-02

- Buffered auto-mode Docling/hybrid PDF attempts so a later page failure falls back to the Ollama PDF route without emitting duplicate partial Docling results.
- Applied selected preprocessing presets to Docling and hybrid image jobs, including rendered PDF page image jobs, and preserved the applied preprocessing steps in result evidence.
- Added focused regressions for auto PDF fallback buffering and Docling/hybrid image preprocessing.

## 2026-05-02

- Fixed the Streamlit custom extraction field defaults so selecting an invoice, receipt, or table profile seeds the extraction fields from that profile instead of the generic invoice placeholder.
- Fixed Docling image stream naming for rendered PDF pages so per-page Docling/hybrid PDF routes send PNG bytes with an image filename suffix.
- Added focused regressions for profile-driven UI custom extraction defaults and rendered PDF page image stream names.

## 2026-05-02

- Restored UI profile preprocessing defaults by adding a Profile default option that routes `None` into `BatchConfig`.
- Made Docling and hybrid PDF routes honor per-page PDF processing by fanning rendered pages into Docling image jobs with page metadata.
- Added focused regressions for Streamlit profile-default preprocessing and Docling/hybrid PDF page fan-out.

## 2026-05-02

- Preserved explicit description mode when a non-generic profile is selected by disabling profile field expansion for CLI/UI description runs.
- Skipped unnecessary Ollama model preflight for hybrid description runs while keeping preflight for hybrid extraction, including profile-driven extraction fields.
- Added focused regressions for CLI/UI profile description routing, hybrid description preflight skipping, and profile-field extraction preflight behavior.

## 2026-05-01

- Routed evaluation through the shared `BatchConfig`/`run_batch` OCR path so backend, profile, and preprocessing selections match CLI/UI behavior.
- Added deterministic metrics JSON output with run metadata, field and aggregate accuracy, backend/profile/preprocess group metrics, detailed rows, and evidence path metadata.
- Added evaluator evidence JSON export backed by `write_evidence`, including deterministic mock `Result` objects so mock runs work without Ollama and never mutate the README.
- Updated README docs for backend modes, profiles, preprocessing presets, optional local Docling setup, CLI evidence export, and evaluator metrics/evidence flags.

## 2026-05-01

- Added Streamlit sidebar controls for document profile, backend mode, and preprocessing while preserving the existing upload, model, options, schema, appearance, and export flows.
- Routed the selected UI profile/backend/preprocessing values into `BatchConfig` and skipped Ollama model preflight for pure Docling UI runs.
- Added compact result evidence metadata plus expandable field evidence, and added an Evidence JSON download alongside the existing CSV, structured CSV, and JSONL exports.
- Added focused UI/helper regressions for sidebar defaults/options, Streamlit processing routing, result evidence formatting, and download button coverage.

## 2026-05-01

- Added CLI backend/profile/preprocessing flags and routed them into `BatchConfig` while keeping the existing JSON stdout summary shape unchanged.
- Skipped Ollama model preflight for pure Docling CLI runs and retained preflight for Ollama, hybrid, and auto modes.
- Added detailed evidence JSON export via `--out-evidence`, including OCR/evidence metadata while excluding binary preview image bytes.
- Added focused CLI/export regressions for flag routing, Docling preflight skipping, clean JSON stdout, and detailed evidence serialization.

## 2026-05-01

- Implemented `run_batch` routing for Docling, hybrid, and auto OCR backend modes while preserving the default Ollama path and lazy Docling imports.
- Added hybrid extraction that uses Docling OCR/layout text as context and a text-only Ollama chat call for semantic field extraction, preserving OCR text/blocks and field evidence metadata.
- Added deterministic auto routing so PDFs try Docling first and fall back to Ollama on Docling availability/configuration errors, while images continue through the existing Ollama route.
- Added focused routing regressions for pure Docling, Docling errors, hybrid extraction/description behavior, auto fallback, auto image routing, and the Docling-free default Ollama path.

## 2026-05-01

- Closed the Docling adapter fallback path so missing local-only configuration symbols raise an unsupported-version error before `DocumentConverter` can be instantiated with defaults.

## 2026-05-01

- Hardened the optional Docling adapter so conversion requires `DOCLING_ARTIFACTS_PATH` to point at a local model artifacts directory before constructing the converter.
- Configured Docling conversion with local artifacts, disabled remote services/plugins, kind-specific allowed formats, and safe stream names for extensionless sources.
- Pinned the optional Docling dependency range to `docling>=2.88,<3` and added focused regressions for local-only configuration behavior.

## 2026-05-01

- Added an optional Docling adapter with lazy imports, availability status, and a clear install message for missing optional dependencies.
- Added local-only Docling conversion helpers for PDF bytes and PIL images that return the shared `Result`/OCR block metadata shape without changing the default Ollama route.
- Added `requirements-docling.txt` so Docling remains outside the base and constrained dependency sets.
- Added focused adapter regressions that mock Docling and verify base imports stay Docling-free.

## 2026-05-01

- Added schema-first Ollama extraction by building JSON Schema from explicit or profile-provided fields and forwarding it through the existing extraction path.
- Added simple per-field extraction evidence on `Result.field_evidence` while keeping `Result.fields` as plain extracted values and preserving missing-field behavior.
- Enabled built-in profile fields, such as the invoice profile, to trigger extraction when no explicit fields are supplied while generic profile runs remain descriptive by default.
- Added focused regressions for schema generation, adapter format forwarding, evidence separation, missing requested fields, and profile-driven extraction mode.

## 2026-05-01

- Fixed backend routing metadata so PDF batch results report deterministic preprocessing steps without changing the existing 9-value `process_pdf` tuple.
- Applied document profile preprocessing defaults when `BatchConfig.preprocess` is omitted while preserving explicit `preprocess="none"` overrides.
- Added focused regressions for invoice profile defaults, explicit no-preprocess overrides, and PDF/image preprocessing metadata parity.

## 2026-05-01

- Added an OCR backend routing module with recognized `ollama`, `docling`, `hybrid`, and `auto` backend modes while keeping only the Ollama path implemented.
- Extended `BatchConfig` with defaulted backend, profile, and preprocessing routing fields; `BatchConfig()` now resolves the default model from settings and keeps Ollama/generic/none defaults.
- Preserved the existing image, PDF, and batch result shapes while adding Result metadata for the active engine/profile on the default Ollama route.
- Added focused pipeline tests for backend defaults, default Ollama routing, and explicit non-Ollama unsupported-mode errors.

## 2026-05-01

- Added core OCR evidence, document profile, and preprocessing result dataclasses while preserving existing `Result` constructor compatibility.
- Added built-in generic, invoice, receipt, and table profile selection with clear invalid-profile errors.
- Added Pillow-only preprocessing presets for none, document cleanup, and high-accuracy scans without mutating source images.
- Added focused tests for model defaults/serialization, profile lookup, and deterministic preprocessing behavior.

## 2026-04-28

- Kept CLI `--json` stdout parseable when `--quiet` is omitted by suppressing human progress/model-status lines in JSON mode.
- Added an Ollama model inventory status path so the sidebar can distinguish "Ollama reachable with no local models" from "Ollama not reachable".
- Added focused regressions for non-quiet JSON output, Ollama empty-list reachability, and sidebar setup inventory wiring.

## 2026-04-28

- Refreshed the Streamlit UI with a shared theme layer, accessible focus states, 44px controls, reduced-motion safeguards, result cards, status badges, and a compact empty state.
- Polished sidebar wording, setup readiness badges, and the primary scan action while preserving the existing upload, OCR, preview, structured JSON, and export flows.
- Added focused UI regression tests for theme safeguards and result metadata/status formatting.

## 2026-04-28

- Completed the mypy cleanup for PDF rendering, image resizing, Ollama request payloads, pipeline result metadata, and CLI result modes.
- Added a `make type` target and wired it into `make check` so CI runs mypy with the existing check workflow.
- Added `types-requests` to the development dependency set and constraints so evaluation imports are typed reproducibly.

## 2026-04-28

- Added `constraints.txt` as the reproducible install layer for the verified runtime, development, and stable transitive dependency versions.
- Updated CI to run `make check` across Python 3.10, 3.11, 3.12, and 3.13 while keeping the Ollama evaluation on a single Python 3.11 job.
- Routed CI and `make install` dependency installation through `constraints.txt`, and documented the constrained install path in the README.

## 2026-04-28

- Wired runtime `Settings` into Ollama adapter model listing and request execution.
- Added injectable model-list TTL support through `Settings.model_list_ttl` while preserving existing default cache behavior.
- Added timeout-aware adapter paths for Ollama `list`, `show`, chat, and streaming chat calls via explicit `timeout` or `Settings.request_timeout`.
- Passed `BatchConfig.settings` through the default pipeline inference path so Streamlit and CLI batch runs use the configured adapter timeout.
- Added regression tests for settings-driven TTL, timeout propagation, and pipeline settings forwarding without requiring a live Ollama service.

## 2026-04-28

- Added CLI agent mode with `--json` and `--quiet` for a single machine-readable stdout summary.
- Preserved deterministic CLI result ordering while reporting missing input paths as explicit error rows in JSON and CSV outputs.
- Added focused CLI tests for quiet JSON output, missing-input rows, and all-missing nonzero exit codes.

## 2026-04-28

- Replaced the misleading PDF sidebar option with truthful first-page wording while preserving the existing processing behavior.
- Added a compact sidebar setup check for Ollama reachability, local vision models, and PDF support.
- Replaced the long empty-state instruction block with a short upload/model/run prompt.
- Added focused helper tests for the PDF copy contract and setup readiness summary.
