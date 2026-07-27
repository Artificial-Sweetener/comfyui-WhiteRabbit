# AGENTS.md

## Mission

WhiteRabbit is a high-quality, strongly typed ComfyUI extension for video frame
manipulation, interpolation, scaling, stabilization, looping, and post-processing.
Engineering priorities are behavior safety, strict separation of concerns, complete
feature integrations, deterministic execution, explicit validation, ComfyUI
compatibility, runtime safety, and long-term maintainability.

## Public behavior boundary

- Preserve serialized workflow compatibility unless the maintainer explicitly
  approves a breaking change.
- Treat node identifiers, display names, categories, input keys and order, defaults,
  output types and order, execution return shapes, and user-visible behavior as the
  public contract.
- Change internals freely and completely within that boundary.
- Add characterization tests before structurally changing behavior-critical code.

## Localization policy

- Route every WhiteRabbit-owned visible node label, description, tooltip, output
  label, category, and eligible option label through the canonical v3 schema and
  its locale catalog. Do not create a second English catalog.
- Treat `locales/languages.json` as the sole supported-locale registry. Do not
  duplicate locale inventories in source, tools, tests, or documentation.
- English v3 schemas are the canonical source and fallback language. Every
  release-enabled non-English locale must provide complete direct translations
  for every owned visible schema field in the same change.
- Add or change visible English text and all release-enabled translations
  atomically. Adding a locale requires complete catalog and README coverage.
- Preserve serialized workflow compatibility: node IDs, categories, input IDs,
  option values, output positions, and execution behavior are never translated.
  Locale catalogs translate presentation only.
- Preserve ComfyUI-owned and third-party text as supplied by ComfyUI; do not
  maintain a parallel corpus for text WhiteRabbit does not own.
- Keep each release-enabled localized README structurally and factually aligned
  with `readme.md`, while writing natural, audience-appropriate prose rather
  than mechanical translations.
- Write translations directly. Maintain the locale terminology guide and do not
  weaken locale-registry, coverage, stale-key, or integration validation to
  accept incomplete work.

## Authoritative environment

- Use Windows PowerShell syntax.
- Run all verification from the repository root.
- Use `E:\ComfyUI\.venv`; do not create a repository-local virtual environment.
- Tests: `E:\ComfyUI\.venv\Scripts\python.exe -m pytest -n auto -q`
- Lint: `E:\ComfyUI\.venv\Scripts\ruff.exe check .`
- Format: `E:\ComfyUI\.venv\Scripts\ruff.exe format .`
- Types: `E:\ComfyUI\.venv\Scripts\mypy.exe --strict __init__.py whiterabbit tests`

## Architecture

- `whiterabbit/nodes_v3`: thin Comfy v3 schemas and execution entry points.
- `whiterabbit/services`: application use cases and orchestration.
- `whiterabbit/domain`: stable value objects, plans, policies, and pure behavior.
- `whiterabbit/runtime`: ComfyUI, tensor, model, device, filesystem, image, and
  network adapters.
- `whiterabbit/shared`: small lower-level validation and logging primitives.
- Dependencies flow from nodes to services and from services to domain/runtime.
- Domain modules never import ComfyUI.
- Nodes never own model loading, downloads, filesystem policy, or non-trivial
  algorithms.
- Every concern has one authoritative owner. Do not duplicate policy across layers.
- Complete refactors fully: update all callsites, remove obsolete internal paths,
  and do not leave internal compatibility shims or transitional adapters.
- Preserve compatibility only at ComfyUI-facing and persisted-workflow boundaries.

## ComfyUI nodes

- Comfy v3 is the sole export path.
- The root `comfy_entrypoint()` and `whiterabbit.nodes_v3.get_nodes()` are the
  authoritative registry.
- Keep schemas deterministic and free of expensive IO, network access, or model
  loading.
- Give every node a concise description and every visible input/output an accurate
  tooltip.
- Validate inputs before side effects and raise actionable errors.
- Return exactly the declared output shape.

## RIFE and model behavior

- Resolve model locations through ComfyUI's `frame_interpolation` folder registry.
- WhiteRabbit owns its enhanced RIFE execution behavior, including scale control,
  internal bidirectional ensemble, arbitrary timing, caching, FPS resampling, seam
  analysis, and stabilization.
- Integrate ComfyUI memory management without reducing WhiteRabbit's feature set.
- Automatic downloads are limited to trusted catalog artifacts, use timeouts,
  bounded destinations, temporary files, atomic publication, checksum validation,
  progress reporting, and cleanup on failure.
- Model format detection must fail with an actionable error when unsupported.
- Device, dtype, VRAM, loading, offloading, and cache lifetime are correctness
  concerns and belong to runtime owners.

## Typing

- `mypy --strict` must pass for source and tests.
- Type every function signature and important internal state.
- Prefer dataclasses, enums, `Literal`, `TypedDict`, and `Protocol` over loose
  dictionaries and `Any`.
- Allow `Any` only at genuinely dynamic ComfyUI/plugin boundaries and narrow it
  before core logic uses it.
- Use `torch.Tensor` at tensor boundaries and validate rank, layout, channels,
  dtype, and batch assumptions explicitly.
- Do not use file-wide mypy suppression or unchecked internal modules.

## Code quality

- Write expressive, concise names and cohesive modules.
- Use docstrings for modules, public classes, and non-obvious functions. Do not
  restate obvious mechanics.
- Use inline comments only for invariants, edge cases, or external constraints.
- Use structured logging; do not use `print` for runtime diagnostics.
- Preserve exception context. Do not swallow exceptions or use bare `except`.
- Remove dead code, unused nodes, obsolete fallbacks, and stale vendored behavior.
- Keep security-sensitive paths bounded and validated.

## Testing

- Add or update tests for every behavior change and bug fix.
- Protect registration, schemas, widget order/defaults, output ordering, model IO,
  tensor layouts, download failure paths, and workflow compatibility.
- Prefer pure domain tests, service tests with injected runtime boundaries, and
  focused integration tests over excessive mocking.
- Test success and failure paths.
- Run the complete suite before completion; failures are blocking.

## Definition of done

- Public behavior is protected by tests.
- Architecture ownership and dependency direction are correct.
- All changed code is fully typed and documented appropriately.
- Obsolete paths and temporary bridges are removed.
- Ruff format, Ruff lint, strict mypy, and the full pytest suite pass.
- ComfyUI imports the extension and exposes the intended v3 nodes.
- Relevant model paths and RIFE behavior are directly verified.

## Commit and release policy

- Use Conventional Commits in the form `type(scope): subject`; keep the subject
  imperative, concise, and free of trailing punctuation.
- Use one of `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, or
  `chore`. Keep commits atomic and cohesive.
- Semantic release creates versions from commits after the latest `vX.Y.Z` tag:
  `feat` creates a minor release; `fix` and `perf` create a patch release; an
  exclamation mark after the type or scope, or a `BREAKING CHANGE:` footer,
  creates a major release. The remaining allowed types do not release by
  themselves.
- Mark intentionally breaking public behavior explicitly, for example
  `feat(rife)!: replace legacy timing controls`, and include a `BREAKING CHANGE:`
  footer that states the migration required by workflow authors.
- Do not manually bump release metadata or edit `CHANGELOG.md` for ordinary
  releases. The release workflow owns `package.json`, `package-lock.json`,
  `pyproject.toml`, `whiterabbit.__version__`, and the changelog.
