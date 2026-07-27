# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Derive and validate WhiteRabbit's ComfyUI node-localization contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from . import get_nodes

OptionPolicy = Literal["translated", "technical"]


@dataclass(frozen=True, slots=True)
class LocalizedInputContract:
    """Describe one visible node input and its stable serialized identifier."""

    identifier: str
    option_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalizedNodeContract:
    """Describe one node surface ComfyUI may localize without changing workflow IDs."""

    identifier: str
    category: str
    inputs: tuple[LocalizedInputContract, ...]
    output_count: int


@dataclass(frozen=True, slots=True)
class LocaleDefinition:
    """Declare one supported WhiteRabbit language and its Comfy catalog alias."""

    identifier: str
    native_display_name: str
    comfy_locale: str
    catalog_directory: str | None
    readme_path: str
    release_enabled: bool


@dataclass(frozen=True, slots=True)
class LocaleManifest:
    """Own WhiteRabbit's complete supported-language inventory."""

    default_language_identifier: str
    source_contract_sha256: str
    languages: tuple[LocaleDefinition, ...]

    @property
    def default_language(self) -> LocaleDefinition:
        """Return the authoritative English fallback declaration."""

        for language in self.languages:
            if language.identifier == self.default_language_identifier:
                return language
        raise ValueError("The locale manifest has no declared default language.")

    @property
    def release_languages(self) -> tuple[LocaleDefinition, ...]:
        """Return every language that must ship with full localization coverage."""

        return tuple(
            language for language in self.languages if language.release_enabled
        )


OPTION_POLICIES: Mapping[str, Mapping[str, OptionPolicy]] = {
    "AutocropToLoop": {"distance_metric": "technical"},
    "RIFE_VFI_Opt": {"ckpt_name": "technical"},
    "RIFE_VFI_Advanced": {"ckpt_name": "technical", "t_mode": "translated"},
    "RIFE_SeamTimingAnalyzer": {
        "ckpt_name": "technical",
        "calibrate_metric": "technical",
    },
    "RIFE_FPS_Resample": {"ckpt_name": "technical"},
    "PixelHold": {
        "ref_source": "translated",
        "mode": "translated",
        "score_mode": "translated",
        "apply": "translated",
        "process_on": "translated",
    },
    "UpscaleWithModelAdvanced": {"precision": "technical"},
    "BatchResizeWithLanczos": {
        "resize_mode": "translated",
        "crop_position": "translated",
        "precision": "technical",
    },
    "BatchWatermarkSingle": {
        "watermark": "technical",
        "position": "translated",
        "precision": "technical",
    },
}


def derive_localization_contract() -> tuple[LocalizedNodeContract, ...]:
    """Read the authoritative v3 registry into a stable localization contract."""

    contracts: list[LocalizedNodeContract] = []
    for node_class in cast(Sequence[Any], get_nodes()):
        schema: Any = node_class.define_schema()
        inputs = tuple(
            LocalizedInputContract(
                identifier=str(input_item.id),
                option_values=_string_options(input_item.as_dict()),
            )
            for input_item in schema.inputs
        )
        contracts.append(
            LocalizedNodeContract(
                identifier=str(schema.node_id),
                category=str(schema.category),
                inputs=inputs,
                output_count=len(schema.outputs),
            )
        )
    return tuple(contracts)


def load_locale_manifest(project_root: Path) -> LocaleManifest:
    """Load the single authoritative language registry from UTF-8 JSON."""

    document = _load_json_object(project_root / "locales" / "languages.json")
    default_language = _required_string(document, "default_language")
    source_contract_sha256 = _required_string(document, "source_contract_sha256")
    language_documents = _required_list(document, "languages")
    languages = tuple(_parse_language(item) for item in language_documents)
    identifiers = [language.identifier for language in languages]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("The locale manifest contains duplicate language IDs.")
    manifest = LocaleManifest(default_language, source_contract_sha256, languages)
    if manifest.default_language.identifier not in identifiers:
        raise ValueError("The locale manifest default language is not declared.")
    if not manifest.default_language.release_enabled:
        raise ValueError(
            "The locale manifest default language must be release-enabled."
        )
    if manifest.default_language.catalog_directory is not None:
        raise ValueError("The canonical English language must not duplicate a catalog.")
    return manifest


def localization_coverage_failures(
    project_root: Path,
    *,
    contracts: Sequence[LocalizedNodeContract] | None = None,
    validate_source_contract: bool = True,
) -> tuple[str, ...]:
    """Return every registry, catalog, and schema-localization contract failure."""

    try:
        manifest = load_locale_manifest(project_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return (f"locale manifest: {error}",)

    active_contracts = tuple(contracts or derive_localization_contract())
    failures = list(_manifest_failures(project_root, manifest))
    if (
        validate_source_contract
        and manifest.source_contract_sha256 != localization_source_contract_sha256()
    ):
        failures.append(
            "locales/languages.json: source_contract_sha256 is stale; update the "
            "Chinese catalog in the same change as the canonical English schema."
        )
    for language in manifest.release_languages:
        if language.identifier == manifest.default_language.identifier:
            continue
        if language.catalog_directory is None:
            failures.append(f"{language.identifier}: missing catalog directory")
            continue
        catalog_root = project_root / "locales" / language.catalog_directory
        failures.extend(_catalog_failures(language, catalog_root, active_contracts))
    return tuple(failures)


def localization_source_contract_sha256() -> str:
    """Hash every canonical visible schema string and stable presentation key."""

    nodes: list[dict[str, object]] = []
    for node_class in cast(Sequence[Any], get_nodes()):
        schema: Any = node_class.define_schema()
        inputs: list[dict[str, object]] = []
        for item in schema.inputs:
            options = item.as_dict().get("options")
            if schema.node_id == "BatchWatermarkSingle" and item.id == "watermark":
                options = "<dynamic-input-files>"
            inputs.append(
                {
                    "id": item.id,
                    "display_name": item.display_name or item.id,
                    "tooltip": item.tooltip,
                    "options": options,
                }
            )
        nodes.append(
            {
                "node_id": schema.node_id,
                "display_name": schema.display_name,
                "category": schema.category,
                "description": schema.description,
                "inputs": inputs,
                "outputs": [
                    {
                        "id": output.id,
                        "display_name": output.display_name,
                        "tooltip": output.tooltip,
                    }
                    for output in schema.outputs
                ],
            }
        )
    canonical = json.dumps(
        nodes,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _manifest_failures(project_root: Path, manifest: LocaleManifest) -> tuple[str, ...]:
    """Validate catalog-directory and localized-README ownership from the manifest."""

    failures: list[str] = []
    expected_directories = {
        language.catalog_directory
        for language in manifest.release_languages
        if language.catalog_directory is not None
    }
    actual_directories = {
        path.name for path in (project_root / "locales").iterdir() if path.is_dir()
    }
    if actual_directories != expected_directories:
        failures.append(
            "locale directories differ from languages.json: "
            f"expected {sorted(expected_directories)}, "
            f"found {sorted(actual_directories)}"
        )
    for language in manifest.release_languages:
        if not (project_root / language.readme_path).is_file():
            failures.append(
                f"{language.identifier}: missing localized README "
                f"{language.readme_path}"
            )
        if (
            language.catalog_directory is not None
            and not (
                project_root / "locales" / language.catalog_directory / "STYLE.md"
            ).is_file()
        ):
            failures.append(
                f"{language.identifier}: missing terminology guide STYLE.md"
            )
    return tuple(failures)


def _catalog_failures(
    language: LocaleDefinition,
    catalog_root: Path,
    contracts: Sequence[LocalizedNodeContract],
) -> tuple[str, ...]:
    """Validate one complete Comfy locale payload against the live schema surface."""

    try:
        main = _load_json_object(catalog_root / "main.json")
        node_definitions = _load_json_object(catalog_root / "nodeDefs.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return (f"{language.identifier}: unreadable catalog: {error}",)

    failures: list[str] = []
    expected_categories = {contract.category for contract in contracts}
    categories = _object_value(main, "nodeCategories", failures, language.identifier)
    if categories is not None:
        _validate_exact_keys(
            categories,
            expected_categories,
            f"{language.identifier}: nodeCategories",
            failures,
        )
        for category, translation in categories.items():
            _require_chinese_text(
                translation,
                f"{language.identifier}: category {category}",
                failures,
            )

    expected_nodes = {contract.identifier for contract in contracts}
    _validate_exact_keys(
        node_definitions,
        expected_nodes,
        f"{language.identifier}: nodeDefs",
        failures,
    )
    for contract in contracts:
        node = _object_value(
            node_definitions,
            contract.identifier,
            failures,
            language.identifier,
        )
        if node is None:
            continue
        _validate_node(contract, node, language.identifier, failures)
    return tuple(failures)


def _validate_node(
    contract: LocalizedNodeContract,
    node: Mapping[str, object],
    language_identifier: str,
    failures: list[str],
) -> None:
    """Validate one localized node while retaining every stable Comfy identifier."""

    prefix = f"{language_identifier}: {contract.identifier}"
    _require_chinese_text(node.get("display_name"), f"{prefix}.display_name", failures)
    _require_chinese_text(node.get("description"), f"{prefix}.description", failures)
    inputs = _object_value(node, "inputs", failures, prefix)
    if inputs is not None:
        expected_inputs = {
            input_contract.identifier for input_contract in contract.inputs
        }
        _validate_exact_keys(inputs, expected_inputs, f"{prefix}.inputs", failures)
        for input_contract in contract.inputs:
            input_document = _object_value(
                inputs,
                input_contract.identifier,
                failures,
                prefix,
            )
            if input_document is not None:
                _validate_input(
                    input_contract,
                    input_document,
                    prefix,
                    failures,
                )
    outputs = _object_value(node, "outputs", failures, prefix)
    if outputs is not None:
        expected_outputs = {str(index) for index in range(contract.output_count)}
        _validate_exact_keys(outputs, expected_outputs, f"{prefix}.outputs", failures)
        for index in expected_outputs:
            output_document = _object_value(outputs, index, failures, prefix)
            if output_document is not None:
                _require_chinese_text(
                    output_document.get("name"),
                    f"{prefix}.outputs.{index}.name",
                    failures,
                )


def _validate_input(
    contract: LocalizedInputContract,
    document: Mapping[str, object],
    node_prefix: str,
    failures: list[str],
) -> None:
    """Validate translated labels, tooltips, and deliberately classified options."""

    prefix = f"{node_prefix}.inputs.{contract.identifier}"
    _require_chinese_text(document.get("name"), f"{prefix}.name", failures)
    _require_chinese_text(document.get("tooltip"), f"{prefix}.tooltip", failures)
    if not contract.option_values:
        if "options" in document:
            failures.append(f"{prefix}.options: input has no string options")
        return
    node_identifier = node_prefix.split(": ", maxsplit=1)[1]
    policy = OPTION_POLICIES.get(node_identifier, {}).get(contract.identifier)
    if policy is None:
        failures.append(f"{prefix}.options: string options lack a localization policy")
        return
    if policy == "technical":
        if "options" in document:
            failures.append(f"{prefix}.options: technical values must remain raw")
        return
    options = _object_value(document, "options", failures, prefix)
    if options is not None:
        _validate_exact_keys(
            options, set(contract.option_values), f"{prefix}.options", failures
        )
        for value, translation in options.items():
            _require_chinese_text(
                translation,
                f"{prefix}.options.{value}",
                failures,
            )


def _string_options(input_document: Mapping[str, object]) -> tuple[str, ...]:
    """Return serializable string options while excluding numeric quality controls."""

    raw_options = input_document.get("options")
    if not isinstance(raw_options, list) or not all(
        isinstance(value, str) for value in raw_options
    ):
        return ()
    return tuple(raw_options)


def _parse_language(value: object) -> LocaleDefinition:
    """Parse one strict manifest language entry without loose dictionary access."""

    if not isinstance(value, dict):
        raise ValueError("Each language entry must be a JSON object.")
    catalog_directory = value.get("catalog_directory")
    if catalog_directory is not None and not isinstance(catalog_directory, str):
        raise ValueError("catalog_directory must be a string or null.")
    return LocaleDefinition(
        identifier=_required_string(value, "id"),
        native_display_name=_required_string(value, "native_display_name"),
        comfy_locale=_required_string(value, "comfy_locale"),
        catalog_directory=catalog_directory,
        readme_path=_required_string(value, "readme_path"),
        release_enabled=_required_bool(value, "release_enabled"),
    )


def _load_json_object(path: Path) -> dict[str, object]:
    """Load one UTF-8 JSON object with an actionable structural error."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return cast(dict[str, object], value)


def _required_string(document: Mapping[str, object], key: str) -> str:
    """Read one required non-empty string from a JSON object."""

    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _required_bool(document: Mapping[str, object], key: str) -> bool:
    """Read one required boolean from a JSON object."""

    value = document.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _required_list(document: Mapping[str, object], key: str) -> list[object]:
    """Read one required JSON list from a JSON object."""

    value = document.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list.")
    return value


def _object_value(
    document: Mapping[str, object],
    key: str,
    failures: list[str],
    prefix: str,
) -> Mapping[str, object] | None:
    """Read one required nested JSON object while accumulating diagnostics."""

    value = document.get(key)
    if not isinstance(value, dict):
        failures.append(f"{prefix}.{key}: expected an object")
        return None
    return cast(Mapping[str, object], value)


def _validate_exact_keys(
    document: Mapping[str, object],
    expected: set[str],
    prefix: str,
    failures: list[str],
) -> None:
    """Reject missing and stale localized keys with deterministic diagnostics."""

    actual = set(document)
    for key in sorted(expected - actual):
        failures.append(f"{prefix}: missing {key}")
    for key in sorted(actual - expected):
        failures.append(f"{prefix}: stale {key}")


def _require_chinese_text(value: object, location: str, failures: list[str]) -> None:
    """Require non-empty Simplified Chinese presentation copy, not English fallback."""

    if not isinstance(value, str) or not value.strip():
        failures.append(f"{location}: missing translation")
    elif not any("\u3400" <= character <= "\u9fff" for character in value):
        failures.append(f"{location}: lacks Chinese text")


__all__ = [
    "LocaleDefinition",
    "LocaleManifest",
    "LocalizedInputContract",
    "LocalizedNodeContract",
    "OPTION_POLICIES",
    "derive_localization_contract",
    "load_locale_manifest",
    "localization_coverage_failures",
    "localization_source_contract_sha256",
]
