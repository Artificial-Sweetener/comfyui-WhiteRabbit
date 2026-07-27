# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Characterization and regression tests for WhiteRabbit-owned localization."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from whiterabbit.nodes_v3.localization import (
    LocalizedInputContract,
    LocalizedNodeContract,
    derive_localization_contract,
    load_locale_manifest,
    localization_coverage_failures,
    localization_source_contract_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CODE_BLOCK_PATTERN = re.compile(r"^```[^\r\n]*\r?\n.*?^```$", re.MULTILINE | re.DOTALL)
_EXTERNAL_URL_PATTERN = re.compile(r"https://[^\s\)\"]+")
_HEADING_PATTERN = re.compile(r"^(#{2,4}) ", re.MULTILINE)
_IMAGE_SOURCE_PATTERN = re.compile(r'<img\s+src="([^"]+)"')


def test_every_release_locale_has_complete_schema_derived_coverage() -> None:
    """No WhiteRabbit node UI can silently fall back to English in Chinese mode."""

    failures = localization_coverage_failures(PROJECT_ROOT)

    assert failures == (), "\n".join(failures)


def test_manifest_hash_tracks_every_canonical_visible_schema_string() -> None:
    """An English schema copy change requires a paired catalog review and update."""

    manifest = load_locale_manifest(PROJECT_ROOT)

    assert manifest.source_contract_sha256 == localization_source_contract_sha256()


def test_every_string_option_is_explicitly_classified() -> None:
    """New widget options must be intentionally translated or kept technical."""

    failures = localization_coverage_failures(PROJECT_ROOT)

    assert not any("lack a localization policy" in failure for failure in failures)


def test_new_visible_schema_input_requires_a_translation(tmp_path: Path) -> None:
    """A new public input fails until its Chinese label and tooltip are authored."""

    _write_fixture_project(
        tmp_path,
        node_definitions={
            "DemoNode": {
                "display_name": "演示节点",
                "description": "用于验证本地化覆盖。",
                "inputs": {
                    "existing": {"name": "现有输入", "tooltip": "已经翻译的输入。"},
                },
                "outputs": {"0": {"name": "结果"}},
            }
        },
    )
    contract = LocalizedNodeContract(
        identifier="DemoNode",
        category="demo",
        inputs=(
            LocalizedInputContract("existing", ()),
            LocalizedInputContract("new_input", ()),
        ),
        output_count=1,
    )

    failures = localization_coverage_failures(
        tmp_path,
        contracts=(contract,),
        validate_source_contract=False,
    )

    assert "zh-Hans: DemoNode.inputs: missing new_input" in failures


def test_new_release_locale_requires_its_own_complete_catalog(tmp_path: Path) -> None:
    """The manifest cannot declare a release language without translation files."""

    _write_fixture_project(tmp_path, node_definitions=_complete_demo_node())
    manifest_path = tmp_path / "locales" / "languages.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["languages"].append(
        {
            "id": "ja",
            "native_display_name": "日本語",
            "comfy_locale": "ja",
            "catalog_directory": "ja",
            "readme_path": "README.ja.md",
            "release_enabled": True,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failures = localization_coverage_failures(
        tmp_path,
        contracts=(_demo_contract(),),
        validate_source_contract=False,
    )

    assert "ja: missing localized README README.ja.md" in failures
    assert any(failure.startswith("ja: unreadable catalog:") for failure in failures)


def test_every_release_readme_matches_the_canonical_structure() -> None:
    """Localized docs retain the product facts and navigable structure of English."""

    authority = (PROJECT_ROOT / "readme.md").read_text(encoding="utf-8")
    manifest = load_locale_manifest(PROJECT_ROOT)
    for language in manifest.release_languages:
        localized = (PROJECT_ROOT / language.readme_path).read_text(encoding="utf-8")
        assert _heading_depths(localized) == _heading_depths(authority)
        assert _code_blocks(localized) == _code_blocks(authority)
        assert _image_sources(localized) == _image_sources(authority)
        assert _external_urls(localized) == _external_urls(authority)


def test_every_readme_selector_lists_each_release_language() -> None:
    """Every registered release language remains discoverable from every README."""

    manifest = load_locale_manifest(PROJECT_ROOT)
    for current_language in manifest.release_languages:
        content = (PROJECT_ROOT / current_language.readme_path).read_text(
            encoding="utf-8"
        )
        for language in manifest.release_languages:
            assert language.native_display_name in content
            if language.identifier != current_language.identifier:
                assert f'href="{language.readme_path}"' in content


def test_contract_covers_the_registered_v3_nodes() -> None:
    """Localization derives its node inventory from the public v3 registry only."""

    contracts = derive_localization_contract()

    assert [contract.identifier for contract in contracts] == [
        "PrepareLoopFrames",
        "AssembleLoopFrames",
        "RollFrames",
        "UnrollFrames",
        "AutocropToLoop",
        "TrimBatchEnds",
        "RIFE_VFI_Opt",
        "RIFE_VFI_Advanced",
        "RIFE_SeamTimingAnalyzer",
        "RIFE_FPS_Resample",
        "PixelHold",
        "UpscaleWithModelAdvanced",
        "BatchResizeWithLanczos",
        "BatchWatermarkSingle",
    ]


def _write_fixture_project(
    project_root: Path,
    *,
    node_definitions: dict[str, object],
) -> None:
    """Write the smallest complete manifest-backed Chinese locale fixture."""

    (project_root / "locales" / "zh").mkdir(parents=True)
    (project_root / "readme.md").write_text("# English\n", encoding="utf-8")
    (project_root / "README_zh-CN.md").write_text("# 简体中文\n", encoding="utf-8")
    (project_root / "locales" / "zh" / "STYLE.md").write_text(
        "# 术语\n",
        encoding="utf-8",
    )
    (project_root / "locales" / "languages.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_language": "en",
                "source_contract_sha256": "fixture",
                "languages": [
                    {
                        "id": "en",
                        "native_display_name": "English",
                        "comfy_locale": "en",
                        "catalog_directory": None,
                        "readme_path": "readme.md",
                        "release_enabled": True,
                    },
                    {
                        "id": "zh-Hans",
                        "native_display_name": "简体中文",
                        "comfy_locale": "zh",
                        "catalog_directory": "zh",
                        "readme_path": "README_zh-CN.md",
                        "release_enabled": True,
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (project_root / "locales" / "zh" / "main.json").write_text(
        json.dumps({"nodeCategories": {"demo": "演示"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_root / "locales" / "zh" / "nodeDefs.json").write_text(
        json.dumps(node_definitions, ensure_ascii=False),
        encoding="utf-8",
    )


def _complete_demo_node() -> dict[str, object]:
    """Return one complete translated node used by manifest regression coverage."""

    return {
        "DemoNode": {
            "display_name": "演示节点",
            "description": "用于验证本地化覆盖。",
            "inputs": {"existing": {"name": "现有输入", "tooltip": "已经翻译的输入。"}},
            "outputs": {"0": {"name": "结果"}},
        }
    }


def _demo_contract() -> LocalizedNodeContract:
    """Return the compact schema contract that matches the complete fixture."""

    return LocalizedNodeContract(
        identifier="DemoNode",
        category="demo",
        inputs=(LocalizedInputContract("existing", ()),),
        output_count=1,
    )


def _heading_depths(content: str) -> tuple[int, ...]:
    """Return heading depths without coupling translated headings to English words."""

    return tuple(len(match) for match in _HEADING_PATTERN.findall(content))


def _code_blocks(content: str) -> tuple[str, ...]:
    """Return technical code blocks that must remain exact across READMEs."""

    return tuple(_CODE_BLOCK_PATTERN.findall(content))


def _image_sources(content: str) -> tuple[str, ...]:
    """Return image evidence that localized READMEs must preserve."""

    return tuple(_IMAGE_SOURCE_PATTERN.findall(content))


def _external_urls(content: str) -> Counter[str]:
    """Return external destinations while permitting localized Markdown labels."""

    return Counter(_EXTERNAL_URL_PATTERN.findall(content))
