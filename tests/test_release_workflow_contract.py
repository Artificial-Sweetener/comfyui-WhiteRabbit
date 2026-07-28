"""Contract tests for release automation and version metadata."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DEPENDENCIES = {
    "@semantic-release/changelog",
    "@semantic-release/commit-analyzer",
    "@semantic-release/exec",
    "@semantic-release/git",
    "@semantic-release/release-notes-generator",
    "semantic-release",
}


def test_release_managed_versions_stay_synchronized() -> None:
    """Keep all release-managed source metadata on the same version."""

    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package = json.loads((PROJECT_ROOT / "package.json").read_text("utf-8"))
    package_lock = json.loads(
        (PROJECT_ROOT / "package-lock.json").read_text(encoding="utf-8")
    )
    package_init = (PROJECT_ROOT / "whiterabbit" / "__init__.py").read_text(
        encoding="utf-8"
    )

    version_match = re.search(r'^version = "([^"]+)"$', pyproject_text, re.MULTILINE)

    assert version_match is not None
    version = version_match.group(1)
    assert package["version"] == version
    assert package_lock["version"] == version
    assert package_lock["packages"][""]["version"] == version
    assert f'__version__ = "{version}"' in package_init


def test_release_dependencies_are_locked_and_private() -> None:
    """Keep the JavaScript release toolchain reproducible and non-publishable."""

    package = json.loads((PROJECT_ROOT / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads(
        (PROJECT_ROOT / "package-lock.json").read_text(encoding="utf-8")
    )

    assert package["private"] is True
    assert package["license"] == "AGPL-3.0-only"
    assert set(package["devDependencies"]) == RELEASE_DEPENDENCIES
    assert package_lock["packages"][""]["devDependencies"] == package["devDependencies"]
    for dependency, version in package["devDependencies"].items():
        assert (
            package_lock["packages"][f"node_modules/{dependency}"]["version"] == version
        )


def test_release_configuration_updates_all_metadata_and_commits_atomically() -> None:
    """Require the SimpleSyrup-style semantic-release prepare and Git stages."""

    config = (PROJECT_ROOT / ".releaserc.cjs").read_text(encoding="utf-8")

    assert 'branches: ["main"]' in config
    assert 'tagFormat: "v${version}"' in config
    assert (
        'prepareCmd: "node scripts/update-release-versions.mjs ${nextRelease.version}"'
        in config
    )
    for asset in (
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "whiterabbit/__init__.py",
        "CHANGELOG.md",
    ):
        assert f'"{asset}"' in config
    assert "chore(release): ${nextRelease.version} [skip ci]" in config


def test_release_workflow_verifies_gates_before_publishing() -> None:
    """Make release publication depend on complete Python quality gates."""

    workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "branches:\n      - main" in workflow
    assert "contents: write" in workflow
    assert "fetch-depth: 0" in workflow
    assert "fetch-tags: true" in workflow
    assert "npm ci" in workflow
    assert "ruff format --check ." in workflow
    assert "ruff check ." in workflow
    assert "mypy --strict __init__.py whiterabbit tests" in workflow
    assert "pytest -n auto -q" in workflow
    assert "bootstrap-release-baseline.mjs" in workflow
    assert "npx semantic-release" in workflow
    assert "git fetch origin main --tags" in workflow
    assert "git checkout main" in workflow
    assert "git reset --hard origin/main" in workflow
    assert "Publish Custom Node to Comfy Registry" in workflow


def test_baseline_bootstrap_validates_the_published_release_commit() -> None:
    """Prevent the first automated release from tagging an arbitrary commit."""

    bootstrap = (PROJECT_ROOT / "scripts" / "bootstrap-release-baseline.mjs").read_text(
        encoding="utf-8"
    )

    assert 'const BASELINE_TAG = "v1.1.1"' in bootstrap
    assert 'const BASELINE_VERSION = "1.1.1"' in bootstrap
    assert (
        'const BASELINE_COMMIT = "f82e9d7541bd439858cb076f275d483b1e7424bf"'
        in bootstrap
    )
    assert "verifyPublishedBaseline();" in bootstrap
    assert 'git(["push", "origin", BASELINE_TAG]);' in bootstrap


def test_release_scripts_parse_as_node_modules() -> None:
    """Fail fast when release automation contains invalid JavaScript syntax."""

    for script_name in (
        "bootstrap-release-baseline.mjs",
        "update-release-versions.mjs",
    ):
        result = subprocess.run(
            ["node", "--check", f"scripts/{script_name}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr


def test_version_updater_synchronizes_a_temporary_release_tree(tmp_path: Path) -> None:
    """Exercise the release updater without mutating this working tree."""

    scripts_directory = tmp_path / "scripts"
    scripts_directory.mkdir()
    shutil.copy(
        PROJECT_ROOT / "scripts" / "update-release-versions.mjs",
        scripts_directory / "update-release-versions.mjs",
    )
    (tmp_path / "package.json").write_text('{"version":"0.0.0"}\n', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text(
        '{"version":"0.0.0","packages":{"":{"version":"0.0.0"}}}\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "0.0.0"\n', encoding="utf-8"
    )
    package_directory = tmp_path / "whiterabbit"
    package_directory.mkdir()
    (package_directory / "__init__.py").write_text(
        '__version__ = "0.0.0"\n', encoding="utf-8"
    )

    result = subprocess.run(
        ["node", "scripts/update-release-versions.mjs", "1.2.3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    package = json.loads((tmp_path / "package.json").read_text("utf-8"))
    assert package["version"] == "1.2.3"
    assert (
        json.loads((tmp_path / "package-lock.json").read_text("utf-8"))["packages"][""][
            "version"
        ]
        == "1.2.3"
    )
    assert 'version = "1.2.3"' in (tmp_path / "pyproject.toml").read_text("utf-8")
    assert '__version__ = "1.2.3"' in (package_directory / "__init__.py").read_text(
        "utf-8"
    )
