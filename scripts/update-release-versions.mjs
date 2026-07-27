// WhiteRabbit - video frame manipulation nodes for ComfyUI
// Copyright (C) 2026 Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-only

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Update one JSON package version, including its lockfile root package.
 *
 * @param {URL} filePath Package metadata file.
 * @param {string} nextVersion Semantic-release version.
 */
export function writeJsonVersion(filePath, nextVersion) {
  const metadata = JSON.parse(readFileSync(filePath, "utf8"));
  if (typeof metadata.version !== "string") {
    throw new Error(`Could not find a top-level version field in ${filePath.pathname}.`);
  }
  metadata.version = nextVersion;

  if (metadata.packages !== undefined) {
    const rootPackage = metadata.packages[""];
    if (!rootPackage || typeof rootPackage.version !== "string") {
      throw new Error(`Could not find a root package version in ${filePath.pathname}.`);
    }
    rootPackage.version = nextVersion;
  }

  writeFileSync(filePath, `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
}

/**
 * Replace an expected version field in one structured text file.
 *
 * @param {URL} filePath File whose version field will be replaced.
 * @param {RegExp} pattern Version field matcher.
 * @param {string} replacement Replacement text.
 */
export function replaceVersionField(filePath, pattern, replacement) {
  const originalText = readFileSync(filePath, "utf8");

  if (!pattern.test(originalText)) {
    throw new Error(`Could not find a version field in ${filePath.pathname}.`);
  }

  writeFileSync(filePath, originalText.replace(pattern, replacement), "utf8");
}

/**
 * Synchronize every WhiteRabbit release-managed version field.
 *
 * @param {URL} projectRoot Repository root URL.
 * @param {string} nextVersion Semantic-release version.
 */
export function updateReleaseVersions(projectRoot, nextVersion) {
  writeJsonVersion(new URL("package.json", projectRoot), nextVersion);
  writeJsonVersion(new URL("package-lock.json", projectRoot), nextVersion);
  replaceVersionField(
    new URL("pyproject.toml", projectRoot),
    /^version = "[^"]+"\r?$/m,
    `version = "${nextVersion}"`,
  );
  replaceVersionField(
    new URL("whiterabbit/__init__.py", projectRoot),
    /^__version__ = "[^"]+"\r?$/m,
    `__version__ = "${nextVersion}"`,
  );
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";

if (invokedPath === fileURLToPath(import.meta.url)) {
  const nextVersion = process.argv[2];
  if (!nextVersion) {
    throw new Error("Expected the next release version as the first argument.");
  }
  updateReleaseVersions(new URL("../", import.meta.url), nextVersion);
}
