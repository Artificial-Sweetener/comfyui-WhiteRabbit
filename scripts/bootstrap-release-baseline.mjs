// WhiteRabbit - video frame manipulation nodes for ComfyUI
// Copyright (C) 2026 Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-only

import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const BASELINE_TAG = "v1.1.1";
const BASELINE_VERSION = "1.1.1";
const BASELINE_COMMIT = "f82e9d7541bd439858cb076f275d483b1e7424bf";
const projectRoot = resolve(fileURLToPath(new URL("../", import.meta.url)));
const dryRun = process.argv.slice(2).includes("--dry-run");

/**
 * Execute Git at the repository root and return its trimmed standard output.
 *
 * @param {string[]} argumentsList Git arguments.
 * @returns {string} Git command output.
 */
function git(argumentsList) {
  return execFileSync("git", argumentsList, {
    cwd: projectRoot,
    encoding: "utf8",
  }).trim();
}

/**
 * Confirm the historical release commit contains the published version.
 */
function verifyPublishedBaseline() {
  const pyproject = git(["show", `${BASELINE_COMMIT}:pyproject.toml`]);
  const versionMatch = pyproject.match(/^version = "([^"]+)"\r?$/m);
  if (versionMatch?.[1] !== BASELINE_VERSION) {
    throw new Error(
      `Baseline commit ${BASELINE_COMMIT} does not declare version ${BASELINE_VERSION}.`,
    );
  }
}

/**
 * Return the commit targeted by an existing annotated or lightweight tag.
 *
 * @returns {string | undefined} Tagged commit, if the baseline tag exists.
 */
function existingBaselineCommit() {
  const tags = git(["tag", "--list", BASELINE_TAG]);
  return tags ? git(["rev-list", "-n", "1", BASELINE_TAG]) : undefined;
}

verifyPublishedBaseline();
const taggedCommit = existingBaselineCommit();

if (taggedCommit && taggedCommit !== BASELINE_COMMIT) {
  throw new Error(
    `${BASELINE_TAG} points to ${taggedCommit}, not published baseline ${BASELINE_COMMIT}.`,
  );
}

if (!taggedCommit) {
  git([
    "tag",
    "--annotate",
    BASELINE_TAG,
    BASELINE_COMMIT,
    "--message",
    "chore(release): 1.1.1 baseline",
  ]);
  if (!dryRun) {
    git(["push", "origin", BASELINE_TAG]);
  }
}

const action = taggedCommit ? "verified" : dryRun ? "would publish" : "published";
process.stdout.write(`${action} ${BASELINE_TAG} at ${BASELINE_COMMIT}\n`);
