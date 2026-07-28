// WhiteRabbit - video frame manipulation nodes for ComfyUI
// Copyright (C) 2026 Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-only

module.exports = {
  branches: ["main"],
  tagFormat: "v${version}",
  plugins: [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    ["@semantic-release/changelog", { changelogFile: "CHANGELOG.md" }],
    [
      "@semantic-release/exec",
      {
        prepareCmd: "node scripts/update-release-versions.mjs ${nextRelease.version}",
      },
    ],
    [
      "@semantic-release/git",
      {
        assets: [
          "package.json",
          "package-lock.json",
          "pyproject.toml",
          "whiterabbit/__init__.py",
          "CHANGELOG.md",
        ],
        message:
          "chore(release): ${nextRelease.version} [skip ci]\\n\\n${nextRelease.notes}",
      },
    ],
  ],
};
