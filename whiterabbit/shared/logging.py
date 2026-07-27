# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Consistent logger creation for WhiteRabbit modules."""

from __future__ import annotations

import logging

LOGGER_NAME = "whiterabbit"


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger within WhiteRabbit's logging namespace."""

    suffix = module_name.rsplit(".", maxsplit=1)[-1]
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}")
