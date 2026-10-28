# SPDX-FileCopyrightText: 2023-present zeyus <zeyus@zeyus.com>
#
# SPDX-License-Identifier: MIT
from importlib.metadata import version

# single source of truth is `version` in pyproject.toml (bump with `uv version --bump minor`)
__version__ = version("ragga")
