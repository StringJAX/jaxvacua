# Copyright 2026 Andreas Schachner
#
# This file is part of JAXVacua.
#
# JAXVacua is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# JAXVacua is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with JAXVacua. If not, see <https://www.gnu.org/licenses/>.

r"""Packaging consistency of JAXVacua.

Purpose
-------
The version is stated twice, in ``setup.py`` (what pip installs and PyPI shows) and in
``jaxvacua.__version__`` (what the documentation and downstream provenance records read). The
0.1.3 preparation bumped only the first; this test keeps them equal.
"""

import re
from pathlib import Path

import jaxvacua


def test_version_matches_setup():
    setup = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    m = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", setup)
    assert m is not None
    assert jaxvacua.__version__ == m.group(1)
