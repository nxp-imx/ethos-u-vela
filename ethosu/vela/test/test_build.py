# SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import tarfile
from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory

import build  # noreorder
import pytest


@pytest.fixture(scope="module", name="vela_path")
def fixture_vela_path():
    """Return the path to the source code."""
    current_dir = Path(__file__).parent
    relative_vela_path = Path.joinpath(current_dir, *([".."] * 3))
    vela_path = Path(relative_vela_path).absolute().resolve()
    return vela_path


@pytest.fixture(scope="module", name="built_sdist")
def fixture_built_sdist(vela_path: Path) -> TarFile:
    """Fixture that returns the TarFile with the project sdist."""
    # Set pretend version
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = "3.7.0"
    # Build vela
    with TemporaryDirectory() as tmp_dir:
        p_builder = build.ProjectBuilder(vela_path)
        build_path = p_builder.build("sdist", tmp_dir)
        archive_path = Path(build_path)
        tar_archive = tarfile.open(archive_path, "r:gz")
    return tar_archive


@pytest.fixture(scope="module", name="source_readme")
def fixture_source_readme(vela_path: Path) -> str:
    """Return the contents of the README.md file."""
    with open(Path.joinpath(vela_path, "README.md"), encoding="utf-8") as readme:
        readme_content = readme.read()

    return readme_content


def test_build_correct_readme_links(built_sdist: TarFile, source_readme: str):
    """Test that PKG-INFO file contains README.md metadata with correct links."""
    md_link_pattern = r"(!?\[.+?\]\((.+?)\))"
    url = "https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/tags/3.7.0/"
    # Extract the name of the package
    dist_tar_name = str(built_sdist.name).replace(".gz", "")
    package_name = Path(dist_tar_name).stem
    file = built_sdist.extractfile(f"{package_name}/PKG-INFO")
    pkg_info = file.read().decode("utf-8")
    # Ensure that all links were replaced correctly
    for match, link in re.findall(md_link_pattern, source_readme):
        if "http" not in link and "https" not in link:
            assert match not in pkg_info
            assert url + link in pkg_info
