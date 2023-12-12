# SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Copyright 2022-2024 NXP
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
#
# Description:
# Packaging for the Vela compiler
import os
import re

from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext

vela_version = "3.10.0"

class BuildExtension(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        import builtins

        # tell numpy it's not in setup anymore
        builtins.__NUMPY_SETUP__ = False
        import numpy as np

        # add the numpy headers to the mlw_codec extension
        self.include_dirs.append(np.get_include())


# Read the contents of README.md file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    tag = vela_version
    url = f"https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/tags/{tag}/"
    # Find all markdown links that match the format:  [text](link)
    for match, link in re.findall(r"(\[.+?\]\((.+?)\))", long_description):
        # If the link is a file that exists, replace it with the web link to the file instead
        if os.path.exists(os.path.join(this_directory, link)):
            url_link = match.replace(link, url + link)
            long_description = long_description.replace(match, url_link)
    if os.getenv("ETHOSU_VELA_DEBUG"):
        # Verify the contents of the modifications made in a markdown renderer
        with open(os.path.join(this_directory, "PYPI.md"), "wt", encoding="utf-8") as fout:
            fout.write(long_description)

mlw_module = Extension(
    "ethosu.mlw_codec",
    ["ethosu/mlw_codec/mlw_encode.c", "ethosu/mlw_codec/mlw_decode.c", "ethosu/mlw_codec/mlw_codecmodule.c"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")],
)

setup(
    use_scm_version=False,
    version=vela_version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[mlw_module],
    cmdclass={"build_ext": BuildExtension},  # type: ignore[dict-item]
)
