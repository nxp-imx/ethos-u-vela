# Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
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
# Description:
# Packaging for the Vela compiler
import os
import re

from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext


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
    tag = "3.3.0"
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
    name="ethos-u-vela",
    use_scm_version=True,
    description="Neural network model compiler for Arm Ethos-U NPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.mlplatform.org/ml/ethos-u/ethos-u-vela.git/",
    author="Arm Ltd.",
    author_email="mlg-vela@arm.com",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
    keywords=["ethos-u", "vela compiler", "tflite", "npu"],
    packages=[
        "ethosu.vela",
        "ethosu.vela.ethos_u55_regs",
        "ethosu.vela.tflite",
        "ethosu.vela.tosa",
        "ethosu.mlw_codec",
    ],
    python_requires="~=3.8",
    install_requires=[
        "flatbuffers==1.12.0",
        "numpy>=1.16.6,<=1.19.5",
        "numpy>=1.16.6,<1.19.4 ; platform_system=='Windows'",
        "lxml>=4.5.1",
    ],
    entry_points={"console_scripts": ["vela = ethosu.vela.vela:main"]},
    ext_modules=[mlw_module],
    cmdclass={"build_ext": BuildExtension},
    setup_requires=["numpy>=1.16.6,<=1.19.5", "setuptools_scm"],
)
