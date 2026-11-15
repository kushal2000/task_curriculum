from __future__ import annotations

from pathlib import Path
import sys

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    c_opts = {
        "darwin": ["-std=c++17", "-O3"],
        "unix": ["-std=c++17", "-O3"],
    }

    l_opts = {
        "darwin": ["-undefined", "dynamic_lookup"],
        "unix": ["-Wl,--allow-shlib-undefined"],
    }

    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        opts = list(self.c_opts.get(compiler_type, []))
        link_opts = list(self.l_opts.get(compiler_type, []))

        if sys.platform == "darwin":
            opts = list(self.c_opts["darwin"])
            link_opts = list(self.l_opts["darwin"])

        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)

        super().build_extensions()


ext_modules = [
    Extension(
        "aioflexiv._aioflexiv_rt",
        ["src/aioflexiv/cpp/aioflexiv_rt.cpp"],
        include_dirs=["vendor/flexiv_rdk/include", "vendor/eigen"],
        language="c++",
    )
]


def model_data_files():
    model_root = Path("models")
    if not model_root.exists():
        return []
    files_by_dir: dict[Path, list[str]] = {}
    for path in sorted(model_root.rglob("*")):
        if path.is_file():
            target_dir = Path("share/aioflexiv") / path.parent
            files_by_dir.setdefault(target_dir, []).append(str(path))
    return [
        (str(target_dir), files)
        for target_dir, files in sorted(files_by_dir.items())
    ]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    data_files=model_data_files(),
)
