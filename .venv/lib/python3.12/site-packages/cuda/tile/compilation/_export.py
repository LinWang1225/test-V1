# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from typing import Sequence, IO, Literal

from cuda.tile._execution import kernel
from cuda.tile.compilation import KernelSignature


def export_kernel(kernel: kernel,
                  signatures: Sequence[KernelSignature],
                  output_file: IO | str | bytes | os.PathLike,
                  *,
                  gpu_code: str,
                  output_format: Literal["cubin", "tileir_bytecode"],
                  bytecode_version: str | None = None):
    """
    Compile and export a kernel.

    Args:
        kernel (cuda.tile.kernel):
            A kernel function to export.
        signatures (Sequence[cuda.tile.compilation.KernelSignature]):
            A non-empty list of signatures for which to compile the kernel.
        output_file (IO | str | bytes | os.PathLike):
            Either a filename or a binary file-like object to write the output to.
            To save the result in memory, you can pass an instance of the `io.BytesIO`
            standard library class.
        gpu_code (str):
            Name of the target GPU for which to compile the kernel (e.g., `"sm_100"`)
        output_format (str):
            Set to "cubin" to export a CUDA binary file, or "tileir_bytecode"
            to export a TileIR bytecode file.
        bytecode_version (str | None):
            Set to `None` to automatically detect the latest TileIR bytecode version supported
            by the compiler (default). Otherwise, it must be a string of the form "major.minor"
            that specifies the version of the TileIR bytecode to use (e.g., "13.1").
    """

    from cuda.tile._compile import compile_tile, parse_bytecode_version

    if bytecode_version is not None:
        bytecode_version = parse_bytecode_version(bytecode_version)

    if output_format == "cubin":
        return_bytecode = False
        return_cubin = True
    elif output_format == "tileir_bytecode":
        return_bytecode = True
        return_cubin = False
    else:
        raise ValueError(f"Unknown output format '{output_format}'")

    res = compile_tile(kernel._annotated_function, signatures,
                       sm_arch=gpu_code,
                       compiler_options=kernel._compiler_options,
                       return_bytecode=return_bytecode,
                       return_cubin=return_cubin,
                       bytecode_version=bytecode_version)

    with _open(output_file, "wb") as f:
        if return_bytecode:
            f.write(res.bytecode)
        else:
            f.write(res.cubin)


@contextmanager
def _open(name_or_file: IO | str | bytes | os.PathLike, mode: str):
    if isinstance(name_or_file, str | bytes):
        with open(name_or_file, mode) as f:
            yield f
    elif isinstance(name_or_file, os.PathLike):
        with open(name_or_file.__fspath__(), mode) as f:
            yield f
    else:
        yield name_or_file
