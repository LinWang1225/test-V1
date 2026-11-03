# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import dataclass
from types import FunctionType
from typing import (get_origin, get_args, Annotated, Any, Sequence)

from cuda.tile._stub import ConstantAnnotation


@dataclass
class AnnotatedFunction:
    pyfunc: FunctionType
    pysig: inspect.Signature
    constant_parameter_mask: Sequence[bool]


def get_annotated_function(pyfunc: FunctionType) -> AnnotatedFunction:
    sig = inspect.signature(pyfunc)
    constant_parameter_mask = tuple(_has_constant_annotation(param.annotation)
                                    for param in sig.parameters.values())
    return AnnotatedFunction(pyfunc=pyfunc,
                             pysig=sig,
                             constant_parameter_mask=constant_parameter_mask)


def _has_constant_annotation(annotation: Any) -> bool:
    if get_origin(annotation) is Annotated:
        _, *metadata = get_args(annotation)
        return any(isinstance(m, ConstantAnnotation) for m in metadata)
    return False
