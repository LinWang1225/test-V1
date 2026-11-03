# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import os.path

from cuda.tile._ir.ir import Var, ArrayValue
from cuda.tile._ir.type import TileTy, ArrayTy, ClosureTy
from ._stub import Tile, Array
from . import TileValueError


class Symbol:
    def __init__(self, var: Var):
        self._var = var


class SymbolicTile(Symbol, Tile):
    def __init__(self, var: Var):
        Symbol.__init__(self, var)

    @property
    def dtype(self):
        ty = self._var.get_type()
        assert isinstance(ty, TileTy)
        return ty.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        ty = self._var.get_type()
        assert isinstance(ty, TileTy)
        return ty.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __bool__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to boolean")

    def __int__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to an integer")

    def __float__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to a float")

    def __index__(self):
        raise TileValueError("Symbolic tile has no concrete value and thus cannot be converted"
                             " to an integer")

    def __repr__(self):
        return f"<tile[{self.dtype}, {self.shape}]>"


class SymbolicArray(Symbol, Array):
    def __init__(self, var: Var):
        Symbol.__init__(self, var)

    @property
    def dtype(self):
        ty = self._var.get_type()
        assert isinstance(ty, ArrayTy)
        return ty.dtype

    @property
    def shape(self):
        agg = self._var.get_aggregate()
        assert isinstance(agg, ArrayValue)
        from cuda.tile._ir.ops import var2sym
        return tuple(var2sym(v) for v in agg.shape)

    @property
    def strides(self):
        agg = self._var.get_aggregate()
        assert isinstance(agg, ArrayValue)
        from cuda.tile._ir.ops import var2sym
        return tuple(var2sym(v) for v in agg.strides)

    @property
    def ndim(self) -> int:
        ty = self._var.get_type()
        assert isinstance(ty, ArrayTy)
        return ty.ndim

    def __repr__(self):
        ty = self._var.get_type()
        assert isinstance(ty, ArrayTy)

        shape_str = ", ".join("?" if s is None else str(s) for s in ty.shape)

        return f"<array[{ty.dtype}, ({shape_str})]>"


class SymbolicClosure(Symbol):
    def __repr__(self):
        ty = self._var.get_type()
        assert isinstance(ty, ClosureTy)
        desc = ty.func_hir.desc
        what = "lambda" if desc.name is None else f"function '{desc.name}'"
        filename = os.path.basename(desc.filename)
        return f"<{what} @{filename}:{desc.line}>"

    def __call__(self, *args, **kwargs):
        from cuda.tile._dispatch_mode import DispatchMode
        return DispatchMode().get_current().call_tile_function_from_host(self, args, kwargs)
