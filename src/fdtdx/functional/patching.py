from fastcore.foundation import copy_func
import jax

from fdtdx.units.unitful import (
    multiply,
    divide,
    add,
    subtract,
    lt,
    le,
    eq,
    ne,
    ge,
    gt,
    matmul,
    pow,
    min,
    max,
    mean,
    sum,
    abs_impl,
    astype,
    squeeze,
    reshape,
    prod,
)

from fdtdx.functional.numpy import (
    sqrt,
    roll,
    square,
    cross,
    conj,
    dot,
    transpose,
    pad,
    stack,
    isfinite,
    roll,
    real,
    imag,
    sin,
    cos,
    tan,
    asarray,
    array,
    exp,
    expand_dims,
    where,
    arange,
    meshgrid,
    floor,
    ceil,
)
from fdtdx.functional.linalg import (
    norm,
)

from fdtdx.functional.jax import (
    jit
)

def patch_fn_to_module(
    f,
    mod,
    fn_name: str | None = None,
) -> None:
    fn_copy = copy_func(f)
    if fn_name is None:
        fn_name = f.__name__
    assert fn_name is not None
    fn_copy.__qualname__ = f"{mod.__name__}.{fn_name}"
    original_name = '_orig_' + fn_name
    if hasattr(mod, fn_name) and not hasattr(mod, original_name):
        setattr(mod, original_name, getattr(mod, fn_name))
    setattr(mod, fn_name, fn_copy)


def patch_all_functions():
    ## add to original jax.numpy ###################
    _full_patch_list_numpy = [
        (multiply, None),
        (divide, None),
        (divide, "true_divide"),
        (add, None),
        (subtract, None),
        (lt, "less"),
        (le, "less_equal"),
        (eq, "equal"),
        (ne, "not_equal"),
        (ge, "greater_equal"),
        (gt, "greater"),
        (matmul, None),
        (pow, None),
        (sqrt, None),
        (min, None),
        (max, None),
        (mean, None),
        (sum, None),
        (roll, None),
        (square, None),
        (abs_impl, "abs"),
        (abs_impl, "absolute"),
        (cross, None),
        (conj, None),
        (conj, "conjugate"),
        (dot, None),
        (transpose, None),
        (pad, None),
        (stack, None),
        (isfinite, None),
        (roll, None),
        (real, None),
        (imag, None),
        (astype, None),
        (squeeze, None),
        (sin, None),
        (cos, None),
        (tan, None),
        (asarray, None),
        (array, None),
        (exp, None),
        (expand_dims, None),
        (where, None),
        (reshape, None),
        (arange, None),
        (meshgrid, None),
        (floor, None),
        (ceil, None),
        (prod, None),
    ]
    for fn, orig in _full_patch_list_numpy:
        patch_fn_to_module(
            f=fn, 
            mod=jax.numpy,
            fn_name=orig,
        )

    ## add to jax.lax ###################
    _full_patch_list_lax = [
        (lt, None),
        (le, None),
        (eq, None),
        (ne, None),
        (ge, None),
        (gt, None),
        (pow, None),
        (sqrt, None),
    ]
    for fn, orig in _full_patch_list_lax:
        patch_fn_to_module(
            f=fn, 
            mod=jax.lax,
            fn_name=orig,
        )
        
    ## add to jax.mumpy.linalg ###################
    _full_patch_list_linalg = [
        (norm, None)
    ]
    for fn, orig in _full_patch_list_linalg:
        patch_fn_to_module(
            f=fn,
            mod=jax.numpy.linalg,
            fn_name=orig,
        )

    ## add to jax ###################
    _full_patch_list_jax = [
        (jit, None),
    ]
    for fn, orig in _full_patch_list_jax:
        patch_fn_to_module(
            f=fn,
            mod=jax,
            fn_name=orig,
        )