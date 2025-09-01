from fastcore.foundation import copy_func
import jax

from fdtdx.units.unitful import (
    multiply,
    divide,
    add,
    subtract,
    remainder,
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
    abs,
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
        (remainder, None),
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
        (abs, None),
        (abs, "absolute"),
        (cross, None),
        (conj, None),
        (conj, "conjugate"),
        (dot, None),
        (transpose, None),
        (pad, None),
        (stack, None),
        (isfinite, None),
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