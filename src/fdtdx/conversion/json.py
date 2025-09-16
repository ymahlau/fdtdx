import importlib
import json
from typing import Any, Literal, Sequence

import jax

from fdtdx.core.jax.pytrees import TreeClass
from fdtdx.core.null import NULL
from fdtdx.typing import JAX_DTYPES


  
def _export_json(obj: Any) -> dict | float | int | str | bool | None:
    if type(obj).__name__ != type(obj).__qualname__:
        raise NotImplementedError()
    if obj is NULL:
        raise Exception("Object should not contain NULL")
    if obj is None:
        return None
    if isinstance(obj, float | int | str | bool):
        # basic data types
        return obj
    # jax data types
    if obj in JAX_DTYPES:
        str_name = str(obj).split("'")[1]
        return {
            "__dtype__": str_name,
        }
    # composite types need module and name 
    result_dict: dict = {
        '__module__': f"{type(obj).__module__}",
        '__name__': f"{type(obj).__name__}"
    }
    # tree classes
    if isinstance(obj, TreeClass):
        public_fields = obj.get_public_fields()
        for field in public_fields:
            name, val = field.name, field.value
            result_dict[name] = _export_json(val)
        return result_dict
    # dictionaries
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str): raise NotImplementedError()
            assert k not in ["__module__", "__name__"]
            result_dict[k] = _export_json(v)
        return result_dict
    # list / tuples / etc.
    if isinstance(obj, Sequence):
        val_list = [_export_json(v) for v in obj]
        result_dict["__value__"] = val_list
        return result_dict
    raise NotImplementedError()
        

def export_json(obj: Any) -> dict:
    """
    Create a dictionary from the given object for exporting to JSON. Can be used for serialization.

    Args:
        obj (Any): The object to serialize.

    Returns:
        dict: Dictionary representing the object
    """
    if isinstance(obj, float | int | str):
        raise Exception(f"Cannot convert python obj to json: {obj}")
    result = _export_json(obj)
    assert isinstance(result, dict)
    return result


def _json_dict_to_str(d: dict) -> str:
    return json.dumps(d, sort_keys=True, indent=4)


def export_json_str(obj: Any) -> str:
    """Create a json string from the given object. Can be used for serialization.

    Args:
        obj (Any): Object to serialize

    Returns:
        str: JSON string
    """
    d = export_json(obj)
    return _json_dict_to_str(d)


def _import_obj_from_json(obj: dict | float | int | str | bool | None) -> Any:
    if obj is None:
        return None
    if isinstance(obj, int | float | str | bool):
        return obj
    assert isinstance(obj, dict)
    # jax data types
    if "__dtype__" in obj:
        dtype_str = obj["__dtype__"]
        raw_dtype_str = dtype_str.split(".")[2]
        dtype = getattr(jax.numpy, raw_dtype_str)
        return dtype
    assert "__module__" in obj
    assert "__name__" in obj
    mod, name = obj["__module__"], obj["__name__"]
    module = importlib.import_module(mod)
    cls = getattr(module, name)
    # sequence
    if "__value__" in obj:
        vals = obj["__value__"]
        imported_vals = [_import_obj_from_json(v) for v in vals if v not in ["__module__", "__name__"]]
        return cls(imported_vals)
    # dictionary
    if "__name__" == "dict":
        return {
            k: _import_obj_from_json(v)
            for k, v in obj.items() if k not in ["__module__", "__name__"]
        }
    # other classes
    kwargs_dict = {
        k: _import_obj_from_json(v)
        for k, v in obj.items() if k not in ["__module__", "__name__"]
    }
    return cls(**kwargs_dict)


def import_from_json(json_str: str) -> Any:
    d = json.loads(json_str)
    assert isinstance(d, dict)
    return _import_obj_from_json(d)
