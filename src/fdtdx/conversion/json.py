import dataclasses
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import TreeClass
from fdtdx.core.null import NULL
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.typing import JAX_DTYPES


@dataclass(frozen=True)
class JsonSetup:
    """
    Serializable payload for `fdtdx.place_objects`.

    Encapsulates:
      - config
      - object_list
      - constraints
    """

    config: SimulationConfig
    object_list: List[SimulationObject]
    constraints: List[PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint]
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def load_json(cls, path: str | Path) -> "JsonSetup":
        """Load setup from JSON file."""
        s = Path(path).read_text()
        return cls.loads(s)

    @classmethod
    def loads(cls, json_str: str) -> "JsonSetup":
        """Load setup from JSON string produced by `export_json_str`."""
        data = import_from_json(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonSetup":
        """Construct setup from decoded dictionary."""
        config = data.get("config", data.get("SimulationConfig", None))
        if config is None:
            raise ValueError("Missing 'config' or 'SimulationConfig'.")

        object_list = data.get("object_list", None)
        if object_list is None:
            raise ValueError("Missing 'object_list'.")

        constraints = data.get("constraints", [])
        if constraints is None:
            raise ValueError("Missing 'constraints'.")

        meta = data.get("meta", None)

        object_list = cls._unwrap_list(object_list, "object_list")
        constraints = cls._unwrap_list(constraints, "constraints")

        settings = cls(config=config, object_list=object_list, constraints=constraints, meta=meta)
        settings.validate()
        return settings

    def dumps(self) -> str:
        """
        Serialize setup to JSON string.

        Formatting depends on `export_json_str` implementation.
        """
        export_format = {
            "__schema__": "fdtdx.place_objects.v1",
            "__fdtdx_version__": "0.6.0",
            "config": self.config,
            "object_list": self.object_list,
            "constraints": self.constraints,
        }
        if self.meta:
            export_format["meta"] = self.meta
        return export_json_str(export_format)

    def export_json(self, path: str | Path) -> None:
        """Write setup to JSON file."""
        Path(path).write_text(self.dumps())

    def validate(self) -> None:
        """
        Perform structural validation before calling `place_objects`.
        """
        errors: List[str] = []

        if not isinstance(self.object_list, list):
            errors.append(f"'object_list' must be a list, got {type(self.object_list)}")
        if not isinstance(self.constraints, list):
            errors.append(f"'constraints' must be a list, got {type(self.constraints)}")

        valid_object_names = {
            # config-related objects / scene objects
            "SimulationVolume",
            "UniformMaterialObject",
            # sources
            "ModePlaneSource",
            "GaussianPlaneSource",
            "UniformPlaneSource",
            "LinearlyPolarizedPlaneSource",
            # detectors
            "EnergyDetector",
            "FieldDetector",
            "ModeOverlapDetector",
            "PhasorDetector",
            "PoyntingFluxDetector",
            # boundary
            "PerfectlyMatchedLayer",
            # misc used inside objects (may appear as standalone nodes in JSON)
            "OnOffSwitch",
            "SingleFrequencyProfile",
            "GaussianPulseProfile",
            "WaveCharacter",
        }

        valid_constraint_names = {
            # constraints
            "PositionConstraint",
            "SizeConstraint",
            "SizeExtensionConstraint",
            "GridCoordinateConstraint",
        }

        for i, o in enumerate(self.object_list):
            name = type(o).__name__
            if type(o).__module__.startswith("fdtdx.materials"):
                continue

            if name not in valid_object_names:
                errors.append(f"object_list[{i}] has unsupported type '{name}'")

        for i, c in enumerate(self.constraints):
            name = type(c).__name__
            if name not in valid_constraint_names:
                errors.append(f"constraints[{i}] has unsupported type '{name}'")

        # must contain a SimulationVolume
        vols = [o for o in self.object_list if type(o).__name__ == "SimulationVolume"]
        if len(vols) == 0:
            errors.append("No SimulationVolume found in object_list.")
        elif len(vols) > 1:
            errors.append(f"Multiple SimulationVolume found ({len(vols)}). Expected exactly one.")

        # names should be unique
        names = []
        for o in self.object_list:
            n = getattr(o, "name", None)
            if isinstance(n, str) and n:
                names.append(n)
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            errors.append(f"Duplicate object names: {dup}")
        name_set = set(names)

        # constraints often have .object / .other_object as strings
        for i, c in enumerate(self.constraints):
            for field in ("object", "other_object"):
                if hasattr(c, field):
                    v = getattr(c, field)
                    if isinstance(v, str) and v not in name_set:
                        errors.append(f"Constraint[{i}] references unknown {field}='{v}'")

        if errors:
            raise ValueError("Invalid place_objects payload:\n- " + "\n- ".join(errors))

    @staticmethod
    def _unwrap_list(x: Any, name: str) -> List[Any]:
        """Ensure value is a plain Python list."""
        if isinstance(x, list):
            return x
        if isinstance(x, dict) and x.get("__name__") == "list" and "__value__" in x:
            v = x["__value__"]
            if isinstance(v, list):
                return v
        raise ValueError(f"'{name}' must be a list (or list wrapper), got {type(x)}")


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
    # dataclass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type) and hasattr(obj, "__dict__"):
        return {
            "__module__": type(obj).__module__,
            "__name__": type(obj).__name__,
            "__value__": {k: _export_json(v) for k, v in obj.__dict__.items() if not k.startswith("_")},
        }
    result_dict: dict = {"__module__": f"{type(obj).__module__}", "__name__": f"{type(obj).__name__}"}
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
            if not isinstance(k, str):
                raise NotImplementedError()
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
        # dataclass
        if isinstance(vals, dict):
            kwargs = {k: _import_obj_from_json(v) for k, v in vals.items()}
            return cls(**kwargs)
        imported_vals = [_import_obj_from_json(v) for v in vals]
        return cls(imported_vals)
    # dictionary
    if name == "dict":
        return {k: _import_obj_from_json(v) for k, v in obj.items() if k not in ["__module__", "__name__"]}
    # other classes
    kwargs_dict = {k: _import_obj_from_json(v) for k, v in obj.items() if k not in ["__module__", "__name__"]}
    return cls(**kwargs_dict)


def import_from_json(json_str: str) -> Any:
    d = json.loads(json_str)
    assert isinstance(d, dict)
    return _import_obj_from_json(d)
