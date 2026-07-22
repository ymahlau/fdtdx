import math
import warnings
from typing import cast

import numpy as np

from fdtdx import constants
from fdtdx.core.jax.pytrees import TreeClass, frozen_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.dispersion import DispersionModel, compute_pole_coefficients_tensor


def _normalize_material_property(
    value: float
    | tuple[float, float, float]
    | tuple[float, float, float, float, float, float, float, float, float]
    | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Normalize material property to a 9-tuple for (xx, xy, xz, yx, yy, yz, zx, zy, zz) components.

    Args:
        value: Either a scalar (isotropic), 3-tuple (diagonally anisotropic), 9-tuple (fully anisotropic),
               or nested 3x3 tuple ((xx, xy, xz), (yx, yy, yz), (zx, zy, zz)) material property

    Returns:
        tuple[float, float, float, float, float, float, float, float, float]: Material property as (xx, xy, xz, yx, yy, yz, zx, zy, zz) components
    """
    if isinstance(value, tuple):
        if len(value) == 3:
            # Check if it's a nested tuple (3x3 matrix format)
            if isinstance(value[0], tuple) and isinstance(value[1], tuple) and isinstance(value[2], tuple):
                # Nested tuple: ((xx, xy, xz), (yx, yy, yz), (zx, zy, zz))
                if len(value[0]) != 3 or len(value[1]) != 3 or len(value[2]) != 3:
                    raise ValueError(
                        f"Nested tuple must have 3 elements in each row, got ({len(value[0])}, {len(value[1])}, {len(value[2])})"
                    )
                # Flatten the nested tuple to row-major order
                return (
                    value[0][0],
                    value[0][1],
                    value[0][2],
                    value[1][0],
                    value[1][1],
                    value[1][2],
                    value[2][0],
                    value[2][1],
                    value[2][2],
                )
            elif isinstance(value[0], float) and isinstance(value[1], float) and isinstance(value[2], float):
                # Diagonally anisotropic: 3-tuple (x, y, z)
                return (value[0], 0.0, 0.0, 0.0, value[1], 0.0, 0.0, 0.0, value[2])
            else:
                raise ValueError(f"Invalid material property tuple: {value}. Expected a tuple of 3 tuples or 3 floats.")
        elif len(value) == 9:
            return value
        else:
            raise ValueError(f"Material property tuple must have exactly 3 or 9 elements, got {len(value)}")
    else:
        # Isotropic: broadcast scalar to all three components
        return (value, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0, value)


class Material(TreeClass):
    """
    Represents an electromagnetic material with specific electrical and magnetic properties.

    This class stores the fundamental electromagnetic properties of a material for use
    in electromagnetic simulations. Supports both isotropic and anisotropic materials.

    Note:
        All material properties are stored internally as 9-tuples (xx, xy, xz, yx, yy, yz, zx, zy, zz components).
        Scalar inputs are automatically broadcast to all diagonal components.

    """

    #: The relative permittivity (dielectric constant) of the material, which describes how the electric field is affected by the material.
    #: Higher values indicate greater electric polarization in response to an applied electric field.
    #: For isotropic materials, provide a scalar float.
    #: For diagonally anisotropic materials, provide a tuple of 3 floats (εx, εy, εz).
    #: For fully anisotropic materials, provide either:
    #:
    #:   - A tuple of 9 floats (εxx, εxy, εxz, εyx, εyy, εyz, εzx, εzy, εzz), or
    #:   - A nested tuple ((εxx, εxy, εxz), (εyx, εyy, εyz), (εzx, εzy, εzz))
    #:
    #: Stored internally as a 9-tuple. Defaults to (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).
    permittivity: tuple[float, float, float, float, float, float, float, float, float] = frozen_field(
        default=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        on_setattr=[_normalize_material_property],
    )

    #: The relative permeability of the material, which describes how the magnetic field is affected by the material.
    #: Higher values indicate greater magnetic response to an applied magnetic field.
    #: For isotropic materials, provide a scalar float.
    #: For diagonally anisotropic materials, provide a tuple of 3 floats (μx, μy, μz).
    #: For fully anisotropic materials, provide either:
    #:
    #:   - A tuple of 9 floats (μxx, μxy, μxz, μyx, μyy, μyz, μzx, μzy, μzz), or
    #:   - A nested tuple ((μxx, μxy, μxz), (μyx, μyy, μyz), (μzx, μzy, μzz))
    #:
    #: Stored internally as a 9-tuple. Defaults to (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).
    permeability: tuple[float, float, float, float, float, float, float, float, float] = frozen_field(
        default=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        on_setattr=[_normalize_material_property],
    )

    #: The electrical conductivity of the material in siemens per meter (S/m), which describes how easily electric current can flow through it.
    #: Higher values indicate materials that conduct electricity more easily.
    #: For isotropic materials, provide a scalar float.
    #: For diagonally anisotropic materials, provide a tuple of 3 floats (σx, σy, σz).
    #: For fully anisotropic materials, provide either:
    #:
    #:   - A tuple of 9 floats (σxx, σxy, σxz, σyx, σyy, σyz, σzx, σzy, σzz), or
    #:   - A nested tuple ((σxx, σxy, σxz), (σyx, σyy, σyz), (σzx, σzy, σzz))
    #:
    #: Stored internally as a 9-tuple. Defaults to (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).
    electric_conductivity: tuple[float, float, float, float, float, float, float, float, float] = frozen_field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        on_setattr=[_normalize_material_property],
    )

    #: The magnetic conductivity, or magnetic loss of the material.
    #: This is an artificial parameter for numerical applications and does not represent an actual physical unit,
    #: even though often described in Ohm/m. The naming can be misleading, because it does not actually describe
    #: a conductivity, but rather an "equivalent magnetic loss parameter".
    #: For isotropic materials, provide a scalar float.
    #: For diagonally anisotropic materials, provide a tuple of 3 floats (σx, σy, σz).
    #: For fully anisotropic materials, provide either:
    #:
    #:   - A tuple of 9 floats (σxx, σxy, σxz, σyx, σyy, σyz, σzx, σzy, σzz), or
    #:   - A nested tuple ((σxx, σxy, σxz), (σyx, σyy, σyz), (σzx, σzy, σzz))
    #:
    #: Stored internally as a 9-tuple. Defaults to (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).
    magnetic_conductivity: tuple[float, float, float, float, float, float, float, float, float] = frozen_field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        on_setattr=[_normalize_material_property],
    )

    #: Optional dispersion model. When set, :attr:`permittivity` represents the
    #: high-frequency permittivity :math:`\varepsilon_\infty`; the full
    #: :math:`\varepsilon(\omega)` is :math:`\varepsilon_\infty + \chi(\omega)`
    #: from the dispersion model. Defaults to ``None`` (non-dispersive).
    dispersion: DispersionModel | None = frozen_field(default=None)

    def __init__(
        self,
        *,
        permittivity: float
        | tuple[float, float, float]
        | tuple[float, float, float, float, float, float, float, float, float]
        | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        permeability: float
        | tuple[float, float, float]
        | tuple[float, float, float, float, float, float, float, float, float]
        | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        electric_conductivity: float
        | tuple[float, float, float]
        | tuple[float, float, float, float, float, float, float, float, float]
        | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        magnetic_conductivity: float
        | tuple[float, float, float]
        | tuple[float, float, float, float, float, float, float, float, float]
        | tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        dispersion: DispersionModel | None = None,
    ) -> None:
        # Normalize inputs and set attributes using object.__setattr__ for frozen fields
        object.__setattr__(self, "permittivity", _normalize_material_property(permittivity))
        object.__setattr__(self, "permeability", _normalize_material_property(permeability))
        object.__setattr__(self, "electric_conductivity", _normalize_material_property(electric_conductivity))
        object.__setattr__(self, "magnetic_conductivity", _normalize_material_property(magnetic_conductivity))
        object.__setattr__(self, "dispersion", dispersion)
        eps = self.permittivity
        diag = (eps[0], eps[4], eps[8])
        if any(isinstance(v, (int, float)) and v <= 0.0 for v in diag):
            warnings.warn(
                "Material has a non-positive static permittivity on the diagonal, which is "
                "unconditionally unstable in explicit FDTD. To model Re(eps) < 0 in-band (metals, "
                "hyperbolic media), keep eps_inf >= 1 and use a DispersionModel (e.g. a DrudePole).",
                UserWarning,
                stacklevel=2,
            )

    @property
    def is_all_isotropic(self) -> bool:
        """Check if all material properties are isotropic (all components equal and no off-diagonal components).

        Returns:
            bool: True if material is isotropic, False if anisotropic
        """

        return (
            _is_property_isotropic(self.permittivity)
            and _is_property_isotropic(self.permeability)
            and _is_property_isotropic(self.electric_conductivity)
            and _is_property_isotropic(self.magnetic_conductivity)
        )

    @property
    def is_all_diagonally_anisotropic(self) -> bool:
        """Check if all material properties are diagonally anisotropic (no off-diagonal components).

        Returns:
            bool: True if material is diagonally anisotropic
        """
        return (
            _is_property_diagonally_anisotropic(self.permittivity)
            and _is_property_diagonally_anisotropic(self.permeability)
            and _is_property_diagonally_anisotropic(self.electric_conductivity)
            and _is_property_diagonally_anisotropic(self.magnetic_conductivity)
        )

    @property
    def is_isotropic_permittivity(self) -> bool:
        """Check if material has isotropic permittivity (all components equal and no off-diagonal components).

        Returns:
            bool: True if material has isotropic permittivity
        """
        return _is_property_isotropic(self.permittivity)

    @property
    def is_diagonally_anisotropic_permittivity(self) -> bool:
        """Check if material has diagonally anisotropic permittivity (no off-diagonal components).

        Returns:
            bool: True if material has diagonally anisotropic permittivity
        """
        return _is_property_diagonally_anisotropic(self.permittivity)

    @property
    def is_isotropic_permeability(self) -> bool:
        """Check if material has isotropic permeability (all components equal and no off-diagonal components).

        Returns:
            bool: True if material has isotropic permeability
        """
        return _is_property_isotropic(self.permeability)

    @property
    def is_diagonally_anisotropic_permeability(self) -> bool:
        """Check if material has diagonally anisotropic permeability (no off-diagonal components).

        Returns:
            bool: True if material has diagonally anisotropic permeability
        """
        return _is_property_diagonally_anisotropic(self.permeability)

    @property
    def is_isotropic_electric_conductivity(self) -> bool:
        """Check if material has isotropic electric conductivity (all components equal and no off-diagonal components).

        Returns:
            bool: True if material has isotropic electric conductivity
        """
        return _is_property_isotropic(self.electric_conductivity)

    @property
    def is_diagonally_anisotropic_electric_conductivity(self) -> bool:
        """Check if material has diagonally anisotropic electric conductivity (no off-diagonal components).

        Returns:
            bool: True if material has diagonally anisotropic electric conductivity
        """
        return _is_property_diagonally_anisotropic(self.electric_conductivity)

    @property
    def is_isotropic_magnetic_conductivity(self) -> bool:
        """Check if material has isotropic magnetic conductivity (all components equal and no off-diagonal components).

        Returns:
            bool: True if material has isotropic magnetic conductivity
        """
        return _is_property_isotropic(self.magnetic_conductivity)

    @property
    def is_diagonally_anisotropic_magnetic_conductivity(self) -> bool:
        """Check if material has diagonally anisotropic magnetic conductivity (no off-diagonal components).

        Returns:
            bool: True if material has diagonally anisotropic magnetic conductivity
        """
        return _is_property_diagonally_anisotropic(self.magnetic_conductivity)

    @property
    def is_magnetic(self) -> bool:
        """Check if material has magnetic properties (permeability != 1.0 for any component).

        Returns:
            bool: True if material is magnetic
        """
        perm = self.permeability
        if isinstance(perm[0], complex) or isinstance(perm[1], complex) or isinstance(perm[2], complex):
            return True
        return not (
            math.isclose(perm[0], 1.0)
            and math.isclose(perm[1], 0.0)
            and math.isclose(perm[2], 0.0)
            and math.isclose(perm[3], 0.0)
            and math.isclose(perm[4], 1.0)
            and math.isclose(perm[5], 0.0)
            and math.isclose(perm[6], 0.0)
            and math.isclose(perm[7], 0.0)
            and math.isclose(perm[8], 1.0)
        )

    @property
    def is_electrically_conductive(self) -> bool:
        """Check if material is electrically conductive (conductivity != 0.0 for any component).

        Returns:
            bool: True if material is electrically conductive
        """
        cond = self.electric_conductivity
        return not (
            math.isclose(cond[0], 0.0)
            and math.isclose(cond[1], 0.0)
            and math.isclose(cond[2], 0.0)
            and math.isclose(cond[3], 0.0)
            and math.isclose(cond[4], 0.0)
            and math.isclose(cond[5], 0.0)
            and math.isclose(cond[6], 0.0)
            and math.isclose(cond[7], 0.0)
            and math.isclose(cond[8], 0.0)
        )

    @property
    def is_magnetically_conductive(self) -> bool:
        """Check if material has magnetic conductivity (magnetic loss != 0.0 for any component).

        Returns:
            bool: True if material has magnetic conductivity
        """
        cond = self.magnetic_conductivity
        return not (
            math.isclose(cond[0], 0.0)
            and math.isclose(cond[1], 0.0)
            and math.isclose(cond[2], 0.0)
            and math.isclose(cond[3], 0.0)
            and math.isclose(cond[4], 0.0)
            and math.isclose(cond[5], 0.0)
            and math.isclose(cond[6], 0.0)
            and math.isclose(cond[7], 0.0)
            and math.isclose(cond[8], 0.0)
        )

    @property
    def is_dispersive(self) -> bool:
        """Check if the material has a non-trivial dispersion model.

        Returns:
            bool: True if a :class:`DispersionModel` with at least one pole is attached.
        """
        return self.dispersion is not None and self.dispersion.num_poles > 0

    @property
    def has_isotropic_dispersion(self) -> bool:
        """Check whether the material's dispersion (if any) is isotropic.

        Returns:
            bool: True if the material is non-dispersive or every pole of its
            dispersion model applies the same parameters to all three axes.
        """
        return self.dispersion is None or self.dispersion.is_isotropic

    @property
    def has_axis_aligned_dispersion(self) -> bool:
        """Check whether the material's dispersion (if any) is free of oriented poles.

        Returns:
            bool: True if the material is non-dispersive or no pole of its
            dispersion model carries an orientation (i.e. the susceptibility
            tensor is diagonal in the grid frame).
        """
        return self.dispersion is None or not self.dispersion.has_off_diagonal_coupling

    @classmethod
    def from_complex_permittivity(
        cls,
        permittivity: complex | float | tuple,
        *,
        reference: WaveCharacter | None = None,
        wavelength: float | None = None,
        frequency: float | None = None,
        permeability: complex | float | tuple = 1.0,
    ) -> "Material":
        r"""Build a non-dispersive lossy material from a complex permittivity.

        The imaginary part of the permittivity is converted to an equivalent
        electric conductivity :math:`\sigma = \omega_0 \varepsilon_0 \varepsilon''`
        at the reference angular frequency :math:`\omega_0 = 2\pi f_0`. Likewise,
        a complex ``permeability`` maps its imaginary part to a magnetic
        conductivity :math:`\sigma_m = \omega_0 \mu_0 \mu''`.

        Sign convention: :math:`e^{-i\omega t}`, so a *positive* imaginary part
        denotes loss (:math:`\varepsilon = \varepsilon' + i\varepsilon''`),
        matching the dispersion model used elsewhere in FDTDX.

        .. warning::
            This reproduces the requested complex permittivity *exactly only at*
            the reference frequency. A constant conductivity yields
            :math:`\varepsilon''(\omega) = \sigma / (\varepsilon_0 \omega)`, which
            falls off as :math:`1/\omega` away from :math:`\omega_0` — the causal
            behaviour of a conductor, not a frequency-flat loss. For a material
            whose :math:`\varepsilon(\omega)` must be matched across a band, use a
            dispersion model instead. The reference frequency is consumed only to
            pick :math:`\sigma`; it is not stored on the material and need not
            match any source.

        Anisotropy: every 3x3 tensor component may be complex. The real parts
        form the permittivity tensor and the imaginary parts map component-wise
        to a real conductivity tensor :math:`\sigma_{ij} = \omega_0 \varepsilon_0
        \varepsilon''_{ij}`, both handled by the full-tensor update equations.
        In particular a Hermitian :math:`\varepsilon` (imaginary off-diagonals,
        e.g. gyrotropic media) maps to an antisymmetric real :math:`\sigma`.

        Args:
            permittivity: Complex relative permittivity :math:`\varepsilon' + i\varepsilon''`.
                Scalar (isotropic), flat 3-tuple (diagonally anisotropic),
                flat 9-tuple ``(xx, xy, xz, yx, yy, yz, zx, zy, zz)`` or nested
                3x3 tuple (fully anisotropic).
            reference: Reference wave characteristic. Provide exactly one of
                ``reference``, ``wavelength``, or ``frequency``.
            wavelength: Free-space reference wavelength (m).
            frequency: Reference frequency (Hz).
            permeability: Optional complex relative permeability
                :math:`\mu' + i\mu''`, in the same formats as ``permittivity``.
                Defaults to ``1.0`` (non-magnetic, lossless).

        Returns:
            Material: Material with real permittivity/permeability and the derived
            electric/magnetic conductivities.
        """
        omega = _resolve_reference_omega(reference, wavelength, frequency)
        eps_real, sigma_e = _split_complex_property(permittivity, omega, constants.eps0)
        mu_real, sigma_m = _split_complex_property(permeability, omega, constants.mu0)
        for name, real_part in (("permittivity", eps_real), ("permeability", mu_real)):
            mat = np.array(_normalize_material_property(real_part), dtype=np.float64).reshape(3, 3)
            det = float(np.linalg.det(mat))
            if abs(det) < 1e-9 * max(1.0, float(np.abs(mat).max()) ** 3):
                raise ValueError(
                    f"The real part of the complex {name} tensor is singular (determinant ~ 0). "
                    f"FDTDX stores the inverse {name}, so the real part must be an invertible 3x3 tensor."
                )
        return cls(
            permittivity=eps_real,
            permeability=mu_real,
            electric_conductivity=sigma_e,
            magnetic_conductivity=sigma_m,
        )

    @classmethod
    def from_refractive_index(
        cls,
        refractive_index: complex | float | tuple,
        *,
        reference: WaveCharacter | None = None,
        wavelength: float | None = None,
        frequency: float | None = None,
    ) -> "Material":
        r"""Build a non-dispersive lossy material from a complex refractive index.

        The complex refractive index :math:`\tilde{n} = n + i\kappa` maps to the
        relative permittivity :math:`\varepsilon = \tilde{n}^2 = (n^2 - \kappa^2)
        + i\,2 n \kappa`, whose imaginary part becomes an equivalent electric
        conductivity at the reference frequency (see
        :meth:`from_complex_permittivity`). Assumes a non-magnetic medium
        (:math:`\mu_r = 1`); for magnetic materials specify :math:`\varepsilon`
        and :math:`\mu` directly via :meth:`from_complex_permittivity`.

        Sign convention: :math:`e^{-i\omega t}`, so a *positive* extinction
        coefficient :math:`\kappa` denotes loss.

        .. warning::
            The loss is matched exactly only at the reference frequency (see
            :meth:`from_complex_permittivity`).

        Args:
            refractive_index: Complex refractive index :math:`n + i\kappa`. Scalar
                (isotropic) or a flat 3-tuple (diagonally anisotropic). Full
                tensors are rejected: the permittivity is the *matrix* square of
                the refractive-index tensor, not the elementwise square — build
                the permittivity tensor directly via
                :meth:`from_complex_permittivity` instead.
            reference: Reference wave characteristic. Provide exactly one of
                ``reference``, ``wavelength``, or ``frequency``.
            wavelength: Free-space reference wavelength (m).
            frequency: Reference frequency (Hz).

        Returns:
            Material: The equivalent lossy material.
        """
        if isinstance(refractive_index, tuple):
            if len(refractive_index) != 3 or any(isinstance(n, tuple) for n in refractive_index):
                raise ValueError(
                    "from_refractive_index accepts a scalar or a flat 3-tuple; for full tensors build the "
                    "permittivity directly via from_complex_permittivity (matrix vs elementwise square)."
                )
            permittivity: complex | tuple = tuple(complex(n) ** 2 for n in refractive_index)
        else:
            permittivity = complex(refractive_index) ** 2
        return cls.from_complex_permittivity(
            permittivity,
            reference=reference,
            wavelength=wavelength,
            frequency=frequency,
        )

    @classmethod
    def from_loss_tangent(
        cls,
        permittivity: float | tuple,
        loss_tangent: float | tuple,
        *,
        reference: WaveCharacter | None = None,
        wavelength: float | None = None,
        frequency: float | None = None,
        permeability: complex | float | tuple = 1.0,
    ) -> "Material":
        r"""Build a non-dispersive lossy material from a real permittivity and loss tangent.

        The loss tangent :math:`\tan\delta = \varepsilon''/\varepsilon'` defines
        the imaginary part :math:`\varepsilon'' = \varepsilon' \tan\delta`, which
        maps to an equivalent electric conductivity
        :math:`\sigma = \omega_0 \varepsilon_0 \varepsilon' \tan\delta` at the
        reference frequency (see :meth:`from_complex_permittivity`).

        .. warning::
            The loss is matched exactly only at the reference frequency (see
            :meth:`from_complex_permittivity`).

        Args:
            permittivity: Real relative permittivity :math:`\varepsilon'`. Scalar
                (isotropic), flat 3-tuple (diagonally anisotropic), or flat
                9-tuple (fully anisotropic).
            loss_tangent: Loss tangent :math:`\tan\delta`. Scalar (applied to
                every tensor component, i.e. :math:`\varepsilon'' = \varepsilon'
                \tan\delta` as a tensor scaling), or a flat tuple matching
                ``permittivity``.
            reference: Reference wave characteristic. Provide exactly one of
                ``reference``, ``wavelength``, or ``frequency``.
            wavelength: Free-space reference wavelength (m).
            frequency: Reference frequency (Hz).
            permeability: Optional complex relative permeability. Defaults to ``1.0``.

        Returns:
            Material: The equivalent lossy material.
        """
        if isinstance(permittivity, tuple) or isinstance(loss_tangent, tuple):
            if (isinstance(permittivity, tuple) and any(isinstance(v, tuple) for v in permittivity)) or (
                isinstance(loss_tangent, tuple) and any(isinstance(v, tuple) for v in loss_tangent)
            ):
                raise ValueError(
                    "from_loss_tangent accepts scalars or flat tuples (3 or 9 components), not nested tuples."
                )
            n = len(permittivity) if isinstance(permittivity, tuple) else len(cast(tuple, loss_tangent))
            eps_t = permittivity if isinstance(permittivity, tuple) else (permittivity,) * n
            tan_t = loss_tangent if isinstance(loss_tangent, tuple) else (loss_tangent,) * n
            if len(eps_t) != len(tan_t):
                raise ValueError(
                    f"permittivity and loss_tangent must have matching lengths, got {len(eps_t)} and {len(tan_t)}."
                )
            complex_eps: complex | tuple = tuple(float(e) * (1.0 + 1j * float(t)) for e, t in zip(eps_t, tan_t))
        else:
            complex_eps = float(permittivity) * (1.0 + 1j * float(loss_tangent))
        return cls.from_complex_permittivity(
            complex_eps,
            reference=reference,
            wavelength=wavelength,
            frequency=frequency,
            permeability=permeability,
        )


def _resolve_reference_omega(
    reference: WaveCharacter | None,
    wavelength: float | None,
    frequency: float | None,
) -> float:
    """Resolve a reference angular frequency ω = 2πf from exactly one input.

    Args:
        reference: Reference wave characteristic, or None.
        wavelength: Free-space reference wavelength (m), or None.
        frequency: Reference frequency (Hz), or None.

    Returns:
        float: Angular frequency ω = 2πf in rad/s.

    Raises:
        ValueError: If not exactly one of the three inputs is provided.
    """
    num_given = sum(x is not None for x in (reference, wavelength, frequency))
    if num_given != 1:
        raise ValueError(
            f"Specify exactly one of 'reference' (WaveCharacter), 'wavelength', or 'frequency'; got {num_given}."
        )
    if reference is not None:
        freq = reference.get_frequency()
    elif wavelength is not None:
        freq = constants.c / wavelength
    else:
        assert frequency is not None
        freq = frequency
    return 2.0 * math.pi * freq


def _split_complex_property(
    value: float | complex | tuple,
    omega: float,
    vacuum_constant: float,
) -> tuple:
    r"""Split a (possibly complex) material property into a real part and conductivity.

    For a complex permittivity :math:`\varepsilon = \varepsilon' + i\varepsilon''`
    the loss maps to an electric conductivity
    :math:`\sigma = \omega \varepsilon_0 \varepsilon''` (use
    ``vacuum_constant = eps0``). For a complex permeability
    :math:`\mu = \mu' + i\mu''` the magnetic loss maps to
    :math:`\sigma_m = \omega \mu_0 \mu''` (use ``vacuum_constant = mu0``). The
    sign convention is :math:`e^{-i\omega t}`, so a positive imaginary part
    denotes loss. A negative imaginary part (gain) is permitted and yields a
    negative conductivity.

    Args:
        value: Scalar or flat tuple of real/complex components.
        omega: Angular frequency (rad/s) at which the imaginary part is matched.
        vacuum_constant: ``eps0`` for permittivity, ``mu0`` for permeability.

    Returns:
        tuple: A ``(real_part, conductivity)`` pair, each matching the input
        shape (scalar float or flat tuple of floats; a nested 3x3 input is
        flattened to a row-major 9-tuple).
    """

    def _component(v) -> tuple[float, float]:
        cv = complex(v)
        return float(cv.real), omega * vacuum_constant * float(cv.imag)

    if isinstance(value, tuple):
        if any(isinstance(v, tuple) for v in value):
            # Nested 3x3 tensor: flatten to a row-major 9-tuple first.
            if len(value) != 3 or not all(isinstance(v, tuple) and len(v) == 3 for v in value):
                raise ValueError(
                    "Nested complex material properties must be a 3x3 tuple ((xx, xy, xz), (yx, yy, yz), (zx, zy, zz))."
                )
            value = tuple(entry for row in value for entry in row)
        reals: list[float] = []
        conds: list[float] = []
        for v in value:
            r, s = _component(v)
            reals.append(r)
            conds.append(s)
        return tuple(reals), tuple(conds)
    r, s = _component(value)
    return r, s


def _is_property_isotropic(prop: tuple[float, float, float, float, float, float, float, float, float]) -> bool:
    return (
        math.isclose(prop[0], prop[4])
        and math.isclose(prop[4], prop[8])
        and math.isclose(prop[1], 0.0)
        and math.isclose(prop[2], 0.0)
        and math.isclose(prop[3], 0.0)
        and math.isclose(prop[5], 0.0)
        and math.isclose(prop[6], 0.0)
        and math.isclose(prop[7], 0.0)
    )


def isotropic_property_value(
    prop: tuple[float, float, float, float, float, float, float, float, float],
    name: str = "material property",
) -> float:
    """Return the scalar value represented by a finite real isotropic material property."""
    if any(isinstance(value, (bool, np.bool_)) for value in prop):
        raise ValueError(f"{name} must be a finite real isotropic value.")
    raw_array = np.asarray(prop)
    if raw_array.shape != (9,) or not np.issubdtype(raw_array.dtype, np.number):
        raise ValueError(f"{name} must be a finite real isotropic value.")
    if np.iscomplexobj(raw_array) and np.any(np.imag(raw_array) != 0.0):
        raise ValueError(f"{name} must be real.")
    array = raw_array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    property_values = cast(
        tuple[float, float, float, float, float, float, float, float, float],
        tuple(float(value) for value in array),
    )
    if not _is_property_isotropic(property_values):
        raise ValueError(f"{name} must be isotropic.")
    return property_values[0]


def _is_property_diagonally_anisotropic(
    prop: tuple[float, float, float, float, float, float, float, float, float],
) -> bool:
    return (
        math.isclose(prop[1], 0.0)
        and math.isclose(prop[2], 0.0)
        and math.isclose(prop[3], 0.0)
        and math.isclose(prop[5], 0.0)
        and math.isclose(prop[6], 0.0)
        and math.isclose(prop[7], 0.0)
    )


def compute_ordered_material_name_tuples(
    materials: dict[str, Material],
) -> list[tuple[str, Material]]:
    """
    Returns a list of materials ordered by their properties.

    The ordering priority is:
    1. Permittivity (ascending, using first component for ordering)
    2. Permeability (ascending, using first component for ordering)
    3. Electric conductivity (ascending, using first component for ordering)
    4. Magnetic conductivity (ascending, using first component for ordering)

    Args:
        materials (dict[str, Material]): Dictionary mapping material names to Material objects.

    Returns:
        list[tuple[str, Material]]: List of Material objects ordered by their properties.
    """
    return sorted(
        materials.items(),
        key=lambda m: (
            m[1].permittivity[0],
            m[1].permeability[0],
            m[1].electric_conductivity[0],
            m[1].magnetic_conductivity[0],
        ),
        reverse=False,
    )


def compute_allowed_permittivities(
    materials: dict[str, Material],
    isotropic: bool = False,
    diagonally_anisotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of permittivity tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples (εx,)
        diagonally_anisotropic: If True, return 3-tuples (εx, εy, εz)

    Returns:
        list[tuple[float, ...]]: List of permittivity tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].permittivity[0],) for o in ordered_materials]
    elif diagonally_anisotropic:
        return [(o[1].permittivity[0], o[1].permittivity[4], o[1].permittivity[8]) for o in ordered_materials]
    else:  # fully anisotropic
        return [o[1].permittivity for o in ordered_materials]


def compute_allowed_permeabilities(
    materials: dict[str, Material],
    isotropic: bool = False,
    diagonally_anisotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of permeability tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples (μx,)
        diagonally_anisotropic: If True, return 3-tuples (μx, μy, μz)

    Returns:
        list[tuple[float, ...]]: List of permeability tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].permeability[0],) for o in ordered_materials]
    elif diagonally_anisotropic:
        return [(o[1].permeability[0], o[1].permeability[4], o[1].permeability[8]) for o in ordered_materials]
    else:  # fully anisotropic
        return [o[1].permeability for o in ordered_materials]


def compute_allowed_electric_conductivities(
    materials: dict[str, Material],
    isotropic: bool = False,
    diagonally_anisotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of electric conductivity tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples (sigma_x,)
        diagonally_anisotropic: If True, return 3-tuples (sigma_x, sigma_y, sigma_z)

    Returns:
        list[tuple[float, ...]]: List of electric conductivity tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].electric_conductivity[0],) for o in ordered_materials]
    elif diagonally_anisotropic:
        return [
            (o[1].electric_conductivity[0], o[1].electric_conductivity[4], o[1].electric_conductivity[8])
            for o in ordered_materials
        ]
    else:  # fully anisotropic
        return [o[1].electric_conductivity for o in ordered_materials]


def compute_allowed_magnetic_conductivities(
    materials: dict[str, Material],
    isotropic: bool = False,
    diagonally_anisotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of magnetic conductivity tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples (sigma_x,)
        diagonally_anisotropic: If True, return 3-tuples (sigma_x, sigma_y, sigma_z)

    Returns:
        list[tuple[float, ...]]: List of magnetic conductivity tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].magnetic_conductivity[0],) for o in ordered_materials]
    elif diagonally_anisotropic:
        return [
            (o[1].magnetic_conductivity[0], o[1].magnetic_conductivity[4], o[1].magnetic_conductivity[8])
            for o in ordered_materials
        ]
    else:  # fully anisotropic
        return [o[1].magnetic_conductivity for o in ordered_materials]


def compute_max_dispersive_poles(materials: dict[str, Material]) -> int:
    """Return the maximum number of dispersive poles across a set of materials.

    Args:
        materials: Dictionary mapping material names to Material objects.

    Returns:
        int: ``max(m.dispersion.num_poles)`` over dispersive materials, or 0.
    """
    n = 0
    for m in materials.values():
        if m.dispersion is not None:
            n = max(n, m.dispersion.num_poles)
    return n


def compute_allowed_dispersive_coefficients(
    materials: dict[str, Material],
    dt: float,
    max_num_poles: int,
    num_components: int,
    coupling_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-material discrete-time dispersive recurrence coefficients.

    For each material (in the canonical sorted order used elsewhere), returns
    the ``c1``, ``c2``, ``c3``, ``c4`` coefficients from
    :func:`fdtdx.dispersion.compute_pole_coefficients_tensor`. Materials with
    fewer poles than ``max_num_poles``, or non-dispersive materials, get
    zero-padded slots (so the polarization term automatically vanishes on those
    voxels). ``c4`` is zero for Lorentz/Drude poles and only non-zero for CCPR
    poles.

    Args:
        materials: Dictionary mapping material names to Material objects.
        dt: Simulation time step (seconds).
        max_num_poles: Maximum pole count to pad every material to. When 0,
            returns zero-pole arrays.
        num_components: Size of the trailing component axis of the recurrence
            coefficients ``c1``/``c2`` — 1 when all materials have isotropic
            dispersion, 3 for per-axis dispersion.
        coupling_components: Size of the trailing axis of the field couplings
            ``c3``/``c4`` — additionally allows 9 (row-major 3x3 coupling
            tensor) when any material has oriented poles. Defaults to
            ``num_components``.

    Returns:
        Four numpy arrays: ``c1``, ``c2`` of shape
        ``(num_materials, max_num_poles, num_components)`` and ``c3``, ``c4``
        of shape ``(num_materials, max_num_poles, coupling_components)``.
    """
    if coupling_components is None:
        coupling_components = num_components
    if num_components not in (1, 3):
        raise ValueError(f"num_components must be 1 or 3, got {num_components}.")
    if coupling_components not in (1, 3, 9):
        raise ValueError(f"coupling_components must be 1, 3 or 9, got {coupling_components}.")
    ordered = compute_ordered_material_name_tuples(materials)
    num_mats = len(ordered)
    c1 = np.zeros((num_mats, max_num_poles, num_components), dtype=np.float64)
    c2 = np.zeros((num_mats, max_num_poles, num_components), dtype=np.float64)
    c3 = np.zeros((num_mats, max_num_poles, coupling_components), dtype=np.float64)
    c4 = np.zeros((num_mats, max_num_poles, coupling_components), dtype=np.float64)
    if max_num_poles == 0:
        return c1, c2, c3, c4
    diag_entries = (0, 4, 8)
    for m_idx, (_, mat) in enumerate(ordered):
        if mat.dispersion is None:
            continue
        if num_components == 1 and not mat.has_isotropic_dispersion:
            raise ValueError("num_components=1 requires isotropic dispersion, but a material has per-axis poles.")
        if coupling_components == 1 and not mat.has_isotropic_dispersion:
            raise ValueError("coupling_components=1 requires isotropic dispersion, but a material has per-axis poles.")
        if coupling_components < 9 and not mat.has_axis_aligned_dispersion:
            raise ValueError(
                "coupling_components < 9 requires axis-aligned dispersion, but a material has oriented poles."
            )
        poles = mat.dispersion.poles
        c1_vals, c2_vals, c3_vals, c4_vals = compute_pole_coefficients_tensor(poles, dt)
        n = len(poles)
        # For num_components == 1 the isotropy check above guarantees all three
        # axis columns are identical, so keeping only the first is exact. For
        # coupling_components < 9 the axis-alignment check guarantees the
        # coupling tensors are diagonal, so keeping diagonal entries is exact.
        c1[m_idx, :n] = c1_vals[:, :num_components]
        c2[m_idx, :n] = c2_vals[:, :num_components]
        if coupling_components == 9:
            c3[m_idx, :n] = c3_vals
            c4[m_idx, :n] = c4_vals
        else:
            c3[m_idx, :n] = c3_vals[:, diag_entries[:coupling_components]]
            c4[m_idx, :n] = c4_vals[:, diag_entries[:coupling_components]]
    return c1, c2, c3, c4


def _min_dispersive_divisor(mat: Material, dt: float) -> tuple[float, int]:
    r"""Minimum per-cell implicit-update divisor for a dispersive material.

    Mirrors the ``divisor`` computed in :func:`fdtdx.fdtd.update.update_E` for a
    cell fully occupied by ``mat``:

    .. math::
        \text{divisor}_a = 1 + \varepsilon_{\infty,a}^{-1} \sum_p c_{4,p,a}
        + \tfrac{1}{2}\, \sigma_a\, c_0\, \Delta t\, \eta_0\, \varepsilon_{\infty,a}^{-1}

    per axis ``a``. The ``courant_number`` that scales both the conductivity term
    and the (resolution-scaled) conductivity array cancels, so the divisor depends
    only on ``dt``, the diagonal ``eps_inf``, the physical conductivity, and the
    pole ``c4`` coefficients. The conductivity term is always ``>= 0``, so the
    binding constraint is ``1 + inv_eps * sum(c4)``.

    Args:
        mat: A dispersive material (``mat.dispersion`` must not be ``None``).
        dt: Simulation time step (seconds).

    Returns:
        tuple: ``(min_divisor, worst_axis)`` — the smallest divisor over the three
        grid axes and the axis index at which it occurs.
    """
    assert mat.dispersion is not None
    # tensor variant handles every pole type; oriented poles have no dE/dt
    # coupling, so their diagonal c4 entries are zero and drop out here
    _, _, _, c4 = compute_pole_coefficients_tensor(mat.dispersion.poles, dt)
    sum_c4 = c4[:, (0, 4, 8)].sum(axis=0)  # (3,) diagonal c4 summed over poles, per axis
    eps_inf = (mat.permittivity[0], mat.permittivity[4], mat.permittivity[8])
    sigma = (mat.electric_conductivity[0], mat.electric_conductivity[4], mat.electric_conductivity[8])
    min_div, worst_ax = math.inf, 0
    for ax in range(3):
        inv_eps = 1.0 / eps_inf[ax]
        cond = 0.5 * sigma[ax] * constants.c * dt * constants.eta0 * inv_eps  # >= 0
        div = 1.0 + inv_eps * float(sum_c4[ax]) + cond
        if div < min_div:
            min_div, worst_ax = div, ax
    return min_div, worst_ax


def _max_safe_courant_factor(
    mat: Material,
    dt: float,
    courant_factor: float,
    threshold: float,
    scan_steps: int = 512,
) -> float:
    """Largest ``courant_factor`` below which ``mat`` stays stable at every step.

    ``dt`` is proportional to ``courant_factor`` on every grid type, so scaling
    the time step by ``s`` scales directly to a ``courant_factor``. Each pole's
    ``c4_p(s) = b_p * (s * dt) / (1 + gamma_p * s * dt / 2)`` is individually
    monotonic in ``s`` (its sign is that of ``b_p``), but the *sum* over poles
    with mixed ``b_p`` signs and dampings — plus the linearly growing
    conductivity term — is not necessarily monotonic and may cross ``threshold``
    several times. To keep it genuinely safe (every value below ``X``
    stable), we search for the *first* crossing from ``s = 0`` (where the
    divisor is ``~ 1``): a forward scan brackets it, then bisection refines.
    Recomputing the divisor at each ``s * dt`` handles the weak
    ``D = 1 + gamma * dt / 2`` nonlinearity exactly.
    """
    n = max(1, scan_steps)
    prev = 0.0  # last safe scale (divisor >= threshold); s = 0 gives divisor ~ 1
    hi = None
    for k in range(1, n + 1):
        s = k / n
        div, _ = _min_dispersive_divisor(mat, s * dt)
        if div < threshold:
            hi = s
            break
        prev = s
    if hi is None:
        # No sub-threshold scan point found. Unreachable via the validator (it
        # only calls this when the divisor at s = 1 is already below threshold),
        # but stay conservative if invoked directly.
        return 0.5 * courant_factor
    lo = prev  # bracket [lo, hi]: divisor(lo) >= threshold, divisor(hi) < threshold
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        div, _ = _min_dispersive_divisor(mat, mid * dt)
        if div >= threshold:
            lo = mid
        else:
            hi = mid
    result = lo * courant_factor
    if result <= 0.0:
        return 0.0
    # Floor to 3 significant figures so the reported "<= X" is never rounded
    # ABOVE the true first-crossing boundary (nearest-rounding could over-promise).
    scale = 10.0 ** (math.floor(math.log10(result)) - 2)
    return math.floor(result / scale) * scale


def validate_dispersive_divisor_stability(
    materials: dict[str, Material],
    dt: float,
    courant_factor: float,
    near_zero_threshold: float = 0.01,
) -> None:
    r"""Validate the per-cell implicit-update divisor of CCPR dispersive materials.

    For CCPR poles (non-zero ``dE/dt`` coupling) the E-field update divides by a
    per-cell factor ``1 + inv_eps * sum(c4) [+ conductivity term]`` (see
    :func:`fdtdx.fdtd.update.update_E`). This must stay positive; as it approaches
    ``0^+`` the transient gain (``~ 1/divisor``) explodes and accuracy collapses.
    Lorentz and Drude poles have ``c4 = 0``, so their divisor is always ``>= 1``
    and they are skipped.

    Checked per distinct material rather than per assembled cell: a cell holds one
    material's coefficients, so this covers every static cell, names the offending
    material, and — because a CONTINUOUS device's interpolated divisor
    ``1 + sum(c4)(t) / eps(t)`` is a monotonic (Moebius) function of the mix ``t``
    — the bracketing materials bound the device-interpolated cells too.

    Args:
        materials: Mapping of label -> :class:`Material` for every material in the
            simulation. Labels appear verbatim in the error/warning messages.
        dt: Simulation time step (seconds).
        courant_factor: The configured ``courant_factor`` (for the remediation hint).
        near_zero_threshold: Warn when ``0 < divisor < near_zero_threshold``; raise
            when ``divisor <= 0``.

    Raises:
        ValueError: If any material's divisor is non-positive (unconditionally
            unstable / NaN).
    """
    for name, mat in materials.items():
        if mat.dispersion is None:
            continue
        # c4 != 0 only for CCPR / non-zero dE/dt coupling; Lorentz & Drude are safe.
        if not any(b != 0.0 for p in mat.dispersion.poles for b in p.coupling_edot_axes):
            continue
        min_div, worst_ax = _min_dispersive_divisor(mat, dt)
        if min_div >= near_zero_threshold:
            continue
        cf_max = _max_safe_courant_factor(mat, dt, courant_factor, near_zero_threshold)
        axis_note = f" on axis {'xyz'[worst_ax]}" if not (mat.is_all_isotropic and mat.dispersion.is_isotropic) else ""
        detail = (
            f"Dispersive material '{name}' has an implicit E-update divisor = {min_div:.4g}{axis_note} "
            "(1 + inv_eps * sum(c4) [+ conductivity term]); this factor divides the E-field update, so as "
            "it approaches 0 the transient gain (~1/divisor) explodes and accuracy collapses. The divisor "
            f"scales with the time step, so lower courant_factor to <= {cf_max:.3g} (currently {courant_factor:.3g})."
        )
        if min_div <= 0.0:
            raise ValueError(
                detail + " The divisor is non-positive, which makes the ADE update unconditionally unstable (NaN)."
            )
        warnings.warn(detail, UserWarning, stacklevel=2)


def compute_ordered_names(
    materials: dict[str, Material],
) -> list[str]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[0] for o in ordered_materials]


def compute_ordered_materials(
    materials: dict[str, Material],
) -> list[Material]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[1] for o in ordered_materials]
