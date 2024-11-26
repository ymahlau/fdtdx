import math

c: float = 299792458.0
""" speed of light """

mu0: float = 4e-7 * math.pi
""" vacuum permeability """

eps0: float = 1.0 / (mu0 * c**2)
""" vacuum permittivity """

eta0: float = mu0 * c
""" free space impedance """

# Relative Permittivities of different materials
relative_permittivity_air: float = 1.0
relative_permittivity_substrate: float = 2.1025
relative_permittivity_polymer: float = 2.368521
relative_permittivity_silicon: float = 12.25
relative_permittivity_silica: float = 2.25
relative_permittivity_SZ_2080: float = 2.1786
relative_permittivity_ma_N_1400_series: float = 2.6326
relative_permittivity_bacteria: float = 1.96
relative_permittivity_water: float = 1.737
relative_permittivity_fused_silica: float = 2.13685924
relative_permittivity_coated_silica: float = 1.69
relative_permittivity_resin: float = 2.202256
relative_permittivity_ormo_prime: float = 1.817104

silicon_permittivity_config = (
    ("Si", relative_permittivity_silicon),
    ("Air", relative_permittivity_air),
)

standard_permittivity_config = (
    ("SZ2080", relative_permittivity_SZ_2080),
    ("Air", relative_permittivity_air),
)

higher_permittivity_config = (
    ("ma-N 1400", relative_permittivity_ma_N_1400_series),
    ("Air", relative_permittivity_air),
)

silica_permittivity_config = (
    ("Polymer", relative_permittivity_silica),
    ("Air", relative_permittivity_air),
)

multi_material_permittivity_config = (
    ("ma-N 1400", relative_permittivity_ma_N_1400_series),
    ("SZ2080", relative_permittivity_SZ_2080),
    ("Air", relative_permittivity_air),
)

multi_material_qd_permittivity_config = (
    ("SZ2080", relative_permittivity_resin),
    ("SZ2080+q", relative_permittivity_resin + 0.01), # TODO: check whether this is enough
    ("Air", relative_permittivity_air),
)

SHARD_STR = "shard"


def wavelength_to_period(wavelength: float):
    return wavelength / c
