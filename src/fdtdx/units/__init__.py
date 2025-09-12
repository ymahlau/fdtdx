from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unit, Unitful, UnitfulIndexer

s_unit = Unit(scale=0, dim={SI.s: 1})
ms_unit = Unit(scale=-3, dim={SI.s: 1})
us_unit = Unit(scale=-6, dim={SI.s: 1})
ns_unit = Unit(scale=-9, dim={SI.s: 1})
ps_unit = Unit(scale=-12, dim={SI.s: 1})
fs_unit = Unit(scale=-15, dim={SI.s: 1})
s = Unitful(val=1.0, unit=s_unit)
ms = Unitful(val=1.0, unit=ms_unit)
us = Unitful(val=1.0, unit=us_unit)
ns = Unitful(val=1.0, unit=ns_unit)
ps = Unitful(val=1.0, unit=ps_unit)
fs = Unitful(val=1.0, unit=fs_unit)

Hz_unit = Unit(scale=0, dim={SI.s: -1})
kHz_unit = Unit(scale=3, dim={SI.s: -1})
MHz_unit = Unit(scale=6, dim={SI.s: -1})
GHz_unit = Unit(scale=9, dim={SI.s: -1})
THz_unit = Unit(scale=12, dim={SI.s: -1})
PHz_unit = Unit(scale=15, dim={SI.s: -1})
Hz = Unitful(val=1.0, unit=Hz_unit)
kHz = Unitful(val=1.0, unit=kHz_unit)
MHz = Unitful(val=1.0, unit=MHz_unit)
GHz = Unitful(val=1.0, unit=GHz_unit)
THz = Unitful(val=1.0, unit=THz_unit)
PHz = Unitful(val=1.0, unit=PHz_unit)

km_unit = Unit(scale=3, dim={SI.m: 1})
m_unit = Unit(scale=0, dim={SI.m: 1})
mm_unit = Unit(scale=-3, dim={SI.m: 1})
um_unit = Unit(scale=-6, dim={SI.m: 1})
nm_unit = Unit(scale=-9, dim={SI.m: 1})
pm_unit = Unit(scale=-12, dim={SI.m: 1})
km = Unitful(val=1.0, unit=km_unit)
m = Unitful(val=1.0, unit=m_unit)
mm = Unitful(val=1.0, unit=mm_unit)
um = Unitful(val=1.0, unit=um_unit)
nm = Unitful(val=1.0, unit=nm_unit)
pm = Unitful(val=1.0, unit=pm_unit)

MW_unit = Unit(scale=6, dim={SI.kg: 1, SI.m: 2, SI.s: -3})
kW_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -3})
W_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3})
mW_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.s: -3})
uW_unit = Unit(scale=-6, dim={SI.kg: 1, SI.m: 2, SI.s: -3})
MW = Unitful(val=1.0, unit=MW_unit)
kW = Unitful(val=1.0, unit=kW_unit)
W = Unitful(val=1.0, unit=W_unit)
mW = Unitful(val=1.0, unit=mW_unit)
uW = Unitful(val=1.0, unit=uW_unit)

kV_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})
V_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})
mV_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})
uV_unit = Unit(scale=-6, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})
kV = Unitful(val=1.0, unit=kV_unit)
V = Unitful(val=1.0, unit=V_unit)
mV = Unitful(val=1.0, unit=mV_unit)
uV = Unitful(val=1.0, unit=uV_unit)

kA_unit = Unit(scale=3, dim={SI.A: 1})
A_unit = Unit(scale=0, dim={SI.A: 1})
mA_unit = Unit(scale=-3, dim={SI.A: 1})
uA_unit = Unit(scale=-6, dim={SI.A: 1})
kA = Unitful(val=1.0, unit=kA_unit)
A = Unitful(val=1.0, unit=A_unit)
mA = Unitful(val=1.0, unit=mA_unit)
uA = Unitful(val=1.0, unit=uA_unit)

F_unit = Unit(scale=0, dim={SI.s: 4, SI.A: 2, SI.m: -2, SI.kg: -2})
mF_unit = Unit(scale=-3, dim={SI.s: 4, SI.A: 2, SI.m: -2, SI.kg: -2})
uF_unit = Unit(scale=-6, dim={SI.s: 4, SI.A: 2, SI.m: -2, SI.kg: -2})
nF_unit = Unit(scale=-9, dim={SI.s: 4, SI.A: 2, SI.m: -2, SI.kg: -2})
pF_unit = Unit(scale=-12, dim={SI.s: 4, SI.A: 2, SI.m: -2, SI.kg: -2})
fF_unit = Unit(scale=-15, dim={SI.s: 4, SI.A: 2, SI.m: -2, SI.kg: -2})
F = Unitful(val=1.0, unit=F_unit)
mF = Unitful(val=1.0, unit=mF_unit)
uF = Unitful(val=1.0, unit=uF_unit)
nF = Unitful(val=1.0, unit=nF_unit)
pF = Unitful(val=1.0, unit=pF_unit)
fF = Unitful(val=1.0, unit=fF_unit)

N_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
N = Unitful(val=1.0, unit=N_unit)

J_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2})
J = Unitful(val=1.0, unit=J_unit)

W_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3})
W = Unitful(val=1.0, unit=W_unit)

m_per_s_unit = Unit(scale=0, dim={SI.s: -1, SI.m: 1})
m_per_s = m / s

V_per_m_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -3, SI.A: -1})
V_per_m = V / m

A_per_m_unit = Unit(scale=0, dim={SI.A: 1, SI.m: -1})
A_per_m = A / m

__all__ = [
    "SI",
    "Unitful",
    "Unit",
    "UnitfulIndexer",
]