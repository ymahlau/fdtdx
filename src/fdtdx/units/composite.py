from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unit, Unitful

s_unit = Unit(scale=0, dim={SI.s: 1})
ms_unit = Unit(scale=-3, dim={SI.s: 1})
us_unit = Unit(scale=-6, dim={SI.s: 1})
ns_unit = Unit(scale=-9, dim={SI.s: 1})
ps_unit = Unit(scale=-12, dim={SI.s: 1})
fs_unit = Unit(scale=-15, dim={SI.s: 1})

s = Unitful(val=1, unit=s_unit)
ms = Unitful(val=1, unit=ms_unit)
us = Unitful(val=1, unit=us_unit)
ns = Unitful(val=1, unit=ns_unit)
ps = Unitful(val=1, unit=ps_unit)
fs = Unitful(val=1, unit=fs_unit)

Hz_unit = Unit(scale=0, dim={SI.s: -1})
kHz_unit = Unit(scale=3, dim={SI.s: -1})
MHz_unit = Unit(scale=6, dim={SI.s: -1})
GHz_unit = Unit(scale=9, dim={SI.s: -1})
THz_unit = Unit(scale=12, dim={SI.s: -1})
PHz_unit = Unit(scale=15, dim={SI.s: -1})

Hz = Unitful(val=1, unit=Hz_unit)
kHz = Unitful(val=1, unit=kHz_unit)
MHz = Unitful(val=1, unit=MHz_unit)
GHz = Unitful(val=1, unit=GHz_unit)
THz = Unitful(val=1, unit=THz_unit)
PHz = Unitful(val=1, unit=PHz_unit)

km_unit = Unit(scale=3, dim={SI.m: 1})
m_unit = Unit(scale=0, dim={SI.m: 1})
mm_unit = Unit(scale=-3, dim={SI.m: 1})
um_unit = Unit(scale=-6, dim={SI.m: 1})
nm_unit = Unit(scale=-9, dim={SI.m: 1})
pm_unit = Unit(scale=-12, dim={SI.m: 1})

km = Unitful(val=1, unit=km_unit)
m = Unitful(val=1, unit=m_unit)
mm = Unitful(val=1, unit=mm_unit)
um = Unitful(val=1, unit=um_unit)
nm = Unitful(val=1, unit=nm_unit)
pm = Unitful(val=1, unit=pm_unit)

m_per_s = Unitful(val=1, unit=Unit(scale=0, dim={SI.s: -1, SI.m: 1}))