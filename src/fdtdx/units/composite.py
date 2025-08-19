from fdtdx.units.unitful import SI, Unit, Unitful

s = Unitful(val=1, unit=Unit(scale=0, dim={SI.s: 1}))
ms = Unitful(val=1, unit=Unit(scale=-3, dim={SI.s: 1}))
us = Unitful(val=1, unit=Unit(scale=-6, dim={SI.s: 1}))
ns = Unitful(val=1, unit=Unit(scale=-9, dim={SI.s: 1}))
ps = Unitful(val=1, unit=Unit(scale=-12, dim={SI.s: 1}))
fs = Unitful(val=1, unit=Unit(scale=-15, dim={SI.s: 1}))

m_per_s = Unitful(val=1, unit=Unit(scale=0, dim={SI.s: -1, SI.m: 1}))

Hz = Unitful(val=1, unit=Unit(scale=0, dim={SI.s: -1}))
kHz = Unitful(val=1, unit=Unit(scale=3, dim={SI.s: -1}))
MHz = Unitful(val=1, unit=Unit(scale=6, dim={SI.s: -1}))
GHz = Unitful(val=1, unit=Unit(scale=9, dim={SI.s: -1}))
THz = Unitful(val=1, unit=Unit(scale=12, dim={SI.s: -1}))
PHz = Unitful(val=1, unit=Unit(scale=15, dim={SI.s: -1}))
