from fdtdx.units.unitful import SI, Unit, Unitful

s = Unitful(val=1, unit=Unit(0, {SI.s: 1}))
ms = Unitful(val=1, unit=Unit(-3, {SI.s: 1}))
µs = Unitful(val=1, unit=Unit(-6, {SI.s: 1}))
ns = Unitful(val=1, unit=Unit(-9, {SI.s: 1}))
ps = Unitful(val=1, unit=Unit(-12, {SI.s: 1}))
fs = Unitful(val=1, unit=Unit(-15, {SI.s: 1}))

m_per_s = Unitful(val=1, unit=Unit(0, {SI.s: -1, SI.m: 1}))

Hz = Unitful(val=1, unit=Unit(0, {SI.s: -1}))
kHz = Unitful(val=1, unit=Unit(3, {SI.s: -1}))
MHz = Unitful(val=1, unit=Unit(6, {SI.s: -1}))
GHz = Unitful(val=1, unit=Unit(9, {SI.s: -1}))
THz = Unitful(val=1, unit=Unit(12, {SI.s: -1}))
PHz = Unitful(val=1, unit=Unit(15, {SI.s: -1}))
