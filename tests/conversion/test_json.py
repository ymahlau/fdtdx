from fdtdx.conversion.json import export_json, export_json_str, import_from_json
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.sources.linear_polarization import UniformPlaneSource
from fdtdx.objects.static_material.static import SimulationVolume


def test_energy_detector_json():
    det = EnergyDetector(
        partial_real_shape=(1e-6, None, None),
        name="Detector",
    )
    d = export_json(det)
    assert d["name"] == "Detector"
    s = export_json_str(det)
    rec = import_from_json(s)
    assert rec.name == "Detector"
    assert rec.partial_real_shape == (1e-6, None, None)
    

def test_object_container_json():
    obj_list = [
        EnergyDetector(),
        UniformPlaneSource(wave_character=WaveCharacter(wavelength=1e-6), direction="-"),
        SimulationVolume(),
    ]
    container = ObjectContainer(object_list=obj_list, volume_idx=2)
    s = export_json_str(container)
    rec = import_from_json(s)
    assert isinstance(rec.object_list, list)
    assert rec.object_list[1].wave_character.wavelength == 1e-6
    assert rec.object_list[1].direction == "-"

