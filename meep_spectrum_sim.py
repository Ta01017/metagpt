# meep_spectrum_sim.py
import meep as mp
import numpy as np
from meep_builder import MeepGeometryBuilder


class MeepSpectrumSimulator:
    def __init__(self, resolution=40):
        # resolution 太低 → 曲线太光滑（你的同事指出的）
        # 40~60 是最合理范围
        self.resolution = resolution
        self.builder = MeepGeometryBuilder()

    # --------------------------------------------------
    def simulate(self, struct_dict, lambda_um_min=0.9, lambda_um_max=1.7, N=161):
        """
        输入：
            struct_dict: 结构字典
        输出：
            wavelengths, R, T, A
        """
        geometry, sx, sy, sz = self.builder.build_unit_cell(struct_dict)

        wavelengths = np.linspace(lambda_um_min, lambda_um_max, N)
        freqs = 1 / wavelengths

        cell = mp.Vector3(sx, sy, sz)
        pml_layers = [mp.PML(0.5)]

        src = [mp.Source(
            mp.GaussianSource(frequency=1/lambda_um_min, fwidth=0.2),
            component=mp.Ez,
            center=mp.Vector3(0, 0, 1.0),   # 上方入射
        )]

        sim = mp.Simulation(
            cell_size=cell,
            resolution=self.resolution,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=src,
            default_material=mp.air
        )

        # ----------------------------
        # Flux monitors
        # ----------------------------
        refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, 0.5), size=mp.Vector3(sx, sy, 0))
        tran_fr = mp.FluxRegion(center=mp.Vector3(0, 0, -sz/2 + 0.3))

        refl = sim.add_flux(freqs[0], freqs[-1], N, refl_fr)
        tran = sim.add_flux(freqs[0], freqs[-1], N, tran_fr)

        # ----------------------------
        # Baseline incident power
        # ----------------------------
        sim.run(until=50)
        inc_flux = mp.get_fluxes(refl)

        sim.reset_meep()

        # ----------------------------
        # Real simulation
        # ----------------------------
        sim = mp.Simulation(
            cell_size=cell,
            resolution=self.resolution,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=src,
            default_material=mp.air
        )

        refl = sim.add_flux(freqs[0], freqs[-1], N, refl_fr)
        tran = sim.add_flux(freqs[0], freqs[-1], N, tran_fr)

        sim.load_minus_flux_data(refl, inc_flux)

        sim.run(until=200)

        refl_flux = np.array(mp.get_fluxes(refl))
        tran_flux = np.array(mp.get_fluxes(tran))
        inc = np.maximum(np.array(inc_flux), 1e-12)

        R = refl_flux / inc
        T = tran_flux / inc
        A = 1 - R - T

        return wavelengths, R, T, A
