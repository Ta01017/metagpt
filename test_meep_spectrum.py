# test_meep_spectrum.py
from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from meep_spectrum_sim import MeepSpectrumSimulator
import matplotlib.pyplot as plt

# Example structure
tokens = [
    '[BOS]', 'PX_600', 'PY_600', 'SUB_Glass_Substrate',
    'L1_MAT_SiO2', 'L1_SHAPE_CYL',
    'L1_H_300', 'L1_R_120',
    '[EOS]'
]

# parse
parser = StructureParser()
struct_dict = parser.parse(tokens)

sim = MeepSpectrumSimulator(resolution=50)
wl, R, T, A = sim.simulate(struct_dict)

plt.figure(figsize=(6,4))
plt.plot(wl, R, label="R")
plt.plot(wl, T, label="T")
plt.plot(wl, A, label="A")
plt.legend(); plt.xlabel("λ (um)"); plt.ylabel("value")
plt.title("Meep spectrum test")
plt.grid()
plt.show()
