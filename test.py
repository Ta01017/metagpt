from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator

tokens = [
    '[BOS]', 'PX_590', 'PY_590', 'SUB_Glass_Substrate',
    'L1_MAT_SiO2', 'L1_SHAPE_RECT',
    'L1_H_1170', 'L1_W_290', 'L1_L_370', '[EOS]'
]

# tokenizer
tk = StructureTokenizer()
ids = tk.encode(tokens)
print("ids:", ids)
print("decoded:", tk.decode(ids))

# parser
parser = StructureParser()
struct_dict = parser.parse(tokens)
print("dict:", struct_dict)

# validator
validator = StructureValidator()
ok, reason = validator.validate(struct_dict)
print("valid:", ok, "reason:", reason)
