# meep_builder.py
import meep as mp


class MaterialLib:
    """
    统一管理折射率 → Meep 材料
    后续可接入你真实 n(λ) 数据
    """
    def __init__(self):
        self.materials = {
            "SiO2": 1.46,
            "TiO2": 2.40,
            "Ta2O5": 2.15,
            "HfO2": 1.98,
            "Glass_Substrate": 1.52,
        }

    def get(self, name):
        if name not in self.materials:
            raise ValueError(f"[MaterialLib] Unknown material: {name}")
        n = self.materials[name]
        return mp.Medium(index=n)


class MeepGeometryBuilder:
    """
    将结构 dict → Meep 几何物体列表
    支持 CYL 和 RECT
    """
    def __init__(self):
        self.matlib = MaterialLib()

    def build_unit_cell(self, struct_dict):
        P_x, P_y = struct_dict["P"]
        layer = struct_dict["layer1"]

        sx = P_x / 1000  # convert nm→um
        sy = P_y / 1000
        sz = 3.0         # 一般 3~5 um 用来容纳结构

        geometry = []

        # -----------------------------------
        # substrate
        # -----------------------------------
        substrate_n = self.matlib.get("Glass_Substrate")
        substrate = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, sz),
            center=mp.Vector3(0, 0, -sz/2),
            material=substrate_n,
        )
        geometry.append(substrate)

        # -----------------------------------
        # layer1 shape
        # -----------------------------------
        mat = self.matlib.get(layer["mat"])
        h = layer["h_nm"] / 1000

        if layer["shape"] == "CYL":
            r = layer["r_nm"] / 1000
            geom = mp.Cylinder(
                radius=r,
                height=h,
                center=mp.Vector3(0, 0, h/2),
                material=mat,
            )
            geometry.append(geom)

        elif layer["shape"] == "RECT":
            w = layer["w_nm"] / 1000
            l = layer["l_nm"] / 1000
            geom = mp.Block(
                size=mp.Vector3(w, l, h),
                center=mp.Vector3(0, 0, h/2),
                material=mat,
            )
            geometry.append(geom)

        else:
            raise ValueError(f"Unknown shape {layer['shape']}")

        # -----------------------------------
        return geometry, sx, sy, sz
