# structure_lang/validator.py

class StructureValidator:
    """
    验证结构是否“物理合法 + 可被 Meep 建模”
    """

    def __init__(self, min_feature_nm=20, margin_nm=30):
        self.min_feature = min_feature_nm
        self.margin = margin_nm

    def validate(self, struct_dict):
        P_x, P_y = struct_dict["P"]
        if P_x is None or P_y is None:
            return False, "missing_period"

        layer = struct_dict["layer1"]

        # material
        if layer["mat"] is None:
            return False, "missing_material"

        # height
        H = layer["h_nm"]
        if H is None or H < self.min_feature:
            return False, "bad_height"

        # shape
        shape = layer["shape"]
        if shape not in ["CYL", "RECT"]:
            return False, "bad_shape"

        # CYL case
        if shape == "CYL":
            R = layer["r_nm"]
            if R is None or R < self.min_feature:
                return False, "bad_radius"
            if R > (P_x/2 - self.margin):
                return False, "radius_too_large"

        # RECT case
        if shape == "RECT":
            W = layer["w_nm"]
            L = layer["l_nm"]
            if (W is None) or (L is None):
                return False, "missing_rect_dims"
            if W < self.min_feature or L < self.min_feature:
                return False, "too_small_rect"
            if W > (P_x - 2*self.margin):
                return False, "rect_width_exceed"
            if L > (P_y - 2*self.margin):
                return False, "rect_length_exceed"

        return True, "ok"
