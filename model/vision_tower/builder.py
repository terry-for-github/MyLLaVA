from .clip_tower import CLIPVisionTower


def build_vision_tower(vision_tower_type: str,
                       mm_vision_select_layer: int,
                       mm_vision_select_feature: str):
    if "clip" in vision_tower_type and \
        (vision_tower_type.startswith("openai") or
            vision_tower_type.startswith("laion")):
        return CLIPVisionTower(vision_tower_type, mm_vision_select_layer,
                               mm_vision_select_feature)
