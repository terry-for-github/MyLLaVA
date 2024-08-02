from .clip_tower import CLIPVisionTower


def build_vision_tower(vision_tower_type: str, select_layer: int, select_feature: str):
    if "clip" in vision_tower_type and (vision_tower_type.startswith("openai") or
                                        vision_tower_type.startswith("laion")):
        return CLIPVisionTower(vision_tower_type, select_layer, select_feature)
