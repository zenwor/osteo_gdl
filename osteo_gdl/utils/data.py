import os
from typing import List, Tuple, Union

import pandas as pd

from osteo_gdl.utils.const import FEATS_COLS


# Formatting
def fmt_name(name: str) -> str:
    if "Case 48" not in name:
        return name.replace(" ", "-")
    else:
        return name.replace("Case 48 - P5 ", "Case-48-P5-")


def fmt_cls(cls: str, cls_map: dict) -> Union[str, None]:
    return cls_map.get(cls, None)


# Images
def load_imgs(path: str) -> List[str]:
    imgs = []
    path = os.path.expandvars(path)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".jpg"):
                imgs.append(os.path.join(root, file))
    return imgs


# Features
def load_feats(path: str, cls_map: dict) -> pd.DataFrame:
    path = os.path.expandvars(path)

    feats = pd.read_csv(path)
    feats = feats[FEATS_COLS]
    feats["image.name"] = feats["image.name"].apply(fmt_name)
    feats["classification"] = feats["classification"].apply(
        lambda x: fmt_cls(x, cls_map)
    )
    feats = feats.dropna()
    return feats


def load_imgs_feats(
    imgs_path: str, feats_path: str, cls_map: dict
) -> Tuple[List[str], pd.DataFrame]:
    return load_imgs(imgs_path), load_feats(feats_path, cls_map)
