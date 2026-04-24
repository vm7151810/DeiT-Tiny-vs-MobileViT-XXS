import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import yaml

class EverydayDataset(Dataset):
    def __init__(self, rows_df: pd.DataFrame, split: str, transform=None, attr_names: List[str] = None):
        self.df = rows_df[rows_df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.attr_names = attr_names or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.loc[idx]
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except Exception as e:
            # Fallback for missing images in development
            img = Image.new('RGB', (224, 224), color='gray')
            
        if self.transform:
            img = self.transform(img)
            
        class_idx = int(r["class_idx"])
        
        # attribute targets
        attrs = [r["attrs_idx"][a] for a in self.attr_names]
        attrs = torch.tensor(attrs, dtype=torch.long)
        
        return img, class_idx, attrs

def parse_attrs_field(s: str) -> Dict[str, str]:
    """Parses attribute string format 'key:val;key2:val2'"""
    out = {}
    if not isinstance(s, str):
        return out
    for kv in s.split(";"):
        if ":" in kv:
            k, v = kv.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def prepare_dataframe(labels_csv: Path, attrs_yaml: Path, classes_file: Path, image_root: Path):
    df = pd.read_csv(labels_csv)
    with open(attrs_yaml, "r") as f:
        attrs_spec = yaml.safe_load(f)

    with open(classes_file, "r") as f:
        classes = [x.strip() for x in f.readlines() if x.strip()]
    
    class2idx = {c: i for i, c in enumerate(classes)}
    attr_names = list(attrs_spec.keys())
    attr_value2idx = {}
    
    for a in attr_names:
        vals = list(attrs_spec[a])
        if "unknown" not in vals:
            vals = vals + ["unknown"]
        attr_value2idx[a] = {v: i for i, v in enumerate(vals)}

    rows = []
    for _, r in df.iterrows():
        # Clean path logic
        image_path = Path(r["image_path"].split("/")[-1]) 
        
        class_label = r["class_label"].split("_")[0]
        if class_label not in class2idx:
            continue
            
        parsed = parse_attrs_field(r.get("attributes", ""))
        attr_idx_map = {}
        for a in attr_names:
            v = parsed.get(a, "unknown")
            attr_idx_map[a] = attr_value2idx[a].get(v, attr_value2idx[a]["unknown"])
            
        rows.append({
            "image_path": r["image_path"],
            "class_idx": class2idx[class_label],
            "split": r.get("split", "train"),
            "attrs_idx": attr_idx_map,
        })
        
    return pd.DataFrame(rows), classes, attr_names, attr_value2idx
