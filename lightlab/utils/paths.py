from pathlib import Path
from glob import glob

ROOT = Path(__file__).parent.parent
ASSETS_DIR = ROOT / "assets"
CFG_DIR = ASSETS_DIR / "cfgs"
PRETRAIN_DIR = ASSETS_DIR / "pretrain"

MODELS_CFG_PATH = {}
for file in glob(f"{CFG_DIR}/*/*.yaml"):
    MODELS_CFG_PATH[Path(file).stem] = file

PRETRAIN_PATH = {}
for file in glob(f"{PRETRAIN_DIR}/*.pth"):
    PRETRAIN_PATH[Path(file).stem] = file

if __name__ == "__main__":
    print(ROOT)
    print(CFG_DIR)
    print(MODELS_CFG_PATH)
    print(PRETRAIN_DIR)
    print(PRETRAIN_PATH)
