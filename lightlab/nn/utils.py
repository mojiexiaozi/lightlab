import ast
from torch import nn
import torch
import timm
import yaml
from lightlab.utils.torch_utils import make_divisible, intersect_dicts
from lightlab.utils.paths import MODELS_CFG_PATH, PRETRAIN_PATH
from lightlab.nn.modules import (
    Conv,
    Classify,
    Concat,
    C2f,
    SPPF,
    Detect,
    SkipConn,
    Semantic,
    OBB,
    Pose,
)


def load_yaml_model(cfg_path: str):
    file = MODELS_CFG_PATH.get(cfg_path) or cfg_path
    if file is None:
        raise FileNotFoundError(f"Model config file not found: {cfg_path}")
    with open(file) as f:
        return yaml.safe_load(f)


def parse_model(cfg: dict, scale="n", verbose=True):
    nc, scales, kpt_shape = (cfg.get(x) for x in ("nc", "scales", "kpt_shape"))
    skip_make_divisible = cfg.get("skip_make_divisible", False)
    if scales:
        depth, width, max_channels = scales[scale]
    else:
        depth, width, max_channels = 1.0, 1.0, float("inf")

    if verbose:
        print(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )
    ch = [3]
    layers, save, c2 = [], [], ch[-1]
    pre_len = 0
    for i, (f, n, m, args) in enumerate(cfg["backbone"] + cfg["head"]):
        # from, repeats, module, args
        if "nn." in m:
            m = getattr(nn, m[3:])
        elif "tf_" in m:
            pass
        else:
            m = globals()[m]

        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except Exception as e:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Conv, Classify, C2f, SPPF, Semantic):
            c1, c2 = ch[f], args[0]
            if c2 != nc and not skip_make_divisible:
                c2 = make_divisible(c2 * width, 8)
            args = [c1, c2, *args[1:]]
            if m in [C2f]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is SkipConn:
            c2 = ch[f[0]]
        elif m in (Detect, OBB, Pose):
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        if isinstance(m, str):
            # timm model
            t = m  # module type
            m_: nn.Module = timm.create_model(
                m,
                pretrained=False,
                features_only=True,
                exportable=True,
                scriptable=True,
            )
            if len(args) > 0 and PRETRAIN_PATH.get(args[0]):
                state_dict = torch.load(PRETRAIN_PATH.get(args[0]), map_location="cpu")
                csd = m_.float().state_dict()
                csd = intersect_dicts(csd, state_dict)  # intersect
                m_.load_state_dict(csd)
                if verbose:
                    print(
                        f"Transferred {len(csd)}/{len(m_.state_dict())} items from pretrained weights"
                    )
            if len(args) > 1:
                m_._stage_out_idx = {s: i for i, s in enumerate(args[1])}
            # 获取通道数
            features = m_(torch.randn(1, 3, 224, 224))
            ch = [p.shape[1] for p in features]
            m_.timm = True
            pre_len = len(ch) - 1
        else:
            # create module instance
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            m.timm = False  # timm model flag
            ch = [] if i == 0 else ch  # 去掉图像输入通道数
            ch.append(c2)

        # number of parameters
        m_.np = sum(x.numel() for x in m_.parameters())
        # attach index, 'from' index, type
        m_.i, m_.f, m_.type = i + pre_len, f, t
        if verbose:
            print(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")
        save.extend(
            x if x >= 0 else m_.i + x
            for x in ([f] if isinstance(f, int) else f)
            if x != -1
        )  # append to savelist
        layers.append(m_)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    import yaml
    from lightlab.utils.paths import ASSETS_DIR

    cfg_path = ASSETS_DIR / "cfgs" / "yolov8" / "yolov8-cls.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        parse_model(cfg, "n")
