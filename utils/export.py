from omegaconf import OmegaConf
import argparse
import os
import torch
import importlib.util
import sys

from models.pose_estimation_model import PoseEstimationModel

def parse_args():
    parser = argparse.ArgumentParser(
        description='Export to TorchScript using trace (supports .pt/.pth and .ckpt)'
    )
    parser.add_argument(
        '--cfg', required=True,
        help='Path to model config file (YAML)'
    )
    parser.add_argument(
        '--ext', default='ckpt',
        help="Weight file extension: 'pt', 'pth' or 'ckpt'"
    )
    return parser.parse_args()


def make_output_path(weights_path):
    dirname = os.path.dirname(weights_path)
    basename = os.path.basename(weights_path)
    return os.path.join(dirname, 'x' + basename)


def load_model_class(model_file, class_name):
    # Dynamically load a module from file and retrieve the named class
    spec = importlib.util.spec_from_file_location('model_module', model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules['model_module'] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def export_pt(cfg):
    m = cfg.model
    weights = m.pretrained
    if not weights or not weights.endswith(('.pt', '.pth')):
        raise ValueError("Set 'model.pretrained' to your .pt/.pth file path in the config.")
    print(f"Loading PT weights from {weights}")

    ModelClass = load_model_class(m.model_file, m.name)
    init_args = {k: v for k, v in dict(m).items()
                 if k not in ['model_file', 'name', 'pretrained']}
    model = ModelClass(**init_args)

    sd = torch.load(weights, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # strip any "model." prefixes
    sd = {k.replace('model.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    w, h = m.image_size
    dummy = torch.randn(1, 3, h, w)
    print(f"Tracing with dummy input (1,3,{h},{w})...")
    traced = torch.jit.trace(model, dummy)

    out = make_output_path(weights)
    traced.save(out)
    print(f"✅ TorchScript saved to {out}")


def export_ckpt(cfg):
    m = cfg.model
    weights = m.pretrained
    if not weights or not weights.endswith('.ckpt'):
        raise ValueError("Set 'model.pretrained' to your .ckpt file path in the config.")
    print(f"Loading checkpoint manually from {weights}")

    model = PoseEstimationModel.load_from_checkpoint(checkpoint_path=cfg.model.pretrained, cfg=cfg, strict=False)
    model = model.model.cpu()
    model.eval()

    w, h = m.image_size
    dummy = torch.randn(1, 3, h, w)
    print(f"Tracing with dummy input (1,3,{h},{w})...")
    traced = torch.jit.trace(model, dummy)

    out = make_output_path(weights)
    traced.save(out)
    print(f"✅ TorchScript saved to {out}")


if __name__ == '__main__':
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    ext = args.ext.lower()

    if ext == 'ckpt':
        export_ckpt(cfg)
    elif ext in ['pt', 'pth']:
        export_pt(cfg)
    else:
        print(f"[ERROR] '{args.ext}' not supported. Use 'ckpt', 'pt' or 'pth'.")
