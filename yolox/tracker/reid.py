import os
import sys
import importlib.util
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


_REID_EXTRACTOR_CACHE = {}


def _add_local_torchreid():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torchreid_root = os.path.join(root, "deep-person-reid")
    if os.path.isdir(torchreid_root) and torchreid_root not in sys.path:
        sys.path.insert(0, torchreid_root)


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _add_local_fastreid():
    root = _project_root()
    fastreid_root = os.path.join(root, "fast-reid")
    if not os.path.isdir(fastreid_root):
        raise FileNotFoundError("FastReID repo not found: {}".format(fastreid_root))
    if fastreid_root not in sys.path:
        sys.path.insert(0, fastreid_root)

    import fastreid

    # Keep compatibility with code that imports fast_reid.fastreid while the
    # local repo and its internal imports use the canonical fastreid package.
    if "fast_reid" not in sys.modules:
        package = types.ModuleType("fast_reid")
        package.__path__ = [fastreid_root]
        sys.modules["fast_reid"] = package
    sys.modules["fast_reid"].fastreid = fastreid
    sys.modules["fast_reid.fastreid"] = fastreid
    return fastreid_root


class ReIDExtractor(object):
    def __init__(
        self,
        model_name="osnet_x1_0",
        model_path="",
        device="cuda",
        image_size=(256, 128),
    ):
        _add_local_torchreid()
        from torchreid.models import build_model
        from torchreid.utils import check_isfile, load_pretrained_weights

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = build_model(
            model_name,
            num_classes=1,
            pretrained=False,
            use_gpu=self.device.type == "cuda",
        )
        if model_path:
            if not check_isfile(model_path):
                raise FileNotFoundError("ReID model path not found: {}".format(model_path))
            load_pretrained_weights(self.model, model_path)
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, crops):
        if len(crops) == 0:
            return np.empty((0, 0), dtype=np.float32)

        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensors.append(self.preprocess(image))
        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
            features = F.normalize(features, dim=1)
        return features.detach().cpu().numpy()

    def extract(self, frame, tlbrs):
        crops = crop_tlbrs(frame, tlbrs)
        if len(crops) == 0:
            return np.empty((0, 0), dtype=np.float32)
        return self(crops)


class FastReIDExtractor(object):
    def __init__(
        self,
        config_file,
        model_path,
        device="cuda",
        batch_size=16,
    ):
        fastreid_root = _add_local_fastreid()
        interface_path = os.path.join(fastreid_root, "fast_reid_interfece.py")
        if not os.path.isfile(interface_path):
            raise FileNotFoundError("FastReID interface not found: {}".format(interface_path))
        if not config_file:
            raise ValueError("--fast-reid-config is required when using --fast-reid")
        if not model_path:
            raise ValueError("--fast-reid-weights or --reid-model-path is required when using --fast-reid")
        if not os.path.isfile(config_file):
            raise FileNotFoundError("FastReID config not found: {}".format(config_file))
        if not os.path.isfile(model_path):
            raise FileNotFoundError("FastReID weights not found: {}".format(model_path))

        spec = importlib.util.spec_from_file_location("fast_reid_interfece", interface_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.FastReIDInterface(config_file, model_path, device, batch_size=batch_size)

    def extract(self, frame, tlbrs):
        if isinstance(frame, str):
            frame = cv2.imread(frame)
        if frame is None or len(tlbrs) == 0:
            return np.empty((0, 0), dtype=np.float32)
        height, width = frame.shape[:2]
        features = [None] * len(tlbrs)
        valid_boxes = []
        valid_indices = []
        for idx, tlbr in enumerate(np.asarray(tlbrs, dtype=np.float32)):
            x1, y1, x2, y2 = tlbr[:4]
            x1 = max(0, min(width - 1, int(np.floor(x1))))
            y1 = max(0, min(height - 1, int(np.floor(y1))))
            x2 = max(0, min(width, int(np.ceil(x2))))
            y2 = max(0, min(height, int(np.ceil(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            valid_indices.append(idx)
            valid_boxes.append([x1, y1, x2, y2])

        if len(valid_boxes) == 0:
            return features

        extracted = self.model.inference(frame, np.asarray(valid_boxes, dtype=np.float32))
        for idx, feat in zip(valid_indices, extracted):
            features[idx] = np.asarray(feat, dtype=np.float32)
        return features


def build_reid_extractor(args):
    backend = getattr(args, "reid_backend", "deep")
    if backend == "fast":
        fast_weights = getattr(args, "fast_reid_weights", "") or getattr(args, "reid_model_path", "")
        cache_key = (
            "fast",
            getattr(args, "fast_reid_config", ""),
            fast_weights,
            getattr(args, "reid_device", "cuda"),
            getattr(args, "fast_reid_batch_size", 16),
        )
        if cache_key not in _REID_EXTRACTOR_CACHE:
            _REID_EXTRACTOR_CACHE[cache_key] = FastReIDExtractor(
                config_file=getattr(args, "fast_reid_config", ""),
                model_path=fast_weights,
                device=getattr(args, "reid_device", "cuda"),
                batch_size=getattr(args, "fast_reid_batch_size", 16),
            )
        return _REID_EXTRACTOR_CACHE[cache_key]
    if backend == "deep":
        cache_key = (
            "deep",
            getattr(args, "reid_model", "osnet_x1_0"),
            getattr(args, "reid_model_path", ""),
            getattr(args, "reid_device", "cuda"),
        )
        if cache_key not in _REID_EXTRACTOR_CACHE:
            _REID_EXTRACTOR_CACHE[cache_key] = ReIDExtractor(
                model_name=getattr(args, "reid_model", "osnet_x1_0"),
                model_path=getattr(args, "reid_model_path", ""),
                device=getattr(args, "reid_device", "cuda"),
            )
        return _REID_EXTRACTOR_CACHE[cache_key]
    raise ValueError("Unsupported ReID backend: {}".format(backend))


def crop_tlbrs(frame, tlbrs):
    if isinstance(frame, str):
        frame = cv2.imread(frame)
    if frame is None:
        return []

    height, width = frame.shape[:2]
    crops = []
    for tlbr in tlbrs:
        x1, y1, x2, y2 = tlbr
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width, int(round(x2))))
        y2 = max(0, min(height, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            crops.append(np.zeros((2, 2, 3), dtype=np.uint8))
        else:
            crops.append(frame[y1:y2, x1:x2].copy())
    return crops
