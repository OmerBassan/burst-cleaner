# platform/windows_embeddings.py
from typing import Optional, Tuple, TYPE_CHECKING
from types import ModuleType
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from ..core.embeddings_core import EmbeddingBackend

if TYPE_CHECKING:  # hinting only; avoids hard dependency at import time
    import torchvision.models as _tv_models
    import torchvision.transforms as _tv_transforms


def _lazy_load_torchvision() -> Tuple[ModuleType, ModuleType]:
    """
    Import torchvision lazily so editors don't flag missing optional deps.
    Raises a clear error at runtime if torchvision is not installed.
    """
    try:
        import torchvision.models as tv_models
        import torchvision.transforms as tv_transforms
    except ImportError as exc:  # pragma: no cover - dependency management
        raise ImportError(
            "torchvision is required for DesktopTorchEmbeddingBackend. "
            "Install it with `pip install torchvision`."
        ) from exc
    return tv_models, tv_transforms


class DesktopTorchEmbeddingBackend(EmbeddingBackend):
    """
    Desktop implementation using torchvision (ResNet).
    Suitable for Windows/Linux/Mac with Python + PyTorch.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        models, transforms = _lazy_load_torchvision()
        self._tv_models = models
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name, embedding_dim).to(self.device)
        self.model.eval()

        # סטנדרט ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self, model_name: str, embedding_dim: Optional[int]) -> nn.Module:
        """
        Load a pretrained model and strip off the classification head,
        so the output is a feature vector.
        """
        models = self._tv_models
        if model_name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # מסירים את הראש (fc) ומשאירים את ה־backbone
            modules = list(base.children())[:-1]  # remove last FC
            model = nn.Sequential(*modules)
        elif model_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            modules = list(base.children())[:-1]
            model = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # אם תרצה PCA/Projection בעתיד – אפשר להוסיף כאן
        return model

    def _load_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0)  # shape: (1, C, H, W)

    def embed_single(self, image_id: str) -> np.ndarray:
        """
        image_id כאן הוא path על הדיסק.
        """
        with torch.no_grad():
            x = self._load_image(image_id).to(self.device)
            feat = self.model(x)          # shape: (1, F, 1, 1) or (1, F)
            feat = torch.flatten(feat, 1)  # (1, F)
            vec = feat[0].cpu().numpy().astype("float32")
        return vec

    # במידת הצורך אפשר override ל־embed_batch כדי לעשות inference ב־batch
