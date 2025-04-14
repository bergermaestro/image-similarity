from typing import cast
from torchvision import models, transforms
import torch
import torch.nn.functional as F
from cv2.typing import MatLike
from torch.types import Number

from utils import cv2_to_pil

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final classifier
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean/std
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def get_embedding(image: MatLike):
    pil_image = cv2_to_pil(image)
    tensor = cast(torch.Tensor, transform(pil_image))
    tensor = tensor.unsqueeze(0)  # Remove batch dimension
    with torch.no_grad():
        embedding = model(tensor).view(-1)  # insead of .squeeze() to flatten input
    return embedding


def resnet_similarity(img1: MatLike, img2: MatLike) -> Number:
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    # Compute cosine similarity
    similarity = F.cosine_similarity(emb1, emb2, dim=0)
    return similarity.item()
