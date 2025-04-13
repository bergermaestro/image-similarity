from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn.functional as F

from cleanup import cleanup_image


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


def get_embedding(image_path):
    img = cleanup_image(image_path, output_size=(224, 224))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).view(-1)  # insead of .squeeze() to flatten input
    return embedding


def resnet_similarity(img1_path, img2_path):
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)

    # Compute cosine similarity
    similarity = F.cosine_similarity(emb1, emb2, dim=0)
    return similarity.item()
