from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn.functional as F


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


def preprocess_image(image_path: Path, output_size=(256, 256)):
    img = Image.open(image_path).convert("RGBA")  # Ensure RGBA mode

    # Create a white background image with the desired output size
    white_bg = Image.new("RGBA", output_size, (255, 255, 255, 255))

    # Scale the image while maintaining aspect ratio
    img.thumbnail(output_size, Image.Resampling.LANCZOS)

    # Calculate position to center the image
    x_offset = (output_size[0] - img.size[0]) // 2
    y_offset = (output_size[1] - img.size[1]) // 2

    # Paste the scaled image onto the white background
    white_bg.paste(img, (x_offset, y_offset), img)

    # Convert back to RGB (remove alpha channel)
    img_rgb = white_bg.convert("RGB")

    file_name = str(image_path).split("/")[-1]

    print("preprocessing image:", file_name)

    Path("tmp").mkdir(parents=True, exist_ok=True)
    img_rgb.save(f"tmp/{file_name}")

    return img_rgb


def get_embedding(image_path):
    img = preprocess_image(image_path, output_size=(224, 224))
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
