import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
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


def preprocess_image(image_path, output_size=(256, 256)):
    img = Image.open(image_path).convert("RGBA")  # Ensure RGBA mode

    # Create a white background image
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # Paste the image onto the white background (handles transparency)
    white_bg.paste(img, (0, 0), img)

    # Convert back to RGB (remove alpha channel)
    img_rgb = white_bg.convert("RGB")

    # Resize for consistency
    img_rgb = img_rgb.resize(output_size)

    file_name = image_path.split("/")[-1]

    img_rgb.save(f"tmp/{file_name}")

    return img_rgb


def match_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)  # Read as grayscale
    img2 = cv2.imread(img2_path, 0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # print("matches", matches)

    return len(matches)


def draw_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.show()


def compute_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))  # Resize for consistency

    score, _ = ssim(img1, img2, full=True)
    return score


def compare_hashes(img1_path, img2_path):
    hash1 = imagehash.phash(Image.open(img1_path))
    hash2 = imagehash.phash(Image.open(img2_path))
    return 1 - (hash1 - hash2) / len(hash1.hash) ** 2  # Normalize similarity


def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
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
