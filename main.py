from tokenize import group
import compare_images
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import split_logos


def import_folder(folder_path):
    """
    Recursively imports all images from a given folder and its subfolders,
    returning a list of image paths.
    """
    import os

    image_paths = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_paths.append(Path(root) / filename)
    return image_paths


def compute_scores(logos, test_image, method="ssim"):
    scores = []
    for logo in logos:
        score = compare_images.resnet_similarity(logo, test_image)
        scores.append((score, logo))

    return scores


def plot_grouped_scores(grouped_scores):
    """
    Plots similarity score distributions for each group.
    """
    plt.figure(figsize=(10, 6))

    for group_name, scores in grouped_scores.items():
        only_scores = [score for score, _ in scores]
        plt.hist(only_scores, bins=20, alpha=0.6, label=group_name)

    plt.xlabel("SSIM Score")
    plt.ylabel("Frequency")
    plt.title("Similarity Scores by Group")
    plt.legend()
    plt.show()


def main():
    REFERENCE_IMAGE = "logos/canada/canada_1.png"
    logo_dirs = [
        "augmented_canada_logos",
        "american_airlines",
        "Durea",
        "Embraer",
        "Esso",
    ]
    # logo_dirs = ["augmented_canada_logos", "american_airlines"]
    logos = []
    for logo_dir in logo_dirs:
        logos.extend(import_folder(Path("logos") / logo_dir))

    print(f"Found {len(logos)} logos in the directories.")

    logos_split = []
    for logo in logos:
        s_logos, _ = split_logos.split_logos_from_file(logo)
        logos_split.extend(s_logos)

    print(f"Found {len(logos_split)} logos after splitting.")

    grouped_scores = defaultdict(list)
    for logo in logos_split:
        group_name = logo.parent.name
        score = compare_images.resnet_similarity(logo, REFERENCE_IMAGE)
        grouped_scores[group_name].append((score, logo))

    plot_grouped_scores(grouped_scores)

    # Sort scores within each group
    canada_scores = grouped_scores["augmented_canada_logos"]
    other_scores = [
        pair
        for group, scores in grouped_scores.items()
        if group != "augmented_canada_logos"
        for pair in scores
    ]

    bottom_5_canada = sorted(canada_scores, key=lambda x: x[0])[:5]
    top_5_others = sorted(other_scores, key=lambda x: x[0], reverse=True)[:5]

    print("Bottom 5 matches in 'augmented_canada_logos':")
    for score, path in bottom_5_canada:
        print(f"{path}: {score:.4f}")

    print("Top 5 matches in other folders:")
    for score, path in top_5_others:
        print(f"{path}: {score:.4f}")


if __name__ == "__main__":
    main()
