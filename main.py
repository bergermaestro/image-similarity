import compare_images
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


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


def compute_scores_by_group(logos, test_image, method="ssim"):
    """
    Compute similarity scores grouped by top-level folder using the specified matching method.
    Supported methods: "ssim", "orb", "hash", "resnet".
    """
    grouped_scores = defaultdict(list)

    for logo in logos:
        group_name = Path(logo).parts[1]

        score = compare_images.resnet_similarity(logo, test_image)

        # if method == "ssim":
        #     score = compare_images.compute_ssim(logo, test_image)
        # elif method == "orb":
        #     score = compare_images.match_images(logo, test_image)
        # elif method == "hash":
        #     score = compare_images.compare_hashes(logo, test_image)
        # elif method == "resnet":
        #     score = compare_images.resnet_similarity(logo, test_image)
        # else:
        #     raise ValueError(f"Unknown method: {method}")

        grouped_scores[group_name].append(score)

    return grouped_scores


def plot_grouped_scores(grouped_scores):
    """
    Plots similarity score distributions for each group.
    """
    plt.figure(figsize=(10, 6))

    for group_name, scores in grouped_scores.items():
        plt.hist(scores, bins=20, alpha=0.6, label=group_name)

    plt.xlabel("SSIM Score")
    plt.ylabel("Frequency")
    plt.title("Similarity Scores by Group")
    plt.legend()
    plt.show()


def main():
    logo_dirs = ["augmented_canada_logos", "american_airlines", "Dollar Tree", "Durea"]
    # logo_dirs = ["augmented_canada_logos", "american_airlines"]
    logos = []
    for logo_dir in logo_dirs:
        logos.extend(import_folder(Path("logos") / logo_dir))

    print(f"Found {len(logos)} logos in the directories.")

    test_image = "logos/canada/canada_1.png"

    # methods = ["ssim", "orb", "hash", "resnet"]
    methods = ["resnet"]

    for method in methods:
        print(f"\n=== {method.upper()} Results ===")
        grouped_scores = compute_scores_by_group(logos, test_image, method=method)
        plot_grouped_scores(grouped_scores)


if __name__ == "__main__":
    main()
