from typing import DefaultDict, Tuple, List

from matplotlib.pylab import f
import cleanup
import compare_images
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import logo_splitter
import utils
from torch.types import Number


def plot_grouped_scores(grouped_scores: DefaultDict[str, List[Number]]) -> None:
    """
    Plots similarity score distributions for each group.
    """
    plt.figure(figsize=(10, 6))

    for (
        group_name,
        scores,
    ) in grouped_scores.items():
        only_scores = [score for score in scores]
        plt.hist(only_scores, bins=20, alpha=0.6, label=group_name)

    plt.xlabel("SSIM Score")
    plt.ylabel("Frequency")
    plt.title("Similarity Scores by Group")
    plt.legend()
    plt.show()


def main():
    REFERENCE_IMAGE_PATH = Path("sample_logos/canada/canada_1.png")
    REFERENCE_IMAGE = cleanup.cleanup_image(
        utils.load_image(REFERENCE_IMAGE_PATH), output_size=(256, 256)
    )

    input_dirs = [
        "augmented_canada_logos",
        "american_airlines",
        "Durea",
        "Embraer",
        "Esso",
    ]
    # logo_dirs = ["augmented_canada_logos", "american_airlines"]
    logo_dirs: list[Tuple[str, list[Path]]] = []
    for dir in input_dirs:
        logo_dirs.append((dir, utils.import_folder(Path("logos") / dir)))

    grouped_scores: DefaultDict[str, List[float]] = defaultdict(list)

    for dir_name, logo_paths in logo_dirs:
        for logo_path in logo_paths:
            logo = utils.load_image(logo_path)

            # For checking if Canada flag
            red_parts = keep_red_parts(logo, output_path=Path(output_path))
            red_logo_with_bg = cleanup.add_white_bg(red_parts, save_image=False)
            red_resized_logo = cleanup.fit_to_square(
                red_logo_with_bg, output_size=(256, 256), save_image=False
            )
            red_split_logos, _ = logo_splitter.split_logo(red_resized_logo)

            # For all other logos
            logo_with_bg = cleanup.add_white_bg(logo, save_image=False)
            resized_logo = cleanup.fit_to_square(
                logo_with_bg, output_size=(256, 256), save_image=False
            )
            split_logos, _ = logo_splitter.split_logo(resized_logo)

            print(f"Processing {logo_path}..., {len(split_logos)} logos found")

            for split_logo in split_logos:
                # Compare each split logo with the reference image
                split_logo_fitted = cleanup.fit_to_square(
                    split_logo, output_size=(256, 256)
                )
                score = compare_images.resnet_similarity(
                    split_logo_fitted, REFERENCE_IMAGE
                )
                grouped_scores[dir_name].append(float(score))

            score = compare_images.resnet_similarity(
                cleanup.cleanup_image(logo), REFERENCE_IMAGE
            )
            grouped_scores[dir_name].append(float(score))

    plot_grouped_scores(grouped_scores)

    # # Sort scores within each group
    # canada_scores = grouped_scores["augmented_canada_logos"]
    # other_scores = [
    #     pair
    #     for group, scores in grouped_scores.items()
    #     if group != "augmented_canada_logos"
    #     for pair in scores
    # ]

    # bottom_5_canada = sorted(canada_scores)[:5]
    # top_5_others = sorted(other_scores reverse=True)[:5]

    # print("Bottom 5 matches in 'augmented_canada_logos':")
    # for score, path in bottom_5_canada:
    #     print(f"{path}: {score:.4f}")

    # print("Top 5 matches in other folders:")
    # for score, path in top_5_others:
    #     print(f"{path}: {score:.4f}")


if __name__ == "__main__":
    main()
