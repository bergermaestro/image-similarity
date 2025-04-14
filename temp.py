from pathlib import Path
import compare_images
import utils


def main():
    REFERENCE_IMAGE_PATH = Path("logos/canada/canada_1.png")
    REFERENCE_IMAGE = utils.load_image(REFERENCE_IMAGE_PATH)

    logo = utils.load_image(Path("logos/canada/canada_1.png"))

    score = compare_images.resnet_similarity(logo, REFERENCE_IMAGE)

    print(f"Score: {score:.4f}")


if __name__ == "__main__":
    main()
