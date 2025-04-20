from pathlib import Path
import cleanup
import compare_images
import logo_splitter
import utils


def main():
    REFERENCE_IMAGE_PATH = Path("sample_logos/pei/pei.png")
    REFERENCE_IMAGE = cleanup.cleanup_image(
        utils.load_image(REFERENCE_IMAGE_PATH), output_size=(256, 256)
    )

    logo = utils.load_image(Path("sample_logos/pei/pei_wordmark_eng_91x69.png"))

    logo_with_bg = cleanup.add_white_bg(logo, save_image=False)
    resized_logo = cleanup.fit_to_square(
        logo_with_bg, output_size=(256, 256), save_image=False
    )

    utils.save_cv2_image(
        resized_logo, Path("sample_logos/pei/pei_wordmark_eng_91x69_resized.png")
    )

    split_logos, boxes = logo_splitter.split_logo(resized_logo, draw_boxes=True)

    if boxes is not None:
        utils.save_cv2_image(boxes, Path("pei_wordmark_eng_91x69_boxes.png"))

    print(f"Processing... {len(split_logos)} logos found")

    for split_logo in split_logos:
        # Compare each split logo with the reference image
        split_logo_fitted = cleanup.fit_to_square(split_logo, output_size=(256, 256))

        utils.save_cv2_image(
            split_logo_fitted,
            Path(
                f"split_sample_logos/pei_wordmark_eng_91x69_split_{id(split_logo_fitted)}.png"
            ),
        )

        score = compare_images.resnet_similarity(split_logo_fitted, REFERENCE_IMAGE)
        print(score)

    score = compare_images.resnet_similarity(
        cleanup.cleanup_image(logo), REFERENCE_IMAGE
    )
    print("the score is")
    print(score)


if __name__ == "__main__":
    main()
