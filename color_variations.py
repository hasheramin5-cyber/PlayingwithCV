import os
import cv2
import numpy as np

# SETTINGS
INPUT_IMAGE = "image_assets/image1.jpg" # Put your image name or full path here
OUTPUT_FOLDER = "color_versions" # All results will be saved here


def save_image(image, name):
    path = os.path.join(OUTPUT_FOLDER, name)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """
    brightness: from -100 to +100
    contrast: from 0.5 to 2.0
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def adjust_saturation(image, factor=1.0):
    """
    factor: 0 = almost gray, 1 = original, 2 = highly saturated
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def shift_hue(image, shift=0):
    """
    shift: from -90 to +90
    In OpenCV HSV, hue ranges from 0 to 179.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def make_sepia(image):
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia = cv2.transform(image, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def make_negative(image):
    return 255 - image


def main():
    # Read the input image
    image = cv2.imread(INPUT_IMAGE)

    if image is None:
        print(f"Error: Could not find '{INPUT_IMAGE}'. Please provide a valid path.")
        return

    # Create the output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save the original image too
    save_image(image, "00_original.png")

    # Color Space Versions
    color_spaces = {
        "01_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        "02_gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        "03_hsv": cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        "04_lab": cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
        "05_ycrcb": cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb),
        "06_hls": cv2.cvtColor(image, cv2.COLOR_BGR2HLS),
        "07_xyz": cv2.cvtColor(image, cv2.COLOR_BGR2XYZ),
    }

    for name, img in color_spaces.items():
        save_image(img, f"{name}.png")

    # Brightness Versions
    save_image(adjust_brightness_contrast(image, brightness=50), "08_brightness_plus50.png")
    save_image(adjust_brightness_contrast(image, brightness=-50), "09_brightness_minus50.png")

    # Contrast Versions
    save_image(adjust_brightness_contrast(image, contrast=1.5), "10_contrast_high.png")
    save_image(adjust_brightness_contrast(image, contrast=0.6), "11_contrast_low.png")

    # Saturation Versions
    save_image(adjust_saturation(image, factor=0.0), "12_saturation_zero.png")
    save_image(adjust_saturation(image, factor=0.5), "13_saturation_low.png")
    save_image(adjust_saturation(image, factor=2.0), "14_saturation_high.png")

    # Hue Shift Versions
    save_image(shift_hue(image, shift=30), "15_hue_shift_30.png")
    save_image(shift_hue(image, shift=60), "16_hue_shift_60.png")
    save_image(shift_hue(image, shift=90), "17_hue_shift_90.png")
    save_image(shift_hue(image, shift=120), "18_hue_shift_120.png")
    save_image(shift_hue(image, shift=150), "19_hue_shift_150.png")

    save_image(make_sepia(image), "20_sepia.png")
    save_image(make_negative(image), "21_negative.png")

    print("\nDone! All versions were saved here:")
    print(os.path.abspath(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()