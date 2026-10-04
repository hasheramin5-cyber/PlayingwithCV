import os
import cv2
import numpy as np

# ================= SETTINGS =================
INPUT_IMAGE = "image_assets/image1.jpg"
OUTPUT_FOLDER = "unique_versions"    # Results

NUMBER_OF_POSTER_COLORS = 6          # Poster effect
DUOTONE_DARK_COLOR = (20, 20, 80)    # BGR format: dark blue
DUOTONE_LIGHT_COLOR = (255, 230, 180) # BGR format: light cream


def save_image(image, name):
    path = os.path.join(OUTPUT_FOLDER, name)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")


def make_thermal(image):

    # Thermal / heatmap effect.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal


def make_posterized(image, num_colors=6):

    # Posterized effect.

    data = image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        data,
        num_colors,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    posterized = quantized.reshape(image.shape)

    return posterized


def make_duotone(image, dark_color, light_color):
  
    # Duotone effect.
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_normalized = gray.astype(np.float32) / 255.0

    dark = np.array(dark_color, dtype=np.float32)
    light = np.array(light_color, dtype=np.float32)

    duotone = dark + (light - dark) * gray_normalized[:, :, None]
    duotone = np.clip(duotone, 0, 255).astype(np.uint8)

    return duotone


def main():
    image = cv2.imread(INPUT_IMAGE)

    if image is None:
        print(f"Error: Could not find '{INPUT_IMAGE}'. Please provide a valid image path.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Thermal / Heatmap version
    thermal = make_thermal(image)
    save_image(thermal, "01_thermal_heatmap.png")

    # 2. Posterized / limited-color version
    posterized = make_posterized(image, NUMBER_OF_POSTER_COLORS)
    save_image(posterized, "02_posterized_colors.png")

    # 3. Duotone version
    duotone = make_duotone(
        image,
        DUOTONE_DARK_COLOR,
        DUOTONE_LIGHT_COLOR
    )
    save_image(duotone, "03_duotone.png")

    print("\nDone! 3 unique versions saved here:")
    print(os.path.abspath(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()