import cv2
import os

def main():
    input_path = input("Enter the path to the input image: ").strip()

    if not os.path.isfile(input_path):
        print("ERROR: File not found. Please check the path and try again.")
        return

    image = cv2.imread(input_path)

    if image is None:
        print("ERROR: Couldn't load the image. The file may be corrupted or in an unsupported format.")
        return

    print("\nImage loaded successfully.")

    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    cv2.imshow('Hue-Lightness-Saturation (HLS)', hls)
    print("Press any key on the image window to close it")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = input("\nEnter the output file name (e.g., output_hls.jpg): ").strip()
    if not output_path:
        output_path = "output_hls.jpg"
        print("No output name given. Using default: output_hls.jpg")

    success = cv2.imwrite(output_path, hls)

    if success:
        h, w, c = hls.shape
        print(f"\nImage Details:")
        print(f"Height: {h}")
        print(f"Width: {w}")
        print(f"Color Channels: {c}")
        print(f"\nImage saved as: {output_path}")
    else:
        print("ERROR: Failed to save the image.")

if __name__ == "__main__":
    main()
