import time
from pathlib import Path

import cv2
import numpy as np


# Image Utilities

def letterbox(
    image,
    target_size=(640, 640),
    color=(114, 114, 114)
):
    # Resize while preserving aspect ratio and add padding
    target_w, target_h = target_size
    original_h, original_w = image.shape[:2]

    scale = min(
        target_w / original_w,
        target_h / original_h
    )

    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, (left, top)


def resize_keep_aspect(image, width=None, height=None):
    # Resize an image while preserving its aspect ratio
    original_h, original_w = image.shape[:2]

    if width is None and height is None:
        return image.copy()

    if width is not None:
        scale = width / original_w
        new_size = (
            int(width),
            int(round(original_h * scale))
        )
    else:
        scale = height / original_h
        new_size = (
            int(round(original_w * scale)),
            int(height)
        )

    return cv2.resize(
        image,
        new_size,
        interpolation=cv2.INTER_AREA
    )


def resize_to_fit(image, max_width=1280, max_height=720):
    # Resize an image so it fits inside the given dimensions
    h, w = image.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    if scale == 1.0:
        return image.copy()

    return cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def normalize_image(image, scale=1.0 / 255.0):
    # Convert image values to float32 and scale them.
    return image.astype(np.float32) * scale


def denormalize_image(image, scale=255.0):
    # Convert normalized image values back to uint8
    return np.clip(
        image * scale,
        0,
        255
    ).astype(np.uint8)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def grayscale(image):
    # Convert BGR/RGB image to grayscale.
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def is_valid_image(image):
    return (
        image is not None
        and isinstance(image, np.ndarray)
        and image.size > 0
    )


# Bounding Box Utilities

def calculate_iou(box_a, box_b):
    
    # Calculate Intersection over Union
    # Box format: [x1, y1, x2, y2]
    
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_iou_matrix(boxes_a, boxes_b):
    # Calculate pairwise IoU between two sets of boxes
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros(
            (len(boxes_a), len(boxes_b)),
            dtype=np.float32
        )

    ax1 = boxes_a[:, 0][:, None]
    ay1 = boxes_a[:, 1][:, None]
    ax2 = boxes_a[:, 2][:, None]
    ay2 = boxes_a[:, 3][:, None]

    bx1 = boxes_b[:, 0][None, :]
    by1 = boxes_b[:, 1][None, :]
    bx2 = boxes_b[:, 2][None, :]
    by2 = boxes_b[:, 3][None, :]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = (
        np.maximum(0, ax2 - ax1)
        * np.maximum(0, ay2 - ay1)
    )

    area_b = (
        np.maximum(0, bx2 - bx1)
        * np.maximum(0, by2 - by1)
    )

    union = area_a + area_b - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )


def xyxy_to_xywh(boxes):
    # Convert [x1, y1, x2, y2] to [cx, cy, width, height]
    boxes = np.asarray(boxes, dtype=np.float32)

    result = np.empty_like(boxes)

    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    result[:, 2] = boxes[:, 2] - boxes[:, 0]
    result[:, 3] = boxes[:, 3] - boxes[:, 1]

    return result


def xywh_to_xyxy(boxes):
    # Convert [cx, cy, width, height] to [x1, y1, x2, y2]
    boxes = np.asarray(boxes, dtype=np.float32)

    result = np.empty_like(boxes)

    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return result


def clip_boxes(boxes, image_shape):
    # Keep bounding boxes inside image boundaries
    h, w = image_shape[:2]

    boxes = np.asarray(boxes).copy()

    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

    return boxes


def scale_boxes(
    boxes,
    from_shape,
    to_shape
):
    # Scale boxes from one image resolution to another
    from_h, from_w = from_shape[:2]
    to_h, to_w = to_shape[:2]

    scale_x = to_w / from_w
    scale_y = to_h / from_h

    result = np.asarray(boxes, dtype=np.float32).copy()

    result[:, [0, 2]] *= scale_x
    result[:, [1, 3]] *= scale_y

    return result


def non_max_suppression(
    boxes,
    scores,
    iou_threshold=0.5
):
    # Apply Non-Maximum Suppression to bounding boxes
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]

        ious = calculate_iou_matrix(
            boxes[current:current + 1],
            boxes[remaining]
        )[0]

        order = remaining[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int32)


# Drawing Utilities

def draw_boxes(
    image,
    boxes,
    labels=None,
    scores=None,
    color=(0, 255, 0),
    thickness=2
):
    # Draw bounding boxes with optional labels and scores
    output = image.copy()

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        text = []

        if labels is not None:
            text.append(str(labels[index]))

        if scores is not None:
            text.append(f"{scores[index]:.2f}")

        if text:
            put_text(
                output,
                " ".join(text),
                (x1, max(20, y1 - 8)),
                color=color
            )

    return output


def draw_center(
    image,
    center,
    radius=5,
    color=(0, 255, 0)
):
    # Draw a point representing an object's center
    output = image.copy()

    x, y = map(int, center)

    cv2.circle(
        output,
        (x, y),
        radius,
        color,
        -1
    )

    return output


def draw_grid(
    image,
    rows=3,
    cols=3,
    color=(100, 100, 100),
    thickness=1
):
    # Draw a simple grid over an image
    output = image.copy()

    h, w = output.shape[:2]

    for row in range(1, rows):
        y = int(h * row / rows)

        cv2.line(
            output,
            (0, y),
            (w, y),
            color,
            thickness
        )

    for col in range(1, cols):
        x = int(w * col / cols)

        cv2.line(
            output,
            (x, 0),
            (x, h),
            color,
            thickness
        )

    return output


def put_text(
    image,
    text,
    position,
    color=(255, 255, 255),
    font_scale=0.6,
    thickness=2,
    background=None
):
    # Draw readable text with an optional background
    output = image.copy()

    x, y = map(int, position)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if background is not None:
        (text_w, text_h), baseline = cv2.getTextSize(
            str(text),
            font,
            font_scale,
            thickness
        )

        cv2.rectangle(
            output,
            (x - 4, y - text_h - baseline - 4),
            (x + text_w + 4, y + 4),
            background,
            -1
        )

    cv2.putText(
        output,
        str(text),
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return output


def draw_fps(
    image,
    fps,
    position=(10, 30),
    color=(0, 255, 0)
):
    # Draw FPS information on an image
    return put_text(
        image,
        f"FPS: {fps:.1f}",
        position,
        color=color,
        background=(0, 0, 0)
    )


# Video Utilities

def open_video(source=0):
    # Open a webcam, video path, or camera source
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video source: {source}"
        )

    return capture


def read_video_frames(
    source=0,
    max_frames=None
):
    # Yield frames from a webcam or video
    capture = open_video(source)

    frame_count = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            yield frame

            frame_count += 1

            if (
                max_frames is not None
                and frame_count >= max_frames
            ):
                break
    finally:
        capture.release()


def get_video_info(source):
    # Return basic video information
    capture = open_video(source)

    info = {
        "width": int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        ),
        "height": int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ),
    }

    capture.release()

    return info


# Image & Video Output

def save_image(
    path,
    image,
    quality=95
):
    # Save an image and create its parent directory
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    extension = path.suffix.lower()

    params = []

    if extension in {".jpg", ".jpeg"}:
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]

    success = cv2.imwrite(
        str(path),
        image,
        params
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {path}"
        )

    return path


def make_mosaic(
    images,
    columns=2,
    cell_size=None,
    background=(0, 0, 0)
):
    # Arrange multiple images into a single grid
    if not images:
        raise ValueError("images cannot be empty")

    processed = []

    for image in images:
        if cell_size is not None:
            image = cv2.resize(
                image,
                cell_size,
                interpolation=cv2.INTER_AREA
            )

        processed.append(image)

    if cell_size is None:
        max_h = max(image.shape[0] for image in processed)
        max_w = max(image.shape[1] for image in processed)
        cell_size = (max_w, max_h)

    cell_w, cell_h = cell_size
    rows = int(np.ceil(len(processed) / columns))

    mosaic = np.full(
        (
            rows * cell_h,
            columns * cell_w,
            3
        ),
        background,
        dtype=np.uint8
    )

    for index, image in enumerate(processed):
        row = index // columns
        col = index % columns

        h, w = image.shape[:2]

        h = min(h, cell_h)
        w = min(w, cell_w)

        mosaic[
            row * cell_h:row * cell_h + h,
            col * cell_w:col * cell_w + w
        ] = image[:h, :w]

    return mosaic


# Performance Utilities

class FPSMeter:
    # Calculate smoothed FPS for real-time applications

    def __init__(self, smoothing=10):
        self.previous_time = None
        self.fps = 0.0
        self.history = []
        self.smoothing = max(1, int(smoothing))

    def update(self):
        current_time = time.perf_counter()

        if self.previous_time is not None:
            elapsed = current_time - self.previous_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed

                self.history.append(current_fps)

                if len(self.history) > self.smoothing:
                    self.history.pop(0)

                self.fps = sum(self.history) / len(self.history)

        self.previous_time = current_time

        return self.fps

    def reset(self):
        self.previous_time = None
        self.fps = 0.0
        self.history.clear()


class Timer:
    # Simple performance timer for CV pipelines

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")

        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None

        return self.elapsed

    def reset(self):
        self.start_time = None
        self.elapsed = 0.0
        
