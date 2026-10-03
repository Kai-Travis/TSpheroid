import cv2
import numpy as np
from pathlib import Path

THRESHOLD = 65
DISTANCE_THRESHOLD = 50
MIN_CONTOUR_AREA = 10000
MIN_CORE_AREA = 1000
# Final crop size
CROP_SIZE = 650

# Extra space between spheroid and crop edge
# This is already included in CROP_SIZE,
# but useful when checking whether the spheroid fits.
MIN_PADDING = 20

def crop_image(image_path, show_debug=False, save=False):
    image_path = Path(image_path)

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print("ERROR: Could not read image")
        return None

    img8 = cv2.normalize( img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur( img8, (9, 9), 0)

    _, mask = cv2.threshold(
        blurred,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Spheroid is the dark region
    mask = cv2.bitwise_not(mask)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mask_filled = np.zeros_like(mask)

    valid_contours = []

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        valid_contours.append(contour)

        # Fill entire foreground contour
        cv2.drawContours(
            mask_filled,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    dist = cv2.distanceTransform(
        mask_filled,
        cv2.DIST_L2,
        5
    )

    dist_display = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    core = np.zeros_like(mask_filled)

    core[dist >= DISTANCE_THRESHOLD] = 255

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            core,
            connectivity=8
        )
    )

    result = cv2.cvtColor(
        img8,
        cv2.COLOR_GRAY2BGR
    )

    best_core = None
    best_area = 0

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_CORE_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]

        y = stats[i, cv2.CC_STAT_TOP]

        w = stats[i, cv2.CC_STAT_WIDTH]

        h = stats[i, cv2.CC_STAT_HEIGHT]

        cx, cy = centroids[i]

        component_mask = np.uint8(
            labels == i
        ) * 255

        component_contours, _ = cv2.findContours(component_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw core
        cv2.drawContours(
            result,
            component_contours,
            -1,
            (0, 255, 0),
            2
        )

        # Draw core bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Draw centroid
        cv2.circle(
            result,
            (int(cx), int(cy)),
            8,
            (0, 0, 255),
            -1
        )

        if area > best_area:

            best_area = area
            best_core = i


    crop = None

    if best_core is not None:

        cx, cy = centroids[best_core]

        cx = int(round(cx))
        cy = int(round(cy))

        half = CROP_SIZE // 2

        x1 = cx - half
        y1 = cy - half

        x2 = x1 + CROP_SIZE
        y2 = y1 + CROP_SIZE

        image_h, image_w = img8.shape

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - image_w)
        pad_bottom = max(0, y2 - image_h)

        if (
            pad_left > 0
            or pad_top > 0
            or pad_right > 0
            or pad_bottom > 0
        ):

            padded = cv2.copyMakeBorder(
                img8,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0
            )

            # Shift coordinates because of padding
            x1_padded = x1 + pad_left
            x2_padded = x2 + pad_left

            y1_padded = y1 + pad_top
            y2_padded = y2 + pad_top

            crop = padded[
                y1_padded:y2_padded,
                x1_padded:x2_padded
            ]

        else:

            crop = img8[
                y1:y2,
                x1:x2
            ]

        # Clip crop box to image for visualization
        draw_x1 = max(0, x1)
        draw_y1 = max(0, y1)
        draw_x2 = min(image_w, x2)
        draw_y2 = min(image_h, y2)

        cv2.rectangle(
            result,
            (draw_x1, draw_y1),
            (draw_x2, draw_y2),
            (0, 0, 255),
            3
        )

        # Mark centre again
        cv2.circle(
            result,
            (cx, cy),
            10,
            (0, 0, 255),
            2
        )

    else:

        print(
            "\nNo suitable core found."
    )

    if crop is not None and save:

        output_path = (
            image_path.parent
            / f"{image_path.stem}_cropped.tif"
        )

        success = cv2.imwrite(
            str(output_path),
            crop
        )

        if not success:
            print(f"Error saving: {output_path}")

    if show_debug:
        cv2.imshow(
            "1 - Original",
            img8
        )

        cv2.imshow(
            "2 - Binary mask",
            mask
        )

        cv2.imshow(
            "3 - Filled foreground contours",
            mask_filled
        )

        cv2.imshow(
            "4 - Distance transform",
            dist_display
        )

        cv2.imshow(
            "5 - Distance threshold core",
            core
        )

        cv2.imshow(
            "6 - Core + centroid + crop",
            result
        )

        if crop is not None:

            cv2.imshow(
                "7 - CROPPED SPHEROID",
                crop
            )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return crop

if __name__ == "__main__":
    IMAGE_PATH = Path(r"imagepath")

    crop_image(IMAGE_PATH, show_debug=True, save=True)

