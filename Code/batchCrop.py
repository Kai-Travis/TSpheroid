from pathlib import Path
from singleCrop import crop_image

INPUT_FOLDER = Path(r"D:\TrainingData\raw\exp01\7h")
OUTPUT_FOLDER = Path(r"D:\TrainingData\processed\exp01\7h")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

image_paths = sorted(p for p in INPUT_FOLDER.glob("*.tif")
                     if not p.stem.endswith("_cropped"))

successful = 0
failed = 0

for i, image_path in enumerate(image_paths, start=1):
    crop = crop_image(
        image_path,
        show_debug=False,
        save=False
    )

    if crop is None:
        failed += 1
        continue

    output_path = (OUTPUT_FOLDER / f"{image_path.stem}_cropped.tif")

    import cv2

    success = cv2.imwrite(str(output_path), crop)

    if success:
        successful += 1
    else:
        failed += 1

print()
print("Finished")
print(f"Successful: {successful}")
print(f"Failed: {failed}")