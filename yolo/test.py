import os
import cv2
import numpy as np
import glob
from ultralytics import YOLO

def visualize_mask(image, mask, color, alpha=0.5):
    """Overlay mask to image"""
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    mask = mask.astype(bool)
    image[mask] = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)[mask]
    return image

def main():
    # load trained segmentation model
    model = YOLO("runs/segment/train/weights/best.pt")  

    image_folder = "test/images"  # segmentation 默认要 images 目录

    image_list = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                 glob.glob(os.path.join(image_folder, "*.png")) + \
                 glob.glob(os.path.join(image_folder, "*.jpeg"))

    os.makedirs("runs/seg_test", exist_ok=True)

    print(f"Found {len(image_list)} test images")

    for img_path in image_list:
        results = model(img_path)

        img = cv2.imread(img_path)

        for result in results:
            # iter所有 objects
            masks = result.masks
            boxes = result.boxes
            names = model.names

            if masks is None:
                continue

            for i, mask in enumerate(masks.data):
                mask = mask.cpu().numpy()

                # random color for mask
                color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()

                # draw mask
                img = visualize_mask(img, mask[0], color)

                # bbox + label
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])

                label = f"{names[cls]} {conf:.2f}"

                cv2.rectangle(img, (xyxy[0], xyxy[1]),
                                   (xyxy[2], xyxy[3]),
                                   color, 2)
                cv2.putText(img, label,
                            (xyxy[0], xyxy[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        save_path = f"runs/seg_test/{os.path.basename(img_path)}"
        cv2.imwrite(save_path, img)
        print("saved:", save_path)


if __name__ == "__main__":
    main()
