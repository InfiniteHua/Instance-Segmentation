import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks

def overlay_mask(image, mask, color=(0,255,255), alpha=0.5):
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:] = color
    mask = mask.astype(bool)
    image[mask] = cv2.addWeighted(image[mask], 1-alpha, color_mask[mask], alpha, 0)
    return image

def main():
    # load segmentation model
    model = YOLO("runs/flower_seg/flower_segmentation/weights/best.pt")

    test_dir = "flowers/exceptions"
    os.makedirs("flowers/exceptions/results", exist_ok=True)

    imgs = sorted(
        glob.glob(os.path.join(test_dir, "*.jpg"))
        + glob.glob(os.path.join(test_dir, "*.png"))
        + glob.glob(os.path.join(test_dir, "*.jpeg"))
    )

    print("Found", len(imgs), "test images")

    for img_path in imgs:
        img = cv2.imread(img_path)
        # results = model(img_path)
        results = model(img_path, imgsz=1280, retina_masks=True, verbose=False)

        result = results[0]
        # masks = result.masks
        rmasks = result.masks.data
        boxes = result.boxes
        names = model.names

        H, W = img.shape[:2]
        # print(rmasks.shape)
        rmasks = rmasks.unsqueeze(1) 
        rmasks = scale_masks(rmasks, (H, W)).cpu().numpy()
        if rmasks is None:
            print("no mask", img_path)
            continue

        labels = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        bboxes = boxes.xyxy.cpu().numpy()

        # 'Stem' class index
        stem_index = None
        for cid, cname in names.items():
            if cname.lower() == "stem":
                stem_index = cid
                break

        if stem_index is None:
            print("Stem not found in:", names)
            continue

        # filtering detection results
        filtered = []

        for i in range(len(labels)):
            # score threshold
            if scores[i] < 0.4:
                continue

            # class filter (Stem only)
            if labels[i] != stem_index:
                continue

            filtered.append({
                "score": scores[i],
                "mask": rmasks[i, 0],
                "bbox": bboxes[i],
            })

        # iterate valid mask objs
        stem_counter = 0

        for i, det in enumerate(filtered):
            mask = det["mask"]

            # 转 numpy
            mask = np.array(mask)

            # binarize
            mask = (mask > 0.5).astype(np.uint8)

            bbox = det["bbox"].astype(int)

            x1,y1,x2,y2 = bbox
            w, h = x2-x1, y2-y1

            # bbox too small → ignore
            if w < 5 or h < 18:
                continue

            ys, xs = np.where(mask > 0)

            # pixel threshold
            if len(xs) <= 95:
                print(f"stem_{stem_counter} removed with {len(xs)} pixels")
                continue

            # draw mask
            img = overlay_mask(img, mask)

            # center text
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            cv2.putText(
                img, 
                str(stem_counter),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),2
            )

            stem_counter += 1

        # save image
        save_name = os.path.join(
            "flowers/exceptions/results",
            "labeled_" + os.path.basename(img_path)
        )
        cv2.imwrite(save_name, img)
        print("saved:", save_name)


if __name__ == '__main__':
    main()
