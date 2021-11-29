import cv2
import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":
    test_images = glob("test_images/*")
    model = tf.keras.models.load_model("unet.h5")
    k = 1;
    for path in tqdm(test_images, total=len(test_images)):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        img_contours = x.copy()
        original_image = x;
        h, w, _ = x.shape

        x = cv2.resize(x, (256, 256))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        pred_mask = model.predict(x)

        pred_mask = pred_mask[0]

        mask = pred_mask[0]
        mask = mask * 255
        mask = cv2.resize(mask, (256, 256))
        mask = np.uint8(mask)
        edges = cv2.Canny(image=mask, threshold1=100, threshold2=200)
        kernel = np.ones((4, 4))
        dilate = cv2.dilate(edges, kernel)
        img_contours = cv2.resize(img_contours, (256, 256))
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        calories = 42
        mass = 3700
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #     cv2.drawContours(img_contours, cnt, -1, (0,255,0), 2)
            print("in")
            if area > 1000:
                cal_estimation = (area * 42) // mass
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_contours, str(cal_estimation), (x + 10, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                            (0, 0, 255), 2)

        pred_mask = np.concatenate(
            [
                pred_mask, pred_mask, pred_mask
            ], axis=2
        )

        pred_mask[pred_mask > 0.5] = 255
        pred_mask = pred_mask.astype(np.float32)

        pred_mask = cv2.resize(pred_mask, (w, h))
        original_image = original_image.astype(np.float32)
        alpha = 0.5
        cv2.addWeighted(pred_mask, alpha, original_image, 1 - alpha, 0, original_image)

        # cv2.imwrite(f"mask/{k}.png", original_image)
        cv2.imwrite(f"mask/{k + 10}.png", pred_mask)
        # cv2.imwrite(f"mask/{k + 50}.png", img_contours)

        k += 1
