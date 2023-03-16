from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import cv2
import numpy as np
import sys
import argparse

def predict_on_image(model, img, conf):
    '''
    runs the model on an image and converts the detection masks to np.arrays 

    params:
        - model: yolov8 model instance
        - img: image to predict on
        - conf: confidence threshold

    outputs:
        - boxes: list of bounding boxs
        - masks: list of detection masks
        - cls: detection classes
        - probs: confidence scores
    '''

    result = model(img, conf=conf, verbose=False)[0]
    if result.masks == None:
        return None, None, None, None
    else:

        # detection
        # result.boxes.xyxy   # box with xyxy format, (N, 4)
        cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
        probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
        boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

        # segmentation
        masks = result.masks.masks.cpu().numpy()     # masks, (N, H, W) converted to numpy array
        masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
        # rescale masks to original image
        masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

        return boxes, masks, cls, probs

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help="Model to use", type=str, required=True)

args=parser.parse_args()

model_name = args.model

model = YOLO(model_name)
model.info(verbose=False)

cap = cv2.VideoCapture(1)

if not(cap.isOpened()):
    cap.open()

while(cap.isOpened()):
    ret, frame = cap.read()
    img = np.copy(frame)

    # predict by YOLOv8
    boxes, masks, cls, probs = predict_on_image(model, img, conf=0.55)

    if not masks is None:
        image_with_masks = np.copy(frame)

        for mask_i in masks:
            image_with_masks = overlay(image_with_masks, mask_i, color=(255,0,0), alpha=0.7)
    else:
        image_with_masks = frame

    cv2.imshow('frame',image_with_masks)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()