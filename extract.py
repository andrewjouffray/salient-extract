from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import cv2
import numpy as np
import sys




'''
runs the model on an image and converts the detection masks to np.arrays 
'''
def predict_on_image(model, img, conf):

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

'''
cuts out the part of the image that is detected by the model, and makes it fill the frame
'''
def cut_out_detection(image, mask):

    '''the following oprations assume that the mask size is smaller than the original image size'''

    # convert the mask to unit8 to make cv2 happy
    mask = mask.astype(np.uint8)
    
    # get only outer contours (ignore contours inside other contours)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    # find the large contours 
    large_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            large_contours.append(contour)

    # make sure there are large contours before we keep going
    if len(large_contours) == 0:
        return None
    
    # get the bounding box around the contour
    rect = cv2.boundingRect(large_contours[0])

    # start colum, start row, width, height of the rectangle
    (x, y, w, h)= rect

    # black image with segmented area being in color
    masked = cv2.bitwise_and(image, image, mask=mask) 

    # cut out the area in color (the same as the area of the mask)s
    img_masked_cropped = masked[y:y+h, x:x+w]

    # get og size of image
    og_h, og_w, _ = masked.shape

    # get the aspect ratio of the og image
    og_aspect = og_w / og_h

    # get the cut out shape
    h, w, _ = img_masked_cropped.shape
    if h <= 5 or w <= 5:
        return None

    # get the cutout aspect ratio
    aspect = w / h

    edge_margin = 4

    # if the cut out has a more vertical aspect ratio
    if aspect < og_aspect:

        # scale up the cutout so that the height is the same as the og image
        new_width = int((w / h) * (og_h)) - edge_margin
        new_height = og_h - edge_margin
        scaled_cropped = cv2.resize(img_masked_cropped, (new_width, new_height))
    
    # else if the cut out has a wider aspect ratio
    else:

        # scale up the cutout so that the width is the same as the og image
        new_height = int((h / w) * (og_w)) - edge_margin
        new_width = og_w - edge_margin
        scaled_cropped = cv2.resize(img_masked_cropped, (new_width, new_height))

    # create a black image
    result_image = np.zeros((og_h,og_w,3), np.uint8)

    # get scaled cut out shape
    s_h, s_w, _ = scaled_cropped.shape

    # add the scaled cut out to the balck image
    result_image[2:2+s_h, 2:2+s_w] = scaled_cropped

 
    return result_image

# stacks 3 video frame together
def merge_videos (original, predicted, extracted):

    # resize the video frames
    h, w, _ = original.shape
    new_h = int(h/3)
    new_w = int(w/3)
    new_size = (new_w, new_h)

    resized_original = cv2.resize(original, (new_size))
    resized_predicted = cv2.resize(predicted, (new_size))
    resized_extracted = cv2.resize(extracted, (new_size))

    # create the blank stack
    stack = np.zeros((h+1,new_w+1,3), np.uint8)

    stack[0:new_h, 0:new_w] = resized_original

    stack[new_h:new_h*2, 0:new_w] = resized_predicted

    stack[new_h*2:new_h*3, 0:new_w] = resized_extracted

    return stack



'''
Entry point of the script

example:
python extract.py yolov8x-seg.pt /home/andrew/Videos/quince.mp4 output.mp4

inspired from https://github.com/ultralytics/ultralytics/issues/561

'''
if __name__ == "__main__":
 
    model_name = sys.argv[1]
    input_name = sys.argv[2]
    output_name = sys.argv[3]

    model = YOLO(model_name)
    model.info(verbose=False)
    vid_capture = cv2.VideoCapture(input_name)

    width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    length = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(width/3)+1, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_name, fourcc, 60.0, size)

    count = 0

    while(vid_capture.isOpened()):

        ret, frame = vid_capture.read()
        if ret == True:
            count += 1
            img = np.copy(frame)

            # predict by YOLOv8
            boxes, masks, cls, probs = predict_on_image(model, img, conf=0.55)

            if masks is None:
                pass
            else:


                # overlay each mask on the original image independently
                image_with_masks = np.copy(frame)
                for mask_i in masks:
                    image_with_masks = overlay(image_with_masks, mask_i, color=(255,0,0), alpha=0.7)
                
                # cut out the detections out of the image
                for mask_i in masks:
                    scaled_cutout = np.copy(frame)
                    scaled_cutout = cut_out_detection(scaled_cutout, mask_i)

                    if not scaled_cutout is None:
                        final_frame = merge_videos(frame, image_with_masks, scaled_cutout)
                        out.write(final_frame)
            print("Processed frames: " + str(count) + "/" + str(length), end="\r")        
        else:
            break

    # release the writer
    out.release()

