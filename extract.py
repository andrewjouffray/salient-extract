'''
    Salient Extract, salient feature extraction model
    Copyright (C) 2023  Andrew Jouffray

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

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



def cut_out_detection(image, mask):
    '''
    cuts out the part of the image that is detected by the model, and makes it fill the frame.

    params:
        - image: original image that the detection was run on
        - mask: binary 2D mask

    output:
        - Color image on the masked object cut out and scaled to fit the frame

    The following oprations assume that the mask size is smaller than the original image size.
    '''

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

    # merge all the contours together
    join_cnts = np.concatenate(large_contours)
    
    # get the bounding box around the contour
    rect = cv2.boundingRect(join_cnts)

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
def merge_videos (original, predicted, extracted, orientation):
    """Stacks the original image, the image with detection mask and the cut out image side by side.

    Params:
        original: original image 
        predicted: image with the prediction mask overlayed
        extracted: cut out of the masked area
        orientation: vertical / horizontal


    Returns:
        merged_images: image of the 3 stacked images side by side

    """
    # resize the video frames
    h, w, _ = original.shape
    new_h = int(h/3)
    new_w = int(w/3)
    new_size = (new_w, new_h)

    resized_original = cv2.resize(original, (new_size))
    resized_predicted = cv2.resize(predicted, (new_size))

    # if None was passed, we assume no detection was made and therefore we create a black frame
    resized_extracted = extracted
    if resized_extracted is None:
        resized_extracted = np.zeros((new_h,new_w,3), np.uint8)
    else:
        resized_extracted = cv2.resize(extracted, (new_size))

    if orientation == "horizontal":

        # create the blank stack
        stack = np.zeros((new_h+1,w+1,3), np.uint8)

        stack[0:new_h, 0:new_w] = resized_original

        stack[0:new_h, new_w:new_w*2] = resized_predicted

        stack[0:new_h, new_w*2:new_w*3] = resized_extracted

    else:

        # create the blank stack
        stack = np.zeros((h+1,new_w+1,3), np.uint8)

        stack[0:new_h, 0:new_w] = resized_original

        stack[new_h:new_h*2, 0:new_w] = resized_predicted

        stack[new_h*2:new_h*3, 0:new_w] = resized_extracted

    return stack




if __name__ == "__main__":
    '''
    Entry point of the script

    example: python extract.py yolov8x-seg.pt <path to video>/yourvid.mp4 output.mp4 yes no

    implementation inspired from https://github.com/ultralytics/ultralytics/issues/561

    '''

    print("\nSalient Extract  Copyright (C) 2023  Andrew Jouffray\n", 
    "This program comes with ABSOLUTELY NO WARRANTY.\n",
    "This is free software, and you are welcome to redistribute it\n",
    "under certain conditions.\n")
    
    # # quick input check
    # if len(sys.argv) < 5:
    #     print("Missing arguments: \npython extract.py --model <model name> --input <path to video> --output <output name.mp4> --smooth <True / False> --stitch <True / False")
    #     exit()

    # # arguments


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help="Model to use", type=str, required=True)
    parser.add_argument('--input', '-i', help="Path to input video", type=str, required=True)
    parser.add_argument('--output', '-o', help="Path to output video", type=str, required=True)
    parser.add_argument('--smooth', '-s', help="Merges masks together (reduce jitter and outline precision)", type= bool, default= True)
    parser.add_argument('--stitch', '-t', help="Stiches input frames, detection frame and extracted frames together", type= bool, default= False)
    args=parser.parse_args()


    model_name = args.model
    input_name = args.input
    output_name = args.output

    merge = args.smooth
    stitch = args.stitch

    model = YOLO(model_name)
    model.info(verbose=False)
    vid_capture = cv2.VideoCapture(input_name)

    # setting up the video writer
    width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    length = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # defines how to stack the videos in debug mode
    if width < length:
        orientation = "vertical"
        size = (int(width/3)+1, height)
    else:
        orientation = "horizontal"
        size = (width, int(height/3)+1)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_name, fourcc, 60.0, size)

    # init tracking variables
    count = 0
    features_extracted = 0
    frames_with_masks = 0

    last_4_masks = []

    # main video processing loop
    while(vid_capture.isOpened()):

        ret, frame = vid_capture.read()
        if ret == True:
            count += 1
            img = np.copy(frame)

            # predict by YOLOv8
            boxes, masks, cls, probs = predict_on_image(model, img, conf=0.55)

            if masks is None:
                if stitch:
                    final_frame = merge_videos(frame, frame, None, orientation)
                    out.write(final_frame)
            else:
                frames_with_masks += 1
                # overlay each mask on the original image independently
                image_with_masks = np.copy(frame)
                merged_masks = np.zeros((height,width,1), np.uint8)

                for mask_i in masks:
                    image_with_masks = overlay(image_with_masks, mask_i, color=(255,0,0), alpha=0.7)

                    # merge all the masks from that frame 
                    mask_i = mask_i.astype(np.uint8)
                    merged_masks = cv2.add(merged_masks, mask_i)

                    # add that merged mask to the list of the last 3
                    last_4_masks.append(merged_masks)

                    # if there are more than 3, remove the first one
                    if len(last_4_masks) > 4:
                        last_4_masks.pop(0)

                # if there are 4 merged masks, add them all up together
                if len(last_4_masks) == 4 and merge:
                    last_4_merged_masks = np.zeros((height,width,1), np.uint8)
                    for mask in last_4_masks:
                        last_4_merged_masks = cv2.add(last_4_merged_masks, mask)
                    masks = [last_4_merged_masks]
                    
                
                # cut out the detections out of the image
                for mask_i in masks:
                    scaled_cutout = np.copy(frame)
                    scaled_cutout = cut_out_detection(scaled_cutout, mask_i)

                    if not scaled_cutout is None:
                        features_extracted += 1
                        if stitch:
                            final_frame = merge_videos(frame, image_with_masks, scaled_cutout, orientation)
                        else:
                            final_frame = scaled_cutout
                        out.write(final_frame)
                    
                    # all the masks were too small and we are in stitch mode
                    elif stitch:
                        final_frame = merge_videos(frame, image_with_masks, None, orientation)
                        out.write(final_frame)


            print("Processed frames: " + str(count) + "/" + str(length), end="\r")        
        else:
            break

    # release the writer
    out.release()
    
    print("\n============== extraction metrics ===============\n")

    # print the stats 
    detection_precent = (frames_with_masks / length) * 100
    print("Percenteage of frames with masks: {:>14.2f}%".format(detection_precent))

    print("Number of salient features extracted: {:>11}".format(features_extracted))

    features_pre_frames = features_extracted / frames_with_masks
    print("Salient feature per frames with masks: {:>10.2f}".format(features_pre_frames))