from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import numpy as np
from ultralytics.yolo.utils.ops import scale_image
#from ultralytics.yolo import YOLO

model = YOLO("yolov8x-seg.pt")  # load an official model
vid_capture = cv2.VideoCapture('/home/andrew/Videos/quince.mp4')
 
if (vid_capture.isOpened() == False):
  print("Error opening the video file")
# Read fps and frame count
else:
  # Get frame rate information
  # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
  fps = vid_capture.get(5)
  print('Frames per second : ', fps,'FPS')
 
  # Get frame count
  # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
  frame_count = vid_capture.get(7)
  print('Frame count : ', frame_count)
 
while(vid_capture.isOpened()):
  # vid_capture.read() methods returns a tuple, first element is a bool 
  # and the second is frame
  ret, frame = vid_capture.read()
  if ret == True:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)
    image = np.copy(img)
    for r in results:
        
        annotator = Annotator(frame)

        masks = r.masks.masks.cpu().numpy()     # masks, (N, H, W)
        masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
        # rescale masks to original image
        masks = scale_image(masks.shape[:2], masks, r.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

        
        color = (0,255,0)
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        resize = None
        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1, image_overlay, 1, 0)

        cv2.imShow("prediction", image)


    # 20 is in milliseconds, try to increase the value, say 50 and observe
    key = cv2.waitKey(200)
     
    if key == ord('q'):
      break
  else:
    break
 
# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()

