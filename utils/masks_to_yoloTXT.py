import sys
import os
import cv2
import numpy as np


def get_mask_files(path):

    mask_files = []
    files = os.listdir(path)

    for file in files:
        if file.endswith("png"):
            mask_files.append(os.path.join(path, file))

    return mask_files


def get_norm_contour_from_mask(mask):

    mask = mask.astype(np.uint8)
    height, width = mask.shape
    (thresh, mask) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    norm_contours = []

    for contour in contours:
        contour_line = "0 "

        for point in contour:

            x = point[0][0] / width
            y = point[0][1] / height
            contour_line += str(x)+" "+str(y)+" "

        norm_contours.append(contour_line)

    return norm_contours


def write_contours_to_file(contours, path):



    pre, ext = os.path.splitext(path)

    path = pre + ".txt"



    with open(path, 'w') as txtfile:
        for line in contours:
            txtfile.write(f"{line}\n")
    
        txtfile.close()
    print("wrote points to ", path )


if __name__ == "__main__":

    source = sys.argv[1]
    dest = sys.argv[2]

    mask_files = get_mask_files(source)

    for mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        norm_contours = get_norm_contour_from_mask(mask)
        file_name = os.path.basename(mask_path)
        write_contours_to_file(norm_contours, os.path.join(dest, file_name))



