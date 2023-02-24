import os
import sys
import subprocess

# sudo python split_test_to_val.py test val

# splits the file list into 2
def get_half_files(input_path):

    img_path = os.path.join(input_path, "images")

    files = os.listdir(img_path)
    half = int(len(files) / 2)
    half_files = files[:half]
    return half_files


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    files_to_move = get_half_files(input_path)

    print("found", len(files_to_move), "files to move")

    for file in files_to_move:

        img_path = os.path.join(input_path, "images", file)
        no_extention_name = file[:-4]
        txt_path = os.path.join(input_path, "labels", no_extention_name+".txt")

        img_dest = os.path.join(output_path, "images", file)
        txt_dest = os.path.join(output_path, "labels", no_extention_name+".txt")

        print("moving img", img_path, "to", img_dest)
        print("moving txt", txt_path, "to", txt_dest)

        bashCommand = "mv " + img_path + " " + img_dest
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        bashCommand = "mv " + txt_path + " " + txt_dest
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()