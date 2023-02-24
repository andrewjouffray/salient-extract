import os
import sys
import subprocess

# sudo python move_img_to_dir.py ../workdir_salient_object/input_data/fold0/test_img test/images


def find_imgs(input_path):

    imgs = []
    files = os.listdir(input_path)
    for file in files:
        if file.endswith("jpg"):
            name = os.path.join(input_path, file)
            imgs.append(name)

    return imgs


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    imgs = find_imgs(input_path)

    print("found", len(imgs), "images")
    print(imgs[0])

    for img in imgs:

        bashCommand = "mv " + img + " " + output_path
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


    