import sys
import os
import json
'''
converts the data in labelme Json format to yolov5 format 


yolo v5 format:
.txt file
<class id> <point ccordinates like this x y x y x y x y>

python convert_to_yolo.py /run/media/andrew/research/workdir_salient_object/input_data/fold0/test_img test/labels
'''

def get_json_files(path):


    json_files = []

    if os.path.isdir(path):

        files = os.listdir(path)

        for file in files:

            if file.endswith("json"):

                json_files.append(file)
        
        if len(json_files) > 0:
            return json_files
        else:
            return None

    else:
        return None



def check_output_path(output_path):

    if not os.path.isdir(output_path):

        print("creating: ", output_path)
        os.makedirs(output_path)

    else:
        print("found ", output_path)

def process_json_file(file_name, output_path):

    # Opening JSON file
    f = open(file_name)

    name = os.path.basename(file_name)[:-4] + "txt"
    name = os.path.join(output_path, name)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    lines = []
    shapes = data["shapes"]
    for shape in shapes:
        line = "0 "
        for point in shape["points"]:
            x = point[0] / int(data["imageWidth"])
            y = point[1] / int(data["imageHeight"])
            coord = str(x) + " " + str(y) + " "
            line += coord
        lines.append(line)
    
    with open(name, 'w') as txtfile:
        for line in lines:
            txtfile.write(f"{line}\n")
    
        txtfile.close()
    print("wrote points to ", name )


if __name__ == "__main__":

    path_to_json_files = sys.argv[1]
    output_path = sys.argv[2]

    # get the json files
    files = get_json_files(path_to_json_files)

    if files is None:
        print("could not get json files form the directory")
        exit
    
    else:
        print("found ", len(files), " json files in ", path_to_json_files)


    # check the output path
    check_output_path(output_path)

    for file in files:

        full_path = os.path.join(path_to_json_files, file)
        process_json_file(full_path, output_path)
