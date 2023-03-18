
# Salient Extract

Yolov8 based feature extraction model. 

The goal of this work is to develop a set of tools that enables users to easily collect training data from the field. Whether it be plants, minerals or other objects. The features of interest can then be extracted by the salient extraction model and superimposed on divers backgrounds. Creating synthetic dataset only comprised of images taken in the real world. 

This project is in very early development stages, expect bugs and frequent updates. 


![example](docs/images/field6.gif)

## Getting started (Linux):


Check your GPU install / drivers (optional):

    nvidia-smi

Your output should look something like this:

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:07:00.0  On |                  N/A |
    | 25%   38C    P8    24W / 260W |   1236MiB / 11264MiB |      5%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+


Install:

    git clone git@github.com:andrewjouffray/salient-extract.git

    cd salient-extract

    pip install -r  requirements.txt

Run:

    python extract.py --model models/salient_extract_n.pt --input yourVideo.mp4 --output output.mp4 --smooth --stitch

## Access the models (YoloV8)

name | mAP@50 | location
--- | --- | ---
salient_n | 0.829 | models/salient_extract_n.pt
salient_s | 0.877 | [patreon.com/SalientExtractAi (Free)](https://patreon.com/SalientExtractAi)
salient_m | 0.723 | models/salient_extract_m.pt
salient_m2 | 0.874 | [patreon.com/SalientExtractAi (Free)](https://patreon.com/SalientExtractAi)
salient_l | 0.888 | [patreon.com/SalientExtractAi (Tier2)](https://patreon.com/SalientExtractAi)
salient_x | 0.811 | [patreon.com/SalientExtractAi (Tier2)](https://patreon.com/SalientExtractAi)
salient_x2 | 0.899 | [patreon.com/SalientExtractAi (Tier2)](https://patreon.com/SalientExtractAi)

## Access the Datasets (YoloV8 format):

Some of them are __free__, so no need to subscribe to my Patreon to gain access.

name | # of images | location
--- | --- | ---
synthetic salient objects | 120,000 + | [patreon.com/SalientExtractAi (Tier3)](https://patreon.com/SalientExtractAi)
sample synthetic dataset | 355 | [patreon.com/SalientExtractAi (Free)](https://patreon.com/SalientExtractAi)
validation | 204 | [patreon.com/SalientExtractAi (Free)](https://patreon.com/SalientExtractAi)
MSRA_10K (yolo) | 10,000 | [patreon.com/SalientExtractAi (Free)](https://patreon.com/SalientExtractAi)


## How it works:

This salient feature extractor is based on the yolov8-seg model, trained on synthetic data comprised of salient objects in a focused foreground superimposed over random blurred and in-focus background images. Therefore the model has a strong bias for in-focused objects, that are not your hands. 

## arguments 

**--model -m:**

example: `--model models/salient_extract_n.pt`. This is the path to the model you want to use.

**--input -i:**

example: `--input yourVideo.mp4`. This is the path to the video that you want to extract features out of.

**--output -o:**

example: `--output output.mp4`. This is the path and filename that you want to use to save all the extracted feature. It must be a video 

**--smooth -s:**

If you use the `--smooth` flag, the script will stack the masks of every 4 frames together, attempting to compensate for the Jitteriness of the detection masks. This means that the cutouts might not be as accurate if the object moves a lot in the frame. 

**--stitch -t:**

If you use the `--stitch` flag, the script will stitch the input frame, prediction frame and cut-out mask frames side by side. Omitting this flag will just output the cut-out frames. 

## Current limitations:

- Need for in-focus foreground and out-of-focus background
- jitteriness of the detection masks
- mask inaccuracies when using smooth mode
- struggles with objects that are not one uniform mass

## generate semi-synthetic data:

You can use the sister-project to salient extract [Composite Image Generator](https://github.com/andrewjouffray/Composite-Image-Generator), to generate images using the "copy paste" method.

Note: this project needs a bit of work and a few updates.

## Special thanks

Although this specific project has been developed on my own free time using my own resources. I would like to thank Dr. Rakesh Kaundal and the [KAABiL Lab](http://bioinfo.usu.edu/) for providing hardware and assistance in the development of early models that eventually led me to start this project. 