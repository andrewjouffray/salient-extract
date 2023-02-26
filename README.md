
# Salien Extract

Yolov8 based feature extraction model. 

The goal of this work is to develop a set of tools that enables users to easily collect training data from the field. Whether it be plants, minerals or other objects. The features of interest can then be extracted by the salient extraction model and superimposed on divers backgrounds. Creating synthetic dataset only comprised of images taken in the real world. 


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

    python extract.py --model free_models/salient_extract_n.pt --input yourVideo.mp4 --output output.mp4 --smooth <True / False> --stitch <True / False>


## Support this project:

**This project is currently self-funded; access the larger Yolo x model, dataset and newest models by supporting this project on Patreon**.
Your support will also allow me to purchase better hardware and accelerate my work. 

## How it works:

This salient feature extractor was trained on synthetic data comprised of salient objects in a focused foreground superimposed over random blurred and in-focus background images. Therefore the model has a strong bias for in-focused objects, that are not your hands. 

### Smooth mode:

If you use the `--smooth True` argument, the script will stack the masks of every 4 frames together, attempting to compensate for the Jitteriness of the detection masks. This means that the cutouts might not be as accurate if the object moves a lot in the frame. 

### Stictch mode:

If you use the `--stitch True` argument, the script will stitch the input frame, prediction frame and cut-out mask frames side by side. Setting it to `False` will just output the cut-out frames. 

## Current limitations:

- Need for in-focus foreground and out-of-focus background
- jitteriness of the detection masks
- mask inaccuracies when using smooth mode


