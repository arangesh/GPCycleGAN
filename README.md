# Gaze Preserving CycleGAN (GPCyceGAN)
PyTorch implementation for the training procedure described in [Driver Gaze Estimation in the Real World: Overcoming the Eyeglass Challenge](http://cvrr.ucsd.edu/publications/2020/GPCycleGAN.pdf).

Parts of the CycleGAN code have been adapted from the [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN) respository.

## Installation
1) Clone this repository
2) Install Pipenv:
```shell
pip3 install pipenv
```
3) Install all requirements and dependencies in a new virtual environment using Pipenv:
```shell
cd GPCycleGAN
pipenv install
```
4) Get link for desired PyTorch and Torchvision wheel from [here](https://download.pytorch.org/whl/torch_stable.html) and install it in the Pipenv virtual environment as follows:
```shell
pipenv install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
pipenv install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

## Dataset
1) Download the complete IR dataset for driver gaze classification using [this link](https://drive.google.com/file/d/1iJTlVytGsmQu9EeB1Iw1-cYwPlOx4-XW/view?usp=sharing).
2) Unzip the file.
3) Prepare the train, val and test splits as follows:
```shell
python prepare_gaze_data.py --dataset-dir=/path/to/lisat_gaze_data
```

## Training (IR data)
The prescribed three-step training procedure can be carried out as follows:
### Step 1: Train the gaze classifier on images without eyeglasses
```shell
pipenv shell # activate virtual environment
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data/ir_no_glasses/ --version=1_1 --snapshot=./weights/squeezenet1_1_rgb.pth --random-transforms
```
### Step 2: Train the GPCycleGAN model using the gaze classifier from Step 1
```shell
python gpcyclegan.py --dataset-root-path=/path/to/lisat_gaze_data/ --data-type=ir --version=1_1 --snapshot-dir=/path/to/trained/gaze-classifier/directory/ --random-transforms
```
### Step 3.1: Create fake images using the trained GPCycleGAN model
```shell
python create_fake_images.py --dataset-root-path=/path/to/lisat_gaze_data/ir_all_data/ --snapshot-dir=/path/to/trained/gpcyclegan/directory/
cp /path/to/lisat_gaze_data/ir_all_data/mean_std.mat /path/to/lisat_gaze_data/ir_all_data_fake/mean_std.mat # copy over dataset mean/std information to fake data folder
```
### Step 3.2: Finetune the gaze classifier on all fake images
```shell
python gazenet-ft.py --dataset-root-path=/path/to/lisat_gaze_data/ir_all_data_fake/ --version=1_1 --snapshot-dir=/path/to/trained/gpcyclegan/directory/ --random-transforms
exit # exit virtual environment
```

## Training (RGB data)
The prescribed three-step training procedure can be carried out as follows:
### Step 1: Train the gaze classifier on images without eyeglasses
```shell
pipenv shell # activate virtual environment
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data/rgb_no_glasses/ --version=1_1 --snapshot=./weights/squeezenet1_1_rgb.pth --random-transforms
```
### Step 2: Train the GPCycleGAN model using the gaze classifier from Step 1
```shell
python gpcyclegan.py --dataset-root-path=/path/to/lisat_gaze_data/ --data-type=rgb --version=1_1 --snapshot-dir=/path/to/trained/gaze-classifier/directory/ --random-transforms
```
### Step 3.1: Create fake images using the trained GPCycleGAN model
```shell
python create_fake_images.py --dataset-root-path=/path/to/lisat_gaze_data/rgb_all_data/ --snapshot-dir=/path/to/trained/gpcyclegan/directory/
cp /path/to/lisat_gaze_data/rgb_all_data/mean_std.mat /path/to/lisat_gaze_data/rgb_all_data_fake/mean_std.mat # copy over dataset mean/std information to fake data folder
```
### Step 3.2: Finetune the gaze classifier on all fake images
```shell
python gazenet-ft.py --dataset-root-path=/path/to/lisat_gaze_data/rgb_all_data_fake/ --version=1_1 --snapshot-dir=/path/to/trained/gpcyclegan/directory/ --random-transforms
exit # exit virtual environment
```

## Inference (IR data)
Inference can be carried out using [this](https://github.com/arangesh/GPCycleGAN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --dataset-root-path=/path/to/lisat_gaze_data/ir_all_data/ --split=test --version=1_1 --snapshot-dir=/path/to/trained/ir-models/directory/
exit # exit virtual environment
```

## Inference (RGB data)
Inference can be carried out using [this](https://github.com/arangesh/GPCycleGAN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --dataset-root-path=/path/to/lisat_gaze_data/rgb_all_data/ --split=val --version=1_1 --snapshot-dir=/path/to/trained/rgb-models/directory/
exit # exit virtual environment
```

### Pre-trained Weights
You can download our pre-trained (GPCycleGAN + gaze classifier) weights for both IR and RGB data using [this link](https://drive.google.com/file/d/1FbYhyoSbCSo6l0b08a6kMPIgLwf7FHC-/view?usp=sharing).

Config files, logs, results and snapshots from running the above scripts will be stored in the `GPCycleGAN/experiments` folder by default.