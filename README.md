# Gaze Preserving CycleGAN (GPCyceGAN) \& Other Driver Gaze Estimation Models and Datasets
PyTorch implementation of the training and inference procedures described in the papers: 
* [On Generalizing Driver Gaze Zone Estimation using Convolutional Neural Networks," IEEE Intelligent Vehicles Symposium, 2017](http://cvrr.ucsd.edu/publications/2017/IV2017-VoraTrivedi-OnGeneralizingGazeZone.pdf)
* [Driver Gaze Zone Estimation using Convolutional Neural Networks: A General Framework and Ablative Analysis," IEEE Transactions on Intelligent Vehicles, 2018](http://cvrr.ucsd.edu/publications/2018/sourabh_gaze_zone.pdf)
* [Driver Gaze Estimation in the Real World: Overcoming the Eyeglass Challenge, IEEE Intelligent Vehicles Symposium, 2020](http://cvrr.ucsd.edu/publications/2020/GPCycleGAN.pdf).
* [Gaze Preserving CycleGANs for Eyeglass Removal & Persistent Gaze Estimation, arXiv:2002.02077, 2020](http://cvrr.ucsd.edu/publications/2020/GPCycleGAN-extended.pdf).

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

## Datasets
### LISA Gaze Dataset v0
This is the dataset introduced in the papers [On Generalizing Driver Gaze Zone Estimation using Convolutional Neural Networks](http://cvrr.ucsd.edu/publications/2017/IV2017-VoraTrivedi-OnGeneralizingGazeZone.pdf) and [Driver Gaze Zone Estimation using Convolutional Neural Networks: A General Framework and Ablative Analysis](http://cvrr.ucsd.edu/publications/2018/sourabh_gaze_zone.pdf).
To use this dataset, do the following:
1) Download the complete RGB dataset for driver gaze classification using [this link](https://drive.google.com/file/d/1Ez-pHW0v-5bRdz8NjTLlzWZPT0GS2rYT/view?usp=sharing).
2) Unzip the file.

### LISA Gaze Dataset v1
This is the second dataset introduced in the paper [Driver Gaze Zone Estimation using Convolutional Neural Networks: A General Framework and Ablative Analysis](http://cvrr.ucsd.edu/publications/2018/sourabh_gaze_zone.pdf).
To use this dataset, do the following:
1) Download the complete RGB dataset for driver gaze classification using [this link](https://drive.google.com/file/d/1YvFzqfDkC2NLX8s0YX0XiMi8SOp_eINx/view?usp=sharing).
2) Unzip the file.

### LISA Gaze Dataset v2
This is the dataset introduced in the paper [Driver Gaze Estimation in the Real World: Overcoming the Eyeglass Challenge](http://cvrr.ucsd.edu/publications/2020/GPCycleGAN.pdf).
To use this dataset, do the following:
1) Download the complete IR+RGB dataset for driver gaze classification using [this link](https://drive.google.com/file/d/1iJTlVytGsmQu9EeB1Iw1-cYwPlOx4-XW/view?usp=sharing).
2) Unzip the file.
3) Prepare the train, val and test splits as follows:
```shell
python prepare_gaze_data.py --dataset-dir=/path/to/lisat_gaze_data_v2
```

## Training (v0 RGB data)
The best performing SqueezeNet gaze classifier can be trained using the following command:
```shell
pipenv shell # activate virtual environment
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data_v0/ --version=1_1 --snapshot=./weights/squeezenet1_1_imagenet.pth --random-transforms
```

## Training (v1 RGB data)
The best performing SqueezeNet gaze classifier can be trained using the following command:
```shell
pipenv shell # activate virtual environment
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data_v1/ --version=1_1 --snapshot=./weights/squeezenet1_1_imagenet.pth --random-transforms
```

## Training (v2 IR data)
The prescribed three-step training procedure can be carried out as follows:
### Step 1: Train the gaze classifier on images without eyeglasses
```shell
pipenv shell # activate virtual environment
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data_v2/ir_no_glasses/ --version=1_1 --snapshot=./weights/squeezenet1_1_imagenet.pth --random-transforms
```
### Step 2: Train the GPCycleGAN model using the gaze classifier from Step 1
```shell
python gpcyclegan.py --dataset-root-path=/path/to/lisat_gaze_data_v2/ --data-type=ir --version=1_1 --snapshot-dir=/path/to/trained/gaze-classifier/directory/ --random-transforms
```
### Step 3.1: Create fake images using the trained GPCycleGAN model
```shell
python create_fake_images.py --dataset-root-path=/path/to/lisat_gaze_data_v2/ir_all_data/ --snapshot-dir=/path/to/trained/gpcyclegan/directory/
cp /path/to/lisat_gaze_data_v2/ir_all_data/mean_std.mat /path/to/lisat_gaze_data_v2/ir_all_data_fake/mean_std.mat # copy over dataset mean/std information to fake data folder
```
### Step 3.2: Finetune the gaze classifier on all fake images
```shell
python gazenet-ft.py --dataset-root-path=/path/to/lisat_gaze_data_v2/ir_all_data_fake/ --version=1_1 --snapshot-dir=/path/to/trained/gpcyclegan/directory/ --random-transforms
exit # exit virtual environment
```

## Training (v2 RGB data)
The prescribed three-step training procedure can be carried out as follows:
### Step 1: Train the gaze classifier on images without eyeglasses
```shell
pipenv shell # activate virtual environment
python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_no_glasses/ --version=1_1 --snapshot=./weights/squeezenet1_1_imagenet.pth --random-transforms
```
### Step 2: Train the GPCycleGAN model using the gaze classifier from Step 1
```shell
python gpcyclegan.py --dataset-root-path=/path/to/lisat_gaze_data_v2/ --data-type=rgb --version=1_1 --snapshot-dir=/path/to/trained/gaze-classifier/directory/ --random-transforms
```
### Step 3.1: Create fake images using the trained GPCycleGAN model
```shell
python create_fake_images.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_all_data/ --snapshot-dir=/path/to/trained/gpcyclegan/directory/
cp /path/to/lisat_gaze_data_v2/rgb_all_data/mean_std.mat /path/to/lisat_gaze_data_v2/rgb_all_data_fake/mean_std.mat # copy over dataset mean/std information to fake data folder
```
### Step 3.2: Finetune the gaze classifier on all fake images
```shell
python gazenet-ft.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_all_data_fake/ --version=1_1 --snapshot-dir=/path/to/trained/gpcyclegan/directory/ --random-transforms
exit # exit virtual environment
```

## Inference (v0 RGB data)
Inference can be carried out using [this](https://github.com/arangesh/GPCycleGAN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --dataset-root-path=/path/to/lisat_gaze_data_v0/ --split=val --version=1_1 --snapshot-dir=/path/to/trained/rgb-model/directory/ --save-viz
exit # exit virtual environment
```

## Inference (v1 RGB data)
Inference can be carried out using [this](https://github.com/arangesh/GPCycleGAN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --dataset-root-path=/path/to/lisat_gaze_data_v0/ --split=val --version=1_1 --snapshot-dir=/path/to/trained/rgb-model/directory/ --save-viz
exit # exit virtual environment
```

## Inference (v2 IR data)
Inference can be carried out using [this](https://github.com/arangesh/GPCycleGAN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --dataset-root-path=/path/to/lisat_gaze_data_v2/ir_all_data/ --split=test --version=1_1 --snapshot-dir=/path/to/trained/ir-models/directory/ --save-viz
exit # exit virtual environment
```

## Inference (v2 RGB data)
Inference can be carried out using [this](https://github.com/arangesh/GPCycleGAN/blob/master/infer.py) script as follows:
```shell
pipenv shell # activate virtual environment
python infer.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_all_data/ --split=val --version=1_1 --snapshot-dir=/path/to/trained/rgb-models/directory/ --save-viz
exit # exit virtual environment
```

## Pre-trained Weights
You can download our pre-trained model weights using [this link](https://drive.google.com/file/d/1FbYhyoSbCSo6l0b08a6kMPIgLwf7FHC-/view?usp=sharing).

Config files, logs, results, snapshots, and visualizations from running the above scripts will be stored in the `GPCycleGAN/experiments` folder by default.

## Citations
If you find our data, code, and/or models useful in your research, please consider citing the following papers:

    @inproceedings{vora2017generalizing,
      title={On generalizing driver gaze zone estimation using convolutional neural networks},
      author={Vora, Sourabh and Rangesh, Akshay and Trivedi, Mohan M},
      booktitle={2017 IEEE Intelligent Vehicles Symposium (IV)},
      pages={849--854},
      year={2017},
      organization={IEEE}
    }

    @article{vora2018driver,
      title={Driver gaze zone estimation using convolutional neural networks: A general framework and ablative analysis},
      author={Vora, Sourabh and Rangesh, Akshay and Trivedi, Mohan Manubhai},
      journal={IEEE Transactions on Intelligent Vehicles},
      volume={3},
      number={3},
      pages={254--265},
      year={2018},
      publisher={IEEE}
    }

    @article{rangesh2020driver,
      title={Gaze Preserving CycleGANs for Eyeglass Removal & Persistent Gaze Estimation},
      author={Rangesh, Akshay and Zhang, Bowen and Trivedi, Mohan M},
      journal={arXiv preprint arXiv:2002.02077},
      year={2020}
    }
