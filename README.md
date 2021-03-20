# Contour-aware Contrastive learning for Image Manipulation Localization


This is a PyTorch implement of the CaCL-Net for IML described in the paper
["Contour-Aware Contrastive Learning for Image Manipulation Localization."]


## Compatibility
The code is tested using PyTorch 0.4.1 under Windows10 with Python 3.6. 
The test cases can be found [here](https://github.com/davidsandberg/facenet/tree/master/test)
and the results can be found [here](http://travis-ci.org/davidsandberg/facenet).

<!--
### TODO
- [x] Support different backbones
- [x] Support CASIA, NIST16, Coverage and Columbia datasets
- [x] Multi-GPU training

-->

## Pre-trained models
| Model name      | f1 score | AUC | Training dataset 
|-----------------|----------|----|------------------|
| [20210320-847](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.768    | 0.738   | CASIA    | 
| [20210317-010](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.950    | 0.962   | NIST     |

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.

## Inspiration
The code is heavily inspired by the 
[DeepLab v3+](https://github.com/jfzhang95/pytorch-deeplab-xception) implementation.



## Training data
| DataSet        |  [NIST16](https://www.nist.gov/itl/iad/mig/nimble-challenge-2017-evaluation.) |  [Casia](http://forensics.idealtest.org/.) |  [Coverage](https://github.com/wenbihan/coverage) |  [Columbia](http://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm) |
|----------------|------------ |------------| --------------|--------------|
|   Num   | 564  | 921(1.0) + 5123(2.0)  |100         | 180    |
 
The above four datasets are used for training. Each of them is consist of tampered images with corresponding ground-truth masks. 
The binary ground-truth masks must be prepared for training, in which 1 denotes manipulated pixels while 0 represents authentic pixels.

<!--Except Columbia, the other datasets provides binary ground-truth mask. 
To train Columbia, we transform the edge masks into binary ground-truth masks, 
-->

<!--
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. 
This training set consists of total of 453 453 images over 10 575 identities after face detection. 
Some performance improvement has been seen if the dataset has been filtered before training. 
Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset 
consisting of ~3.3M faces and ~9000 classes.
-->
<!--
## Pre-processing

    别的方法是怎么做的, 训练之前的准备

Current models adopt  
### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples 
(partial occlusion, silhouettes, etc). This makes the training set too "easy" which causes the model 
to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. 
One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). 
A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) 
and this has been used for face alignment with very good results. A Python/Tensorflow implementation of MTCNN 
can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align). 
This implementation does not give identical results to the Matlab/Caffe implementation but the performance is very similar.
-->

## Running training
Based on the encoder-decoder structure, the model combined contrastive learning and contour binary cross-entropy achieves
the best results. Details on how to train a model using CaCL-Net on the test dataset can be found on the page
[Training of CaCL-Net]().

<!--
Currently, the best results are achieved by training the model using softmax loss. 
Details on how to train a model using softmax loss on the CASIA-WebFace dataset can be found 
on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1).
-->

## Pre-trained models
A couple of pretrained models are provided. They are trained with the CaCL-Net on different datasets. 
The pre-prepared datasets has been provided on [pre-prepared dataset](). 

<!---
A couple of pretrained models are provided. They are trained using softmax loss with the Inception-Resnet-v1 model. 
The datasets has been aligned using [MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).
-->

## Performance
The accuracy on NIST16 for the model [20210317-010]() is 0.961909. 
The `test.py` can be used to test the provided model(change the option `--in-path` and `--ckpt`
according to the directory of test data and pre-trained model.)

<!---
The accuracy on LFW for the model [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) 
is 0.99650+-0.00252. A description of how to run the test can be found on the page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw). 
Note that the input images to the model need to be standardized using fixed image standardization 
(use the option `--use_fixed_image_standardization` when running e.g. `validate_on_lfw.py`).
--->








































#######==========================================================================================#############

### Introduction
This is a PyTorch(0.4.1) implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. Currently, we train DeepLab V3 Plus
using Pascal VOC 2012, SBD and Cityscapes datasets.

![Results](doc/results.png)


### Installation
The code was tested with Anaconda and Python 3.6. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```
### Training
Follow steps below to train your model:

0. Configure your dataset path in [mypath.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/mypath.py).

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    ```

2. To train deeplabv3+ using Pascal VOC dataset and ResNet as backbone:
    ```Shell
    bash train_voc.sh
    ```
3. To train deeplabv3+ using COCO dataset and ResNet as backbone:
    ```Shell
    bash train_coco.sh
    ```    

### Acknowledgement
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)
