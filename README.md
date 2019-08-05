# Solutions for Tiger2019 Pose Track Challenge

### Note:
- Flip test is used.
- Input size is 384x288
- pose_unetplus is our work.


## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 8 TITAN V GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── Tiger2019
   ├     ├──coco
   ├     ├──pose
   ├     ├──inference.sh
   ├     ├──train.sh
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet101-5d3b4d8f.pth
            |   `-- resnet152-b121ed2d.pth

   ```
   
### Data preparation
**For CVWC 2019 Tiger Pose Track data**, please make them look like this:
```
${POSE_ROOT}
|-- Tiger2019
`-- |-- pose
    `-- |-- annotations
        |   |-- keypoint_train.json
        |   |-- keypoint_val.json
        |   |-- image_info_test.json
        `-- train
            |-- 001163.jpg
            |-- 003072.jpg
        `-- val
            |-- 001163.jpg
            |-- 003072.jpg
        `-- test
            |-- 001163.jpg
            |-- 003072.jpg
```
you can get our handled json files from dir 'pose/annotations', and then put them in dir 'pose/annotations/' like this:
```
pose/annotations/tiger_keypoints_train.json
```

### Training
```
bash train.sh or . ./train.sh
```

#### Testing on Tiger pose dataset using model zoo's models here
([BaiduDrive](https://pan.baidu.com/s/15r5aL6j94R0Nc0rWNnDO8Q)) (Extract Code: r916)
download our trained model and put it in dir '/share/db/Pose-Estimation-Baseline/Tiger2019/work_dir' like this:
```
/share/db/Pose-Estimation-Baseline/Tiger2019/work_dir/unetplus_r152_256x192/tiger/pose_unetplus/res152_384x288_d256x3_adam_lr1e-3/model_best.pth
```
### Testing

```
bash inference.sh or . ./inference.sh
```





