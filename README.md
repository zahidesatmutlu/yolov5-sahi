# YOLOv5 + SAHI (Slicing Aided Hyper Inference)

🙌 Using [SAHI](https://github.com/obss/sahi) with [YOLOv5](https://github.com/ultralytics/yolov5) algorithm.

## Overview

💡 Object detection and instance segmentation are by far the most important fields of applications in Computer Vision. However, detection of small objects and inference on large images are still major issues in practical usage. Here comes the SAHI to help developers overcome these real-world problems with many vision utilities.

<p align="center">
  <img src="https://i.hizliresim.com/ljh8i5u.jpg" />
Standard Inference with a YOLOv5 Model
</p>
<p align="center">
  <img src="https://i.hizliresim.com/7mdgcuq.png" />
Sliced Inference with a YOLOv5 Model (YOLOv5 + SAHI)
</p>

## Installations ⬇️

✔️ A virtual environment is created for the system. (Assuming you have [Anaconda](https://www.anaconda.com/) installed.)

```bash
conda create -n yolov5sahi python -y
conda activate yolov5sahi
```

✔️ Clone repo and install [requirements.txt](https://github.com/zahidesatmutlu/yolov5-sahi/blob/master/requirements.txt) in a [Python>=3.7.0](https://www.python.org/downloads/) (3.9 recommended) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/) (1.9.0 recommended).

```bash
git clone https://github.com/zahidesatmutlu/yolov5-sahi  # clone
cd yolov5-sahi
pip install -r requirements.txt  # install
```

✔️ Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-6-0-download-archive) version 11.6 and install [PyTorch](https://pytorch.org/get-started/previous-versions/) version 1.9.0.

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

✔️ Copy the test folder containing the images you will detect and your best.pt weight file to the project folder.

```bash
./yolov5-sahi/%here%
```

✔️ The file structure should be like this:

```bash
yolov5-sahi/
    .idea
    sahi
    test
    venv
    yolov5
    sahi_predict.py
```

## Usage 🔷

```python
from sahi.predict import get_prediction, get_sliced_prediction, predict

yolov5_model_path = 'yolov5s.pt' # if you have a pre-trained weight file copy it to the project folder and replace it

model_type = "yolov5"
model_path = yolov5_model_path
model_device = "0" # cuda device, i.e. 0 or 0,1,2,3 or cpu
model_confidence_threshold = 0.8

slice_height = 512
slice_width = 512
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "test/"

predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)
```

## Resources 🤝

🔸 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

🔸 [https://github.com/obss/sahi](https://github.com/obss/sahi)
