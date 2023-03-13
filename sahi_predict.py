from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from sahi.models.yolov5 import Yolov5DetectionModel

# comment lines are for detection with pre-trained model
"""
from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)
"""

"""
download YOLOV5S6 model to 'models/yolov5s6.pt'
download_yolov5s6_model(destination_path=yolov5_model_path)
"""

"""
detection_model = Yolov5DetectionModel(
    #model_type='yolov5',
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cuda:0", # or 'cpu'
)
"""

yolov5_model_path = 'yolov5/yolov5s.pt'

model_type = "yolov5"
model_path = yolov5_model_path
model_device = "0" # or 'cpu'
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
