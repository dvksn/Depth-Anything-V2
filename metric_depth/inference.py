import torch 
import numpy as np
from PIL import Image
import requests

import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from torchvision import transforms
## load the model
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
for name, param in model.named_parameters():
    print(name, param.shape)

state_dict =  torch.hub.load_state_dict_from_url('https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu') 
model.load_state_dict(state_dict)

import os
def download_image(image_url, file_dir):
    response = requests.get(image_url)

    if response.status_code == 200:
        directory = os.path.dirname(file_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_dir, "wb") as fp:
            fp.write(response.content)
        print("Image downloaded successfully.")
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
file_dir = "/content/download.jpg"
download_image(image_url, file_dir)

## load the image
raw_img = cv2.imread(file_dir)
# ## directly use the image2tensor fucntion
image, (h, w) = model.image2tensor(raw_img)

## use model directly on raw image
#depth2 = model.infer_image(raw_image)
## preprocess the image
patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
        
features = model.pretrained.get_intermediate_layers(image, model.intermediate_layer_idx[model.encoder], 
                                                    return_class_token=True)
# print("features: ", features)      

for item in features:
    if isinstance(item[0], torch.Tensor):
        print("registered_tokens", item[0].shape)
    else:
        print("registered_tokens", item[0])
    if isinstance(item[1], torch.Tensor):
        print("class_tokens", item[1].shape)
    else:
        print("class_tokens", item[1])

depth = model.forward(image)
print("depth: ", depth)

depth = model.infer_image(raw_img)
print("depth using infer image:", depth)
# model.eval()

# raw_img = cv2.imread('your/image/path')
# depth = model.infer_image(raw_img) 