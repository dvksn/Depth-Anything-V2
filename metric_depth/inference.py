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

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
# for name, param in model.named_parameters():
#     print(name, param.shape)

state_dict =  torch.hub.load_state_dict_from_url('https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu') 
model.load_state_dict(state_dict)

## load the image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw)
## directly use the image2tensor fucntion
image, (h, w) = model.image2tensor(raw_image)



transform = transforms.Compose([
            transforms.Resize(
                width=512,
                height=512,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            transforms.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PrepareForNet(),
        ])
h, w = raw_image.shape[:2]
image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
image1 = image.to(DEVICE)
depth1 = model(image)

## use model directly on raw image
depth2 = model.infer_image(raw_image)
## preprocess the image
patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
        
features = model.pretrained.get_intermediate_layers(image, model.intermediate_layer_idx[model.encoder], 
                                                    return_class_token=True)
print("features: ", features)      

# model.eval()

# raw_img = cv2.imread('your/image/path')
# depth = model.infer_image(raw_img) 