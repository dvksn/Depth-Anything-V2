import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

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

state_dict =  torch.hub.load_state_dict_from_url('https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu') 
model.load_state_dict(state_dict)
# model.eval()

# raw_img = cv2.imread('your/image/path')
# depth = model.infer_image(raw_img) 