import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
from model import AU, DSAU
import cv2

def Run(model,img):
    W_, H_ = 176, 96
    img = cv2.resize(img, (W_, H_))
    img_rs=img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    x0=img_out

    _,da_predict=torch.max(x0, 1)
    DA = da_predict.byte().cpu().data.numpy()[0]*255

    mask_indices = DA > 100
    new_color = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.65  # Full blending, can adjust for transparency
    img_rs[mask_indices] = (alpha * new_color + (1 - alpha) * img_rs[mask_indices]).astype(np.uint8)

    # img_rs[DA>100]=[255, 0, 0]

    return img_rs

config = 'DSAU'
num_model = '6'

print("config: ", config)
print(f"path: best_model/model_{config}_{num_model}.pth")

if config == 'AU':
    model = AU.attention_unet()
elif config == 'DSAU':
    model = DSAU.DS_attention_unet()

model = model.cuda()

model.load_state_dict(torch.load(f'best_model/model_{config}_{num_model}.pth'))
model.eval()

img_path = '/home/ceec/tri/Lightweight-Attention-Unet/Datasets/AugmentedDataset_new/images/val'
image_list = os.listdir(img_path)
save_path = f'results_{config}_{num_model}'

# if os.path.isdir(save_path):
#     shutil.rmtree(save_path)
#     os.mkdir(save_path)
# else:
#     os.mkdir(save_path)

img = cv2.imread('/home/ceec/tri/Lightweight-Attention-Unet/Datasets/AugmentedDataset/images/val/image190.jpg')
img = img[:, :180, :]

# # for i, imgName in enumerate(image_list):
#     img = cv2.imread(os.path.join(img_path,imgName))
img = Run(model,img)
cv2.imwrite(os.path.join(save_path, 'chokhoi.jpg'),img)