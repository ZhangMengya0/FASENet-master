import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from .model.FASENet import FASENet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FASENet()

model.load_state_dict(torch.load(r".checkpoint\epoch-099.pth", map_location=device))
model.eval().to(device)

input_folder = r'.VOCdevkit\VOC2007\test\img'
output_folder = r'.output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

with torch.no_grad():
    for image_file in image_files:

        img_path = os.path.join(input_folder, image_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        prediction = model(img_tensor)
        # prediction,_,_,_ = model(img_tensor)
        prediction = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
        prediction = (prediction * 255).astype(np.uint8)

        mask_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_mask.png')
        Image.fromarray(prediction.astype(np.uint8)).save(mask_path)


