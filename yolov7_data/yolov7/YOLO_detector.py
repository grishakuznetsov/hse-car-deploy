import torch
import os
from .models.yolo import Model
from .utils.torch_utils import select_device
import cv2


def custom(path_or_model, autoshape=True):
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)


class YoloDetector:
    def __init__(self):
        self.model = custom(path_or_model='yolov7_data/best.pt')

    def predict(self, img_path):
        if img_path.endswith('.jpg'):
            new_img_path = img_path.replace('.jpg', '.jpeg')
            try:
                os.rename(img_path, new_img_path)
            except:
                print('file exists')
                pass
            img_path = new_img_path
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.model(img)  # batched inference
        results.print()
        results.save(save_dir='results')
        return None



if __name__ == '__main__':
    cls = YoloDetector()
    # cls.predict('../car_damage_dataset/images/train/37.jpeg')
    cls.predict('../car_damage_dataset/images/train/38.jpeg')

