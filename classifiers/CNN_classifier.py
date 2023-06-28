import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn


class BasicBlockNet4(nn.Module):  # best
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64)
        )
        self.con11 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1))
        self.finrelu = nn.Sequential(nn.ReLU())
        self.pool_out = nn.Sequential(nn.AvgPool2d(64))
        self.classifier = nn.Linear(in_features=576, out_features=7)

    def forward(self, x):
        con = self.con11(x)
        feat_map = self.net(x)
        out = self.finrelu(con + feat_map)
        out = self.pool_out(out)
        out = torch.flatten(out, start_dim=1, end_dim=3)
        out = self.classifier(out)

        return out

class CNNClassifier:
    def __init__(self):
        # self.model = torch.jit.load('CNN_TORCH/CNN_FIN_MODEL.pt', map_location=torch.device('cpu'))


        self.model = torch.load('CNN_TORCH/CNN_fin.pt', map_location=torch.device('cpu'))

        self.label2id = {
            'unknown': 0,
            'tail_lamp': 7,
            'head_lamp': 6,
            'glass_shatter': 5,
            'door_scratch': 4,
            'door_dent': 3,
            'bumper_scratch': 2,
            'bumper_dent': 1
        }
        self.id2label = {
            1: 'bumper_dent',
            2: 'bumper_scratch',
            3: 'door_dent',
            4: 'door_scratch',
            5: 'glass_shatter',
            6: 'head_lamp',
            7: 'tail_lamp',
            0: 'unknown'
        }

    @staticmethod
    def _preproc_base(img_path):
        img = Image.open(img_path)
        return np.array([np.array(img)])

    @staticmethod
    def transform_single_image(s_img):

        img_as_img = Image.open(s_img).resize((224, 224))
        # img_as_img = Image.open(s_img)
        my_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img_as_tensor = my_transform(img_as_img)
        return img_as_tensor.unsqueeze(0)

    def predict(self, name):
        tens = self.transform_single_image(name)
        rez = self.model.forward(tens)
        probs = torch.nn.functional.softmax(rez, dim=1)
        conf, pred = torch.max(probs, 1)
        dict_ans = {'bumper_dent': 1, 'bumper_scratch': 2, 'door_dent': 3, 'door_scratch': 4,
                    'glass_shatter': 5, 'head_lamp': 6, 'tail_lamp': 0}
        ans = None
        for k in dict_ans.keys():
            if dict_ans[k] == pred:
                ans = k
        return ans, conf.item()


if __name__ == '__main__':
    i = '../all_images/0.jpg'
    c = CNNClassifier()
    pred = c.predict(i)
    print(pred)
    #
    # m = np.argmax(pred)
    # print(c.id2label[m])

