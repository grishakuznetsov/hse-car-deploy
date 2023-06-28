from PIL import Image
import numpy as np
import joblib


class SVCClassifier:
    def __init__(self):
        with open('linear_models/model_checkpoints/SVC_base.pkl', 'rb') as f:
            checkpoint = joblib.load(f)
        self.model = checkpoint

        self.label2id = {
        "bumper_dent": 1,
        "bumper_scratch": 2,
        "door_dent": 3,
        "door_scratch": 4,
        "glass_shatter": 5,
        "head_lamp": 6,
        "tail_lamp": 7,
        "unknown": 0,
    }

    @staticmethod
    def _preproc_base(img_path):
        img = Image.open(img_path)
        return np.array(img)[:, :, 0].reshape(1, -1)

    def predict(self, img):
        pred = self.model.predict(self._preproc_base(img))
        return pred[0]


if __name__ == '__main__':
    i = '../all_images/0.jpg'
    l = SVCClassifier()
    print(l.predict(i))





