from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch


class VITClassifier:
    def __init__(self):
        self.hf_hub = 'google/vit-base-patch16-384'
        self.model_path = 'dl_models/model_checkpoints/VIT'

        self.model = ViTForImageClassification.from_pretrained(self.model_path)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.hf_hub)

    def predict(self, img_path):
        inputs = self.feature_extractor(Image.open(img_path).resize((480, 480)), return_tensors='pt', size=({'width':384, 'height':384}))

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.nn.functional.softmax(logits, dim=1)

        return self.model.config.id2label[logits.argmax(-1).item()], predictions.max().item()


if __name__ == '__main__':
    i = '../all_images/0.jpg'
    v = VITClassifier()

    print(v.predict(i))