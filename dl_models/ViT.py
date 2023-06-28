import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from transformers import (
    ViTImageProcessor,
    BeitFeatureExtractor,
    ViTForImageClassification,
    BeitForImageClassification,
)
from datasets import Dataset, DatasetDict
import torch
import numpy as np
from datasets import load_metric
from transformers import Trainer, TrainingArguments

np.random.seed(42)

#  обучали в кегле поэтому пути такие

df = pd.read_csv(
    "../all_data.csv"
)


def fix_path(x):
    return (
        "../all_images"
        + x.split("/")[-1]
    )


df["image_path"] = df["image"].apply(fix_path)


df = df[df["label"] != "unknown"]
labels = set(df["label"])

# id2l и l2id для файнтюнинга инференса

id2label = {i: c for i, c in enumerate(labels)}
label2id = {c: i for i, c in enumerate(labels)}


df["labels"] = df["label"].apply(lambda x: label2id[x])


df_train, df_test = train_test_split(df)

tds = Dataset.from_pandas(df_train)
vds = Dataset.from_pandas(df_test)

ds = DatasetDict()
ds["train"] = tds
ds["test"] = vds

# предобученная модель
model_name_or_path = "google/vit-base-patch16-384"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)


def transform_one(x):
    inputs = feature_extractor(
        Image.open(x["image_path"]), return_tensors="pt", size=(384, 384)
    )
    inputs["labels"] = x["labels"]
    return inputs


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(
        [Image.open(x).resize((480, 480)) for x in example_batch["image_path"]],
        return_tensors="pt",
        size=({"width": 384, "height": 384}),
    )

    inputs["labels"] = example_batch["labels"]
    return inputs


prepared_ds = ds.with_transform(transform)


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


metric = load_metric("f1")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="micro",
    )


# модель и конфиг для трейнера
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


training_args = TrainingArguments(
    output_dir="./vit-base",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=feature_extractor,
)


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


metrics = trainer.evaluate(prepared_ds["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
