import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split

np.random.seed(42)

data = pd.read_csv("../all_data.csv")
train_df, test_df = train_test_split(data, test_size=0.3)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(test_df.head(5))


np_imgs = []
for i in train_df["image"]:
    img = cv2.cvtColor(
        cv2.imread("../all_images/" + i), cv2.COLOR_BGR2RGB
    )
    np_imgs.append(img)
np_imgs = np.array(np_imgs)
train_dataset = np_imgs

np_imgs = []
for i in test_df["image"]:
    img = cv2.cvtColor(
        cv2.imread("../all_images" + i), cv2.COLOR_BGR2RGB
    )
    np_imgs.append(img)
np_imgs = np.array(np_imgs)
test_dataset = np_imgs


test_data = test_dataset
train_data = train_dataset
train_data_norm, test_data_norm = train_dataset / 255.0, test_dataset / 255.0
train_df.replace(
    {
        "bumper_dent": 1,
        "bumper_scratch": 2,
        "door_dent": 3,
        "door_scratch": 4,
        "glass_shatter": 5,
        "head_lamp": 6,
        "tail_lamp": 7,
        "unknown": 0,
    },
    inplace=True,
)
test_df.replace(
    {
        "bumper_dent": 1,
        "bumper_scratch": 2,
        "door_dent": 3,
        "door_scratch": 4,
        "glass_shatter": 5,
        "head_lamp": 6,
        "tail_lamp": 7,
        "unknown": 0,
    },
    inplace=True,
)


plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data_norm[i])
plt.show()


model4 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(224, 224, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation="softmax"),
    ]
)

optim = tf.keras.optimizers.Adam()

model4.compile(
    optimizer=optim,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history4 = model4.fit(
    train_data_norm,
    train_df["label"],
    epochs=10,
    validation_data=(test_data_norm, test_df["label"]),
)


model3 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(224, 224, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation="softmax"),
    ]
)

optim = tf.keras.optimizers.Adam()

model3.compile(
    optimizer=optim,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history3 = model3.fit(
    train_data_norm,
    train_df["label"],
    epochs=10,
    validation_data=(test_data_norm, test_df["label"]),
)


model2 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(224, 224, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(8, activation="softmax"),
    ]
)

optim = tf.keras.optimizers.Adam()

model2.compile(
    optimizer=optim,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history2 = model2.fit(
    train_data_norm,
    train_df["label"],
    epochs=10,
    validation_data=(test_data_norm, test_df["label"]),
)


model1 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(224, 224, 3)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(8, activation="softmax"),
    ]
)

optim = tf.keras.optimizers.Adam()

model1.compile(
    optimizer=optim,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history1 = model1.fit(
    train_data_norm,
    train_df["label"],
    epochs=10,
    validation_data=(test_data_norm, test_df["label"]),
)


answ = np.array(test_df["label"])
from sklearn.metrics import classification_report


def f1_score_handy(model, y_true, test_img):
    pred = model.predict(test_img)
    predfin = []
    for i in range(pred.shape[0]):
        m = np.argmax(pred[i])
        predfin.append(m)
    print(classification_report(y_true, predfin))
    return


print("Model with 2 conv layers, dense 128, no reg")
print(f1_score_handy(model1, answ, test_data_norm))
print()
print("Model with 4 conv layers, dense 256, no reg")
print(f1_score_handy(model2, answ, test_data_norm))
print()
print("Model with 4 conv layers, dense 512, dropout")
print(f1_score_handy(model3, answ, test_data_norm))
print()
print("Model with 4 conv layers, dense 512, droput + reg l2")
print(f1_score_handy(model4, answ, test_data_norm))


model_json1 = model1.to_json()
with open("model_base.json", "w") as json_file:
    json_file.write(model_json1)
model1.save_weights("model_base.h5")

model_json2 = model2.to_json()
with open("model_more_layers_more_neurons.json", "w") as json_file:
    json_file.write(model_json2)
model2.save_weights("model_more_layers_more_neurons.h5")

model_json3 = model3.to_json()
with open("model_more_layers_more_neurons_dropout.json", "w") as json_file:
    json_file.write(model_json3)
model3.save_weights("model_more_layers_more_neurons_dropout.h5")

model_json4 = model4.to_json()
with open("model_more_layers_more_neurons_dropout_l2.json", "w") as json_file:
    json_file.write(model_json4)
model4.save_weights("model_more_layers_more_neurons_dropout_l2.h5")
