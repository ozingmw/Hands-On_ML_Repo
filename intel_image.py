import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from tqdm import tqdm
import os
from sklearn.utils import shuffle

class_name = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
class_labels = {class_name:i for i, class_name in enumerate(class_name)}

image_size = (150, 150)

def load_data():
    datasets = ["./datasets/intel/seg_train/seg_train", "./datasets/intel/seg_test/seg_test"]
    output = []
    
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        for folder in os.listdir(dataset):
            label = class_labels[folder]
            
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size) 
                
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
train_images = train_images / 255.
test_images = test_images / 255.

keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.Conv2D(32, 3, activation = 'relu', padding="same", input_shape = (150, 150, 3)), 
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, 3, activation = 'relu', padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(6, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

cb_earlystopping = keras.callbacks.EarlyStopping(patience=10)
cb_modelcheckpoint = keras.callbacks.ModelCheckpoint("./models/intel_image.h5", save_best_only=True)
model.fit(train_images, train_labels, epochs=100, validation_split=0.15, callbacks=[cb_earlystopping, cb_modelcheckpoint])

print(model.evaluate(test_images, test_labels))