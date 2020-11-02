from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import imutils
import numpy as np
import cv2
import os
from keras import preprocessing
import random

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        model = Sequential()
        inputShape = (height, width, depth)

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Output
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        return model
    
from google.colab import drive
drive.mount("/content/drive")

train_file = "/content/drive/My Drive/Colab Notebooks/NORMAL.zip"
import zipfile
with zipfile.ZipFile(train_file, 'r') as z:
  z.extractall()
  
data = []
labels = [] 
EPOCHS = 30
INIT_LR = 1e-3
BS = 9
image_dims = (96, 96 , 3)

def load_image(file_path ):

  label = file_path.split(os.path.sep)[-2].split("_")
  image = cv2.imread(file_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (image_dims[1], image_dims[0]))

 #update the data and labels lists, respectively
  data.append(image)
  labels.append(label)

train_path = "./NORMAL"

image_files = os.listdir(train_path)
for file in image_files:
  print(train_path + "/" + file)
train_images = [load_image(train_path + "/" + file) for file in image_files]

test_file = "/content/drive/My Drive/Colab Notebooks/PNEUMONIA-BACTERIAL.zip"
import zipfile
with zipfile.ZipFile(test_file, 'r') as z:
  z.extractall()
  
def load_image(file_path):

  label = file_path.split(os.path.sep)[-2].split("_")
  image = cv2.imread(file_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (image_dims[1], image_dims[0]))

  #update the data lists, respectively
  data.append(image)
  labels.append(label)
  
test_path = "./PNEUMONIA-BACTERIAL"

image_files = os.listdir(test_path)
for file in image_files:
  print(test_path + "/" + file)
test_images = [load_image(test_path + "/" + file) for file in image_files]

COVID_file = "/content/drive/My Drive/Colab Notebooks/PNEUMONIA-VIRAL.zip"
import zipfile
with zipfile.ZipFile(COVID_file, 'r') as z:
  z.extractall()
  
def load_image(file_path):

  label = file_path.split(os.path.sep)[-2].split("_")
  image = cv2.imread(file_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(image_dims[1], image_dims[0]))

  #update the data lists, respectively
  data.append(image)
  labels.append(label)
  
Coivd_path = "./PNEUMONIA-VIRAL"

image_files = os.listdir(Coivd_path)
for file in image_files:
  print(Coivd_path + "/" + file)
Covid_images = [load_image(Coivd_path + "/" + file) for file in image_files]

COVID_file = "/content/drive/My Drive/Colab Notebooks/COVID.zip"
import zipfile
with zipfile.ZipFile(COVID_file, 'r') as z:
  z.extractall()
  
def load_image(file_path):

  label = file_path.split(os.path.sep)[-2].split("_")
  image = cv2.imread(file_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(image_dims[1], image_dims[0]))

  #update the data lists, respectively
  data.append(image)
  labels.append(label)
  
Coivd_path = "./COVID-19"

image_files = os.listdir(Coivd_path)
for file in image_files:
  print(Coivd_path + "/" + file)
Covid_images = [load_image(Coivd_path + "/" + file) for file in image_files]

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
data.shape

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

print("class labels:")
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))
    
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.4, train_size = 0.6, random_state=42)
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

model = SmallerVGGNet.build(
    width=image_dims[1], height=image_dims[0],
    depth=image_dims[2], classes=len(mlb.classes_),
    finalAct="sigmoid")

model.compile(loss="binary_crossentropy", optimizer='adam',
              metrics=["accuracy"])

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

print("[INFO] saving COVID-19 detector model...")
model.save_weights("model.tf")

model.save('saved_model\VGGNET16.h5')
from google.colab import files
uploads = files.upload()

uploads.keys()
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import tensorflow as tf


eval_model = model
eval_model.load_weights("model.tf")
pics = []

# load an image from file
for ima in uploads.keys():
    img = cv2.imread(ima)
    img = cv2.resize(img, (96, 96))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    proba = eval_model.predict(img)[0]
    pics.append(img)

    print(proba)
    idxs = np.argsort(proba)[::-1][:2]
    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
      print("{}: {:.2f}%".format(label, p * 100))

    # plot image and label
    plt.figure(figsize=(10,10))
    for (i, j) in enumerate(idxs):
      # build the label and draw the label on the image
      label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
      plt.text(10, (i * 30) + 25, label, fontsize=16, color='y')

      output = load_img(ima)    
      plt.xticks([])
      plt.yticks([])
      plt.imshow(output)