from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
import tensorflow.keras as keras
import sklearn
from sklearn import linear_model
from keras.utils import to_categorical
import matplotlib.pyplot as pltpath
from pydub.playback import play
from pydub import AudioSegment   
from scipy.io import wavfile 
from scipy import signal
import soundfile as sf
import numpy as np
import os.path
import os

Audio_file = os.path.dirname("C:\\Users\\Umer Naveed\\Desktop\\Audio")

audio = []
labels = []

# importing phenumonia audio 
def load_audio(file_path):
    extension = os.path.splitext(file_path)[1][1:]
    if extension in ('.wav'):
        label = file_path.split(os.path.sep)[-2].split("_")
    data, samplerate = sf.read(file_path)
    frequencies, times, spectrogram = signal.spectrogram(data, samplerate)
    labels.append(label)
    audio.append(spectrogram)
    
audio_path = "./Audio"
audio_files = os.listdir(audio_path)
train_images = [load_audio(audio_path + "/" + file) for file in audio_files]

##Importing healthy audio

HealthyAudio_file =  os.path.dirname("C:\\Users\\Umer Naveed\\Desktop\\HealthyAudio")
  
def load_audio(file_path):
    extension = os.path.splitext(file_path)[1][1:]
    if extension in ('.wav'):
        label = file_path.split(os.path.sep)[-2].split("_")
        data, samplerate = sf.read(file_path)
    if data.size == 882000:
        frequencies, times, spectrogram = signal.spectrogram(data, samplerate)
        labels.append(label)
        audio.append(spectrogram)
      
audio_path = "./HealthyAudio"
audio_files = os.listdir(audio_path)
train_images = [load_audio(audio_path + "/" + file) for file in audio_files]

audio = np.array(audio) / 255.0
labels = np.array(labels)

print(audio.size)
print(labels.size)

## Prepaing Dataset

def prepare_Datasets(test_size,validation_size):
    x_train, x_test, y_train, y_test = train_test_split(audio, labels, test_size = test_size)
    # Create train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = validation_size)
    #3d array => (130, 13, 1)
    x_train = x_train[..., np.newaxis] # 4d array -> [num_Smaples, 130, 13, 1]
    
    x_validation = x_validation[..., np.newaxis] # 4d array -> [num_Smaples, 130, 13, 1]

    x_test = x_test[..., np.newaxis] # 4d array -> [num_Smaples, 130, 13, 1]
    
    return x_train, x_validation, x_test, y_train, y_validation, y_test


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

## create Train, Validation and Test sets 
x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_Datasets(0.25, 0.2) 

def build_modal(input_shape):
  #Create Model
  model = keras.Sequential()
  #1st conv layer 
  #relu =  rectified linear unit
  model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape ))   
  model.add(keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = 'same'))
  model.add(keras.layers.BatchNormalization())

  #2nd conv layer 
  model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu'))   
  model.add(keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = 'same'))
  model.add(keras.layers.BatchNormalization())

  #3rd conv layer 
  model.add(keras.layers.Conv2D(32, (2, 2), activation = 'relu' ))   
  model.add(keras.layers.MaxPool2D((2, 2), strides = (2, 2), padding = 'same'))
  model.add(keras.layers.BatchNormalization())

  #flatten the output
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(64, activation = 'relu'))
  model.add(keras.layers.Dropout(0.3))

  #Output
  # model.add(keras.layers.Dense(#number of objects,activation))
  model.add(keras.layers.Dense(2, activation = 'softmax' ))

  return model

#Build the CNN Net
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
model = build_modal(input_shape)

#compile Network
optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer = optimizer,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])


print(x_train.shape)
#train 
model.fit(x_train, y_train, validation_data = (x_validation, y_validation), batch_size = 3, epochs = 30)

print("[INFO] saving COVID-19 detector model...")
model.save_weights("model.tf")

#evaluate the cnn on the test set
test_error, test_accuracy = model.evaluate(x_test, y_test, verbose = 1)
print("Accuracy on test set is: {}".format(test_accuracy))

def predict(model, x, y):
  x = x[np.newaxis, ...]
  x = x[..., np.newaxis]
  # prediction = [ [0.1, 0.2, ...] ]
  predictions = model.predict(x) # X -> (1, 130, 13, 1) 
  # extract index with max value
  predicted_index = np.argmax( predictions, axis = -1) # [4]
  print( "Expected index: {}, Predicted index: {}".format(y, predicted_index) )


  cols = 4
  rows = np.ceil(len(x)/cols)
  
  for i in range(len(x)):
    pltpath.title("Normal" if np.argmax(predicted_index[i]) == 0 else "Pnuemonia")
    pltpath.axis('off')
    
    
    
from google.colab import files
uploads = files.upload()
uploads.keys()

# running our model
eval_model = model
eval_model.load_weights("model.tf")

for ima in uploads.keys():
    data, samplerate = sf.read(ima)
    frequencies, times, spectrogram = signal.spectrogram(data, samplerate)
    
    #make prediction on a sample
    predict( model , spectrogram,  spectrogram )