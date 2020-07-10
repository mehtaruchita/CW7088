#Import Libraries
import numpy as np
import math,  os
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Activation, Dense,Flatten, Conv2D, Reshape,Dropout
from keras.models import Model
from keras import optimizers
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from random import randint
from sklearn.utils import shuffle
import pandas as pd
import time
import tensorflow as tf
import datetime
tf.compat.v1.get_default_graph()
'exec(%matplotlib inline)'
#Loading data by giving path
train_data_dir = "C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/bloodCells_dataset/dataset2-master/images/TRAIN"
test_data_dir = "C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/bloodCells_dataset/dataset2-master/images/TEST"
data_list     = os.listdir(train_data_dir)
NUM_CLASSES   = len(data_list)
batch_size    = 25
img_width, img_height = 80,80
Blood_cells    = ['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
#Splitting training data into training and validation data
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15)
datagen = ImageDataGenerator(rescale=1. / 255)
#Train, Validation and test data image generator
generator_tr = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset = 'training')

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

generator_ts = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
print(generator_ts)
#Train_label and step_size for training data
nb_train_samples = len(generator_tr.filenames)
num_classes = len(generator_tr.class_indices)
predict_size_train = int(math.ceil(nb_train_samples / batch_size))
print(predict_size_train)
train_labels = generator_tr.classes
######################################################################################
#Validationa label and step_size for validation data
nb_validation_samples = len(validation_generator.filenames)
num_classes_val = len(validation_generator.class_indices)
predict_size_val = int(math.ceil(nb_validation_samples / batch_size))
print(predict_size_val)
validation_labels = validation_generator.classes
####################################################################################
#Test label and stepsize for test data
nb_test_samples = len(generator_ts.filenames)
num_classes_test = len(generator_ts.class_indices)
predict_size_ts = int(math.ceil(nb_test_samples / batch_size))
test_labels = generator_ts.classes
#######################################################################################

model_MLP = keras.models.Sequential([
    keras.layers.Dense(250, activation=keras.activations.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(125, activation=keras.activations.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(75, activation=keras.activations.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation=keras.activations.softmax)
])
#Compilation of Multilayer perceptron model
model_MLP.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])
#Evaluating Traing time
start = datetime.datetime.now()
#Fitting MLP model
history = model_MLP.fit(generator_tr, epochs=55,validation_data=validation_generator, steps_per_epoch=predict_size_train)
end = datetime.datetime.now()
elapsed = end - start
#saving model weight and model MLP
model_MLP.save_weights('model_mlp_weight_1.h5')
model_MLP.save('C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/model_MLP_1.h5')
#Loading MLP model
model = load_model('C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/model_1.h5')
(eval_loss, eval_accuracy) = model_MLP.evaluate(
    validation_generator,verbose=1)
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))
print ('Time: ', elapsed)
#####################################################################################
#Plotting training and validation accuracy
train_acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_acc))
plt.plot(epochs, train_acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.figure()
#Plotting training and validation loss
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
################################################################################

# Prediction on test data
model_MLP.summary()
Y_pred = np.round(model_MLP.predict_generator(generator_ts, predict_size_ts))
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
conf_matrix = confusion_matrix(generator_ts.classes,y_pred)
print(conf_matrix)
###########################################################################################
#Plot confusion matrix
def plot_confusion_matrix(cm, classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

plot_confusion_matrix(conf_matrix,['Eosinophil','Lymphocytre','Monocyte','Neutrophil'], normalize = True)


