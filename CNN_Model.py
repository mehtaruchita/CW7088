#Importing Libraries
import pandas as pd
import numpy as np
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense,MaxPool2D,Conv2D,Activation
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import loadtxt
from keras.models import load_model
import matplotlib.image as mpimg
import math
import datetime
import time
tf.compat.v1.get_default_graph()
'exec(%matplotlib inline)'

img_width, img_height = 80, 80

# Load dataset by giving oath to it
train_data_dir =  'C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/bloodCells_dataset/dataset2-master/images/TRAIN/'
test_data_dir = 'C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/bloodCells_dataset/dataset2-master/images/TEST/'
# number of epochs for training
epochs = 60
# batch size for epochs
batch_size = 30
###################################################################
#Splitting Validation data from training data
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15)
datagen = ImageDataGenerator(rescale=1. / 255)
#Training data generator
generator_tr = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset = 'training')
print(generator_tr)
#Validation data generator
validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')
#Test data generator
generator_ts = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
print(generator_ts)
####################################################################
#cell plot
Blood_cells    = ['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']

for k in range(3):
    i=0
    plt.figure(figsize=(25,15))
    for category in Blood_cells:
        plt.subplot(5, 5, i+1)
        plt.yticks([])
        plt.xticks([])
        path=train_dir + '/' + category
        image_p=os.listdir(path)
        plt.title(category , color='red').set_size(15)
        plt.axis('off')
        image = cv2.imread(os.path.join(path, image_p[k]))
        image = image[:, :, [2, 1, 0]]
        plt.imshow(image)
        i+=1

########################################################################
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


#######################################################################################
#Model creation
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(80, 80, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

###################################################################
#Comilation of model
model.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])
####################################################################
#Training and fitting CNN model
start = datetime.datetime.now()
history = model.fit_generator(generator_tr,
   epochs=60,
   validation_data=validation_generator, steps_per_epoch=predict_size_train)
end = datetime.datetime.now()
elapsed = end-start
model.save_weights(top_model_weights_path)
model.save('C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/model.h5')
model = load_model('C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/model.h5')
(eval_loss, eval_accuracy) = model.evaluate(
    validation_generator,verbose=1)
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))
print ('Time: ', elapsed)
########################################################################
#Graphing our training and validation

train_acc = model.history['acc']
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
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
########################################################################################
#loading model and prediction on test data

# summarize model.
model.summary()
Y_pred = np.round(model.predict_generator(generator_ts, predict_size_ts))
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
conf_matrix = confusion_matrix(generator_ts.classes,y_pred)
print(conf_matrix)
Blood_cells = ['Eosinophil','Lymphocyte','Monocyte','Neutrophil']
###########################################################################################
#Plot confusion matrix
def plot_confusion_matrix(cm, classes,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues):
    '''Add normalisation option'''
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













