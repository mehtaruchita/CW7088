#importing libraries
import pandas as pd
import numpy as np
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
#%matplotlib inline
import math
import datetime
import time
import tensorflow_datasets as tfds
tf.compat.v1.get_default_graph()
'exec(%matplotlib inline)'
#Width and Height of Image
img_width, img_height = 240,240
#Bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5'
# Loading dataset by giving path
train_data_dir =  'C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/bloodCells_dataset/dataset2-master/images/TRAIN/'
test_data_dir = 'C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/bloodCells_dataset/dataset2-master/images/TEST/'
epochs = 70
# Batch_size
batch_size = 16
#Using Pretrained Model VGG-16
start = datetime.datetime.now()
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
#################################################################################
datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
nb_train_samples = len(generator.filenames)
num_classes = len(generator.class_indices)
predict_size_train = int(math.ceil(nb_train_samples / batch_size))
print(predict_size_train)
bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train)
print(bottleneck_features_train)
np.save('bottleneck_features_train.npy', bottleneck_features_train)
train_data = np.load('bottleneck_features_train.npy')
print(train_data)
# Labels for training data
train_labels = generator.classes
#Converting Labels to categorical Values
train_labels = to_categorical(train_labels, num_classes=num_classes)
print(train_labels)
######################################################################################
generator_top = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical',
    shuffle=False)
print(generator_top)
print(len(generator_top.filenames))
print(len(generator_top.class_indices))
nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)
#############################################################################
#Building a model
start = datetime.datetime.now()
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.5))
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
#Compilation of the model
model.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])
#Fitting model
history = model.fit(train_data, train_labels,
   epochs=70,
   batch_size=batch_size)
end = datetime.datetime.now()
elapsed = end - start
#Savong model weights and saving model
model.save_weights(top_model_weights_path)
model.save('C:/Ruchita/MSc_Data_Science/Module_6-7088-ANN/CW-7088/model_VGG16_bottleneck.h5')
#time to fit the model

print ('Time: ', elapsed)
#Plotting accuracy and loss
acc = history.history['acc']
loss = history.history['loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.title('Training  accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training  loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
####################################################################################
#Prediction using test dataset
generator_1 = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical',
    shuffle=False)
nb_test_samples = len(generator_1.filenames)
num_classes_1 = len(generator_1.class_indices)
predict_size_test = int(math.ceil(nb_test_samples / batch_size))
bottleneck_features_test = vgg16.predict_generator(generator_1, predict_size_test)
np.save('bottleneck_features_test.npy', bottleneck_features_test)
test_data = np.load('bottleneck_features_test.npy')
test_labels = generator_1.classes
# converting test label into categorical values
test_labels = to_categorical(test_labels, num_classes=num_classes)
print(test_labels)
####################################################################################
#Prediction of blood cell category by using vgg model
preds = np.round(model.predict((test_data),0))
categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)
#Calculation of confusion matrix
conf_matrix= confusion_matrix(categorical_test_labels,categorical_preds)
print(conf_matrix)
#Plotting Confusion matrix
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




