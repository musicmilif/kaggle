import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Activation
from keras.optimizers import RMSprop
from .hadamard import HadamardClassifier

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception


def vgg16_model(input_shape=(197, 197, 3), num_classes=1, learning_rate=0.0001, weight_path=None):
    base_model=VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(Flatten())
    model.add(HadamardClassifier(output_dim=num_classes))
    model.add(Activation('softmax'))
    
    if weight_path:
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), 
                  metrics=['accuracy'])
    return model


def vgg19_model(input_shape=(197, 197, 3), num_classes=1, learning_rate=0.0001, weight_path=None):
    base_model=VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(Flatten())
    model.add(HadamardClassifier(output_dim=num_classes))
    model.add(Activation('softmax'))
    
    if weight_path:
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), 
                  metrics=['accuracy'])
    
    return model


def resnet50_model(input_shape=(197, 197, 3), num_classes=1, learning_rate=0.0001, weight_path=None):
    base_model=ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    # Remove top average pooling layer
    base_model.layers.pop()
    base_model.layers.pop()
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(AveragePooling2D(pool_size=4, strides=1))
    model.add(Flatten())
    model.add(HadamardClassifier(output_dim=num_classes))
    model.add(Activation('softmax'))
    
    if weight_path:
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), 
                  metrics=['accuracy'])
    
    return model


def xception_model(input_shape=(197, 197, 3), num_classes=1, learning_rate=0.0001, weight_path=None):
    base_model=Xception(include_top=False, weights=None, input_shape=input_shape)
    for i in range(10):
        base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(AveragePooling2D(pool_size=6, strides=2))
    model.add(Flatten())
    model.add(HadamardClassifier(output_dim=num_classes))
    model.add(Activation('softmax'))
    
    if weight_path:
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), 
                  metrics=['accuracy'])

    return model
