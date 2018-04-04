import numpy as np
import os
import tensorflow as tf
from keras.models import Model

from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, Dropout, Conv2DTranspose, UpSampling2D, Lambda
from keras.layers.normalization import BatchNormalization as bn
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import add
import numpy as np
from keras.regularizers import l2
import cv2
import glob
import h5py


#loading data

x_train = []
y_train = []
shapes = []
names=[]
X_test_data = []

TRAINING_PATH = "/home/parya/datascience/p4/data/train/data"
TEST_FOLDER = "/home/parya/datascience/p4/data/test/data"
OUTPUT_FOLDER = "/home/parya/datascience/p4/output/"

import random

X_train_data = []
Y_Train_data =[]

for folder in sorted( os.listdir( TRAINING_PATH ) ):
    xTrainFiles = random.sample( sorted(glob.glob ( os.path.join( TRAINING_PATH , folder+ "/*.png" ) )) , 10 )

    Y_train_image = cv2.imread(os.path.join(TRAINING_PATH, folder + "/mask.png") , 0 )
    Y_train_image = cv2.resize(Y_train_image, (512, 512))
    Y_train_image = (Y_train_image == 2)
    Y_train_image = Y_train_image.astype(int)

    for myFile in xTrainFiles:
        image = cv2.imread (myFile,1)
        image=cv2.resize(image, (512, 512))
        X_train_data.append (image)

        Y_Train_data.append( Y_train_image )


for folder in sorted(os.listdir(TEST_FOLDER)):
    XTestFiles = sorted( glob.glob(os.path.join(TEST_FOLDER,  folder+ "/frame0050.png") ))

    for myFile in XTestFiles :
        names.append( myFile[ -78 :-14 ] )

        image = cv2.imread(myFile, 1)
        shapes.append( (image.shape[0], image.shape[1]) )

        image = cv2.resize(image, (512, 512))
        X_test_data.append(image)
        

X_train = np.array( X_train_data )
Y_train = np.array( Y_Train_data )
X_test = np.array(X_test_data)


Y_train=Y_train.reshape(Y_train.shape+(1,))

print( X_train.shape , Y_train.shape , X_test.shape )

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f )
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def UNet(input_shape,learn_rate=1e-3):
    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape)
    input_prob=Input(input_shape)
    input_prob_inverse=Input(input_shape)
    #Conv3D(filters,(3,3,3),sjfsjf)
    conv1 = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    conv1 = bn()(conv1)
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv1)
    conv1 = bn()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(DropP)(pool1)



    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
    conv2 = bn()(conv2)
    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv2)
    conv2 = bn()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(DropP)(pool2)



    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)
    conv3 = bn()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv3)
    conv3 = bn()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(DropP)(pool3)



    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(pool3)
    conv4 = bn()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv4)
    conv4 = bn()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(DropP)(pool4)



    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
    conv5 = bn()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv5)
    conv5 = bn()(conv5)
    
    up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5), conv4],name='up6', axis=3)
    up6 = Dropout(DropP)(up6)
    conv6 = Conv2D(256,(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
    conv6 = bn()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv6)

    conv6 = bn()(conv6)
    up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6), conv3],name='up7', axis=3)
    up7 = Dropout(DropP)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(up7)
    conv7 = bn()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv7)
    conv7 = bn()(conv7)

    up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7), conv2],name='up8', axis=3)
    up8 = Dropout(DropP)(up8)
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(up8)
    conv8 = bn()(conv8)
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv8)
    conv8 = bn()(conv8)

    up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8), conv1],name='up9',axis=3)
    up9 = Dropout(DropP)(up9)
    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(up9)
    conv9 = bn()(conv9)
    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda) )(conv9)
    conv9 = bn()(conv9)   

    conv10 = Conv2D(1, (1, 1), activation='sigmoid',name='conv10')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model


model=UNet(input_shape=(512,512,1))
print(model.summary())

print(X_train.shape)
print(y_train.shape)

if os.path.isfile( 'ds-project4-unet-b16ep620.h5'): 
    model = load_model( 'ds-project4-unet-b16ep620.h5' ,custom_objects=
    { 'BilinearUpSampling2D':BilinearUpSampling2D,'dice_coef_loss':dice_coef_loss,'dice_coef':dice_coef})
else :
# training network
    model.fit([X_train], [Y_train], batch_size=16, epochs=620, shuffle=True)
    model.save('ds-project4-unet-b16ep620.h5')


# post processing :
predicted = model.predict(X_test)
for i in range ( len(shapes) ) :
    image = (predicted[i]*255).astype(np.uint8)
    image = cv2.resize(image, ( shapes[i][1], shapes[i][0]) , interpolation=cv2.INTER_AREA)
    image = (image != 0).astype(int)*2
    cv2.imwrite( os.path.join(OUTPUT_FOLDER,  names[i] +'.png' ) , image )
print('done')
    