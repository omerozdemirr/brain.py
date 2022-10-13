import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')


import cv2
from glob import glob
from itertools import chain 
from skimage.io import imread,imshow,concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras.models import Model, load_model,save_model
from tensorflow.keras.layers import Input,Activation,BatchNormalization, Dropout,Lambda,Conv2D,Conv2DTranspose,MaxPooling2D,concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from unet import *
from util import *
#setting size parameters of images 


im_width =256
im_height =256

#loading the image and mask paths 


image_file_names_train = []
#creating a list of all images containing the word 'mask'

mask_files = glob('lgg-mri-segmentation/kaggle_3m/*/*_mask*')
for i in mask_files:
    image_file_names_train.append(i.replace('_mask', ''))
    
    
    
   #plot few images and mask 
   
   
plot_from_img_path(3, 3, image_file_names_train, mask_files)
    
    
    
 #create data frame and split data on train set validation and test set
   
df = pd.DataFrame(data={'image_file_names_train':image_file_names_train, 'mask':mask_files}) 
df_train, df_test =train_test_split(df, test_size=0.1)
df_train, df_val =train_test_split(df_train,test_size =0.2)

    #data genereator and data augmentation,and adjust data 
    
#unet

def train_generator(
        data_frame,
        batch_size,
        augmentation_dict,
        image_color_mode ="rgb",
        mask_color_mode ='grayscale',
        image_save_prefix = "image",
        mask_save_prefix ="mask",
        save_to_dir =None,
        target_size =(356,256),
        seed =1):
        
    

        image_datagen = ImageDataGenerator(**augmentation_dict)
        mask_datagen = ImageDataGenerator(**augmentation_dict)
    
        image_generator = image_datagen.flow_from_dataframe(
            data_frame,
            x_col ='image_file_names_train',
            class_mode =None,
            color_mode =image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir =save_to_dir,
            save_prefix =image_save_prefix,
            seed= seed,)
        mask_generator = mask_datagen.flow_from_dataframe(
            data_frame,
            x_col ='mask',
            class_mode=None,
            color_mode=mask_color_mode,
            target_size =target_size,
            batch_size =batch_size,
            save_to_dir =save_to_dir,
            save_prefix=mask_save_prefix,
            seed =seed,)
        train_gen =zip(image_generator, mask_generator)
        #final return tuple after image normalization and Diognastik 
        for(img,mask) in train_gen:
            img, amsk =normalize_and_diagnose(img,mask)
            yield(img,mask)
            
def normalize_and_diagnose(img,mask):
    img =img/255
    mask =mask/255
    mask[mask>0.5]=1
    mask[mask<=0.5]=0
    return (img, mask)           
           
EPOCHS =2
BATCH_SIZE =32
learning_rate =1e-4  
smooth =100  
train_generator_param = dict(rotation_range =0.2,
                             width_shift_range =0.05,
                             height_shift_range =0.05,
                             shear_range =0.05,
                             zoom_range = 0.05,
                             horizontal_flip =True,
                             fill_mode ='nearest')  
train_gen =train_generator(df_train,BATCH_SIZE,train_generator_param,target_size=(im_height,im_width))
test_gen = train_generator(df_test,BATCH_SIZE,dict(),target_size=(im_height,im_width))

model =unet(input_size =(im_height,im_width, 3))
decay_rate =learning_rate/EPOCHS
optimizer=Adam(lr =learning_rate,beta_1 =0.9,beta_2=0.999,epsilon=None,decay=decay_rate,amsgrad =False)
model.compile(optimizer=optimizer,loss=dice_efficent_loss, metrics = ['binary_accuracy',iou,dice_coefficent])    
callbacks = [ModelCheckpoint('unet.hdf5',verbose =1,save_best_only =True)]    
history =model.fit(train_gen,
                   steps_per_epoch =len(df_train) / BATCH_SIZE,
                   epochs=EPOCHS,
                   callbacks=callbacks,
                   validation_data =test_gen,
                   validation_steps =len(df_val) /BATCH_SIZE
                   )
    
    
#load previously trained model 
model = load_model('unet.hdf5', custon_objects = {'dice_coefficient_loss':dice_efficent_loss, 'iou':iou,"dice_coefficient": dice_coefficent})
test_gen =train_generator(df_test,BATCH_SIZE, dict(),target_Size=(im_height,im_width) )
results = model.evaluate(test_gen,steps=len(df_test)/BATCH_SIZE)
print('Test loss :',results[0])
print('Test IOU',results[1])
print('Test Dice Coefficient', results[2])    
    
#plotting predictes masks segmentation results from the test image st


for i in range(20):
    index =np.random.ranint(1, len(df_test.index))
    img =cv2.imread(df_test['image_file_name_train'].iloc[index])
    img =cv2.resize(img, (im_height,im_width))
    img =img/255
    img =img[np.newaxis, :, :,: ]
    
    predicted_img = model.predict(img)
    plt.figure(figsize =(12, 12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Orginal Image ')
    
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
    plt.title('Orginal Mask ')
    
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(predicted_img)> 0.5)
    plt.title('Prediction')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
