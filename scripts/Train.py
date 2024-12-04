# Import all the desired packages

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import matplotlib.pyplot as plt
import os
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt
import cv2
import scipy.interpolate as interp
import scipy.signal as sign
#import astroscrappy.astroscrappy as lac
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
        
from model_train import train

np.random.seed(0)

# g_image = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_g_image_wd.npy')
# i_image = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_i_image_wd.npy')
# r_image = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_r_image_wd.npy')
# z_image = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_z_image_wd.npy')

# g_mask = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_g_mask_wd.npy')
# i_mask = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_i_mask_wd.npy')
# r_mask = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_r_mask_wd.npy')
# z_mask = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_z_mask_wd.npy')

# g_sky = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_g_sky_wd.npy')
# i_sky = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_i_sky_wd.npy')
# r_sky = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_r_sky_wd.npy')
# z_sky = np.load('/raid/ai19resch11003/srinadh/same_scale/data/decam_z_sky_wd.npy')

# image_data1 = np.vstack((g_image, i_image))
# image_data2 = np.vstack((r_image, z_image))
# image = np.vstack((image_data1, image_data2))

# mask_data1 = np.vstack((g_mask, i_mask))
# mask_data2 = np.vstack((r_mask, z_mask))
# mask = np.vstack((mask_data1, mask_data2))

# sky_data1 = np.hstack((g_sky, i_sky))
# sky_data2 = np.hstack((r_sky, z_sky))
# sky = np.hstack((sky_data1, sky_data2))

# s = np.arange(image.shape[0])   
# np.random.shuffle(s)

# image = image[s]
# mask = mask[s]
# sky = sky[s]
# import torch

# if torch.cuda.is_available():
#     print("cuda is available")

# else :
#     print("cuda is not available")
images = np.load('C:/TransUNet/Attention-Augmented-Cosmic-Ray-Detection-in-Astronomical-Images/Attention_Based_CR_Detection_Models/DECam Data/DECAM_g_img.npy')
masks = np.load('C:/TransUNet/Attention-Augmented-Cosmic-Ray-Detection-in-Astronomical-Images/Attention_Based_CR_Detection_Models/DECam Data/DECAM_g_mask.npy')
skys = np.load('C:/TransUNet/Attention-Augmented-Cosmic-Ray-Detection-in-Astronomical-Images/Attention_Based_CR_Detection_Models/DECam Data/DECAM_g_sky.npy')

s = np.arange(images.shape[0])
np.random.shuffle(s)

## Take only 100 images for training
# images = images

image = images[s][:1000]
mask = masks[s][:1000]
sky = skys[s][:1000]
# Check for the type and shape of the train image and mask data
print('\nType and shape of the train_image_data:')
print(type(image))
print(np.shape(image))

print('\nType and shape of the train_mask_data:')
print(type(mask))
print(np.shape(mask))

print('\nType and shape of the train_sky_data:')
print(type(sky))
print(np.shape(sky))

print('\n')

# Input image and mask
# from deepCR import train
trainer = train(image,mask,ignore=None,sky=sky,aug_sky=(-0.9,3),name = 'base_vit_direct_wa',hidden=32, epoch=100,epoch_phase0=100, batch_size=4, save_after=10,plot_every=61,use_tqdm=False,lr = 0.001)
train_loss = trainer.train()
#filename = trainer.save() # not necessary if save_after is specified

## Save training loss
train_loss = np.array(train_loss)
np.save('training_loss.npy',train_loss)
# After training, you can examine that validation loss has reached its minimum by
trainer.plot_loss()
# plt.grid()
plt.savefig('epoch_vs_validation_loss_base_cc3_direct_wa.png')

# trainer.plot_dice_score()
# plt.grid()
# plt.savefig('epoch_vs_ds_base_cc3_direct_wa.png')

