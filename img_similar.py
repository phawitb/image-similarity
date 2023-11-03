# !pip install tensorflow-hub
# !pip install tensorflow==2.1
# !pip install keras==2.3.1

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from scipy.spatial import distance

def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    file = np.stack((file,)*3, axis=-1)
    file = np.array(file)/255.0

    embedding = model.predict(file[np.newaxis, ...])
    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()

    return flattended_feature

def find_similar(img_init,img_list):
    img0 = extract('imgs/init.jpeg')
    imgs = []
    for i in img_list:
        imgs.append(extract(i))
        
    metric = 'cosine'
    
    DC = []
    for img in imgs:
        dc = distance.cdist([img0], [img], metric)[0]
        DC.append(dc)

    index = DC.index(min(DC))
    return img_list[index]

#init
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
IMAGE_SHAPE = (224, 224)
layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])

#check similarity
img_init = 'imgs/init.jpeg'
img_list = ['imgs/2.jpeg','imgs/3.jpeg','imgs/4.jpeg']
similarity_img = find_similar(img_init,img_list)

print(f'init_img={img_init},similarity_img={similarity_img}')

