from fastai.vision import ImageDataBunch
from fastai.vision.transforms import get_transforms
from fastai.vision.data import imagenet_stats
from fastai.vision import cnn_learner
from fastai.vision import models
from fastai.metrics import error_rate
from fastai.vision import open_image



import pandas as pd
import numpy as np

cluster_path = '/home/jfumanal/SemArt'
df_styles_train = pd.read_csv(cluster_path + '/train_info.csv')
wikiart_path = '/home/jfumanal/WikiArt'
train_path = wikiart_path + '/train'
data = ImageDataBunch.from_df(
  df=df_styles_train, path=train_path, label_col='style', fn_col='new_filename', 
  ds_tfms=get_transforms(), size=299, bs=48).normalize(imagenet_stats)


def get_names_pictures_folder(path):
    '''Returns a list of the names of the files in the folder path'''
    import os
    return [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg'))]

learner = cnn_learner(data, model=models.resnet50, metrics=[error_rate])
learner.load('syle_classifier')


res = pd.DataFrame(np.zeros(len(get_names_pictures_folder(cluster_path), learner.data.classes)), columns=learner.data.classes)
for ix, image in enumerate(get_names_pictures_folder(cluster_path)):
    img = open_image(cluster_path + '/' + image)
    pred = learner.predict(img)
    res.iloc[ix, :] = pred

res.to_csv(cluster_path + '/style_predictions.csv')


