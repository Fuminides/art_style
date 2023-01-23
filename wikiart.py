from fastai.vision import ImageDataBunch
from fastai.vision.transforms import get_transforms
from fastai.vision.data import imagenet_stats
from fastai.vision import cnn_learner
from fastai.vision import models
from fastai.metrics import error_rate



import pandas as pd

cluster_path = '/home/jfumanal/WikiArt'

train_path = cluster_path + '/train'
test_path = cluster_path + '/test'

df_styles_train = pd.read_csv(cluster_path + '/train_info.csv')

data = ImageDataBunch.from_df(
  df=df_styles_train, path=train_path, label_col='style', fn_col='new_filename', 
  ds_tfms=get_transforms(), size=299, bs=48).normalize(imagenet_stats)

learner = cnn_learner(data, models.resnet50, metrics=[error_rate])
learner.fit_one_cycle(6)

learner.lr_find()

learner.unfreeze()
learner.fit_one_cycle(300, max_lr=slice(1e-6,3e-4))

learner.save('syle_classifier')