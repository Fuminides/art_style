import torch.nn as nn
from torchvision import models
import torch
import clip

class ResnetArt(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task

    def __init__(self, num_class, model='resnet'):
        super(ResnetArt, self).__init__()

        
        self.model = model
        # Load pre-trained visual model
        if model == 'resnet':
            resnet = models.resnet50(pretrained=True)
            self.deep_feature_size = 2048
        elif model == 'vgg':
            resnet = models.vgg16(pretrained=True)
            self.deep_feature_size = 25088
        elif model == 'clip':
            resnet, _ = clip.load("ViT-B/32")
            self.deep_feature_size = 512


        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
            
        # Classifiers
        self.final_classifier = nn.Sequential(nn.Linear(self.deep_feature_size, num_class))
        

    def forward(self, img):

        if self.model != 'clip':
            visual_emb = self.resnet(img)
        else:
            visual_emb = self.resnet.encode_image(img)

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out = self.final_classifier(visual_emb)

        return out