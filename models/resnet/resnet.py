import torch
import torchvision
from torch import nn
from collections import OrderedDict
import os

class ResNet(torch.nn.Module):
    def __init__(self, name: str, pretrained: bool):
        super().__init__()

        feature_dims = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048}
        weights = 'DEFAULT' if pretrained else None
        self.resnet = getattr(torchvision.models, name)(weights=weights)
        self.embed_dim = feature_dims[name]

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.resnet.fc = None
        self.resnet.avgpool = None  # original output: [batch, feature_dims, 1, 1]
        del self.resnet.fc
        del self.resnet.avgpool

        # global_pool would be replaced by Identity() if --use_ml_decoder
        global_pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        dic = OrderedDict([('global_pool_layer', global_pool_layer)])
        dic['flatten'] = nn.Flatten(start_dim=1)
        self.global_pool = nn.Sequential(dic) # output: [batch, feature_dims]

        # Embedding_pool (will not be replaced) (for return_embedding)
        embedding_pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        dic_embed = OrderedDict([('embedding_pool_layer', embedding_pool_layer)])
        dic_embed['flatten'] = nn.Flatten(start_dim=1)
        self.embedding_pool = nn.Sequential(dic_embed)

        self.embedding_target_dim = 100
        if self.embed_dim > self.embedding_target_dim:
            print(f"Adding pooling for embeddings: {self.embed_dim} -> {self.embedding_target_dim}")
            self.embedding_reduce = nn.AdaptiveAvgPool1d(self.embedding_target_dim)
        else:
            self.embedding_reduce = None

        fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))


    def _forward_impl(self, x, return_embedding=False, return_label_features=False):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # [batch, 2048, H, W]

        if return_embedding:
            embedding = self.embedding_pool(x)  # [batch, feature_dims]
            return embedding

        if return_label_features:
            # Spatial features for ML-Decoder (same as DINOv3.forward_features_spatial)
            spatial = x.flatten(2).transpose(1, 2)  # [batch, H*W, 2048]
            return self.head(spatial, return_label_features=True)  # [batch, C, decoder_embedding]

        embedding = self.global_pool(x)
        logits = self.head(embedding)
        return logits

    def forward(self, x, return_embedding=False, return_label_features=False):
        return self._forward_impl(x, return_embedding, return_label_features)

def Resnet50_timm(model_params):
    num_classes = model_params['num_classes']
    pretrained = model_params['pretrained']
    model = ResNet('resnet50', False)
    if pretrained:
        if not os.path.exists("./models/resnet/resnet50checkpoint.pth"):
            torch.hub.download_url_to_file(
                url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
                dst="./models/resnet/resnet50checkpoint.pth"
            )
        state = torch.load("./models/resnet/resnet50checkpoint.pth", map_location="cpu")
        model.resnet.load_state_dict(state, strict=True)
        print('Load the backbone pretrained on ImageNet')

    model.reset_classifier(num_classes)
    model.num_features = 2048
    return model
