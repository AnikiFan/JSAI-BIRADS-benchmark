import torch
import torch.nn as nn
import timm


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        model_name="densenet121.ra_in1k", 
        pretrained=True, 
        num_classes=6, 
        **kwargs
    ):
        super(DenseNetClassifier, self).__init__()
        pathToCheckpoints = {
            "densenet121.ra_in1k": "./model_data/densenet121.ra_in1k.bin",
        }
        self.base_model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=pathToCheckpoints[model_name]),
            num_classes=num_classes, 
        )
        
        self.train_transform, self.val_transform = self.get_transform()
    
    def get_transform(self):
        data_config = timm.data.resolve_model_data_config(self.base_model)
        print(data_config)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        return train_transform, val_transform
    
    def forward(self, x):
        if self.training:
            x = self.train_transform(x)
        else:
            x = self.val_transform(x)
        x = x.unsqueeze(0)
        x = self.base_model(x)
        return x
