import torch
import os
from torch import nn


class Model:
    def __init__(self):
        self._model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)

        # Change the last part of the model
        self._model.query_embed = nn.Embedding(10, self._model.transformer.d_model)
        self._model.class_embed = nn.Linear(256, 3)

        trained_params = os.path.join(os.path.dirname(__file__), 'trained_params', 'model.pth')
        self._model.load_state_dict(torch.load(trained_params, map_location=torch.device('cpu')))


        for p in [p for _, p in self._model.named_parameters()]:
            p.requires_grad = False

        self._model.eval()


    def __call__(self, x):
        x = self._model(x)

        """
        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape=[batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the un-normalized bounding box.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        return x

    def to(self, device):
        self.model.to(device)