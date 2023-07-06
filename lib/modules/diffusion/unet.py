import torch
import numpy as np

from ..layers.class_attention import EfficientClassAttentionBlock
from ..segmentation.unet import UNet,crop_to_size

class DiffusionUNet(UNet):
    def __init__(self,
                 classifier_free_guidance:bool=False,
                 classifier_classes:int=2,
                 *args,**kwargs):
        if "n_channels" not in kwargs:
            raise Exception("n_channels must be defined")
        kwargs["n_classes"] = kwargs["n_channels"]
        kwargs["encoding_operations"] = None
        kwargs["bottleneck_classification"] = False
        kwargs["feature_conditioning"] = None
        super().__init__(*args,**kwargs)

        self.classifier_free_guidance = classifier_free_guidance
        self.classifier_classes = classifier_classes

        self.init_eca()

    def init_eca(self):
        if self.classifier_free_guidance == True:
            eca_op = EfficientClassAttentionBlock
        else:
            eca_op = torch.nn.Identity
        self.encoder_eca = torch.nn.ModuleList([
            eca_op(self.classifier_classes,d) for d in self.depth])
        self.decoder_eca = torch.nn.ModuleList([
            eca_op(self.classifier_classes,d) for d in self.depth[:-1][::-1]])
        self.link_eca = torch.nn.ModuleList([
            eca_op(self.classifier_classes,d) for d in self.depth[:-1][::-1]])

    def forward(self,
                X:torch.Tensor,
                cls:torch.Tensor=None)->torch.Tensor:
        """Forward pass for this class.

        Args:
            X (torch.Tensor)
            cls (torch.Tensor): classification for guidance.

        Returns:
            torch.Tensor
        """

        cls = [] if cls is None else [cls]
        encoding_out = []
        curr = X
        for (op,op_ds),eca in zip(self.encoding_operations,self.encoder_eca):
            curr = eca(op(curr),*cls)
            encoding_out.append(curr)
            curr = op_ds(curr)

        deep_outputs = []
        for i in range(len(self.decoding_operations)):
            op = self.decoding_operations[i]
            link_op = self.link_ops[i]
            up = self.upscale_ops[i]
            link_op_input = encoding_out[-i-2]
            eca_link = self.link_eca[i]
            eca_decoder = self.decoder_eca[i]
            encoded = eca_link(link_op(link_op_input),*cls)
            curr = up(curr)
            sh = list(curr.shape)[2:]
            sh2 = list(encoded.shape)[2:]
            if np.prod(sh) < np.prod(sh2):
                encoded = crop_to_size(encoded,sh)
            if np.prod(sh) > np.prod(sh2):
                curr = crop_to_size(curr,sh2)
            curr = torch.concat((curr,encoded),dim=1)
            curr = eca_decoder(op(curr),*cls)
            deep_outputs.append(curr)

        curr = self.final_layer(curr)
        
        if self.deep_supervision is True:
            for i in range(len(deep_outputs)):
                o = deep_outputs[i]
                op = self.deep_supervision_ops[i]
                deep_outputs[i] = op(o)
            return curr,deep_outputs

        return curr
