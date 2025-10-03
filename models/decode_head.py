import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv2DNormalGamma


class decode_head_evidential(nn.Module):
    def __init__(
        self,
        input_transform="resize_concat",
        image_shape=(224, 224),
        in_index=(0, 1, 2, 3),
        upsample=4,
        in_channels=None,
        channels=6144,
        align_corners=False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.image_shape = image_shape
        self.in_index = in_index
        self.upsample = upsample
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners

        self.conv_depth = Conv2DNormalGamma(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1,
        )

    def _transform_inputs(self, inputs):
        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    F.interpolate(
                        x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def _forward_feature(self, inputs):
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.conv_depth(output)
        output = F.interpolate(
            output,
            size=self.image_shape,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return output
