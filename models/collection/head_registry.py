import torch
import torch.nn as nn

HEADS = {}

def register_head(name):
    def decorator(cls):
        HEADS[name] = cls
        return cls
    return decorator


@register_head("upsample_head")
class UpSampleAndConvHead(nn.Module):
    def __init__(self, num_keypoints, output_size, image_size, feature_dim):
        super(UpSampleAndConvHead, self).__init__()
        self.inplanes = feature_dim[-1]
        self.num_keypoints = num_keypoints
        self.output_size = output_size
        self.w, self.h = image_size
        self.feature_w = self.w//(2**len(feature_dim))  #we suppose that each conv from the backbone has halved the size (not true for convnext)
        self.feature_h = self.h//(2**len(feature_dim))

        self.upsample_layers, final_channels = self._make_upsample_blocks(self.feature_h, self.feature_w)

        self.final_layer = nn.Conv2d(
            in_channels=final_channels,
            out_channels=self.num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = x[-1]
        x = self.upsample_layers(x)
        x = self.final_layer(x)
        return x

    def _make_upsample_blocks(self, h, w):
        target_w, target_h = self.output_size
        curr_h, curr_w = h, w
        curr_channels = self.inplanes
        scale_factor = 2

        layers = []

        while curr_h < target_h or curr_w < target_w:
            out_channels = curr_channels // 2

            layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(curr_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            curr_h *= scale_factor
            curr_w *= scale_factor
            curr_channels = out_channels

        return nn.Sequential(*layers), curr_channels

@register_head("deconv_head")
class DeconvHead(nn.Module):
    def __init__(self, num_keypoints, output_size, image_size, feature_dim, num_deconv_layers=3, deconv_channels=256):
        super(DeconvHead, self).__init__()
        self.inplanes = feature_dim[-1]
        self.num_keypoints = num_keypoints
        self.output_size = output_size
        self.image_size = image_size

        self.upsample_layers, final_channels = self._make_deconv_layers(num_deconv_layers, deconv_channels)

        self.final_layer = nn.Conv2d(
            in_channels=final_channels,
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = x[-1]  
        x = self.upsample_layers(x)
        x = self.final_layer(x)
        return x

    def _make_deconv_layers(self, num_layers, deconv_channels):
        layers = []
        in_channels = self.inplanes

        for i in range(num_layers):
            out_channels = deconv_channels
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers), in_channels


@register_head("depthwise_conv_head")
class DepthwiseConvHead(nn.Module):
    def __init__(self, num_keypoints, output_size, image_size, feature_dim):
        super(DepthwiseConvHead, self).__init__()
        self.inplanes = feature_dim[-1]
        self.num_keypoints = num_keypoints
        self.output_size = output_size
        self.w, self.h = image_size
        self.feature_w = self.w // (2 ** len(feature_dim))
        self.feature_h = self.h // (2 ** len(feature_dim))

        self.upsample_layers, final_channels = self._make_upsample_blocks(
            self.feature_h, self.feature_w
        )

        self.final_layer = nn.Conv2d(
            in_channels=final_channels,
            out_channels=self.num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = x[-1]
        x = self.upsample_layers(x)
        x = self.final_layer(x)
        return x

    def _make_upsample_blocks(self, h, w):
        target_w, target_h = self.output_size
        curr_h, curr_w = h, w
        curr_channels = self.inplanes
        scale_factor = 2

        layers = []

        # keep upsampling until we hit output size
        while curr_h < target_h or curr_w < target_w:
            out_channels = curr_channels // 2

            layers.append(nn.Upsample(scale_factor=scale_factor,
                                      mode='bilinear',
                                      align_corners=True))
            layers.append(SeparableConv3x3(curr_channels, out_channels))

            curr_h *= scale_factor
            curr_w *= scale_factor
            curr_channels = out_channels

        return nn.Sequential(*layers), curr_channels

def SeparableConv3x3(in_ch, out_ch, stride=1, padding=1, bias=False):
    """
    Depthwise 3×3 conv followed by 1×1 pointwise conv.
    """
    return nn.Sequential(
        # depthwise
        nn.Conv2d(in_ch, in_ch, kernel_size=3,
                  stride=stride, padding=padding,
                  groups=in_ch, bias=bias),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        # pointwise
        nn.Conv2d(in_ch, out_ch, kernel_size=1,
                  stride=1, padding=0, bias=bias),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )