import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from nets.CSPdarknet53_tiny import darknet53_tiny


#-------------------------------------------------#
#   卷积块
#   CONV+BATCHNORM+LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y= self.sigmoid(y)
        return y.expand_as(x)*x

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage
# class AdapFPN(nn.Module):
#     def __init__(self, level):
#         super(AdapFPN, self).__init__()
#         self.level = level
#         # 输入的三个特征层的channels, 根据实际修改
#         # self.dim = [512, 256, 256]
#         self.dim = [256,256]
#         self.inter_dim = self.dim[self.level]
#         if level == 0:
#             self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
#             self.expand = add_conv(self.inter_dim, 256, 3, 1)
#         elif level == 1:
#             self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
#             self.expand = add_conv(self.inter_dim, 256, 3, 1)
#
#         else:
#             print('wrong asff!!!')
#
#         compress_c = 16
#
#         self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
#         # self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels0 = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
#         # self.weight_levels1 = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
#
#
#     # 尺度大小 level_0 < level_1 < level_2
#     #x_level_0=P5=(19，19，512) & x_level_1=feature1=(38，38，256)
#     def forward(self, x_level_0, x_level_1):
#         if self.level == 0:
#             level_0_resized = x_level_0
#             level_1_resized = self.stride_level_1(x_level_1)
#
#         elif self.level == 1:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
#             level_1_resized = x_level_1
#         else:
#             print('wrong asff!!!')
#
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
#         levels_weight = self.weight_levels0(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)  # alpha等产生
#
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:, :, :]
#         out = self.expand(fused_out_reduced)
#
#         return out

# class AdapFPN(nn.Module):
#     def __init__(self, level):
#         super(AdapFPN, self).__init__()
#         self.level = level
#         # 输入的三个特征层的channels, 根据实际修改
#         # self.dim = [512, 256, 256]
#         self.dim = [256,256]
#         self.inter_dim = self.dim[self.level]

#         self.expand = add_conv(self.inter_dim, 256, 3, 1)
#         compress_c = 16
#
#         self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
#         # self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels0 = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
#         # self.weight_levels1 = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
#
#
#     # 尺度大小 level_0 < level_1 < level_2
#     #x_level_0=P5=(19，19，512) & x_level_1=feature1=(38，38，256)
#     def forward(self, x_level_0, x_level_1):
#         if self.level == 0:
#             level_0_resized = x_level_0
#             level_1_resized = F.max_pool2d(x_level_1, 3, stride=2, padding=1)
#
#         elif self.level == 1:
#             # level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(x_level_0, scale_factor=2, mode='nearest')
#             level_1_resized = x_level_1
#         else:
#             print('wrong asff!!!')
#
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
#         levels_weight = self.weight_levels0(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)  # alpha等产生
#
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:, :, :]
#         out = self.expand(fused_out_reduced)
#         # elif self.level == 1:
#         #     level_0_weight_v = self.weight_level_0(level_0_resized)
#         #     level_1_weight_v = self.weight_level_1(level_1_resized)
#         #     level_2_weight_v = self.weight_level_2(level_2_resized)
#         #     levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v,level_2_weight_v), 1)
#         #     levels_weight = self.weight_levels1(levels_weight_v)
#         #     levels_weight = F.softmax(levels_weight, dim=1)  # alpha等产生
#         #
#         #     fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#         #                         level_1_resized * levels_weight[:, 1:2, :, :] + \
#         #                         level_2_resized * levels_weight[:, 2:, :, :]
#         #
#         #     out = self.expand(fused_out_reduced)
#         return out
class ASFF(nn.Module):
    def __init__(self, level):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [ 256, 384]
        self.inter_dim = self.dim[self.level]
        # 每个level融合前，需要先调整到一样的尺度
        if level == 0:
            self.stride_level_1 = add_conv(384, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 384, 3, 1)

        else:
            print('wrong asff!!!')


        compress_c = 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)



    def forward(self, x_level_0, x_level_1):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            # level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            # level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            # level_2_resized = self.stride_level_2(x_level_2)


        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        # level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # 自适应权重融合
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                                        level_1_resized * levels_weight[:, 1:, :, :]
        out = self.expand(fused_out_reduced)

        return out


#-------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512,256,1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)],256)

        self.upsample = Upsample(256,128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)],384)
        # self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 256)
        self.se1 = SELayer(256)
        self.se2 = SELayer(512)
        self.se3 = SELayer(384)
        self.sa = SpatialAttention()

        # self.asff_level0 = AdapFPN(level=0)
        # self.asff_level1 = AdapFPN(level=1)
        self.asff_level0 = ASFF(level=0)
        self.asff_level1 = ASFF(level=1)

    # def forward(self, x):
    #     # feat0, feat1, feat2 = self.backbone(x)
    #     feat1, feat2 = self.backbone(x)
    #     feat1 = self.se1(feat1)
    #     feat1 = self.sa(feat1)#256
    #     feat2 = self.se2(feat2)
    #     feat2 = self.sa(feat2)#512
    #     P5 = self.conv_for_P5(feat2)
    #     P5 = self.se1(P5)
    #     P5 = self.sa(P5)
    #     results_after_asff0 = self.asff_level0(P5, feat1)
    #     results_after_asff1 = self.asff_level1(P5, feat1)
    #
    #     out0 = self.yolo_headP5(results_after_asff0)
    #     out1 = self.yolo_headP4(results_after_asff1)
    #
    #     return out0, out1



    #
    def forward(self, x):
        #  backbone
        feat1, feat2 = self.backbone(x)
        feat1 = self.se1(feat1)
        feat1 = self.sa(feat1)#256
        feat2 = self.se2(feat2)
        feat2 = self.sa(feat2)#512

        P5 = self.conv_for_P5(feat2)
        P5 = self.se1(P5)
        P5 = self.sa(P5)
        P5_Upsample = self.upsample(P5)
        P4 = torch.cat([feat1,P5_Upsample],axis=1)
        P4 = self.se3(P4)
        P4 = self.sa(P4)

        results_after_asff0 = self.asff_level0(P5, P4)
        results_after_asff1 = self.asff_level1(P5, P4)

        out0 = self.yolo_headP5(results_after_asff0)
        out1 = self.yolo_headP4(results_after_asff1)

        return out0, out1

    # def forward(self, x):
    #     #  backbone
    #     feat1, feat2 = self.backbone(x)
    #     # feat1 = self.se1(feat1)
    #     # feat1 = self.sa(feat1)#256
    #     # feat2 = self.se2(feat2)
    #     # feat2 = self.sa(feat2)#512
    #     P5 = self.conv_for_P5(feat2)
    #     # P5 = self.se1(P5)
    #     # P5 = self.sa(P5)
    #     out0 = self.yolo_headP5(P5)
    #
    #     P5_Upsample = self.upsample(P5)
    #     P4 = torch.cat([feat1,P5_Upsample],axis=1)
    #     # P4 = self.se3(P4)
    #     # P4 = self.sa(P4)
    #
    #     out1 = self.yolo_headP4(P4)
    #
    #     return out0, out1

