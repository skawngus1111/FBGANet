# BGANet code

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

import kornia.filters.sobel as sobel_filter

from IS2D_models import load_backbone_model

class SubDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, scale_factor):
        super(SubDecoder, self).__init__()

        self.subdecoder = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))

    def forward(self, x):
        return self.subdecoder(x)

class RFB_S(nn.Module):
    def __init__(self, in_channels):
        super(RFB_S, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.branch3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        self.branch4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=7, dilation=7)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        att_map = self.conv1x1(torch.cat([branch1, branch2, branch3, branch4], dim=1))

        return att_map

class RFB_C(nn.Module):
    def __init__(self, in_channels):
        super(RFB_C, self).__init__()

        self.branch1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.branch3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        self.branch4 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=7, dilation=7)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels * 4, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(dim=2)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        att_map = self.conv(torch.cat([branch1, branch2, branch3, branch4], dim=1))

        return att_map.unsqueeze(dim=2)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection_channels, group=1):
        super(UpsampleBlock, self).__init__()
        self.group = group
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_channel = RFB_C(in_channels=out_channels)
        self.conv1x1_spatial = RFB_S(in_channels=1)

        in_group_channels = int(in_channels // self.group)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_group_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid())

        in_channels = in_channels + skip_connection_channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.count = 1

    def forward(self, x, skip_connection=None, boundary_guide=None, viz=False):
        x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        B, C, H, W = x.size()

        group_att_map_list = []
        for idx in range(self.group):
            group_x = x[:, int(C // self.group) * idx:int(C // self.group) * (idx + 1)]
            group_att_map = self.conv1(group_x)
            group_att_map_list.append(group_att_map)
        group_att_map = torch.mean(torch.cat(group_att_map_list, dim=1), dim=1, keepdim=True)
        x_for = x * group_att_map + x
        x = torch.cat([x_for, skip_connection], dim=1)
        x = self.conv2(x)

        if boundary_guide is not None:
            B, C, _, _ = boundary_guide.size()
            boundary_channel_descriptor = self.conv1x1_channel(self.average_channel_pooling(boundary_guide))
            x += boundary_channel_descriptor

            boundary_spatial_descriptor = self.conv1x1_spatial(torch.mean(boundary_guide, dim=1, keepdim=True))
            x += boundary_spatial_descriptor
        x = self.conv3(x)

        return x

class DCTFreqUNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 group=2,
                 inter_channels=2048,
                 backbone_name='res2net50_v1b_26w_4s',
                 decoder_filters=(256, 128, 64, 32),
                 parametric_upsampling=False,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        super(DCTFreqUNet, self).__init__()

        self.backbone_name = backbone_name
        self.backbone = load_backbone_model(self.backbone_name, pretrained=True)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True))

        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        self.low_upsample_block1 = UpsampleBlock(bb_out_chs, decoder_filters[0],
                                                 skip_connection_channels=shortcut_chs[3],
                                                 group=group)
        self.low_upsample_block2 = UpsampleBlock(decoder_filters[0], decoder_filters[1],
                                                 skip_connection_channels=shortcut_chs[2],
                                                 group=group)
        self.low_upsample_block3 = UpsampleBlock(decoder_filters[1], decoder_filters[2],
                                                 skip_connection_channels=shortcut_chs[1],
                                                 group=group)
        self.low_upsample_block4 = UpsampleBlock(decoder_filters[2], decoder_filters[3],
                                                 skip_connection_channels=shortcut_chs[0],
                                                 group=group)

        self.high_upsample_block1 = UpsampleBlock(bb_out_chs, decoder_filters[0],
                                                 skip_connection_channels=shortcut_chs[3],
                                                 group=group)
        self.high_upsample_block2 = UpsampleBlock(decoder_filters[0], decoder_filters[1],
                                                 skip_connection_channels=shortcut_chs[2],
                                                 group=group)
        self.high_upsample_block3 = UpsampleBlock(decoder_filters[1], decoder_filters[2],
                                                 skip_connection_channels=shortcut_chs[1],
                                                 group=group)
        self.high_upsample_block4 = UpsampleBlock(decoder_filters[2], decoder_filters[3],
                                                 skip_connection_channels=shortcut_chs[0],
                                                 group=group)

        self.low_stage1_conv = SubDecoder(decoder_filters[0], num_classes, scale_factor=16)
        self.low_stage2_conv = SubDecoder(decoder_filters[1], num_classes, scale_factor=8)
        self.low_stage3_conv = SubDecoder(decoder_filters[2], num_classes, scale_factor=4)
        self.low_stage4_conv = SubDecoder(decoder_filters[3], num_classes, scale_factor=2)

        self.high_stage1_conv = SubDecoder(decoder_filters[0], num_classes, scale_factor=16)
        self.high_stage2_conv = SubDecoder(decoder_filters[1], num_classes, scale_factor=8)
        self.high_stage3_conv = SubDecoder(decoder_filters[2], num_classes, scale_factor=4)
        self.high_stage4_conv = SubDecoder(decoder_filters[3], num_classes, scale_factor=2)

    def forward(self, x, mode='train'):
        features, x = self.backbone.forward_feature(x)

        x_dct = dct.dct_2d(x)
        x_compress = self.conv1x1(x)
        x_dct_compress = dct.dct_2d(x_compress)
        x_dct_compress = (x_dct_compress - torch.mean(x_dct_compress, dim=(2, 3), keepdim=True)) / torch.std(x_dct_compress, dim=(2, 3), keepdim=True)
        low_mask = (torch.sigmoid(x_dct_compress) >= 0.5).type(torch.int)

        x_dct_low = x_dct * low_mask
        x_dct_high = x_dct * (1 - low_mask)

        x_dct_low = dct.idct_2d(x_dct_low)
        x_dct_high = dct.idct_2d(x_dct_high)

        x_dct_high1 = self.high_upsample_block1(x_dct_high, features[-1])
        x_dct_high2 = self.high_upsample_block2(x_dct_high1, features[-2])
        x_dct_high3 = self.high_upsample_block3(x_dct_high2, features[-3])
        x_dct_high4 = self.high_upsample_block4(x_dct_high3, features[-4])

        x_dct_low1 = self.low_upsample_block1(x_dct_low, features[-1], x_dct_high1)
        x_dct_low2 = self.low_upsample_block2(x_dct_low1, features[-2], x_dct_high2)
        x_dct_low3 = self.low_upsample_block3(x_dct_low2, features[-3], x_dct_high3, viz=True)
        x_dct_low4 = self.low_upsample_block4(x_dct_low3, features[-4], x_dct_high4)

        edge_stage1 = self.high_stage1_conv(x_dct_high1)
        edge_stage2 = self.high_stage2_conv(x_dct_high2)
        edge_stage3 = self.high_stage3_conv(x_dct_high3)
        edge_stage4 = self.high_stage4_conv(x_dct_high4)

        region_stage1 = self.low_stage1_conv(x_dct_low1)
        region_stage2 = self.low_stage2_conv(x_dct_low2)
        region_stage3 = self.low_stage3_conv(x_dct_low3)
        region_stage4 = self.low_stage4_conv(x_dct_low4)

        if mode=='train':
            return [region_stage1, region_stage2, region_stage3, region_stage4], \
                   [edge_stage1, edge_stage2, edge_stage3, edge_stage4]
        else:
            return region_stage4

    def _calculate_criterion(self, criterion, y_pred, y_true, mode):
        if mode=='train':
            edge_true = sobel_filter(y_true)
            edge_true[edge_true >= 0.5] = 1; edge_true[edge_true < 0.5] = 0

            region_loss, edge_loss = 0, 0
            for region_pred, edge_pred in zip(y_pred[0], y_pred[1]):
                region_loss += self.structure_loss(region_pred, y_true)
                edge_loss += criterion(edge_pred, edge_true)

            loss = region_loss + edge_loss
        else:
            loss = criterion(y_pred, y_true)

        return loss

    def similarity_loss(self, low_freq, high_freq):
        low_freq, high_freq = low_freq.squeeze(), high_freq.squeeze()
        cos_loss = torch.sum(torch.abs(self.cos(low_freq, high_freq)))

        return cos_loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def infer_skip_channels(self):
        x = torch.zeros(1, 3, 224, 224)

        [x, x1, x2, x3], x4 = self.backbone.forward_feature(x)
        channels, out_channels = [x.shape[1], x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1]], x4.shape[1]

        return channels, out_channels

if __name__=='__main__':
    model = DCTFreqUNet(num_classes=1)
    inp = torch.randn(2, 3, 256, 256)
    oup = model(inp, mode='val')
    print(oup.shape)