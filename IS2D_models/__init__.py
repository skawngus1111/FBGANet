import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import IS2D_models.backbone as backbone

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth'
}

def IS2D_model(model_name, image_size, num_channels, num_classes, device, backbone_name, group, inter_channels) :
    if model_name == 'BGANet':
        from IS2D_models.DCTFreqUNet import DCTFreqUNet
        return DCTFreqUNet(num_classes, group=group, backbone_name=backbone_name, inter_channels=inter_channels)

def load_backbone_model(backbone_name, pretrained=False, **kwargs):
    if backbone_name=='resnet18':
        from IS2D_models.backbone.resnet import ResNet
        model = ResNet(backbone.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    elif backbone_name=='resnet50':
        from IS2D_models.backbone.resnet import ResNet
        model = ResNet(backbone.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    elif backbone_name=='res2net50_v1b_26w_4s':
        from IS2D_models.backbone.res2net import Res2Net
        model = Res2Net(backbone.res2net.Bottle2Neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    elif backbone_name=='res2net101_v1b_26w_4s':
        from IS2D_models.backbone.res2net import Res2Net
        model = Res2Net(backbone.res2net.Bottle2Neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)

    if pretrained:
        print("Complete loading your pretrained backbone {}".format(backbone_name))
        model.load_state_dict(model_zoo.load_url(model_urls[backbone_name]))

    return model
def model_to_device(args, model):
    # multiprocess 설정
    if args.distributed:
        print('Multi GPU activate : {} with DP'.format(torch.cuda.device_count()))
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # when using a single GPU per process and per DDP, we need to divide tha batch size
            # ourselves based on the total number of GPUs we have
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all available GPUs if device_ids are not set
            # 만약에 device_ids를 따로 설정해주지 않으면, 가능한 모든 gpu를 기준으로 ddp가 알아서 배치사이즈와 workers를 나눠준다는 뜻.
            model = nn.parallel.DistributedDataParallel(model)
    elif not args.distributed and torch.cuda.device_count() > 1:
        print('Multi GPU activate : {}'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model).to(args.device)
    else:
        model = model.to(args.device)

    return model