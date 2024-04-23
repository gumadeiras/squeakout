import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1, input_size=512, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, n_class),)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Interpolate(nn.Module):
    """Interpolation implementation to be used in nn.Sequence

    Implements nn.UpSample (deprecated) using nn.functional.interpolate.
    Works inside nn.Sequence (nn.functional.interpolate does not work inside nn.Sequence)

    Extends:
        nn.Module
    """

    def __init__(self, scale, mode, align):
        super(Interpolate, self).__init__()
        self.interpol = F.interpolate
        self.scale = scale
        self.mode = mode
        self.align = align

    def forward(self, x):
        x = self.interpol(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align)
        return x

class SqueakOut(nn.Module):
    def __init__(self, pre_trained=None):
        super(SqueakOut, self).__init__()

        self.backbone = MobileNetV2()

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)

        self.upsample = Interpolate(scale=2, mode="bilinear", align=False)

        self._init_weights()

        if pre_trained is not None:
            self.backbone.load_state_dict(torch.load(pre_trained))

    def forward(self, x):
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x

        for n in range(14, 19):
            x = self.backbone.features[n](x)

        up1 = torch.cat([x4, self.dconv1(x)], dim=1)
        up1 = self.invres1(up1)

        up2 = torch.cat([x3, self.dconv2(up1)], dim=1)
        up2 = self.invres2(up2)

        up3 = torch.cat([x2, self.dconv3(up2)], dim=1)
        up3 = self.invres3(up3)

        up4 = torch.cat([x1, self.dconv4(up3)], dim=1)
        up4 = self.invres4(up4)

        x = self.conv_last(up4)

        x = self.upsample(x)

        x = self.conv_score(x)

        #         x = torch.sigmoid(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SqueakOut_autoencoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = SqueakOut(pre_trained=None)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset_dir = "./dataset/train"
        trans = dataset.AugmentationTransform()
        dataset_set = dataset.SegmentationDataset(dataset_dir, transforms=trans)
        return DataLoader(dataset_set, batch_size=12, shuffle=True, num_workers=0, pin_memory=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss, bce, dice = calc_loss(pred, y, bce_weight=0.3)
        
        tensorboard_logs = {'train_loss': loss, 'train_bce': bce, 'train_dice': dice}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        tensorboard_logs = {'avg_loss': avg_loss}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        dataset_dir = "./dataset/val"
        dataset_set = dataset.SegmentationDataset(dataset_dir, transforms=None)
        return DataLoader(dataset_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss, bce, dice = calc_loss(pred, y, bce_weight=0.3)

        image       = x.data.cpu().numpy()
        mask        = y.data.cpu().numpy()
        output_mask = pred.data.cpu().numpy()
        
        output_path = f'./logs/outputs/val/' + str(self.current_epoch) + '/'
        os.makedirs(output_path, exist_ok=True)
        create_montage(image[0], mask[0], output_mask[0],
                             show_plot=False, 
                             model_dir=output_path, image_name=str(batch_idx))
        
        tensorboard_logs = {'val_loss': loss, 'val_bce': bce, 'val_dice': dice}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_avg_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self):
        dataset_dir = "./dataset/test"
        dataset_set = dataset.SegmentationDataset(dataset_dir, transforms=None)
        return DataLoader(dataset_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss, bce, dice = calc_loss(pred, y, bce_weight=0.3)
        
        image       = x.data.cpu().numpy()
        mask        = y.data.cpu().numpy()
        output_mask = pred.data.cpu().numpy()
        
        output_path = f'./logs/outputs/test/'
        os.makedirs(output_path, exist_ok=True)
        create_montage(image[0], mask[0], output_mask[0],
                             show_plot=False, 
                             model_dir=output_path, image_name=str(batch_idx))
        
        tensorboard_logs = {'test_loss': loss, 'test_bce': bce, 'test_dice': dice}
        return {'test_loss': loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_avg_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

if __name__ == "__main__":
    # Debug
    model = SqueakOut_autoencoder()