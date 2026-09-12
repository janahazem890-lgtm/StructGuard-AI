import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ResNet18UNet(nn.Module):
    def __init__(self, pretrained=True, in_channels=3, num_classes=1):
        super().__init__()
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            base = models.resnet18(weights=weights)
        else:
            base = models.resnet18(weights=None)
            
        # Encoder stages
        self.init_conv = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu
        ) # 64 ch, H/2, W/2
        
        self.maxpool = base.maxpool # H/4, W/4
        self.layer1 = base.layer1  # 64 ch, H/4, W/4
        self.layer2 = base.layer2  # 128 ch, H/8, W/8
        self.layer3 = base.layer3  # 256 ch, H/16, W/16
        self.layer4 = base.layer4  # 512 ch, H/32, W/32

        # Decoder stages
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32 + 64, 32)

        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_dec = ConvBlock(16, 16)
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.init_conv(x) # [B, 64, 128, 128]
        x1 = self.layer1(self.maxpool(x0)) # [B, 64, 64, 64]
        x2 = self.layer2(x1) # [B, 128, 32, 32]
        x3 = self.layer3(x2) # [B, 256, 16, 16]
        x4 = self.layer4(x3) # [B, 512, 8, 8]

        # Decoder
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        d0 = self.final_up(d1)
        d0 = self.final_dec(d0)
        out = self.out_conv(d0) # [B, num_classes, 256, 256]
        return out

if __name__ == '__main__':
    model = ResNet18UNet(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print('Model test forward successful, output shape:', y.shape)
