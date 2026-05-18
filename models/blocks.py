import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础卷积块
class ConvBlock(nn.Module):
    """
    两层卷积 + BN + ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

# 下采样
class DownSample(nn.Module):
    """
    下采样：最大池化
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        return self.pool(x)

# 上采样
class UpSample(nn.Module):
    """
    上采样：转置卷积 + concat + ConvBlock
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 先上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 上采样后接卷积块  跳联
        self.conv = ConvBlock(in_channels, out_channels)  # concat 后 channels = out + skip
    
    def forward(self, x, skip):
        x = self.up(x)
        
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # concat
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# 输出层
class OutConv(nn.Module):
    """
    输出 logits
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

# U-Net
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = DownSample()
        
        self.enc2 = ConvBlock(base_channels, base_channels*2)
        self.down2 = DownSample()
        
        self.enc3 = ConvBlock(base_channels*2, base_channels*4)
        self.down3 = DownSample()
        
        self.enc4 = ConvBlock(base_channels*4, base_channels*8)
        self.down4 = DownSample()
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16)
        
        # Decoder
        self.up4 = UpSample(base_channels*16, base_channels*8)
        self.up3 = UpSample(base_channels*8, base_channels*4)
        self.up2 = UpSample(base_channels*4, base_channels*2)
        self.up1 = UpSample(base_channels*2, base_channels)
        
        # 输出
        self.out_conv = OutConv(base_channels, num_classes)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.down4(e4))
        
        # Decoder
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        
        # 输出
        out = self.out_conv(d1)
        return out

#测试
if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)