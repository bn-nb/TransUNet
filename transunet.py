import gc
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchinfo import summary


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_c, feats, pool):
        super(UNetEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.pool = pool

        for f in feats:
            self.layers.append(DoubleConv(in_c, f))
            in_c = f

    def forward(self, x):
        skips = []

        for l in self.layers:
            x = l(x)
            skips.append(x)
            x = self.pool(x)

        return x, skips


class UNetDecoder(nn.Module):
    def __init__(self, bottle, feats):
        super(UNetDecoder, self).__init__()
        self.layers = ModuleList()

        for f in reversed(feats):
            l = nn.ConvTranspose2d(bottle, f, kernel_size=2, stride=2)
            self.layers.append(l)
            self.layers.append(DoubleConv(bottle, f))
            bottle = f

    def forward(self, x, skips):
        skips = skips[::-1]

        for idx in range(0, len(self.layers), 2):
            x = self.layers[idx](x)
            skip_conn = skips[idx//2]
            x = torch.cat((skip_conn, x), dim=1)
            x = self.layers[idx+1](x)

        return x


class TransUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, bottle=None, feats=[64, 128, 256, 512], *, trainVIT=False):
        super(TransUNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_ = UNetEncoder(in_c, feats, pool=self.pool)

        if bottle is None:
            bottle = feats[-1] * 2

        self.bottle = DoubleConv(feats[-1], bottle)
        self.vtrain = trainVIT
        self.vitenc = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).encoder
        self.hidden = 768 # depends on VIT

        if (not self.vtrain):
            for p in self.vitenc.parameters():
                p.requires_grad = False
            self.vitenc.eval()

        self.toproj = nn.Conv2d(bottle, self.hidden, kernel_size=1)
        self.deproj = nn.Conv2d(self.hidden, bottle, kernel_size=1)
        self.decode = UNetDecoder(bottle, feats)
        self.output = nn.Sequential(
            nn.Conv2d(feats[0], out_c, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        if (not self.vtrain):
            self.vitenc.eval()

        x, skips = self.encode(x)
        x = self.bottle(x)
        B, C, H, W = x.shape

        # EncoderBlock will enforce input restraints, so do manually
        x = self.toproj(x)
        pos_embedding = self.vitenc.pos_embedding[:, 1:, :]
        x = x.flatten(2).transpose(1, 2) + pos_embedding

        if (not self.vtrain):
            with torch.no_grad():
                x = self.vitenc.ln(self.vitenc.layers(x))
        else:
            x = self.vitenc.ln(self.vitenc.layers(self.vitenc.dropout(x)))

        x = x.transpose(1, 2).reshape(B, 768, H, W)
        x = self.deproj(x)

        x = self.decode(x, skips)
        x = self.output(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_size = (2, 1, 448, 448)
    x = torch.randn(in_size).to(device)

    model = TransUNet(1, 1, feats=[32, 64, 128, 256, 512], trainVIT=1).to(device)
    p = model(x)
    print(x.shape)
    print(p.shape)

    summary(model, input_size=in_size, depth=3, col_names=["input_size", "output_size", "num_params", "trainable"])
    del model, p, x
    torch.cuda.empty_cache()
    gc.collect()
