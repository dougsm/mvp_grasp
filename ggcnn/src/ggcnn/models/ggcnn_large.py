import torch.nn as nn
import torch.nn.functional as F


class GGCNN(nn.Module):
    def __init__(self):
        super().__init__()

        filter_sizes = [16, 16, 16, 32, 64, 32, 32]

        self.features = nn.Sequential(
            nn.Conv2d(1, filter_sizes[0], kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(filter_sizes[5], filter_sizes[6], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

        )

        self.pos_output = nn.Conv2d(filter_sizes[6], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[6], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[6], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[6], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.features(x)

        pos_output = F.sigmoid(self.pos_output(x))
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.binary_cross_entropy(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }


if __name__ == '__main__':
    from torchsummary import summary

    module = GGCNN()
    summary(module, (1, 300, 300), device='cpu')
