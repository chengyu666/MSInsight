import torch
import torch.nn.functional as F
import torch.nn as nn


# BFNet model used by bfnet2_ep46_260122a.pt.
# Input: (batch, x, y, z, strike, dip, rake), with each sample shaped
# (8, 8, 8, 24, 7, 24), extracted at the global jSSA brightness peak time.
# Output: [sin(strike), cos(strike), sin(dip), cos(dip), sin(rake), cos(rake)].
# Each pair is L2-normalized; both dip components are nonnegative.
# The checkpoint's original forward computation is preserved, including
# its feature reshape order and the absence of a tanh activation.


class StrikeDipRakeNet(nn.Module):
    pi = 3.14159265358979323846

    def __init__(self, shape):
        super(StrikeDipRakeNet, self).__init__()
        self.name = "bfnet2"
        # Extract dimensions from shape
        self.shape = shape
        x_dim, y_dim, z_dim, s_dim, d_dim, r_dim = shape  # Corrected order
        self.sdr_dim = s_dim * d_dim * r_dim
        self.xyz_dim = x_dim * y_dim * z_dim
        self.xyz_dim_afterconv = (x_dim // 2 // 2) * \
            (y_dim // 2 // 2) * (z_dim // 2 // 2)
        self.xyz_dim_afterfc = 8

        # spatial convolutional layers
        # Depthwise convolution over x, y, z; one channel per mechanism.
        self.spatial_conv1 = nn.Conv3d(
            in_channels=self.sdr_dim, out_channels=self.sdr_dim, kernel_size=3, padding=1,
            groups=self.sdr_dim
        )
        self.spatial_conv2 = nn.Conv3d(
            in_channels=self.sdr_dim, out_channels=self.sdr_dim, kernel_size=3, padding=1,
            groups=self.sdr_dim
        )
        self.spatial_fc1 = nn.Linear(
            in_features=self.xyz_dim_afterconv, out_features=16
        )
        self.spatial_fc2 = nn.Linear(
            in_features=16, out_features=self.xyz_dim_afterfc)

        # fm convolutional layers
        # Convolution over strike, dip, rake dimensions.
        self.fm_conv1 = nn.Conv3d(
            in_channels=self.xyz_dim_afterfc, out_channels=16, kernel_size=(3, 3, 3), padding=1
        )
        self.fm_bn1 = nn.BatchNorm3d(16)
        self.fm_conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=1
        )
        self.fm_bn2 = nn.BatchNorm3d(32)
        self.fm_conv3 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1
        )
        self.fm_bn3 = nn.BatchNorm3d(64)
        self.fm_conv4 = nn.Conv3d(
            in_channels=64, out_channels=8, kernel_size=(3, 3, 3), padding=1
        )
        self.fm_bn4 = nn.BatchNorm3d(8)

        # Fully connected layers
        # Size of the flattened mechanism features.
        self.flattened_size = 8 * s_dim * d_dim * r_dim
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.ln1 = nn.LayerNorm(512)
        self.dp1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.dp2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(256, 64)
        self.ln3 = nn.LayerNorm(64)
        self.dp3 = nn.Dropout(p=0.1)

        self.fc4 = nn.Linear(64, 16)
        self.ln4 = nn.LayerNorm(16)
        self.dp4 = nn.Dropout(p=0.0)

        self.fcfinal = nn.Linear(16, 6)

        # Apply the first-quadrant constraint only to the dip pair.
        self.register_buffer("dip_mask", torch.tensor(
            [[False, True, False]]).view(1, 3, 1))

        with torch.no_grad():
            self.fcfinal.bias.zero_()
            # Bias order: strike_sin/cos, dip_sin/cos, rake_sin/cos.
            self.fcfinal.bias[1] = 1.0  # Initial strike near 0 degrees.
            self.fcfinal.bias[2] = 1.0  # dip_sin
            self.fcfinal.bias[3] = 1.0  # Initial dip near 45 degrees.
            self.fcfinal.bias[5] = 1.0  # Initial rake near 0 degrees.

    def forward(self, x):
        batch_size, x_dim, y_dim, z_dim, s_dim, d_dim, r_dim = x.size()
        minv = x.amin(dim=(1, 2, 3, 4, 5, 6), keepdim=True)
        maxv = x.amax(dim=(1, 2, 3, 4, 5, 6), keepdim=True)
        x = (x - minv) / (maxv - minv + 1e-8)

        # Merge mechanism dimensions into channels, preserving spatial axes.
        x = x.view(batch_size, x_dim, y_dim, z_dim, s_dim * d_dim * r_dim)
        # (batch, channel_sdr, x, y, z)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # (batch, channel_sdr, x, y, z)
        x = F.leaky_relu(self.spatial_conv1(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        # (batch, channel_sdr, x//2, y//2, z//2)
        x = F.leaky_relu(self.spatial_conv2(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        # Flatten the last three spatial dimensions.
        # (batch*s*d*r, x//2//2*y//2//2*z//2//2)
        x = x.view(batch_size*self.sdr_dim, self.xyz_dim_afterconv)
        # Compress spatial features.
        x = F.leaky_relu(self.spatial_fc1(x))
        x = F.leaky_relu(self.spatial_fc2(x))  # (batch*s*d*r, xyz_dim_afterfc)

        # Restore the mechanism tensor using the checkpoint's reshape order.
        x = x.view(batch_size, self.xyz_dim_afterfc, s_dim, d_dim, r_dim)
        x = x.contiguous()

        # Mechanism convolutions.
        x = F.leaky_relu(self.fm_bn1(self.fm_conv1(x)))  # (batch, 16, s, d, r)
        x = F.leaky_relu(self.fm_bn2(self.fm_conv2(x)))
        x = F.leaky_relu(self.fm_bn3(self.fm_conv3(x)))
        x = F.leaky_relu(self.fm_bn4(self.fm_conv4(x)))

        # Flatten.
        x = x.view(batch_size, -1)

        # Final linear layers with LayerNorm and Dropout.
        x = self.dp1(F.leaky_relu(self.ln1(self.fc1(x))))
        x = self.dp2(F.leaky_relu(self.ln2(self.fc2(x))))
        x = self.dp3(F.leaky_relu(self.ln3(self.fc3(x))))
        x = self.dp4(F.leaky_relu(self.ln4(self.fc4(x))))
        x = self.fcfinal(x)  # (batch, 6)

        # Three (sin, cos) pairs: strike, dip, rake.
        grouped_output = x.reshape(-1, 3, 2)  # (batch, 3, 2)

        # Normalize each pair to unit length.
        normalized_grouped = F.normalize(grouped_output, p=2, dim=2, eps=1e-6)

        # Restrict the dip pair to the first quadrant.
        normalized_grouped2 = torch.where(
            self.dip_mask, normalized_grouped.abs(), normalized_grouped)

        x = normalized_grouped2.reshape(-1, 6)
        return x


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (8, 8, 8, 24, 7, 24)
    model = StrikeDipRakeNet(shape).to(device)
    checkpoint = Path(__file__).parent / "model" / "bfnet2_ep46_260122a.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, *shape, device=device)).cpu().numpy()
    angles = np.rad2deg(np.arctan2(output[:, 0::2], output[:, 1::2]))
    print("Device:", device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Output pairs:", output)
    print("Strike, dip, rake (degrees):", angles)
