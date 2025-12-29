import torch
import torch.nn.functional as F
import torch.nn as nn
import torchsummary


# 模型输入：
# 亮度场最大值处的x,y,z,s,d,r六维数据，shape=[16,16,14,12,4,12]
# 模型输出：
# 走向、倾角、滑动角的sin、cos对，经过计算得到的矩张量参数Mxx, Mxy, Mxz, Myz, Myy, Mzz
# 输出和fmnetv1相同
# 增加了sincos平方和为1的限制


class StrikeDipRakeNet(nn.Module):
    def __init__(self, shape):
        super(StrikeDipRakeNet, self).__init__()
        self.name = "bfnet"
        # Extract dimensions from shape
        self.shape = shape
        x_dim, y_dim, z_dim, s_dim, d_dim, r_dim = shape  # Corrected order
        self.sdr_dim = s_dim * d_dim * r_dim
        self.xyz_dim = x_dim * y_dim * z_dim
        self.xyz_dim_afterconv = (x_dim // 2 // 2) * \
            (y_dim // 2 // 2) * (z_dim // 2 // 2)
        self.xyz_dim_afterfc = 8

        # spatial convolutional layers
        # 只在x, y, z空间维度上卷积，保持s, d, r形状
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
        # 在s, d, r维度上卷积
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
        # 计算flattened_size
        self.flattened_size = 8 * s_dim * d_dim * r_dim
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fcfinal = nn.Linear(16, 6)  # 输出sin/cos对

    def normalize_sin_cos_torch(self, sin_vec, cos_vec):
        # 使得vec中对应的sin、cos平方和为1
        # 计算sin和cos的平方和
        norm = torch.sqrt(sin_vec**2 + cos_vec**2)
        print("normshape", norm.shape)
        # 归一化
        sin_norm = sin_vec / norm
        cos_norm = cos_vec / norm
        return sin_norm, cos_norm

    def forward(self, x):
        batch_size, x_dim, y_dim, z_dim, s_dim, d_dim, r_dim = x.size()
        global_min = x.min()
        global_max = x.max()
        # Min-Max归一化到[0,1]
        x_range = global_max - global_min + 1e-8
        x = (x - global_min) / x_range

        # 空间卷积：合并s, d, r到channel维，保持空间结构
        x = x.view(batch_size, x_dim, y_dim, z_dim, s_dim * d_dim * r_dim)
        # (batch, channel_sdr, x, y, z)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # (batch, channel_sdr, x, y, z)
        x = F.leaky_relu(self.spatial_conv1(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        # (batch, channel_sdr, x//2, y//2, z//2)
        x = F.leaky_relu(self.spatial_conv2(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        # 展平最后三个维度
        # (batch*s*d*r, x//2//2*y//2//2*z//2//2)
        x = x.view(batch_size*self.sdr_dim, self.xyz_dim_afterconv)
        # 全连接层
        x = F.leaky_relu(self.spatial_fc1(x))
        x = F.leaky_relu(self.spatial_fc2(x))  # (batch*s*d*r, xyz_dim_afterfc)

        # 恢复s, d, r维度，准备在s, d, r上卷积
        x = x.view(batch_size, self.xyz_dim_afterfc, s_dim, d_dim, r_dim)
        x = x.contiguous()

        # 在s, d, r上卷积
        x = F.leaky_relu(self.fm_bn1(self.fm_conv1(x)))  # (batch, 16, s, d, r)
        x = F.leaky_relu(self.fm_bn2(self.fm_conv2(x)))
        x = F.leaky_relu(self.fm_bn3(self.fm_conv3(x)))
        x = F.leaky_relu(self.fm_bn4(self.fm_conv4(x)))

        # 展平
        x = x.view(batch_size, -1)

        # 最终全连接层
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fcfinal(x)
        x = torch.tanh(x)

        # 分组归一化：对每组(sin, cos)分别归一化（核心步骤）
        grouped_output = x.reshape(-1, 3, 2)  # (batch_size, 3, 2)
        normalized_grouped = F.normalize(grouped_output, p=2, dim=2)
        x = normalized_grouped.reshape(-1, 6)

        # Return sin/cos pairs
        return x


# Example usage
if __name__ == "__main__":
    shape = (8, 8, 8, 24, 7, 24)  # Specify the shape
    model = StrikeDipRakeNet(shape).cuda()
    input_data = torch.randn(2, *shape).cuda()  # Match the shape
    moment_tensor = model(input_data).cpu().detach().numpy()
    print("Moment tensor:", moment_tensor)
    # Print model summary
    torchsummary.summary(model, input_size=shape, device="cuda")
