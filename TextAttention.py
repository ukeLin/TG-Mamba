import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class TextAttention(nn.Module):
    def __init__(self, input_channels, text_embedding_dim):
        super(TextAttention, self).__init__()

        # 设置一个较小的隐藏维度
        hidden_dim = input_channels // 4  # 可以根据需要调整此值，以减少参数

        # 文本嵌入处理使用深度可分离卷积
        self.text_conv = nn.Sequential(
            nn.Conv1d(text_embedding_dim, hidden_dim, kernel_size=1),  # 首先减少维度
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_channels, kernel_size=1),  # 再映射回输入通道数
            # nn.Dropout(0.4)  # 可选的 Dropout
        )

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.Conv1d(input_channels, input_channels // 32, kernel_size=1),  # 减小的压缩维度
            nn.ReLU(inplace=True),
            nn.Conv1d(input_channels // 32, input_channels, kernel_size=1),  # 激励步骤
            nn.Sigmoid()
        )
    
    def forward(self, image_features, text_embedding):
        # 获取图像形状
        B, H, W, C = image_features.shape

        # 处理文本特征
        text_embedding = text_embedding.view(B, -1, 1)  # 改变形状为 (B, text_embedding_dim, 1)

        # print("embedding", text_embedding.shape)
        text_features = self.text_conv(text_embedding)

        # print("features", text_features.shape)
        # 通道注意力机制
        attention_weights = self.channel_attention(text_features)
        enhanced_text_features = text_features * attention_weights
        
        # 变换为 (B, C, 1, 1)
        enhanced_text_features = enhanced_text_features.view(B, C, 1, 1).expand(B, C, H, W)
        output = enhanced_text_features.permute(0, 2, 3, 1)

        # 融合原始文本特征
        text_features = text_features.view(B, C, 1, 1).expand(B, C, H, W).permute(0, 2, 3, 1)
        output = output + text_features
        
        return output

if __name__ == "__main__":
    x = torch.randn(64, 5)  # 输入形状调整为 (batch_size, text_embedding_dim)
    y = torch.randn(64, 56, 56, 96)
    model = TextAttention(96, 5)
    out = model(y, x)
    # print(out.shape)  # 期望输出形状: (batch_size, height, width, channels)
