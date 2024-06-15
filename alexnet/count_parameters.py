import torch
import torch.nn as nn

# 创建一个示例输入张量
input_tensor = torch.randn(1, 3, 64, 64)

# 创建一个Conv2D层
conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)

# 应用Conv2D操作
output_tensor = conv_layer(input_tensor)

# 输出张量的形状和参数数量
print(output_tensor.shape)
print("参数数量：", sum(p.numel() for p in conv_layer.parameters() if p.requires_grad))

# bias = True/False, 参数量差别为10， 为out_channels的数量。 -- 跟inchannel无关。
# 参数量： bias=True时：( IN * K * K + 1 ) * OUT ； bias=False时： ( IN * K * K ) * OUT
conv_layer_2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=4, bias=True)
output_tensor = conv_layer_2(input_tensor)
print(output_tensor.shape)

# numel() : number of element.
sum(p.numel() for p in conv_layer_2.parameters() if p.requires_grad)
print("parameters amount:", sum(p.numel() for p in conv_layer_2.parameters() if p.requires_grad))
print("parameters amount:", sum(p[1].numel() for p in conv_layer_2.named_parameters() if p[1].requires_grad))
for name, param in conv_layer_2.named_parameters():
    print("name is:",name,",value is:",param)
print("#############\n")
for param in conv_layer_2.parameters():
    print(param)