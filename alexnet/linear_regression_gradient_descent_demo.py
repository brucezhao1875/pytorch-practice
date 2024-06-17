import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出都是一维的线性层

    def forward(self, x):
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建一些简单的训练数据
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]], requires_grad=False)

# 进行一次前向传播计算损失
outputs = model(x_train)
loss = criterion(outputs, y_train)
print(f'Initial loss: {loss.item()}')

# 反向传播计算梯度
optimizer.zero_grad()  # 清除之前的梯度
loss.backward()  # 计算当前梯度

# 计算梯度的范数平方
grad_norm_squared = 0.0
for param in model.parameters():
    param_norm = param.grad.norm()  # 计算梯度的L2范数
    grad_norm_squared += param_norm.item() ** 2  # 累加每个参数梯度的范数平方

print(f'Gradient norm squared: {grad_norm_squared}')

# 使用优化器更新参数
optimizer.step()

# 再次计算更新后的损失
outputs = model(x_train)
new_loss = criterion(outputs, y_train)
print(f'New loss: {new_loss.item()}')

# 损失的下降量
delta_loss = loss.item() - new_loss.item()
print(f'Decrease in loss: {delta_loss}')
