import torch
import torch.optim as optim

# 定义一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(1000, 5)

    def forward(self, x):
        return self.fc1(x)

# 创建模型
model = SimpleModel()

# 查看模型参数的存储空间和梯度空间
total_model_param_space = 0
total_model_grad_space = 0

for p in model.parameters():
    total_model_param_space += p.data.nelement() * p.data.element_size()
    if p.grad is not None:
        total_model_grad_space += p.grad.nelement() * p.grad.element_size()

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 运行前向和反向传播
inputs = torch.randn(1, 1000)
outputs = model(inputs)
loss = outputs.sum()
loss.backward()

# 查看优化器的存储空间构成
total_optimizer_space = 0

# 参数存储空间（包括梯度）
for group in optimizer.param_groups:
    for p in group['params']:
        total_optimizer_space += p.data.nelement() * p.data.element_size()
        if p.grad is not None:
            total_optimizer_space += p.grad.nelement() * p.grad.element_size()

# 超参数存储空间（假设每个超参数占用 4 字节）
total_optimizer_space += len(optimizer.defaults) * 4

print(f"Total model parameters storage space: {total_model_param_space} bytes")
print(f"Total model gradients storage space: {total_model_grad_space} bytes")
print(f"Total optimizer storage space: {total_optimizer_space} bytes")
