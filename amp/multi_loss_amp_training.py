import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 == nn.Conv2d(1,20,5)
        self.conv2 == nn.Conv2d(20,64,5)
        self.fc1 = nn.Linear(64*5*5, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、优化器和损失函数
model = SimpleModel().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()  # 假设我们有第二个损失函数

# 初始化 GradScaler
scaler = GradScaler()

# 假输入和目标
input = torch.randn(16, 1, 32, 32).cuda()
target1 = torch.randint(0, 10, (16,)).cuda()
target2 = torch.randn(16, 10).cuda()  # 第二个目标用于 MSELoss

for epoch in range(10):
    optimizer.zero_grad()

    with autocast():
        # 计算第一个损失
        output = model(input)
        loss1 = criterion1(output, target1)

    # 第一次反向传播和缩放
    scaler.scale(loss1).backward()

    with autocast():
        # 计算第二个损失
        loss2 = criterion2(output, target2)

    # 第二次反向传播和缩放
    scaler.scale(loss2).backward()

    # 执行优化步骤
    scaler.step(optimizer)
    scaler.update()

    print(f"Epoch {epoch + 1}, Loss1: {loss1.item()}, Loss2: {loss2.item()}")
