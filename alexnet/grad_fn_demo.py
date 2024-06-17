import torch
# 示例：创建一个张量并执行一些操作
x = torch.randn(3, 3, requires_grad=True)
y = x + 2
z = y.mean()
x.register_hook(lambda grad:print('x grad:',grad))
y.register_hook(lambda grad:print('y grad:',grad))
z.register_hook(lambda grad:print('z grad:',grad))

z.backward()
