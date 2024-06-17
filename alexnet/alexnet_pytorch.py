import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        # 'super' is a built-in function in python that returns an object representing the superclass.
        # it takes 2 arguments: 'AlexNet' to tell 'super' function 怎么找到父类, 'self' represents the instance of AlexNet.
        # it is simple & clear. 2个argument都是necessary: one to determine what is the super class, another to determine the instance.
        # 这里曾经让我困惑的地方是：一般定义中self都是放前面，怎么这里self放后面。
        # 实际上，super的__init__定义中，第1个隐含参数还是self，这是由python隐含处理的。
        # 后面的2个参数，才是需要传入的，其中传入的第2个参数'self'代表的是this instance.
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv Layer 1: 224x224x3 (input) to 55x55x96 (output)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),#(h,w): (w-k+2*padding)/stride+1=(224-11+2*2)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#(h,w): (w-k+2*padding)/stride+1=(55-3)/2+1=27
            # Conv Layer 2: 55x55x96 to 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2), #(h,w): (w-k+2*padding)/stride+1=(27-5+2*2)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #(h,w): (w-k+2*padding)/stride+1=(27-3)/2+1 = 13
            # Conv Layer 3: 27x27x256 to 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1), #(h,w): (w-k+2*padding)/stride+1=(13-3+2*1)/1 + 1 = 13
            nn.ReLU(inplace=True),

            # Conv Layer 4: 13x13x384 to 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1), #这种情况下h,w不变:stride=1,padding*2+1=k
            nn.ReLU(inplace=True),

            # Conv Layer 5: 13x13x384 to 6x6x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1), #(h,w): (w-k+2*padding)/stride+1=(13-3+2*1)/1+1=13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),   #(h,w): (w-k)/stride+1 = (13-3)/2+1 = 6
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # Fully connected layers
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # Pass the input tensor through the AlexNet architecture
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = AlexNet(num_classes=1000)
print(model)
print("parameter amount:",sum(p.numel() for p in model.parameters() if p.requires_grad))

