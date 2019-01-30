import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

EPOCH = 8
BATCH_SIZE = 50
LR = 0.001


# Define Net
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),   # padding=2保证输入输出尺寸相同
            nn.ReLU(),                  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2) # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # =(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def readImg(path):
    return Image.open(path)


trainset = tv.datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

testset = tv.datasets.ImageFolder(root='./digits', transform=transforms.ToTensor(), loader=readImg)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)
# 定义loss function 和优化器（采用SGD）
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
accuracy = []

# train
if __name__ == "__main__":
    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # print("predict:", predicted)
                # print("label:", labels)
                correct += predicted.eq(labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
            accuracy.append(100 * correct / total)
        torch.save(net.state_dict(), '%s/net_%03d.pth' % ("./model/", epoch + 1))

x = range(8)
plt.plot(x, accuracy)
plt.savefig('accuracy.jpg')
