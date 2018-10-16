import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

net = Net()
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0,1],[0,1], [0,1], [1,0]]

# 新建一个优化器, 指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.MSELoss()
for i in  range(100):
    for j in range(4):
        output = net(Variable(torch.Tensor(x[j])))
        # 在训练过程中:
        optimizer.zero_grad()  # 首先梯度清零(与 net.zero_grad() 效果一样)
        loss = criterion(output, Variable(torch.Tensor(y[j])))
        loss.backward()
        optimizer.step()  # 更新参数
        print(loss)


