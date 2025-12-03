'''
 线性回归
 通x去预测我们的y值
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
# 导入
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# 1、准备数据集
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7],
[-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
# 2、数据清洗（无）
# 3、确认标签和特征
# 转换成numpy类型
data = np.array(data)
# 获取特征
# 所有行，的第一列
x = data[:,0]
print("特征\t",x)
# 标签
y = data[:,1]
print("标签\t",y)
plt.ion()
fig, (ax_scatter, ax_fit) = plt.subplots(1, 2, figsize=(10, 4))
ax_scatter.scatter(x, y, c="blue", zorder=1)
ax_scatter.set_title("散点图")
ax_scatter.set_xlabel("x")
ax_scatter.set_ylabel("y")
ax_scatter.set_xlim(x.min(), x.max())
ax_scatter.set_ylim(y.min()*1.1, y.max()*1.1)
ax_fit.scatter(x, y, c="blue", alpha=0.6, zorder=1)
line, = ax_fit.plot([], [], "r-", zorder=2)
ax_fit.set_title("动态拟合线")
ax_fit.set_xlabel("x")
ax_fit.set_ylabel("y")
ax_fit.set_xlim(x.min(), x.max())
ax_fit.set_ylim(y.min()*1.1, y.max()*1.1)
fig.tight_layout()
# 3、划分数据集
# 无
# 一般输入的特征都是，二维数组（矩阵）
x_tensor = torch.tensor(x,dtype=torch.float32).reshape(-1,1)
y_tensor = torch.tensor(y,dtype=torch.float32)
x_plot = np.linspace(x.min(), x.max(), 100).astype(np.float32).reshape(-1,1)
x_plot_tensor = torch.from_numpy(x_plot)
#4、构架我们的网络模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 线性层
        self.linear= nn.Linear(1,1)
    # forward函数
    def forward(self,x):
        x = self.linear(x)
        return x
# 5、超参数的设置
# 创建模型
model = MyModel()
# 学习率
lr = 0.01
# 损失函数的值
mse_loss = torch.nn.MSELoss()
# 优化器,学习率
optim = torch.optim.SGD(model.parameters(), lr=lr)
# 训练轮数
epoches = 500
# 6、训练模型
for epoch in range(1,epoches+1):
    model.train()
    y_hat = model(x_tensor)
    optim.zero_grad()
    loss_value = mse_loss(y_hat.squeeze(-1), y_tensor)
    loss_value.backward()
    optim.step()
    with torch.no_grad():
        y_plot = model(x_plot_tensor).squeeze(-1).detach().numpy()
    line.set_data(x_plot_tensor.numpy().squeeze(), y_plot)
    ax_fit.relim()
    ax_fit.autoscale_view()
    fig.canvas.draw()
    plt.pause(0.001)
    if epoch % 50 == 0:
        print(f"损失值的变换{loss_value:.4f}")
# w和b
# 权重和偏移量
w = model.linear.weight.data.numpy()
b = model.linear.bias.data.numpy()
print(f"得出最优的权重{w}和{b}")
plt.ioff()
plt.show()