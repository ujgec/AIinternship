import cv2
# 获取图片，读图片
img = cv2.imread('image0.jpg')
# 查看图像的维度 高 宽 通道数,B G R(蓝色，绿色，红色)
# 获取宽和高
height,width,chanels = img.shape
print(img.shape)
# 获取宽度的一半
half_width = int(width/2)
print(half_width)
# 从一半到结束的位置
b_channel = img[:,half_width:width,0]
print(b_channel.shape)
g_channel = img[:,half_width:width,1]
r_channel = img[:,half_width:width,2]
b_channel[:,:] = 0
g_channel[:,:] = 0
# 合并
cv2.merge([b_channel,g_channel,r_channel])
cv2.imshow('img_new',img)
# cv2
# 等待手动关闭
cv2.waitKey(0)
# 释放资源
cv2.destroyAllWindows()