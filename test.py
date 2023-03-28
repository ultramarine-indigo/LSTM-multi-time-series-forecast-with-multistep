import torch
import numpy as np

batch_size=2
len=4
feature=3

'''
input = torch.rand(batch_size,len,feature)
pred0=torch.rand(batch_size,1,feature)
pred=torch.rand(batch_size,feature)
pred1 = torch.unsqueeze(pred,1)


output1=torch.transpose()
print()
print(pred0)
print(pred)
print(pred1)
print(pred1.shape)

z=torch.cat([input,pred1],1)
z1=z[:,1:,:]
print(input)
print(pred)
print(z)
print(z1)
'''
output=np.arange(24).reshape(len,batch_size,feature)
output1=output.transpose(1,0,2)
print(output)
print(output1)