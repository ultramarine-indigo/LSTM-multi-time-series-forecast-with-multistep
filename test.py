import torch


batch_size=2
len=4
feature=3

input = torch.rand(batch_size,len,feature)

pred=torch.rand(batch_size,1,feature)


z=torch.cat([input,pred],1)
z1=z[:,1:,:]
print(input)
print(pred)
print(z)
print(z1)