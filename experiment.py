import torch
import torch.nn as nn
import torch.nn.functional as F
# experiment on softmax function

attention = torch.randn(4,4,4)
# print(attention)
b = torch.softmax(attention,dim=-1)
sum_row = torch.sum(b,dim=-1)
print(sum_row)

