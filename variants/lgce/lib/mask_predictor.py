import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from lib.lgm import MSB


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()
        self.msb1 = MSB((512,1024))
        self.msb2 = MSB((256,512))
        self.msb3 = MSB((128,256))

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)
        L = 768

        self.lg_layer1 = nn.Linear(L, 128)
        self.lg_layer2 = nn.Linear(L, 256)
        self.lg_layer3 = nn.Linear(L, 512)
        self.lg_layer4 = nn.Linear(L, 1024)

    def forward(self, x_c4, x_c3, x_c2, x_c1, lguide):
        # fuse Y4 and Y3
        B,C4,H4,W4 = x_c4.shape
        B,C3,H3,W3 = x_c3.shape
        B,C2,H2,W2 = x_c2.shape
        B,C1,H1,W1 = x_c1.shape
        lguide = lguide.mean(-1)
        #lguide = torch.nn.Parameter(torch.ones(lguide.shape,device=lguide.device))

        lg1 = self.lg_layer1(lguide).view([B,1,C1])
        lg2 = self.lg_layer2(lguide).view([B,1,C2])
        lg3 = self.lg_layer3(lguide).view([B,1,C3])
        lg4 = self.lg_layer4(lguide).view([B,1,C4])

        guide_tokens1 = [lg3,lg4]
        guide_tokens2 = [lg2,lg3]
        guide_tokens3 = [lg1,lg2]

        out1 = self.msb1([x_c3.view(B,C3,-1).transpose(1,2),x_c4.view(B,C4,-1).transpose(1,2)],guide_tokens1)
        x_c3 = out1[0][:,1:,:]
        x_c4 = out1[1][:,1:,:]
        x_c3 = x_c3.transpose(1,2).view([B,C3,H3,W3])
        x_c4 = x_c4.transpose(1,2).view([B,C4,H4,W4])
        
                
        '''out3 = self.msb3([x_c1.view(B,C1,-1).transpose(1,2),x_c2.view(B,C2,-1).transpose(1,2)],guide_tokens3)
        x_c1 = out3[0][:,1:,:]
        x_c2 = out3[1][:,1:,:]
        x_c1 = x_c1.transpose(1,2).view([B,C1,H1,W1])
        x_c2 = x_c2.transpose(1,2).view([B,C2,H2,W2])'''


        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)

        '''out2 = self.msb2([x_c2.view(B,C2,-1).transpose(1,2), x_c3.view(B,C3,-1).transpose(1,2)], guide_tokens2)
        x_c2 = out2[0][:,1:,:]
        x_c3 = out2[1][:,1:,:]
        x_c2 = x_c2.transpose(1,2).view([B,C2,H2,W2])
        x_c3 = x_c3.transpose(1,2).view([B,C3,H3,W3])'''
 
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        return self.conv1_1(x)
