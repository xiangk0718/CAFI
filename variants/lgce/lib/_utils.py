from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
from einops import rearrange, repeat


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        #self.gtoken = torch.nn.Parameter(torch.randn([1,768,20]))
        #self.gtoken = torch.nn.Parameter(torch.ones([1,768,20]))

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        B = x.shape[0]
        #gtoken = repeat(self.gtoken, '1 c d -> b c d', b = B)
        gtoken = l_feats
        x = self.classifier(x_c4, x_c3, x_c2, x_c1, gtoken)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        #self.gtoken = torch.Parameter(torch.randn(l_feats.shape))

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1, l_feats)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass


class _LAVTCattSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTCattSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        #self.gtoken = torch.nn.Parameter(torch.randn([1,768,20]))
        #self.gtoken = torch.nn.Parameter(torch.ones([1,768,20]))

    def forward(self, x, l_feats, l_mask, nl_feats, nl_mask):
        input_shape = x.shape[-2:]
        #positive_features = self.backbone(x, l_feats, l_mask)
        #negative_features = self.backbone(x, nl_feats, nl_mask)
        #features = tuple(pf - nf for pf, nf in zip(positive_features, negative_features))
        #层归一化
        features = self.backbone(x, l_feats, l_mask, nl_feats, nl_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        B = x.shape[0]
        #gtoken = repeat(self.gtoken, '1 c d -> b c d', b = B)
        gtoken = l_feats
        x = self.classifier(x_c4, x_c3, x_c2, x_c1, gtoken)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVTCatt(_LAVTCattSimpleDecode):
    pass