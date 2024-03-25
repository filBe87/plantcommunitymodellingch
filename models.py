import torch.nn as nn

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out   
    
class SDMNet(nn.Module):
    def __init__(self, in_features, num_classes, num_filts):
        super(SDMNet, self).__init__()

        # define layers used for geoprior
        self.feats = nn.Sequential(nn.Linear(in_features, num_filts),
                                   nn.BatchNorm1d(num_filts),
                                   nn.ReLU(inplace=True),
                                   ResLayer(num_filts),
                                   ResLayer(num_filts),
                                   ResLayer(num_filts),
                                   ResLayer(num_filts))
        
        self.class_emb = nn.Linear(num_filts, num_classes)
        
    def forward(self, loc_feats):
        loc_emb = self.feats(loc_feats.float())
        prior_pred = self.class_emb(loc_emb)
        
        return prior_pred