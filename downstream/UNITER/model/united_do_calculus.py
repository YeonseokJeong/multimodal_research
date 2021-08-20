from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class Causal_v(nn.Module):
    def __init__(self):
        super(Causal_v, self).__init__()
        self.embedding_size = 768
        self.Wy = nn.Linear(self.embedding_size, self.embedding_size)
        self.Wz = nn.Linear(2048, self.embedding_size)
        self.dic_z = torch.tensor(np.load("./conf_and_prior_1_mrc_from_devlbert/dic_v.npy"), dtype=torch.float16).cuda()
        self.prior = torch.tensor(np.load("./conf_and_prior_1_mrc_from_devlbert/prior_v.npy"), dtype=torch.float16).cuda()
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)

    def forward(self, y):
        temp = []
        # Class = torch.argmax(image_target, 2, keepdim=True)
        # for boxes, cls in zip(y, Class):
        for boxes in y:
            attention = torch.mm(self.Wy(boxes), self.Wz(self.dic_z).t()) / (self.embedding_size ** 0.5)
            attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
            z_hat = attention.unsqueeze(2) * self.dic_z.unsqueeze(0)  # torch.Size([box, 1601, 2048])
            z = torch.matmul(self.prior.unsqueeze(0), z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])
            temp.append(z)
        temp = torch.stack(temp, 0)
        return temp

class Causal_t(nn.Module):
    def __init__(self):
        super(Causal_t, self).__init__()
        self.embedding_size = 768
        self.Wy = nn.Linear(768, 768)
        self.Wz = nn.Linear(768, 768)
        self.dic_z = torch.tensor(np.load("./conf_and_prior_1_mrc_from_devlbert/dic_t.npy"), dtype=torch.float16).cuda()
        self.prior = torch.tensor(np.load("./conf_and_prior_1_mrc_from_devlbert/prior_t.npy"), dtype=torch.float16).cuda()
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)
        # self.id2class = np.load("./dic/id2class.npy", allow_pickle=True).item()

    def forward(self, y):
        temp = []
        for sentence in y:
            attention = torch.mm(self.Wy(sentence), self.Wz(self.dic_z).t()) / (self.embedding_size ** 0.5)
            attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
            z_hat = attention.unsqueeze(2) * self.dic_z.unsqueeze(0)  # torch.Size([box, 1601, 2048])
            z = torch.matmul(self.prior.unsqueeze(0), z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])
            temp.append(z)
        temp = torch.stack(temp, 0)
        return temp