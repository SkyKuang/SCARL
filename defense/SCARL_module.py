from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from ast import Num
from email.message import EmailMessage
import pdb

import numpy as np
from numpy.core.numeric import flatnonzero
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.container import T
import torch.optim as optim
from utils import one_hot_tensor
import math
import torchtext as text
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SCARL_Module(nn.Module):
    def __init__(self, basic_net, args, aux_net=None):
        super(SCARL_Module, self).__init__()
        self.basic_net = basic_net
        self.aux_net = aux_net
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.criterion_mi = SemanticMI(num_classes=args.num_classes)
        self.criterion_g = Geometry_loss()
        self.args = args
        self.num_steps = 10
        self.step_size = 2.0/255
        self.epsilon = 8.0/255
        self.beta = 6.0
        self.norm = 'l_inf'
        self.num_classes = args.num_classes

        self.momentum = 0.9
        self.embedding_dim = 50
        self.feat_dim = 64

        word_list = ['airplane', 'automobile', 'bird','cat','deer','dog','frog','horse','ship','truck']

        vec = text.vocab.GloVe(name='6B', dim=self.embedding_dim)
        self.class_vectors = vec.get_vecs_by_tokens(word_list, lower_case_backup=True).to(device)
        # self.norm_vertors = F.normalize(self.class_vectors,p=2)
        # self.soft_score = torch.mm(self.norm_vertors, self.norm_vertors.transpose(1,0))
        self.stdv = 1. / math.sqrt(self.feat_dim / 3)
        self.register_buffer('memory_feat', torch.rand(self.num_classes, self.feat_dim).mul_(2 * self.stdv).add_(-self.stdv).cuda())

    def train(self, epoch, inputs, targets, index, optimizer):
        ############ generating adversarial examples stage
        self.basic_net.eval()
        self.aux_net.eval()
        batch_size = len(inputs)
        x_adv = inputs.detach() + 0.001 * torch.randn(inputs.shape).cuda().detach()
        logits_nat = self.basic_net(inputs)
        if self.norm == 'l_inf':
            for step in range(self.num_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    logits_adv = self.basic_net(x_adv)
                    loss_adv = self.criterion_kl(F.log_softmax(logits_adv, dim=1),
                                        F.softmax(logits_nat, dim=1))
                grad = torch.autograd.grad(loss_adv, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon), inputs + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(inputs, 0.0, 1.0)
        adv_inputs = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        ############ adversarial training stage
        self.basic_net.train()
        self.basic_net.zero_grad()
        optimizer.zero_grad()

        y_gt = one_hot_tensor(targets, self.num_classes)
        embedding_txt = torch.mm(y_gt, self.class_vectors)

        ############ calculate robust loss
        logits_nat, feat_nat, _ = self.basic_net(inputs, True, embedding_txt)
        logits_adv, feat_adv, feat_txt = self.basic_net(adv_inputs, True, embedding_txt)
        loss_nat = F.cross_entropy(logits_nat, targets)
        loss_adv = F.cross_entropy(logits_adv, targets)

        ############ semantic MI

        feat_txt = F.normalize(feat_txt, p=2)
        feat_nat = F.normalize(feat_nat, p=2)
        feat_adv = F.normalize(feat_adv, p=2)
        alpha = 1
        loss_mi = alpha*self.criterion_mi(feat_adv, feat_txt, targets)

        ########### semantic structure
        self.curt_feat_txt = torch.rand_like(self.memory_feat)
        self.curt_feat_adv = torch.rand_like(self.memory_feat)
        exit_label = []
        for class_ in range(self.num_classes):
            class_index = torch.where(targets == class_)[0]
            if len(class_index) == 0:
                continue
            exit_label.append(class_)
            self.curt_feat_txt[class_] = feat_txt[class_index,:].mean(0)
            self.curt_feat_adv[class_] = feat_adv[class_index,:].mean(0)
        momentum = math.exp(-5 * (1-epoch/self.args.max_epoch)**2)
        self.memory_feat[exit_label] = self.curt_feat_txt[exit_label]* (1 - momentum) +  self.memory_feat[exit_label]*momentum
        adv_nodes = self.curt_feat_adv[exit_label]

        weight = math.exp(-5 * (1-epoch/self.args.max_epoch)**2)
        loss_geo = weight*self.criterion_g(adv_nodes, self.memory_feat)

        ############ semantic KL
        beta = 6
        loss_robust = (beta / batch_size) * self.criterion_kl(F.log_softmax(logits_adv, dim=1),F.softmax(logits_nat, dim=1))

        loss = loss_adv + loss_robust + loss_mi + loss_geo
        loss.backward()
        optimizer.step()
        self.optimizer_aux.step()

        return logits_nat.detach(), logits_adv.detach(), loss.item(), loss_mi.item(), loss_geo.item()

    def test(self, inputs, targets, adversary=None):
        if adversary is not None:
            inputs = adversary.attack(inputs, targets).detach()

        # self.basic_net.eval()
        logits = self.basic_net(inputs)
        loss = self.criterion(logits, targets)

        return logits.detach(), loss.item()


class SemanticMI(torch.nn.Module):
    def __init__(self, num_classes, temperature=0.5, master_rank="cuda", DDP=False):
        super(SemanticMI, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):
        sim_matrix_e2p = self.calculate_similarity_matrix(embed, proxy)
        sim_matrix_e2p = torch.exp(self._remove_diag(sim_matrix_e2p) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        pos_removal_mask = 1 - neg_removal_mask
        sim_neg_only_e2p = pos_removal_mask * sim_matrix_e2p
        loss_pos = -torch.log(1 - (sim_neg_only_e2p/(sim_neg_only_e2p + 0.01))).mean()

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)
        loss_neg = -torch.log(emb2proxy/(emb2proxy + 0.01)).mean()

        return loss_pos + loss_neg

class Geometry_loss(nn.Module):
    def __init__(self, w_d=2, w_a=3):
        super(Geometry_loss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)
        #Distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d)
        #Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        # prod = e @ e.t()
        prod = torch.matmul(e,e.t())
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res