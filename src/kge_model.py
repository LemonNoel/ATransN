import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


class KGEModel(nn.Module):
    """
    github.com/DeepGraphLearning/KnowledgeGraphEmbedding/
    """

    def __init__(self, model_name, num_entities, num_relations, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = gamma
        self.embedding_range = (self.gamma + self.epsilon) / hidden_dim

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.empty(num_entities, self.entity_dim))
        nn.init.uniform_(self.entity_embedding, -self.embedding_range, b=self.embedding_range)

        self.relation_embedding = nn.Parameter(torch.empty(num_relations, self.relation_dim))
        nn.init.uniform_(self.relation_embedding, -self.embedding_range, b=self.embedding_range)

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range]]))

        if model_name == 'ConvE':
            self.input_dropout = nn.Dropout(0.2)
            self.hidden_dropout = nn.Dropout(0.3)
            self.feature_dropout = nn.Dropout(0.2)
            self.criterion = nn.BCELoss()
            self.l2_distance = nn.PairwiseDistance(p=2)
            self.cos_distance = nn.CosineSimilarity(dim=1, eps=1e-6)

            self.conv = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
            self.bn0 = nn.BatchNorm2d(1)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm1d(self.hidden_dim)
            self.register_parameter('b', nn.Parameter(
                torch.zeros(self.num_entities)))
            self.fc = nn.Linear(10368, self.hidden_dim)

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'ConvE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError(
                'ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        head, relation, tail = self.embed(sample, mode)
        score = self.score(head, relation, tail, mode)
        return score

    def infer_all(self, pos_part, mode='head-batch'):
        if mode == 'head-batch':
            head = self.entity_embedding.unsqueeze(0)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=pos_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=pos_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=pos_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=pos_part[:, 1]
            ).unsqueeze(1)

            tail = self.entity_embedding.unsqueeze(0)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.score(head, relation, tail, mode)
        return score

    def embed(self, sample, mode='single'):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(
                0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(
                0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        return head, relation, tail
    
    def score(self, head, relation, tail, mode="single"):
        
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'ConvE': self.ConvE
        }

        if self.model_name in model_func:
            return model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

    def TransE(self, head, relation, tail, mode):
        # entity constraint
        head = head / torch.norm(head, p=2, dim=2, keepdim=True)
        tail = tail / torch.norm(tail, p=2, dim=2, keepdim=True)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range / pi)
        phase_relation = relation / (self.embedding_range / pi)
        phase_tail = tail / (self.embedding_range / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma - score.sum(dim=2) * self.modulus
        return score

    def ConvE(self, head, relation, tail, mode):
        batch_size = head.size(0)

        if mode == 'head-batch':
            negative_sample_size = head.size(1)
            ent_embeds = head.view(-1, 1, 10, 20)
            trg_embeds = tail.repeat(1, negative_sample_size, 1)
            rel_embeds = relation.repeat(
                1, negative_sample_size, 1).view(-1, 1, 10, 20)
        else:
            negative_sample_size = tail.size(1)
            ent_embeds = head.repeat(
                1, negative_sample_size, 1).view(-1, 1, 10, 20)
            rel_embeds = relation.repeat(
                1, negative_sample_size, 1).view(-1, 1, 10, 20)
            trg_embeds = tail

        inputs = torch.cat([ent_embeds, rel_embeds], 2)

        inputs = self.bn0(inputs)
        x = self.input_dropout(inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_dropout(x)
        x = x.view(batch_size*negative_sample_size, -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x).view(batch_size, negative_sample_size, -1)
        # x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
        # x += self.b.expand_as(x)
        # pred = torch.sigmoid(x)
        score = torch.einsum('bne,bne->bn', x, trg_embeds)
        pred = torch.sigmoid(score)
        # score = self.l2_distance(x, trg_embeds) + 1. - self.cos_distance(x, trg_embeds)
        # score = torch.norm(x-trg_embeds, p=2, dim=1) + 1. - self.cos_distance(x, trg_embeds)
        # score = score.view(batch_size, negative_sample_size)

        return pred
