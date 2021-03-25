import os
import sys
import random
import logging
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import shutil
import datetime
from torch.utils.data import DataLoader
from dataset import KGDataset, TrainDataset, TestDataset, BidirectionalOneShotIterator, load_shared_entity, batch_iter
from transfer_module import TransNetwork, Generator, Discriminator
from kge_model import KGEModel
from link_predictor import link_prediction
from util import *
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


def train(args):
    # ----------------
    # Load Data
    # ----------------
    logging.info("loading data..")

    teacher = KGDataset(dict_path=args.teacher_data_path, model_path=args.teacher_model_path)
    student = KGDataset(data_path=args.student_data_path, dict_path=args.student_data_path)
    shared_entity = load_shared_entity(args.shared_entity_path)
    shared_entity = shared_entity[(shared_entity["student"] < student.get_entity_count()) & (shared_entity["teacher"] < teacher.get_entity_count())]
    student2teacher = dict(zip(shared_entity["student"], shared_entity["teacher"]))

    logging.info("train: valid: test = %d: %d: %d" % (
        len(student.train_set), len(student.valid_set), len(student.test_set)))
    num_entities = student.get_entity_count()
    num_relations = student.get_relation_count()

    logging.info("number of entities: %d" % (num_entities))
    logging.info("number of relations: %d" % (num_relations))

    # training data
    train_loader_head = DataLoader(
        TrainDataset(student.train_set,
                     num_entities,
                     num_relations,
                     args.num_neg_samples,
                     'head-batch'),
        batch_size=args.kge_batch,
        shuffle=True,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TrainDataset.collate_fn)
    train_loader_tail = DataLoader(
        TrainDataset(student.train_set,
                     num_entities,
                     num_relations,
                     args.num_neg_samples,
                     'tail-batch'),
        batch_size=args.kge_batch,
        shuffle=True,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TrainDataset.collate_fn)
    train_iterator = BidirectionalOneShotIterator(
        train_loader_head, train_loader_tail)

    # validation data and test data
    all_student_data = student.train_set + student.valid_set + student.test_set
    
    valid_loader_head = DataLoader(
        TestDataset(student.valid_set,
                    all_student_data,
                    num_entities,
                    num_relations,
                    'head-batch'),
        batch_size=args.test_batch,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TestDataset.collate_fn)
    valid_loader_tail = DataLoader(
        TestDataset(student.valid_set,
                    all_student_data,
                    num_entities,
                    num_relations,
                    'tail-batch'),
        batch_size=args.test_batch,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TestDataset.collate_fn)
    valid_dataloaders = [valid_loader_head, valid_loader_tail]

    test_loader_head = DataLoader(
        TestDataset(student.test_set,
                    all_student_data,
                    num_entities,
                    num_relations,
                    'head-batch'),
        batch_size=args.test_batch,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TestDataset.collate_fn)
    test_loader_tail = DataLoader(
        TestDataset(student.test_set,
                    all_student_data,
                    num_entities,
                    num_relations,
                    'tail-batch'),
        batch_size=args.test_batch,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TestDataset.collate_fn)
    test_dataloaders = [test_loader_head, test_loader_tail]

    # ----------------
    # Prepare Data
    # ----------------

    logging.info("preparing data..")

    writer = SummaryWriter(args.save_path)

    if args.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % (args.gpu_id))
    teacher.entity_embedding = teacher.entity_embedding.to(device).requires_grad_(False)
    teacher.relation_embedding = teacher.relation_embedding.to(device).requires_grad_(False)

    learner = KGEModel(model_name=args.kge_model,
                       num_entities=num_entities,
                       num_relations=num_relations,
                       hidden_dim=args.emb_dim,
                       gamma=args.margin,
                       double_entity_embedding=args.kge_model in ["RotatE", "ComplEx"],
                       double_relation_embedding=args.kge_model in ["ComplEx"]).to(device)
    transnet = TransNetwork(teacher.get_embedding_dim(), learner.entity_dim).to(device)
    generator = Generator(learner.entity_dim).to(device)
    discriminator = Discriminator(learner.entity_dim, activation=nn.Sigmoid()).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(list(transnet.parameters()) + list(discriminator.parameters()), lr=args.gan_lr, betas=(0.5, 0.9))
    optimizer_l = optim.Adam(list(transnet.parameters()) + list(learner.parameters()), lr=args.kge_lr)
    # scheduler_g = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer_g, args.steps//100, args.steps, num_cycles=4, min_percent=args.gan_lr*0.001)
    # scheduler_d = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer_d, args.steps//100, args.steps, num_cycles=4, min_percent=args.gan_lr*0.001)
    # scheduler_l = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer_l, args.steps//100, args.steps, num_cycles=4, min_percent=args.kge_lr*0.001)
    scheduler_g = get_constant_schedule_with_warmup(optimizer_g, args.steps//100)
    scheduler_d = get_constant_schedule_with_warmup(optimizer_d, args.steps//100)
    scheduler_l = get_constant_schedule_with_warmup(optimizer_l, args.steps//100)
    optimizer_g.zero_grad()
    optimizer_g.step()
    scheduler_g.step(0)
    optimizer_d.zero_grad()
    optimizer_d.step()
    scheduler_d.step(0)
    optimizer_l.zero_grad()
    optimizer_l.step()
    scheduler_l.step(0)

    bce_loss = nn.BCELoss(reduction='none')
    mse_loss = nn.MSELoss(reduction='none')
    cos_loss = nn.CosineEmbeddingLoss(reduction='none')

    logging.info("begin training..")
    transnet.train()
    generator.train()
    discriminator.train()
    learner.train()
    shared_iterator = batch_iter(shared_entity, args.gan_batch)

    training_logs = []
    valid_best_metrics = {}
    test_best_metrics = {}

    for step in range(1, args.steps+1):
        log = {}

        loss_g_selfs = []
        loss_g_dists = []
        loss_gs = []
        loss_d_reals = []
        loss_d_transes = []
        loss_d_fakes = []
        loss_ds = []
        teacher_entities = []
        student_entities = []
        for gan_step in range(args.gan_steps):
            try:
                shared_batch = next(shared_iterator)
            except StopIteration:
                shared_iterator = batch_iter(shared_entity, args.gan_batch)
                shared_batch = next(shared_iterator)
            teacher_entity = torch.LongTensor(
                shared_batch['teacher'].values).to(device)
            student_entity = torch.LongTensor(
                shared_batch['student'].values).to(device)
            teacher_embed = torch.index_select(
                teacher.entity_embedding, dim=0, index=teacher_entity)
            student_embed = torch.index_select(
                learner.entity_embedding, dim=0, index=student_entity)
            gan_batch_size = teacher_entity.size(0)
            gan_ones = torch.ones((gan_batch_size, 1), device=device, requires_grad=False)
            teacher_entities.append(teacher_entity)
            student_entities.append(student_entity)

            # --------------------
            # Train Discriminator
            # --------------------

            random_z = (torch.rand_like(student_embed) * 2 - 1)
            fake_emb = generator(student_embed.detach(), random_z)
            real_output = discriminator(student_embed.detach(), student_embed.detach())
            trans_output = discriminator(student_embed.detach(), transnet(teacher_embed))
            fake_output = discriminator(student_embed.detach(), fake_emb.detach())

            loss_d_real = bce_loss(real_output, gan_ones).mean()
            loss_d_trans = bce_loss(trans_output, gan_ones).mean()
            loss_d_fake = bce_loss(fake_output, 1 - gan_ones).mean()

            # beta = anneal_fn("cosine", step, args.steps, args.kge_beta, 0)
            beta = 0
            loss_d = loss_d_real + loss_d_fake + beta * loss_d_trans

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            scheduler_d.step(step)

            loss_d_reals.append(loss_d_real.item())
            loss_d_transes.append(loss_d_trans.item())
            loss_d_fakes.append(loss_d_fake.item())
            loss_ds.append(loss_d.item())

            if args.gan_n_critic > 0:
                if gan_step % args.gan_n_critic == 0:

                    # ----------------
                    # Train Generator
                    # ----------------

                    # random_z = (torch.rand_like(student_embed) * 2 - 1)
                    # fake_emb = generator(student_embed.detach(), random_z)
                    loss_g_self = bce_loss(discriminator(student_embed.detach(), fake_emb), gan_ones).mean()
                    loss_g_dist = cos_loss(student_embed.detach(), fake_emb, gan_ones).mean()
                    alpha = anneal_fn("cosine", step, args.steps, args.kge_alpha, 0)
                    loss_g = loss_g_self + alpha * loss_g_dist

                    optimizer_g.zero_grad()
                    loss_g.backward()
                    optimizer_g.step()
                    scheduler_g.step(step)
                    loss_g_selfs.append(loss_g_self.item())
                    loss_g_dists.append(loss_g_dist.item())
                    loss_gs.append(loss_g.item())
            else:
                for gene_step in range(-args.gan_n_critic):

                    # ----------------
                    # Train Generator
                    # ----------------

                    random_z = (torch.rand_like(student_embed) * 2 - 1)
                    fake_emb = generator(student_embed.detach(), random_z)
                    loss_g_self = bce_loss(discriminator(student_embed.detach(), fake_emb), gan_ones).mean()
                    loss_g_dist = cos_loss(student_embed.detach(), fake_emb, gan_ones).mean()
                    alpha = anneal_fn("cosine", step, args.steps, args.kge_alpha, 0)
                    loss_g = loss_g_self + alpha * loss_g_dist

                    optimizer_g.zero_grad()
                    loss_g.backward()
                    optimizer_g.step()
                    scheduler_g.step(step)
                    loss_g_selfs.append(loss_g_self.item())
                    loss_g_dists.append(loss_g_dist.item())
                    loss_gs.append(loss_g.item())

        log["loss_g_self"] = np.mean(loss_g_selfs) if len(loss_gs) > 0 else 0.0
        log["loss_g_dist"] = np.mean(loss_g_dists) if len(loss_gs) > 0 else 0.0
        log["loss_g"] = np.mean(loss_gs) if len(loss_gs) > 0 else 0.0
        log["loss_d_real"] = np.mean(loss_d_reals) if len(loss_ds) > 0 else 0.0
        log["loss_d_trans"] = np.mean(loss_d_transes) if len(loss_ds) > 0 else 0.0
        log["loss_d_fake"] = np.mean(loss_d_fakes) if len(loss_ds) > 0 else 0.0
        log["loss_d"] = np.mean(loss_ds) if len(loss_ds) > 0 else 0.0

        # --------------------
        # Train Learner
        # --------------------

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        teacher_entity = []
        student_entity = []
        transfered_sample = []
        transfered_weight = []
        transfered_index = []
        for i, s in enumerate(positive_sample[:, 0].numpy()):
            t = student2teacher.get(s, -1)
            if t != -1:
                teacher_entity.append(t)
                student_entity.append(s)
                transfered_sample.append(positive_sample[i])
                transfered_weight.append(subsampling_weight[i].item())
                transfered_index.append(i)
        num_transfered_head = len(transfered_sample)
        for i, s in enumerate(positive_sample[:, 2].numpy()):
            t = student2teacher.get(s, -1)
            if t != -1:
                teacher_entity.append(t)
                student_entity.append(s)
                transfered_sample.append(positive_sample[i])
                transfered_weight.append(subsampling_weight[i].item())
                transfered_index.append(i)
        teacher_entity = torch.LongTensor(teacher_entity).to(device)
        student_entity = torch.LongTensor(student_entity).to(device)
        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        transfered_sample = torch.cat(transfered_sample, dim=0).view(-1, 3).to(device)        
        subsampling_weight = subsampling_weight.to(device)
        transfered_weight = torch.tensor(transfered_weight).to(device)
        transfered_index = torch.LongTensor(transfered_index).to(device)
        # teacher_entities.append(teacher_entity)
        # student_entities.append(student_entity)

        positive_score = learner(positive_sample)
        negative_score = learner((positive_sample, negative_sample), mode=mode)

        if transfered_sample.size(0) > 0:
            transfered_negative_score = torch.index_select(
                    negative_score, dim=0, index=transfered_index)
            teacher_embed = torch.index_select(
                teacher.entity_embedding, dim=0, index=teacher_entity)
            student_embed = torch.index_select(
                learner.entity_embedding, dim=0, index=student_entity)

            transfered_head = torch.cat([
                transnet(teacher_embed[:num_transfered_head]),
                torch.index_select(
                    learner.entity_embedding,
                    dim=0,
                    index=transfered_sample[num_transfered_head:, 0])],
                dim=0).unsqueeze(1)
            transfered_relation = torch.index_select(
                    learner.relation_embedding,
                    dim=0,
                    index=transfered_sample[:, 1]).unsqueeze(1)
            transfered_tail = torch.cat([
                torch.index_select(
                    learner.entity_embedding,
                    dim=0,
                    index=transfered_sample[:num_transfered_head, 2]),
                transnet(teacher_embed[num_transfered_head:])],
                dim=0).unsqueeze(1)

            transfered_score = learner.score(transfered_head, transfered_relation, transfered_tail)
            # gan_validity = torch.ones((transfered_weight.size(0), 1), dtype=transfered_weight.dtype, device=transfered_weight.device) # w/o AAM
            gan_validity = discriminator(student_embed, transnet(teacher_embed))
            transfered_weight = gan_validity * transfered_weight
        else:
            transfered_negative_score = torch.tensor([[0.0]]).to(device)
            transfered_score = torch.tensor([[0.0]]).to(device)
            gan_validity = 1
            transfered_weight = torch.tensor([0.0]).to(device)

        if learner.model_name == 'ConvE':
            positive_sample_loss = bce_loss(F.logsigmoid(positive_score), torch.ones_like(positive_score)).mean()
            negative_sample_loss = bce_loss(F.logsigmoid(-negative_score), torch.zeros_like(negative_score)).mean()
            transfered_sample_loss = (gan_validity * bce_loss(F.logsigmoid(transfered_score), torch.ones_like(transfered_score))).mean()
            loss_l_self = positive_sample_loss + negative_sample_loss
            loss_l_trans = transfered_sample_loss
        elif learner.model_name == 'ComplEx' or learner.model_name == 'RotatE':
            positive_sample_loss = - F.logsigmoid(positive_score).mean()
            negative_sample_loss = - F.logsigmoid(-negative_score).mean()
            transfered_sample_loss = - (gan_validity * F.logsigmoid(transfered_score)).mean()
            loss_l_self = positive_sample_loss + negative_sample_loss
            loss_l_trans = transfered_sample_loss
        elif learner.model_name == 'TransE' or learner.model_name == 'DistMult':
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = negative_score.mean()
            transfered_sample_loss = - transfered_score.mean()
            loss_l_self = F.relu(-positive_score+negative_score+args.margin).mean()
            loss_l_trans = (F.relu(-transfered_score+transfered_negative_score+args.margin) * gan_validity.unsqueeze(1)).mean()

        if len(teacher_entities) > 0:
            teacher_entities = torch.cat(teacher_entities, dim=0)
            student_entities = torch.cat(student_entities, dim=0)
            teacher_embeds = torch.index_select(
                teacher.entity_embedding, dim=0, index=teacher_entities)
            student_embeds = torch.index_select(
                learner.entity_embedding, dim=0, index=student_entities)
            trans_embeds = transnet(teacher_embeds)
            # gan_validity = torch.ones((trans_embeds.size(0), 1), dtype=trans_embeds.dtype, device=trans_embeds.device) # w/o AAM
            gan_validity = discriminator(student_embeds, trans_embeds)
            loss_l_dist = (gan_validity * cos_loss(
                student_embeds, trans_embeds, torch.ones(trans_embeds.size(0), device=device))).mean()
        else:
            loss_l_dist = torch.tensor([0.0]).to(device)

        if args.reg != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            reg = args.reg * (
                learner.entity_embedding.norm(p=3) ** 3 +
                learner.relation_embedding.norm(p=3).norm(p=3) ** 3)
            loss_l_self = loss_l_self + reg
            reg_log = {'reg': reg.item()}
            log.update(reg_log)
        else:
            reg_log = {}


        alpha = anneal_fn("cyclical_cosine", step, args.steps//2, args.kge_alpha, 0)
        beta = anneal_fn("cyclical_cosine", step, args.steps//2, args.kge_beta, 0)
        # alpha = args.kge_alpha
        # beta = args.kge_beta
        loss_l = loss_l_self + alpha * loss_l_dist + beta * loss_l_trans

        optimizer_l.zero_grad()
        loss_l.backward()
        optimizer_l.step()
        scheduler_l.step(step)

        log["alpha"] = alpha
        log["beta"] = beta
        log["pos_loss"] = positive_sample_loss.item()
        log["neg_loss"] = negative_sample_loss.item()
        log["loss_l_self"] = loss_l_self.item()
        log["loss_l_dist"] = loss_l_dist.item()
        log["loss_l_transfer"] = transfered_sample_loss.item()
        log["loss_l"] = loss_l.item()

        # ----------------
        # Save logs
        # ----------------
        training_logs.append(log)

        # save summary
        for k, v in log.items():
            writer.add_scalar('loss/%s' % (k), v, step)

        # training average
        if len(training_logs) > 0 and (step % args.print_steps == 0 or step == args.steps):
            logging.info("--------------------------------------")
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum(
                    [log[metric] for log in training_logs]) / len(training_logs)
            log_metrics('training average', step, metrics)

            for k, v in metrics.items():
                writer.add_scalar('train-metric/%s' % (k.replace("@", "_")), v, step)

        # evaluate model
        if len(training_logs) > 0 and (step % args.eval_steps == 0 or step == args.steps):
            logging.info("--------------------------------------")
            logging.info('evaluating on valid dataset...')
            valid_metrics = link_prediction(args, learner, valid_dataloaders)
            log_metrics('valid', step, valid_metrics)
            if len(valid_best_metrics) == 0 or \
                valid_best_metrics["MRR"] + valid_best_metrics["HITS@3"] < valid_metrics["MRR"] + valid_metrics["HITS@3"]:
                valid_best_metrics = valid_metrics.copy()
                valid_best_metrics["step"] = step
                save_model({
                    "model": learner,
                    "transnet": transnet,
                    "generator": generator,
                    "discriminator": discriminator,
                    "optimizer_l": optimizer_l,
                    "optimizer_g": optimizer_g,
                    "optimizer_d": optimizer_d,
                    "scheduler_l": scheduler_l,
                    "scheduler_g": scheduler_g,
                    "scheduler_d": scheduler_d,
                    "step": step,
                    "steps": args.steps},
                    os.path.join(args.save_path, "checkpoint_valid.pt"))

            for k, v in valid_metrics.items():
                writer.add_scalar('valid-metric/%s' % (k.replace("@", "_")), v, step)

            logging.info("--------------------------------------")
            logging.info('evaluating on test dataset...')
            test_metrics = link_prediction(args, learner, test_dataloaders)
            log_metrics('test', step, test_metrics)
            if len(test_best_metrics) == 0 or \
                test_best_metrics["MRR"] + test_best_metrics["HITS@3"] < test_metrics["MRR"] + test_metrics["HITS@3"]:
                test_best_metrics = test_metrics.copy()
                test_best_metrics["step"] = step
                save_model({
                    "model": learner,
                    "transnet": transnet,
                    "generator": generator,
                    "discriminator": discriminator,
                    "optimizer_l": optimizer_l,
                    "optimizer_g": optimizer_g,
                    "optimizer_d": optimizer_d,
                    "scheduler_l": scheduler_l,
                    "scheduler_g": scheduler_g,
                    "scheduler_d": scheduler_d,
                    "step": step,
                    "steps": args.steps},
                    os.path.join(args.save_path, "checkpoint_test.pt"))

            for k, v in test_metrics.items():
                writer.add_scalar('test-metric/%s' % (k.replace("@", "_")), v, step)

            learner.train()

        # save model
        if len(training_logs) > 0 and (step % args.save_steps == 0 or step == args.steps):
            save_model({
                "model": learner,
                "transnet": transnet,
                "generator": generator,
                "discriminator": discriminator,
                "optimizer_l": optimizer_l,
                "optimizer_g": optimizer_g,
                "optimizer_d": optimizer_d,
                "scheduler_l": scheduler_l,
                "scheduler_g": scheduler_g,
                "scheduler_d": scheduler_d,
                "step": step,
                "steps": args.steps},
                os.path.join(args.save_path, "checkpoint_%d.pt" % (step)))

        if len(training_logs) > 0 and (step % args.print_steps == 0 or step == args.steps):
            training_logs.clear()
        
    logging.info("--------------------------------------")
    log_metrics('valid-best', valid_best_metrics["step"], valid_best_metrics)
    log_metrics('test-best', test_best_metrics["step"], test_best_metrics)

    # TODO VISUALIZE EMBEDDING
    # all_embeddings = torch.cat([teacher.relation_embedding.weight.data, teacher.entity_embedding.weight.data, learner.relation_embedding.data, learner.entity_embedding.data], dim=0)
    # all_meta_label = ['teacher_rel' for x in range(teacher.get_relation_count()+1)] + \
    #                 ['teacher_ent' for x in range(teacher.get_entity_count()+1)] + \
    #                 ['student_rel' for x in range(student.get_relation_count()+1)] + \
    #                 ['student_ent' for x in range(student.get_entity_count()+1)]

    # TODO VISUALIZE ALIGNED EMBEDDING
    # align_embeddings = []
    # align_meta_label = []
    # print(np.array(shared_entity))
    # for x, y in np.array(shared_entity):
    #     align_embeddings.append(teacher.entity_embedding(torch.LongTensor([x])))
    #     align_meta_label.append('T')
    #     align_embeddings.append(torch.index_select(learner.entity_embedding, dim=0, index=torch.LongTensor([y])))
    #     align_meta_label.append('S')
    # align_embeddings = torch.cat(align_embeddings, dim=0)
    #
    # writer.add_embedding(align_embeddings, metadata=align_meta_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2020,
                        help="random seed")
    parser.add_argument("--teacher_data_path", type=str,
                        help="teacher data file path")
    parser.add_argument("--teacher_model_path", type=str,
                        help="teacher model file path")
    parser.add_argument("--student_data_path", type=str,
                        help="student data file path")
    parser.add_argument("--shared_entity_path", type=str,
                        help="shared entitiy file path")
    parser.add_argument("--save_path", type=str, default="../dumps",
                        help="checkpoint output path")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of threads")
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help="gpu id to use")

    parser.add_argument("--steps", type=int, default=20000,
                        help="total steps (e.g., 200 epochs)")
    parser.add_argument("--gan_steps", type=int, default=5,
                        help="GAN internal steps")
    parser.add_argument("--print_steps", type=int, default=100,
                        help="print logs frequency")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="save models frequency")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="evaluate models frequency")
    parser.add_argument("--gan_batch", type=int, default=128,
                        help="GAN batch size")
    parser.add_argument("--kge_batch", type=int, default=128,
                        help="KGE batch size")
    parser.add_argument("--test_batch", type=int, default=1024,
                        help="test batch size")

    parser.add_argument("--gan_lr", type=float, default=2e-4,
                        help="GAN learning rate")
    parser.add_argument("--kge_lr", type=float, default=1e-3,
                        help="KGE learning rate")
    parser.add_argument("--reg", type=float, default=0.0,
                        help="coefficient for regularization")
    parser.add_argument("--kge_alpha", type=float, default=1.0,
                        help="coefficient for KGE distance")
    parser.add_argument("--kge_beta", type=float, default=1.0,
                        help="coefficient for KGE transfered triple loss")
    parser.add_argument("--gan_n_critic", type=int, default=1,
                        help="number of iterations to update discriminator in GAN")

    parser.add_argument("--num_neg_samples", type=int, default=256,
                        help="number of negative samples (e.g., 1/50 number of entities)")
    parser.add_argument("--adv_sampling", type=str2bool, default=False,
                        help="whether to enable adversarial sampling")
    parser.add_argument("--adv_sampling_temp", type=float, default=1.0,
                        help="adversarial sampling temperature")
    parser.add_argument("--sampling_weighting", type=str2bool, default=False,
                        help="whether to enable subsampling weighting")

    parser.add_argument("--emb_dim", type=int, default=200,
                        help="embedding dimensions")
    parser.add_argument("--margin", type=float, default=4.0,
                        help="margin for the loss function")
    parser.add_argument("--kge_model", type=str, default="TransE",
                        choices=['TransE', 'DistMult', 'ComplEx',
                                 'RotatE', 'pRotatE', 'ConvE'],
                        help="KGE model")
    args = parser.parse_args()

    args.teacher_data_path = args.teacher_data_path.rstrip("/")
    args.teacher_data_path = args.teacher_data_path.rstrip("\\")
    args.student_data_path = args.student_data_path.rstrip("/")
    args.student_data_path = args.student_data_path.rstrip("\\")
    args.save_path = args.save_path.rstrip("/")
    args.save_path = args.save_path.rstrip("\\")

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    args.save_path = os.path.join(args.save_path, "GAN_%s_T%s_S%s_GLR%.4f_GS%d_KLR%.4f_A%.4f_B%.4f_ED%d_MG%d_NS%d_SD%d_%s" % (
        args.kge_model,
        os.path.split(args.teacher_data_path)[1],
        os.path.split(args.student_data_path)[1],
        args.gan_lr, args.gan_steps, 
        args.kge_lr, args.kge_alpha, args.kge_beta,
        args.emb_dim, args.margin,
        args.num_neg_samples, args.seed, ts))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_config(args, os.path.join(args.save_path, "config.json"))

    set_seed(args.seed)
    set_logger(os.path.join(args.save_path, "log.txt"))

    train(args)
