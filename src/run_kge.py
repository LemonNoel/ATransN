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
import json
import datetime
from torch.utils.data import DataLoader
from dataset import KGDataset, TrainDataset, TestDataset, BidirectionalOneShotIterator, load_shared_entity, batch_iter
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

    data = KGDataset(data_path=args.data_path, dict_path=args.data_path)

    logging.info("train: valid: test = %d: %d: %d" % (
        len(data.train_set), len(data.valid_set), len(data.test_set)))
    num_entities = data.get_entity_count()
    num_relations = data.get_relation_count()

    logging.info("number of entities: %d" % (num_entities))
    logging.info("number of relations: %d" % (num_relations))

    # training data
    train_loader_head = DataLoader(
        TrainDataset(data.train_set,
                     data.get_entity_count(),
                     data.get_relation_count(),
                     args.num_neg_samples,
                     'head-batch'),
        batch_size=args.kge_batch,
        shuffle=True,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TrainDataset.collate_fn)
    train_loader_tail = DataLoader(
        TrainDataset(data.train_set,
                     data.get_entity_count(),
                     data.get_relation_count(),
                     args.num_neg_samples,
                     'tail-batch'),
        batch_size=args.kge_batch,
        shuffle=True,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TrainDataset.collate_fn)
    train_iterator = BidirectionalOneShotIterator(
        train_loader_head, train_loader_tail)

    # validation data and test data
    all_data = data.train_set + data.valid_set + data.test_set
    
    valid_loader_head = DataLoader(
        TestDataset(data.valid_set,
                    all_data,
                    num_entities,
                    num_relations,
                    'head-batch'),
        batch_size=args.test_batch,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TestDataset.collate_fn)
    valid_loader_tail = DataLoader(
        TestDataset(data.valid_set,
                    all_data,
                    num_entities,
                    num_relations,
                    'tail-batch'),
        batch_size=args.test_batch,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TestDataset.collate_fn)
    valid_dataloaders = [valid_loader_head, valid_loader_tail]

    test_loader_head = DataLoader(
        TestDataset(data.test_set,
                    all_data,
                    num_entities,
                    num_relations,
                    'head-batch'),
        batch_size=args.test_batch,
        num_workers=max(0, args.num_workers//2),
        collate_fn=TestDataset.collate_fn)
    test_loader_tail = DataLoader(
        TestDataset(data.test_set,
                    all_data,
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

    learner = KGEModel(model_name=args.kge_model,
                       num_entities=num_entities,
                       num_relations=num_relations,
                       hidden_dim=args.emb_dim,
                       gamma=args.margin,
                       double_entity_embedding=args.kge_model in ["RotatE", "ComplEx"],
                       double_relation_embedding=args.kge_model in ["ComplEx"]).to(device)

    # TODO COPY TOP EMBEDDING
    # top_student_entity = pickle.load(open('../intermediate/top_one_entity_transe_200.pkl', 'rb'))
    # top_shared_entity = shared_entity[shared_entity['student'].isin(top_student_entity)]
    # shared_entity = shared_entity[(True ^ shared_entity['student'].isin(top_student_entity))]
    # for t, s in np.array(top_shared_entity):
    # for t, s in np.array(shared_entity):
    #     learner.entity_embedding.data[s].copy_(teacher.entity_embedding.weight[t])

    # TODO DELETE ONCE ENTITY
    # once_entity = pickle.load(open('top_one_entity_transe_200.pkl', 'rb'))
    # shared_entity = shared_entity[(True ^ shared_entity['student'].isin(once_entity))]

    optimizer = optim.Adam(learner.parameters(), lr=args.kge_lr)
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer, args.steps//100, args.steps, num_cycles=4, min_percent=args.kge_lr*0.001)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.steps//100)
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step(0)

    bce_loss = nn.BCELoss(reduction='none')

    logging.info("begin training..")
    learner.train()

    training_logs = []
    valid_best_metrics = {}
    test_best_metrics = {}

    for step in range(1, args.steps+1):
        # ----------------
        # Configure Input
        # ----------------

        positive_sample, negative_sample, subsampling_weight, mode = next(
            train_iterator)
        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        subsampling_weight = subsampling_weight.to(device)

        log = {}

        # --------------------
        # Train Learner
        # --------------------

        optimizer.zero_grad()

        positive_score = learner(positive_sample)
        negative_score = learner((positive_sample, negative_sample), mode=mode)

        if learner.model_name == 'ConvE':
            positive_sample_loss = bce_loss(F.logsigmoid(positive_score), torch.ones_like(positive_score)).mean()
            negative_sample_loss = bce_loss(F.logsigmoid(-negative_score), torch.zeros_like(negative_score)).mean()
            loss = positive_sample_loss + negative_sample_loss
        elif learner.model_name == 'ComplEx' or learner.model_name == 'RotatE':
            positive_sample_loss = - F.logsigmoid(positive_score).mean()
            negative_sample_loss = - F.logsigmoid(-negative_score).mean()
            loss = positive_sample_loss + negative_sample_loss
        elif learner.model_name == 'TransE' or learner.model_name == 'DistMult':
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = negative_score.mean()
            loss = F.relu(-positive_score+negative_score+args.margin).mean()

        if args.reg != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            reg = args.reg * (
                learner.entity_embedding.norm(p=3) ** 3 +
                learner.relation_embedding.norm(p=3).norm(p=3) ** 3)
            loss = loss + reg
            reg_log = {'reg': reg.item()}
            log.update(reg_log)
        else:
            reg_log = {}

        loss.backward()
        optimizer.step()
        scheduler.step(step)

        log["pos_loss"] = positive_sample_loss.item()
        log["neg_loss"] = negative_sample_loss.item()
        log["loss"] = loss.item()

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
                writer.add_scalar('train-metric/%s' % (k), v, step)

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
                    "learner": learner,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
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
                    "optimizer": optimizer,
                    "scheduler": scheduler,
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
                "optimizer": optimizer,
                "scheduler": scheduler,
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
    parser.add_argument("--data_path", type=str,
                        help="data file path")
    parser.add_argument("--save_path", type=str, default="../dumps",
                        help="checkpoint output path")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of threads")
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help="gpu id to use")

    parser.add_argument("--steps", type=int, default=20000,
                        help="total steps (e.g., 200 epochs)")
    parser.add_argument("--print_steps", type=int, default=100,
                        help="print logs frequency")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="save models frequency")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="evaluate models frequency")
    parser.add_argument("--kge_batch", type=int, default=128,
                        help="KGE batch size")
    parser.add_argument("--test_batch", type=int, default=1024,
                        help="test batch size")

    parser.add_argument("--kge_lr", type=float, default=1e-3,
                        help="KGE learning rate")
    parser.add_argument("--reg", type=float, default=0.0,
                        help="coefficient for regularization")

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

    args.data_path.rstrip("/")
    args.data_path.rstrip("\\")
    args.save_path.rstrip("/")
    args.save_path.rstrip("\\")

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    args.save_path = os.path.join(args.save_path, "%s_%s_KLR%.4f_ED%d_MG%d_NS%d_SD%d_%s" % (
        args.kge_model,
        os.path.split(args.data_path)[1],
        args.kge_lr, args.emb_dim, args.margin,
        args.num_neg_samples, args.seed, ts))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_config(args, os.path.join(args.save_path, "config.json"))

    set_seed(args.seed)
    set_logger(os.path.join(args.save_path, "log.txt"))

    train(args)
