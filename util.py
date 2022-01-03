import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from data.util import *
import copy
from sklearn.metrics import accuracy_score, f1_score


def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def init_para_frompretrained(m, pm, share_para=False):
    m.wte.weight = pm.wte.weight
    m.wpe.weight = pm.wpe.weight

    for i in range(min(len(m.h), len(pm.h))):
        m.h[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
        m.h[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
        m.h[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
        m.h[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
        m.h[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
        m.h[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
        m.h[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
        m.h[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
        m.h[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
        m.h[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
        m.h[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
        m.h[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)

    m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)


def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """

    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s

    return f


def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f

def evaluate_classification(pos_losses, neg_losses, labels):
    predictions = []
    for p, n in zip(pos_losses, neg_losses):
        if p > n:
            predictions.append('negative')
        else:
            predictions.append('positive')
    
    print('accuracy', accuracy_score(predictions, labels))
    #print('f1 score', f1_score(predictions, labels))

def evaluate_classification_civilcomments(pos_losses, neg_losses, labels, domains):
    predictions = []
    for p, n in zip(pos_losses, neg_losses):
        if p > n:
            predictions.append('approved')
        else:
            predictions.append('rejected')

    print('accuracy sanity', accuracy_score(predictions, labels))

    domain_rights = dict()
    domain_preds = dict()
    preds = 0
    rights = 0
    ratings = ['rejected', 'approved']
    demographics = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_reli', 'white', 'black']

    for r in ratings:
        domain_preds[r] = dict()
        domain_rights[r] = dict()
        for d in demographics:
            domain_preds[r][d] = 0
            domain_rights[r][d] = 0

    for p, l, dom in zip(predictions, labels, domains):
        preds += 1
        for i, d in enumerate(dom):
            if d == 1:
                domain_preds[l][demographics[i]] += 1

        if p == l:
            rights += 1
            for i, d in enumerate(dom):
                if d == 1:
                    domain_rights[l][demographics[i]] += 1

    print('accuracy', rights/preds)
    for r in ratings:
        for d in demographics:
            print(f'accuracy {r} {d} {domain_rights[r][d]/domain_preds[r][d]}')
