import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import importlib
import logging
import copy

from apex.optimizers import FusedAdam
from apex import amp
from apex.fp16_utils import FP16_Optimizer

from data.util import *
from util import *

from model import *

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

devices = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = devices


def compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens)
    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta * kl_loss

    return loss, ce_loss, kl_loss


def compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, y_mask=x_mask, y_tokens=x_tokens, from_mean=False, from_prior=False)

    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta* kl_loss

    return loss, ce_loss, kl_loss


def train_step(device, model, optimizer, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta, model_type):
    output = []
    if model_type == 'ae_vae_fusion':
        optimizer.zero_grad()
        loss, ce_loss, kl_loss = compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                              target_tokens, mask, loss_fn, beta)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
        optimizer.step()
        output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))
    """
    optimizer.zero_grad()
    loss, ce_loss, kl_loss = compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                          target_tokens, mask, loss_fn, beta)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    optimizer.step()
    output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))
    """
    return output


def compute_masked_loss(device, model, optimizer, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta, model_type):
    model.eval()
    output = []
    if model_type == 'ae_vae_fusion':

        loss, ce_loss, kl_loss = compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                              target_tokens, mask, loss_fn, beta)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0

    return loss



def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass

    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0



def compute_sentence_losses(model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                            temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, model_type='cvae', lr=0.07, latent_epochs=25):
            
    
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device) # y tokens shape (1, length), e.g (1, 250)
    print(y_tokens.shape)

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    target_tokens = torch.squeeze(y_tokens)

    for param in model.parameters():
        param.requires_grad = False
    
    z = torch.randn((1, 768)).to(device)

    with torch.enable_grad():

        z = z.data.clone().detach().requires_grad_(True) # z dimension (1, 768)
        z_opt = AdamW([z], lr=lr)
        #z, z_opt = amp.initialize([z], z_opt, opt_level='O0')
        z_opt.zero_grad()

        for j in range(latent_epochs):
            sentence_loss = 0
            
            prev = x_tokens[:, -1].view(batch_size, -1)        

            output = prev
            probability = torch.tensor([], dtype=z.dtype, device=device)

            for i, original_token in enumerate(target_tokens.view(-1,1)): #trange

                if i == 0:
                    logits, mem = model.transformer(input_ids=prev, past=None, representations=z)
                else:
                    logits, mem = model.transformer(input_ids=prev, past=mem, representations=z)

                logits = model.lm_head(logits) # logits shape (1,1, 50257)

                if model.add_softmax:
                    logits_rep = model.lm_head_rep(z)
                    logits = logits + logits_rep.unsqueeze(dim=1)

                logits = logits[:, -1, :] / temperature
                logits = top_k_top_p_filtering(logits, top_k, top_p)
                probs = F.softmax(logits, dim=-1) #(1, 50257)

                sentence_loss += loss_fn(probs.cpu(), original_token.cpu())

            
            sentence_loss = sentence_loss / len(target_tokens)
            kl_loss = model.kl_loss(z.cpu(), torch.ones(768), torch.zeros(768), torch.ones(768))
            sentence_loss += kl_loss
            #with amp.scale_loss(sentence_loss, z_opt) as scale_loss:
               # scale_loss.backward()
               # print(f'sentence_loss epoch {j}', sentence_loss.item())
            sentence_loss.backward()
            z_opt.step()
            z_opt.zero_grad()

    #for param in model.parameters():
    #   param.requires_grad = True
            
    return sentence_loss.data#.clone().detach()



def sample_sequence(model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, model_type='cvae'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    with torch.no_grad():
        if model_type == 'cvae':
            try:
                prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            except:
                prior_mean = prior_logvar = torch.zeros([batch_size, model.config.n_embd], device=device)
            latent_mean, latent_logvar = prior_mean, prior_logvar
            z = model.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'
        else:
            posterior_mean, posterior_logvar = model.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]
            latent_mean, latent_logvar = posterior_mean, posterior_logvar
            z = latent_mean
            assert not torch.isnan(z).any(), 'training get nan z'

        _, mem = model.transformer(input_ids=x_tokens[:, :-1], past=None, attention_mask=x_mask[:, :-1], representations=z)
        prev = x_tokens[:, -1].view(batch_size, -1)

        output = prev
        probability = torch.tensor([], dtype=z.dtype, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

        for i in range(length): #trange
            logits, mem = model.transformer(input_ids=prev, past=mem, representations=z)

            logits = model.lm_head(logits)
            if model.add_softmax:
                logits_rep = model.lm_head_rep(z)
                logits = logits + logits_rep.unsqueeze(dim=1)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)

    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])
    parser.add_argument('--iterations', type=int, default=101640 * 4)  # wp 850001  wi 300001 ax 300001 yp 800001
    parser.add_argument('--dataset', type=str, default='wi', choices=['ax', 'yp', 'wp', 'wi'], help="Dataset to use for training")
    parser.add_argument('--warmup', type=int, default=10000,
                        help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1],
                        help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[1024],
                        help='seq length per sample. Lists the schedule.')
    parser.add_argument('--switch-time', type=float, default=0,
                        help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
    parser.add_argument('--fp16_opt_level', default='O0', type=str, required=False)

    # KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
    parser.add_argument('--beta_0', default=1.00, type=float)
    parser.add_argument('--beta_warmup', type=int, default=50000)
    # cyc_vae parameters
    parser.add_argument('--cycle', type=int, default=101640)

    parser.add_argument('--add_input', action="store_true")
    parser.add_argument('--add_attn', action="store_true")
    parser.add_argument('--add_softmax', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    parser.add_argument('--learn_prior', action="store_true")

    args = parser.parse_args() # wi.12.proj_vary_beta_cvae

    if args.model_type == 'cvae':
        args.learn_prior = True
    else:
        args.learn_prior = False

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # logging
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    importlib.reload(logging)
    logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
                        level=logging.INFO, format='%(asctime)s--- %(message)s')
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808
    config = GPT2Config()

    # add special tokens
    # special_tokens_dict = {
    #     'pad_token': '<|startoftext|>',
    #     'cls_token': '<|startofcond|>',
    #     'sep_token': '<|sepofcond|>',
    #     'mask_token': '<|endofcond|>'
    # }
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'special tokens')
    # # Notice: resize_token_embeddings expect to receive the full size of the new vocab
    # gpt2_model.resize_token_embeddings(len(tokenizer))
    # assert tokenizer.pad_token == '<|startoftext|>'

    VAE1 = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)
                 
    VAE2 = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)
    
    init_para_frompretrained(VAE1.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(VAE1.encoder, gpt2_model.transformer, share_para=False)
    
    init_para_frompretrained(VAE2.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(VAE2.encoder, gpt2_model.transformer, share_para=False)
    
    if args.learn_prior:
        init_para_frompretrained(VAE1.encoder_prior, VAE1.encoder, share_para=True)
        VAE1.encoder_prior.averageSelfAttention.attention_weights = VAE1.encoder.averageSelfAttention.attention_weights
        
        init_para_frompretrained(VAE2.encoder_prior, VAE2.encoder, share_para=True)
        VAE2.encoder_prior.averageSelfAttention.attention_weights = VAE2.encoder.averageSelfAttention.attention_weights
        

    VAE1.lm_head.weight = gpt2_model.lm_head.weight
    
    VAE2.lm_head.weight = gpt2_model.lm_head.weight
    

    if VAE1.add_softmax:
        VAE1.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
        # VAE.lm_head_rep = LM_head_rep(*gpt2_model.lm_head.weight.size()[::-1])
    print('VAE_params:', num_params(VAE1))  # 286694400

    
    if VAE2.add_softmax:
        VAE2.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
        # VAE.lm_head_rep = LM_head_rep(*gpt2_model.lm_head.weight.size()[::-1])
    print('VAE_params:', num_params(VAE2))  # 286694400
    

    """
    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(args.load, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        VAE1.load_state_dict(state)
        gc.collect()
    print('Done.')
    """

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = 40000
    tuning_all = False
    

    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    print('Batch schedule', batch_schedule)
    train_loader = prepare_dataset(
        args.data_dir, 'amazon_both', tokenizer,
        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        make_test=True,ratio=0.05,
        num_workers=args.workers, data_type=args.data_type, domains=['dvd']
    )[0]
    test_loader = train_loader
    print('Done.')

    ###
    val_loader = test_loader
    ###

    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    VAE1 = VAE1.to(device)
    VAE1.train()

    VAE2 = VAE2.to(device)
    VAE2.train()

    optimizer1 = AdamW(VAE1.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_schedule)
    VAE1, optimizer1 = amp.initialize(VAE1, optimizer1, opt_level=args.fp16_opt_level)

    optimizer2 = AdamW(VAE2.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_schedule)
    VAE2, optimizer2 = amp.initialize(VAE2, optimizer2, opt_level=args.fp16_opt_level)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')

    print('Begin training iterations')
    logging.info("Begin training iterations")
    max_val_batches = 20000  # max num. of val batches
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    beta = args.beta_0
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")



    def compute_latent_losses_for_classification(test_loader, VAE):
        VAE.eval()
        losses = []
        labels = []
        args.nsamples = 1
        args.batch_size = 1
        args.temperature = 0.95
        args.top_k = 100
        args.top_p = 0.95

        # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
        with tqdm(total=len(test_loader)) as pbar:
            for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, label) in enumerate(
                    test_loader):

                length = -1
                if length == -1:
                    length = VAE.config.n_ctx - x_tokens.size(1) - 1
                elif length > VAE.config.n_ctx - x_tokens.size(1) - 1:
                    raise ValueError("Can't get samples longer than window size: %s" % VAE.config.n_ctx)

                eff_samples = []
                n, l = target_tokens.size()
                storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                storys = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in storys]
                storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                              storys]

                loss = compute_sentence_losses(model=VAE,
                                                tokenizer=tokenizer,
                                                length=length,
                                                batch_size=args.batch_size,
                                                x_mask=x_mask,
                                                x_tokens=x_tokens,
                                                y_mask=y_mask,
                                                y_tokens=y_tokens,
                                                temperature=args.temperature,
                                                top_k=args.top_k,
                                                top_p=args.top_p,
                                                device=device,
                                                eos_token=tokenizer.encoder['<|endoftext|>'],
                                                model_type=args.model_type)

                losses.append(loss)
                labels.append(label)
                torch.cuda.empty_cache()
                pbar.update(1)

        VAE.train()

        return losses, labels



    for e in [2,5,10,18,35]:
        print(f'evaluating models for epoch {e}')
        try:
            VAE1.load_state_dict(torch.load(f'./out/testvanilla/model_positive_books_epoch{e}.pt'))
            VAE2.load_state_dict(torch.load(f'./out/testvanilla/model_negative_books_epoch{e+27}.pt'))
        except FileNotFoundError:
            print('file not existing')
            continue

        losses_pos = []
        losses_neg = []
        labels = []

        losses_pos, labels = compute_latent_losses_for_classification(train_loader, VAE1)
        losses_neg, labels = compute_latent_losses_for_classification(train_loader, VAE2)
        """
        for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, label) in enumerate(train_loader):
            VAE1.eval()
            VAE2.eval()
            if i %100 == 0:
                print(i)
            loss_pos = compute_masked_loss(device, VAE1, optimizer1, x_mask, x_tokens, y_mask, y_tokens,
                                       input_tokens, target_tokens, mask, loss_fn, beta, args.model_type)
            
            loss_neg = compute_masked_loss(device, VAE2, optimizer2, x_mask, x_tokens, y_mask, y_tokens,
                                       input_tokens, target_tokens, mask, loss_fn, beta, args.model_type)
            
            losses_pos.append(loss_pos)
            losses_neg.append(loss_neg)
            labels.append(label)
        """
        evaluate_classification(losses_pos, losses_neg, labels)


if __name__ == "__main__":
    main()
