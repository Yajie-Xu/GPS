""" VAE for Text Generation
This is for Module 1: Candidates Generation.
Usage: python VAE_Text_Generation.py --dataset reddit
Update: 5/9/2025 New version
"""
import argparse
import math
import os
import numpy as np
import torch as T
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utility.VAE_Text_Generation.dataset import MyDataset, collate_batch
from utility.VAE_Text_Generation.helper_functions import get_cuda
from utility.VAE_Text_Generation.model import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_vocab', type=int, default=12000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--n_hidden_G', type=int, default=512)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--n_hidden_E', type=int, default=512)
parser.add_argument('--n_layers_E', type=int, default=1)
parser.add_argument('--n_z', type=int, default=100)
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--rec_coef', type=float, default=7)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_highway_layers', type=int, default=2)
parser.add_argument('--n_embed', type=int, default=300)
parser.add_argument('--out_num', type=int, default=30000)
parser.add_argument('--unk_token', type=str, default="<unk>")
parser.add_argument('--pad_token', type=str, default="<pad>")
parser.add_argument('--start_token', type=str, default="<sos>")
parser.add_argument('--end_token', type=str, default="<eos>")
parser.add_argument('--dataset', type=str, default="reddit")
parser.add_argument('--training', action='store_true')
parser.add_argument('--resume_training', action='store_true')

opt = parser.parse_args()
print(opt)
save_path = "tmp/saved_VAE_models/" + opt.dataset + ".tar"
print(save_path)
if not os.path.exists("tmp/saved_VAE_models"):
    os.makedirs("tmp/saved_VAE_models")
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

candidates_path = opt.dataset + '_for_VAE.txt'
train_dataset = MyDataset('./data/' + candidates_path)
train_iter = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_batch)
val_iter = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_batch)
vocab = train_dataset.get_vocab()
opt.n_vocab = len(vocab)

if opt.training:
    vae = VAE(opt)
    vae = get_cuda(vae)
    trainer_vae = T.optim.Adam(vae.parameters(), lr=opt.lr)
else:
    checkpoint = T.load(save_path)
    vae = checkpoint['vae_dict']
    trainer_vae = checkpoint['vae_trainer']
    if 'opt' in checkpoint:
        opt_old = checkpoint['opt']
        print(opt_old)

def create_generator_input(x, train):
    G_inp = x[:, 0:x.size(1)-1].clone()
    if train == False:
        return G_inp
    r = np.random.rand(G_inp.size(0), G_inp.size(1))
    for i in range(len(G_inp)):
        for j in range(1, G_inp.size(1)):
            if r[i, j] < opt.word_dropout and G_inp[i, j].item() not in [vocab[opt.pad_token], vocab[opt.end_token]]:
                G_inp[i, j] = vocab[opt.unk_token]
    return G_inp

def train_batch(x, G_inp, step, train=True):
    logit, _, kld = vae(x, G_inp, None, None)
    logit = logit.view(-1, opt.n_vocab)
    x = x[:, 1:x.size(1)]
    x = x.contiguous().view(-1)
    rec_loss = F.cross_entropy(logit, x)
    kld_coef = (math.tanh((step - 15000)/1000) + 1) / 2
    loss = opt.rec_coef * rec_loss + kld_coef * kld
    if train:
        trainer_vae.zero_grad()
        loss.backward()
        trainer_vae.step()
    return rec_loss.item(), kld.item()

def training():
    start_epoch = step = 0
    if opt.resume_training:
        step, start_epoch = checkpoint['step'], checkpoint['epoch']
    for epoch in range(start_epoch, opt.epochs):
        vae.train()
        train_rec_loss = []
        train_kl_loss = []
        for batch in train_iter:
            x = get_cuda(batch)
            G_inp = create_generator_input(x, train=True)
            rec_loss, kl_loss = train_batch(x, G_inp, step, train=True)
            train_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            step += 1

        vae.eval()
        valid_rec_loss = []
        valid_kl_loss = []
        for batch in val_iter:
            x = get_cuda(batch)
            G_inp = create_generator_input(x, train=False)
            with T.autograd.no_grad():
                rec_loss, kl_loss = train_batch(x, G_inp, step, train=False)
            valid_rec_loss.append(rec_loss)
            valid_kl_loss.append(kl_loss)

        train_rec_loss = np.mean(train_rec_loss)
        train_kl_loss = np.mean(train_kl_loss)
        valid_rec_loss = np.mean(valid_rec_loss)
        valid_kl_loss = np.mean(valid_kl_loss)

        print("No.", epoch, "T_rec:", '%.2f' % train_rec_loss, "T_kld:", '%.2f' % train_kl_loss,
              "V_rec:", '%.2f' % valid_rec_loss, "V_kld:", '%.2f' % valid_kl_loss)
        if epoch >= 50 and epoch % 10 == 0:
            print('save model ' + str(epoch) + '...')
            T.save({'epoch': epoch + 1, 'vae_dict': vae, 'vae_trainer': trainer_vae, 'step': step, 'opt': opt}, save_path)
            generate_sentences(5)

def generate_sentences(n_examples, save=0):
    vae.eval()
    out = []
    for _ in tqdm(range(n_examples)):
        z = get_cuda(T.randn([1, vae.n_z]))
        h_0 = get_cuda(T.zeros(vae.generator.n_layers_G, 1, vae.generator.n_hidden_G))
        c_0 = get_cuda(T.zeros(vae.generator.n_layers_G, 1, vae.generator.n_hidden_G))
        G_hidden = (h_0, c_0)
        G_inp = T.LongTensor(1, 1).fill_(vocab[opt.start_token])
        G_inp = get_cuda(G_inp)
        out_str = ""
        while (G_inp[0][0].item() != vocab[opt.end_token]) and (G_inp[0][0].item() != vocab[opt.pad_token]):
            with T.autograd.no_grad():
                logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
            probs = F.softmax(logit[0], dim=1)
            G_inp = T.multinomial(probs, 1)
            out_str += (list(vocab.keys())[list(vocab.values()).index(G_inp[0][0].item())] + " ")
        print(out_str[:-6])
        out.append(out_str[:-6])
    if save:
        original = []
        with open('./data/' + candidates_path, 'r') as fin:
            for line in fin:
                original.append(line.strip())
        fname = './data/' + opt.dataset + '_candidates.txt'
        with open(fname, 'w') as fout:
            for i in out + original:
                fout.write(i + '\n')

if __name__ == '__main__':
    if opt.training or opt.resume_training:
        training()
        generate_sentences(opt.out_num, save=1)
    else:
        generate_sentences(opt.out_num, save=1)
