import re
from torch.utils.data import Dataset, DataLoader                     ### NEW: use PyTorch Dataset, not torchtext
from torchtext.vocab import GloVe                                   ### CHANGED: optional for pretrained embeddings
from torch.nn.utils.rnn import pad_sequence                         ### NEW: padding sequences
from torchtext.data.utils import get_tokenizer                      ### NEW: modern tokenizer (optional)
from collections import Counter                                     ### NEW: to build vocab manually
import torch

### CHANGED: no torchtext.data
# from torchtext import data  ❌ removed

# Preprocessing
my_punc = "!\"#$%&\()*+?_/:;[]{}|~,`"
table = dict((ord(char), u' ') for char in my_punc)

def clean_str(string):                                              ### SAME
    string = re.sub(r"\'s ", " ", string)
    string = re.sub(r"\'m ", " ", string)
    string = re.sub(r"\'ve ", " ", string)
    string = re.sub(r"n\'t ", " not ", string)
    string = re.sub(r"\'re ", " ", string)
    string = re.sub(r"\'d ", " ", string)
    string = re.sub(r"\'ll ", " ", string)
    string = re.sub("-", " ", string)
    string = re.sub(r"@", " ", string)
    string = re.sub('\'', '', string)
    string = string.translate(table)
    string = string.replace("..", "").strip()
    return string

def tokenizer_function(text):                                       ### UPDATED (mimics the old logic)
    return [x for x in text.split(" ") if x]

### REPLACED: torchtext.data.Dataset with torch.utils.data.Dataset
class MyDataset(Dataset):                                           ### CHANGED
    def __init__(self, path, opt):
        tokenizer = tokenizer_function                              ### CHANGED: use custom tokenizer

        with open(path, 'r') as f:                                  ### CHANGED: manual file reading
            lines = [clean_str(line.strip()) for line in f if line.strip()]
        tokenized = [tokenizer(line)[:100] for line in lines]       ### NEW: manual tokenization

        # Build vocab manually                                      ### NEW
        counter = Counter()
        for tokens in tokenized:
            counter.update(tokens)
        self.specials = [opt.pad_token, opt.unk_token, opt.start_token, opt.end_token]
        self.vocab = {tok: i for i, tok in enumerate(self.specials)}  ### NEW
        for word in counter:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        self.reverse_vocab = {i: w for w, i in self.vocab.items()}  ### NEW

        self.data = []
        for tokens in tokenized:
            ids = [self.vocab[opt.start_token]] + \
                  [self.vocab.get(t, self.vocab[opt.unk_token]) for t in tokens] + \
                  [self.vocab[opt.end_token]]
            self.data.append(torch.tensor(ids, dtype=torch.long))   ### CHANGED: return tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_vocab(self):                                            ### NEW: for compatibility
        return self.vocab

### NEW: for padding batches
def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

### REPLACED: get_iterators using torchtext iterators
def get_iterators(opt, path, fname):                                ### CHANGED
    dataset = MyDataset(path + fname, opt)
    vocab = dataset.get_vocab()
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader, vocab