import spacy
import random
import torch
import torchtext
import math
from bleu import bleu_score
#from data import corpus
#import data
from HyperParams import *
def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def load_spacy(language):
    spacy_ = spacy.load(language)
    #spacy_en = spacy.load('en')

    return spacy_

class tokenizer():
    def __init__(self,language):
        if language == 'de':
            self.tn = load_spacy('de')
        elif language == 'en':
            self.tn = load_spacy('en')
        else:
            self.tn = None
    def get_tokenizer(self):
        return self.tn

    def get_list(self,text):
        if self.tn is not None:
            return [tok.text for tok in self.tn.tokenizer(text)]
    pass


def create_filed(tn,init_token = '<sos>',eos_token = '<eos>',lower = True,batch_first = True):
    return torchtext.data.Field(tokenize=tn,init_token=init_token,eos_token=eos_token,
                                lower=lower,batch_first=batch_first)
    pass

def model_init(model):
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

def get_pos_encoding(bsz,seq_len,d_model):
    sen = torch.zeros(seq_len,d_model)
    for pos in range(0,seq_len):
        for i in range(0,d_model):
            sen[pos][i] = (math.sin(pos/(10000**(i/d_model))) if i % 2 == 0 else math.cos(pos/(10000**(2*(i//2)/d_model))))

    sen = sen.expand(bsz,seq_len,d_model)
    return sen

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
    pass


def train(model, iterator, optimizer, criterion, clip,corpus):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.optimizer.zero_grad()

        output = model(src, trg[:, :-1])

        # output = [batch size, trg sent len - 1, output dim]
        # trg = [batch size, trg sent len]



        sen_out = torch.max(output.contiguous(),dim = 2)[0]
        score = 0
        score = bleu_score(sen_out,trg[:,1:].contiguous(),corpus)

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg sent len - 1, output dim]
        # trg = [batch size*trg sent len - 1]



        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator),score


def evaluate(model, iterator, criterion,corpus):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])

            # output = [batch size, trg sent len - 1, output dim]
            # trg = [batch size, trg sent len]
            sen_out = torch.max(output.contiguous(), dim=2)[0]
            score = bleu_score(sen_out,trg[:,1:].contiguous(),corpus)

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg sent len - 1, output dim]
            # trg = [batch size * trg sent len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator),score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs







