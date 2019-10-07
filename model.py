import torch
import torch.tensor as tensor
import torch.nn as nn
import torch.nn.functional as F
from utils import get_pos_encoding
class Encoder(nn.Module):
    def __init__(self,vocab_dim,hid_dim,n_layers,n_heads,pf_dim,
                 encode_layer,self_attention,positionwise_feedforward,dropout,device,
                 PositionalEncoding=False):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encode_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.vocab_emb = nn.Embedding(vocab_dim,hid_dim)
        self.pos_emb = nn.Embedding(1000,hid_dim)

        self.layers = nn.ModuleList([self.encoder_layer(hid_dim,pf_dim,self_attention,
                                                        positionwise_feedforward,
                                                        n_heads,dropout,device)]
                                    for _ in self.n_layers
                                    )
        self.Dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]).to(device))
        self.PositionalEncoding = PositionalEncoding
        pass
    def forward(self, src,src_mask):

         #src:        bsz  seq_len
         #src_mask：  bsz  seq_len

        if self.PositionalEncoding == False:

            pos = torch.arange(0,src.shape[1]).unsqueeze(0).repeat(src.shape[0],1).to(self.device)
        else :
            pos = get_pos_encoding(src.shape[0],src.shape[1],self.hid_dim).to(self.device)
        src = self.Dropout((self.vocab_emb(src)*self.scale)+((self.pos_emb(pos)) if self.PositionalEncoding == False else pos))

        for layer in self.layers:
            src = layer(src,src_mask)

        return src
        pass




class EncoderLayer(nn.Module):
    def __init__(self,hid_dim,pf_dim,attention,positionwise_feedforward,n_heads,dropout,device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = attention(hid_dim,n_heads,dropout,device)
        self.pf = positionwise_feedforward(hid_dim=hid_dim,pf_dim=pf_dim,dropout=dropout)
        self.Dropout = nn.Dropout(dropout)
        pass

    def forward(self,src,src_mask):

        src = self.ln(src+self.sa(src,src,src,src_mask))
        src = self.ln(src+self.pf(src))

        return src
        pass
    pass

class SelfAttention(nn.Module):
    def __init__(self,hid_dim,n_heads,dropout,device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        #检查多头是否可以整除
        assert hid_dim % n_heads == 0

        self.Qm = nn.Linear(hid_dim,hid_dim)
        self.Km = nn.Linear(hid_dim, hid_dim)
        self.Vm = nn.Linear(hid_dim, hid_dim)

        self.ln = nn.Linear(hid_dim,hid_dim)
        self.Dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]).to(device))
        pass

    def forward(self,query,key,value,mask = None):
        #bsz seq_len hid_dim

        Q = self.Qm(query)
        K = self.Km(key)
        V = self.Vm(value)

        bsz = Q.shape[0]

        #multihead:

        Q = Q.view(bsz, -1 , self.n_heads,self.hid_dim//self.n_heads).permute(0,2,1,3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)

        # bsz n_heads seq_len hid_dim_head

        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale
        # bsz n_heads seq_len seq_len

        if mask is not None:
            energy = energy.masked_fill(mask== 0 ,-1e10)

        attention = self.Dropout(F.softmax(energy,dim=3))

        X = torch.matmul(attention,V)

        #bsz,n_heads,seq_len,hid_dim_head

        X = X.permute(0,2,1,3).contiguous()
        #bsz seq_len n_heads hid_dim_head

        X = X.view(bsz,-1,self.hid_dim)
        # bsz seq_len hid_dim

        X = self.ln(X)

        return X
        pass
    pass

class PositionFeedForward(nn.Module):
    def __init__(self,hid_dim,pf_dim,dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.l1 = nn.Linear(hid_dim,pf_dim)
        self.l2 = nn.Linear(pf_dim,hid_dim)
        self.Dropout = nn.Dropout(dropout)
        pass

    def forward(self,input):
        # bsz,seq_len,hid_dim
        #input = input.permute(0,2,1)

        input = F.relu(self.l1(input))
        input = self.Dropout(input)
        input = self.l2(input)
        return input

    pass



class Decoder(nn.Module):
    def __init__(self,vocab_dim,hid_dim,n_layers,n_heads,pf_dim,
                 decode_layer,self_attention,positionwise_feedforward,dropout,device,
                 PositionalEncoding=False):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decode_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.vocab_emb = nn.Embedding(vocab_dim,hid_dim)
        self.pos_emb = nn.Embedding(1000,hid_dim)

        self.layers = nn.ModuleList([self.decoder_layer(hid_dim,n_heads,pf_dim,self_attention,
                                                        positionwise_feedforward,dropout,device)])
        self.Dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]).to(device))
        self.PositionalEncoding = PositionalEncoding
        self.out = nn.Linear(self.hid_dim,self.vocab_dim)


    def forward(self, trg,src,trg_mask,src_mask):

        #bsz seq_len
        if self.PositionalEncoding == False:
            pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)
        else:
            pos = get_pos_encoding(trg.shape[0],trg.shape[1],self.hid_dim).to(self.device)
        trg = self.Dropout((self.vocab_emb(trg)*self.scale) + ((self.pos_emb(pos)) if self.PositionalEncoding == False else pos))

        for layer in self.layers :
            trg = layer(trg,src,trg_mask,src_mask)

        return self.out(trg)
        pass
pass


class DecoderLayer(nn.Module):
    def __init__(self,hid_dim,n_heads,pf_dim,self_attention,
                 positionwise_feedforward,dropout,device):
        super().__init__()
        self.sa = self_attention(hid_dim,n_heads,dropout,device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.pf = positionwise_feedforward(hid_dim,pf_dim,dropout)
        self.device = device
        self.Dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(hid_dim)

        pass
    def forward(self,trg,src,trg_mask,src_mask):

        trg = self.ln(trg + self.Dropout(self.sa(trg,trg,trg,trg_mask) ))

        trg = self.ln(trg +self.Dropout(self.ea(trg,src,src,src_mask)))

        trg = self.ln(trg + self.Dropout(self.pf(trg)))

        return trg

        pass


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,pad_idx,device,PositionalEncoding=False):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.PositionalEncoding = PositionalEncoding

        pass
    def mask(self,src,trg):

        #bsz seq_len

        src_mask = (src != self.pad_idx ).unsqueeze(1).unsqueeze(2).to(self.device)

        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3).to(self.device)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones(trg_len,trg_len)).to(self.device).byte()

        trg_mask = trg_sub_mask & trg_pad_mask

        return src_mask,trg_mask

        pass



    def forward(self,src,trg):

        src_mask,trg_mask = self.mask(src,trg)

        enc_src = self.encoder(src,src_mask)

        out = self.decoder(trg,enc_src,trg_mask,src_mask)

        return out
        pass
