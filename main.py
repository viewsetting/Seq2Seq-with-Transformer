from model import *
from torchtext.data import  Field,BucketIterator
from torchtext.datasets import TranslationDataset,Multi30k
import spacy
import torch.optim as optim
import os,math,random,time
from HyperParams import *
from utils import model_init,NoamOpt,train,evaluate,epoch_time
from model import Encoder,Decoder,Seq2Seq,EncoderLayer,PositionFeedForward,SelfAttention,DecoderLayer
from data import corpus



device = torch.device(('cuda' if torch.cuda.is_available() == True else 'cpu'))

#-------------------seq2seq model---------------------
cor = corpus()
SRC,TRG = cor.get_fileds()
train_iterator, valid_iterator, test_iterator = cor.get_iters()
input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)

enc = Encoder(vocab_dim=input_dim,hid_dim=HID_DIM,n_heads=N_HEADS,n_layers=N_LAYERS,
              pf_dim=PF_DIM,positionwise_feedforward=PositionFeedForward,self_attention=SelfAttention,
              encode_layer=EncoderLayer,dropout=DROPOUT,device=device,PositionalEncoding=True).to(device)

dec = Decoder(vocab_dim=input_dim,hid_dim=HID_DIM,n_layers=N_LAYERS,n_heads=N_HEADS,
              pf_dim=PF_DIM,decode_layer=DecoderLayer,self_attention=SelfAttention,
              positionwise_feedforward=PositionFeedForward,dropout=DROPOUT,device=device,PositionalEncoding=True
              ).to(device)

pad_idx = SRC.vocab.stoi['<pad>']

model = Seq2Seq(encoder=enc,decoder=dec,pad_idx=pad_idx,device=device).to(device)

#-------------------------train & valid--------------------------------

model_init(model)

optimizer = NoamOpt(HID_DIM,factor=1,warmup=2000,
                optimizer=optim.Adam(model.parameters(),lr = 0,betas=(0.9,0.98),eps=1e-9))
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


for epoch in range(N_EPOCH):

    start_time = time.time()

    train_loss,train_score = train(model,train_iterator,optimizer,criterion,CLIP,cor)
    valid_loss,valid_score = evaluate(model, valid_iterator, criterion,cor)

    end_time = time.time()

    EpochTime = epoch_time(start_time=start_time,end_time = end_time)

    print("Epoch No.",epoch," Time: ",EpochTime[0],"min",EpochTime[1],"sec\n "," Train Loss: ",train_loss,
          " Train PPL: ",math.exp(train_loss)," Train_BLEU: ",train_score,
          "\nValid Loss: ",valid_loss," Valid PPL: ",math.exp(valid_loss)," Valid_BLEU: ",valid_score
          )


