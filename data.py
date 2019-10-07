#from utils import create_filed
import torch
from torchtext.data import  Field,BucketIterator
from torchtext.datasets import TranslationDataset,Multi30k
from HyperParams import *
import utils
#print(utils.__file__)
class corpus():
    def __init__(self):

        self.token_en = utils.tokenizer('en')
        self.token_de = utils.tokenizer('de')
        device = torch.device(('cuda' if torch.cuda.is_available() == True else 'cpu'))
        print(self.token_en.get_tokenizer()("I LOVE U"))
        self.SRC = utils.create_filed(self.token_de.get_list)
        self.TRG = utils.create_filed(self.token_en.get_list)
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts=('.de', '.en'), fields=(self.SRC, self.TRG))
        self.SRC.build_vocab(self.train_data,min_freq = MIN_FREQ)
        self.TRG.build_vocab(self.train_data,min_freq = MIN_FREQ)
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
                                                        (self.train_data, self.valid_data, self.test_data),
                                                            batch_size=BATCH_SIZE,
                                                            device=device)
    def get_fileds(self):
        return self.SRC,self.TRG
    def get_iters(self):
        return self.train_iterator, self.valid_iterator, self.test_iterator






