import data
import nltk
import torch
def bleu_score(sen, ref, cor):

    # sen =[ ['This', 'is', 'a','test']]
    # ref =[['This', 'is', 'a', 'test']]
    # for (s,r) in zip(sen,ref):
    #     print(s,r)
    #     print(nltk.translate.bleu_score.sentence_bleu([r],s,weights=(1, 0, 0, 0)))

    #sen = torch.tensor([[543,21,432]])
    #ref = torch.tensor([[543,21,12,432]])

    TRG = cor.get_fileds()[1]
    SRC = cor.get_fileds()[0]
    senList = []
    refList = []
    #print(sen.shape,ref.shape)
    #print(SRC.vocab.itos[300])
    for s in sen:
        #print(s)

        s_list = [ TRG.vocab.itos[int(w)] for w in s ]
        senList.append(s_list)
        pass

    for s in ref:
        # print(s)

        s_list = [TRG.vocab.itos[int(w)] for w in s]
        refList.append(s_list)
        pass

    score = 0
    smooth = nltk.translate.bleu_score.SmoothingFunction()
    for (h, r) in zip(senList,refList):
        score += nltk.translate.bleu_score.sentence_bleu([r], h,auto_reweigh=True,
                            smoothing_function=smooth.method4)



    return score

# return ret

#print(bleu_score())







