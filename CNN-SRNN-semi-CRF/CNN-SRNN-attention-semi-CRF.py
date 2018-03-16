import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
import torch.nn.functional as fun
torch.manual_seed(1234)


#####################################################################
# Helper functions to make the code more readable.


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int    
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    #word id sequence
    idxs = []
    for w in seq:
        if not w.lower() in to_ix:
            idxs.append(to_ix['<unk>'])
        else:
            idxs.append(to_ix[w.lower()])
    tensor = torch.LongTensor(idxs).cuda()
    
        
    idxs = []
    for w in seq:
        idxs.append(w)
    return autograd.Variable(tensor), idxs


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    if len(vec) == 1:
        return vec
    max_score = vec[argmax(vec)]
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score)))

            

def calculate_max_point(vec):
    #vec shape: [(pre_tag_end, tag_end, pretag, next_tag, score)]
    max = -1
    index = -1
    for i in range(len(vec)):
        if vec[i][4] > max:
            max = vec[i][4]
            index = i
    return vec[index]

# load word embedding
def load_my_vecs(dim):
    f = open("embedding/glove.6B.100d.txt")
    lines = f.readlines()
    word_vecs = np.zeros([len(lines), dim])
    word_to_ix = {}
    index = 0
    for line in lines:
        values = line.split(" ")
        word = values[0]
        if word in word_to_ix:
            print('wrong loading embedding')
        word_to_ix[word] = index
        vector = []
        for count, val in enumerate(values):
            if count == 0:
                continue
            vector.append(float(val))
        word_vecs[index] = vector
        index += 1

            
    return word_vecs, word_to_ix
#####################################################################
# Create model


class Atten_RNN_semiCRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, max_span_len, word_to_ix, word_vecs, char_to_ix, char_vecs, char_embedding_dim):
        super(Atten_RNN_semiCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)
        self.max_span_len = max_span_len
        self.char_to_ix = char_to_ix
        self.char_vecs = char_vecs
        self.char_embedding_dim = char_embedding_dim
        
        #embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim) 
        self.word_embeds.weight.data.copy_(torch.from_numpy(word_vecs))
        
        self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
        self.char_embeds.weight.data.copy_(torch.from_numpy(char_vecs))
        
        self.U_word = nn.Parameter(torch.nn.init.normal(torch.Tensor(embedding_dim*2,1).cuda(), mean=0, std=1))  
        self.attention_mlp_word = nn.Linear(4*embedding_dim, 2*embedding_dim)
        
        #word_gru layers
        self.gru = nn.GRU(embedding_dim*2, embedding_dim*2,
                            num_layers=1, bidirectional=True, dropout=0.5)
        self.gru2 = nn.GRU(embedding_dim*4, embedding_dim*2,
                            num_layers=1, bidirectional=True, dropout=0.5)

        #char_gru layers
        self.char_gru = nn.GRU(char_embedding_dim, char_embedding_dim,
                            num_layers=1, bidirectional=True, dropout=0.5)
                          
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0)
        #fc layer
        self.fc = nn.Sequential(
                 nn.Linear(embedding_dim*4, embedding_dim*4),
                 nn.ReLU(),
                 nn.Dropout(0.5),
                 nn.Linear(embedding_dim*4, self.target_size)    
        )

        
        #crf features
        self.transitions = nn.Parameter(torch.zeros(len(tag_to_ix), len(tag_to_ix)).cuda())  #randn
        self.transitions_bound = nn.Parameter(torch.zeros(2, len(tag_to_ix)).cuda())  #randn
 
        #semi-CRF weight
        self.w1 = nn.Parameter(torch.Tensor([1.0]).cuda())
        self.w2 = nn.Parameter(torch.Tensor([1.0]).cuda()) 
     
    
    
    


    def _forward_alg(self, feats, seqlen):
        #shape of feat:  #[spanlen, step, numlabel]
        
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.zeros(self.target_size, seqlen).fill_(0).cuda()

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
      
        # Iterate through the sentence
        for step in range(seqlen):
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.target_size):
                if step == 0:
                    emit_score = feats[0][0][next_tag]
                    trans_score = self.transitions_bound[0, next_tag]
                    t = self.w1*trans_score+self.w2*emit_score
                    forward_var[next_tag, 0] = log_sum_exp(t)
                    continue
                if next_tag == self.tag_to_ix['O']:  #look back one step
                    emit_score = feats[0][step][next_tag]
                    trans_score = self.transitions[next_tag]
                    t = forward_var[:,step-1]+self.w1*trans_score+self.w2*emit_score
                    forward_var[next_tag, step] = log_sum_exp(t)
                    continue
                temp_score = [] 
                for span_len in range(1, self.max_span_len+1):
                    if step - span_len >= 0:
                        emit_score = feats[span_len-1][step-span_len+1][next_tag]
                        trans_score = self.transitions[next_tag]
                        t = forward_var[:,step-span_len]+ self.w1*trans_score + self.w2*emit_score 
                        temp_score.append(t)#
                    elif step - span_len == -1:
                        emit_score = feats[span_len-1][0][next_tag]
                        trans_score = self.transitions_bound[0, next_tag]  
                        t = self.w1*trans_score + self.w2*emit_score
                        temp_score.append(t)#+trans_score
                        break
                temp_score = torch.cat(temp_score).view(-1)
                forward_var[next_tag, step] = log_sum_exp(temp_score)
        terminal_var =  forward_var[:, seqlen-1] + self.w1*self.transitions_bound[1]
        alpha = log_sum_exp(terminal_var)

        return alpha
    
    def _get_char_seq(self, word_seq):
        #
        seq = []
        for word in word_seq:
            for char in word:
                if char in self.char_to_ix:
                    seq.append(self.char_to_ix[char])
                else:
                    # '#' stands for unknown character
                    seq.append(self.char_to_ix['$$'])
                    continue
        tensor = torch.LongTensor(seq).cuda()
        return autograd.Variable(tensor)
    
    def _get_gru_features(self, sentence, word_seq):
        #[spanlen, step_start, numlabel]
        if len(sentence) != len(word_seq):
            print('something wrong when extracting gru-cnn features!')
            
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        
        
        #cnn features part
        seq = word_seq
        char_seq = self._get_char_seq(seq)
        #shape is [batchsize, input_channel, height, width]
        char_embeds = self.char_embeds(char_seq).view(len(char_seq), 1, self.char_embedding_dim) 

        start = 0
        end = 0
        char_feats = torch.zeros(len(sentence), 1, self.char_embedding_dim*2).cuda() #[spanlen, step, numlabel]
        char_feats = autograd.Variable(char_feats) ####
        for index in range(len(seq)):
            w = seq[index]
            span = len(w)
            start = end
            end = start + span
            part_input = char_embeds[start:end,:,:]
            part_out, _ = self.char_gru(part_input)
            p1 = part_out[-1, 0, 0:self.char_embedding_dim]
            p2 = part_out[0, 0, self.char_embedding_dim:self.char_embedding_dim*2]
            char_repre = torch.cat([p1,p2]).view(-1)
            char_feats[index, 0, :] = char_repre
        
        final_feats = torch.cat((embeds, char_feats), 2) #
    
        gru_out, _ = self.gru(final_feats)
        gru_out = gru_out.view(len(sentence), 1, self.embedding_dim*4) # [seqlen, 1, hidden_size]
        gru_feats = torch.zeros(self.max_span_len, len(sentence), self.target_size).cuda() #[spanlen, step, numlabel]
        gru_feats = autograd.Variable(gru_feats) ####
        for span_len in range(1, self.max_span_len+1):
            for step in range(len(sentence)-span_len+1):
                start = step
                end = step + span_len
                #gru-feature part
                part_input = gru_out[start:end,:,:]
                part_out, _ = self.gru2(part_input)  #[span_len, 1, hidden] 
                
                            
                    
                x1 = self.tanh(self.attention_mlp_word(part_out))  #[span_len, 1, hidden//2]
                x2 = self.U_word
                atten_value = self.tanh(torch.matmul(x1,x2))
                atten_ratio = self.softmax(atten_value)#[span_len, 1, 1]           
                combine_atten = part_out*atten_ratio  #[span_len, 1, hidden] 

                part_span_feat = torch.sum(combine_atten, 0)
                
                #basic combine
                #part1 = part_out[-1, 0, 0:self.embedding_dim*2]
                #part2 = part_out[0, 0, self.embedding_dim*2:self.embedding_dim*4] 
                #hidden = torch.cat([part1, part2]).view(1,-1)  

                span_feat = self.fc(part_span_feat)                
                gru_feats[span_len-1, start] = span_feat
        return gru_feats  

    def _score_sentence(self, feats, tags):
        #[spanlen, step, numlabel]
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]).cuda())
        for i in range(len(tags)-1):
            span_len = tags[i][1]-tags[i][0]
            start = tags[i][0]
            score = score  + self.w1*self.transitions[self.tag_to_ix[tags[i + 1][2]], self.tag_to_ix[tags[i][2]]] + self.w2*feats[span_len-1,start,self.tag_to_ix[tags[i][2]]]
        span_len = tags[-1][1]-tags[-1][0]
        start =tags[-1][0]
        #feat
        score = score + self.w2*feats[span_len-1,start,self.tag_to_ix[tags[-1][2]]] + self.w1*self.transitions_bound[0, self.tag_to_ix[tags[0][2]]] + self.w1*self.transitions_bound[1, self.tag_to_ix[tags[-1][2]]]
        return score

    def _viterbi_decode(self, feats, sentence):
        #[spanlen, step, numlabel]
        forward_var = np.zeros([self.target_size,len(sentence)])
        backpointer = {}
        for i in range(len(sentence)):
            for next_tag in range(self.target_size):
                temp = []
                if next_tag == self.tag_to_ix["O"]:
                    if i == 0:
                        score = (self.w1*self.transitions_bound[0, next_tag] + self.w2*feats[0, 0, next_tag]).data[0]
                        temp.append((-1, i, -1, next_tag, score))
                        point = calculate_max_point(temp)
                        backpointer[str(next_tag)+" "+str(i)] = point
                        forward_var[next_tag,i] = point[4]
                        continue
                    for pre_tag in range(self.target_size):
                        score = forward_var[pre_tag, i-1] + (self.w1*self.transitions[next_tag, pre_tag] + self.w2*feats[0, i, next_tag]).data[0]
                        temp.append((i-1, i, pre_tag, next_tag, score))
                    point = calculate_max_point(temp)
                    backpointer[str(next_tag)+" "+str(i)] = point
                    forward_var[next_tag, i] = point[4]
                    continue
                for j in range(1, self.max_span_len+1):
                    if i-j == -1:
                        score = (self.w1*self.transitions_bound[0, next_tag] + self.w2*feats[j-1, 0, next_tag]).data[0]
                        temp.append((-1, i, -1, next_tag, score))
                        break
                    for pre_tag in range(self.target_size):                    
                        score = forward_var[pre_tag, i-j] + (self.w1*self.transitions[next_tag, pre_tag] + self.w2*feats[j-1, i-j+1, next_tag]).data[0]
                        temp.append((i-j, i, pre_tag, next_tag, score))
                #calculate max value in temp
                point = calculate_max_point(temp)
                backpointer[str(next_tag)+" "+str(i)] = point
                forward_var[next_tag, i] = point[4]
        terminal_record =  forward_var[:,len(sentence)-1]
  
        score = terminal_record + (self.w1*self.transitions_bound[1]).data[0] 
        index = np.argmax(score, 0)
        step = len(sentence)-1
        r_result = [] 
        while True:
            point = backpointer[str(index)+" "+str(step)]
            start = point[0]+1
            end = point[1]+1
            tag = point[3]
            index = point[2]
            step = point[0]
            r_result.append((start, end, tag))
            if step == -1:
                break
                
        result = list(reversed(r_result))    
        return result

    def neg_log_likelihood(self, sentence, tags, word_seq):
        seqlen = len(sentence)
        feats = self._get_gru_features(sentence, word_seq)
        forward_score = self._forward_alg(feats, seqlen)
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence, word_seq):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiGRU
        gru_feats = self._get_gru_features(sentence, word_seq)

        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(gru_feats, sentence)
        return tag_seq

    
#####################################################################
# Run training
#####################################################################

# Make up some training data

EMBEDDING_DIM = 100
CHAR_EMBEDDING_DIM = 50
HIDDEN_DIM = 200
training_data=[]
testing_data=[]
deving_data=[]
f1= open('data/eng.train')
f2= open('data/eng.testa')
f3= open('data/eng.testb')

tag_to_ix = {}
ix_to_tag = {}
max_span_len = -1

def process_tag(tags):
    global max_span_len
    results = []
    flag = 0
    start = -1
    label = ''
    for i in range(len(tags)):
        t = tags[i].replace('B-','')
        t = t.replace('I-','')
        if not t in tag_to_ix:
            tag_to_ix[t] = len(tag_to_ix)
            ix_to_tag[len(ix_to_tag)] = t
        if tags[i] == 'O':
            if flag == 1: 
                if i-start > max_span_len:
                    max_span_len = i-start
                results.append((start, i, label))
                start = -1
                flag = 0
                label = ''
            results.append((i,i+1,"O"))
        elif tags[i].startswith('B-'):
            if flag == 1:
                if i-start > max_span_len:
                    max_span_len = i-start
                results.append((start, i, label))
            start = i
            label = t
            flag = 1
        elif tags[i].startswith('I-'):
            continue
        else:
            print('wrong in decodeing tags')
    if not label == '':
        if i-start > max_span_len:
                    max_span_len = len(tags)-start
        results.append((start, len(tags), label))
    return results

def read_data(f):
    dataset = []
    line = f.readline()
    while line:
        line = line.strip()
        if line == '-DOCSTART- -X- O O':
            line = f.readline()
            continue
        if len(line) < 1:
            line = f.readline()
            continue           
        words = []
        tags = []
        word = line.split(' ')[0]
        tag = line.split(' ')[-1]
        words.append(word)
        tags.append(tag)
        line = f.readline().strip()
        while len(line) > 1:
            word = line.split(' ')[0]
            tag = line.split(' ')[-1]
            words.append(word)
            tags.append(tag)
            line = f.readline().strip()
        tags = process_tag(tags)
        dataset.append((words,tags))
        line = f.readline()
    return dataset

#training_data = [(["i","am","a","chinese","people","who"],[(0,2,"A"),(2,3,"O"),(3,4,"O"),(4,5,"B"),(5,6,"B")])]

 
training_data = read_data(f1)
deving_data = read_data(f2)
testing_data = read_data(f3)
f1.close()
f2.close()
f3.close()
train_datasize = len(training_data)
dev_datasize = len(deving_data)
test_datasize = len(testing_data)
print("Reading "+str(train_datasize)+" train instance, "+str(dev_datasize)+" valid instance, "+str(test_datasize)+" test instance.")


#prepare word embedding
count = {}
for sentence, tags in training_data:
    for word in sentence:
        if word.lower() not in count:
            count[word.lower()] = 1
        else:
            num = count[word.lower()]+1
            count[word.lower()] = num

word_vecs, word_to_ix = load_my_vecs(EMBEDDING_DIM)            
 
for sentence, tags in training_data:
    for word in sentence:
        if word.lower() not in word_to_ix and count[word.lower()] >= 3:
            word_to_ix[word.lower()] = len(word_to_ix)
            word_vecs = np.row_stack((word_vecs,np.zeros([1, EMBEDDING_DIM])))
            
if len(word_to_ix) != len(word_vecs):
    print('wrong loading word embedding!')
    
    
#prepare char embedding
char_to_ix = {}
char_vecs = []
for sentence, tags in training_data:
    for word in sentence:
        for char in word:
            if not char in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
                char_vecs.append(np.random.uniform(-np.sqrt(3/CHAR_EMBEDDING_DIM), np.sqrt(3/CHAR_EMBEDDING_DIM), [CHAR_EMBEDDING_DIM]))
                #char_vecs.append(np.random.normal([CHAR_EMBEDDING_DIM]))

# '#' stands for unknow character
char_to_ix['$$'] = len(char_to_ix)
char_vecs.append(np.zeros([CHAR_EMBEDDING_DIM]))
char_vecs = np.array(char_vecs)
if len(char_vecs) != len(char_to_ix):
    print('wrong loading char embedding!')
    
    
#word_to_ix['<unk>'] = len(word_to_ix)            
#for sentence, tags in testing_data:
#    for word in sentence:
#        if word not in word_to_ix:
#            word_to_ix[word] = len(word_to_ix)
        
#for sentence, tags in deving_data:
#    for word in sentence:
#        if word not in word_to_ix:
#            word_to_ix[word] = len(word_to_ix)

print("MAM_SPAN_LEN="+str(max_span_len))
print(tag_to_ix)
print(char_to_ix)

model = Atten_RNN_semiCRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, max_span_len, word_to_ix, word_vecs, char_to_ix, char_vecs, CHAR_EMBEDDING_DIM)

model.cuda()



def evaluation(gold, predict):
    if not len(gold) == len(predict):  # Sanity check
        print('something wrong in evaluation 1.')
    right = np.zeros([len(tag_to_ix)])
    find = np.zeros([len(tag_to_ix)])
    all = np.zeros([len(tag_to_ix)])
    l_p = np.zeros([len(tag_to_ix)])
    l_r = np.zeros([len(tag_to_ix)])
    l_f = np.zeros([len(tag_to_ix)])
    for i in range(len(gold)):
        flag = 0
        for j in range(len(gold[i])):
            gold_index = tag_to_ix[gold[i][j][2]]
            g_start = gold[i][j][0]
            g_end = gold[i][j][1]
            all[gold_index] += 1
            for m in range(len(predict[i])):
                predict_index = predict[i][m][2]
                p_start = predict[i][m][0]
                p_end = predict[i][m][1]
                if flag == 0:
                    find[predict_index] += 1
                if p_start == g_start and p_end == g_end and predict_index == gold_index:
                    right[predict_index] += 1
            flag = 1
    global_right = 0
    global_find = 0
    global_all = 0
    # start from 1, don't calculate label O
    for i in range(len(right)):
        if i == tag_to_ix['O']:
            continue
        global_right += right[i]
        global_find += find[i]
        global_all += all[i]
        l_p[i] =  right[i]/all[i]
        l_r[i] = right[i]/find[i]
        l_f[i] = 2*l_p[i]*l_r[i]/(l_p[i]+l_r[i])
    for i in range(len(l_r)):
        if i == tag_to_ix['O']:
            continue
        print(ix_to_tag[i]+" Pre="+str(l_p[i])+" , Rec="+str(l_r[i])+" , F1="+str(l_f[i]))   
    if global_find == 0 and global_right == 0:
        return 0,0,0
    precision = global_right/global_all
    recall = global_right/global_find
    if global_right == 0:
        return 0,0,0
    f = 2*precision*recall/(precision+recall)    
    return precision, recall, f


#hypher parameter setting
cur_best = 0.85
epoch_num = 2000
valid_step = 2000000 
valid_train_step = 20000000
global_step = 0
sample = 100
lr = 0.01

for epoch in range(1, epoch_num+1):  # again, normally you would NOT do 300 epochs, it is toy data
    
    if lr*0.5**(epoch-1) >= 0.0001:
        print('Learning Rate: '+str(lr*0.5**(epoch-1)))
        optimizer = optim.SGD(model.parameters(), lr=lr*0.5**(epoch-1), momentum=0.9)
    else:
        print('Learning Rate: 0.0001')
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    time_start=time.time()
    if epoch >= 16:
        valid_step = 1000  
        valid_train_step = 1000

    for sentence, tags in training_data:
        model.train()
        global_step += 1
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in, word_seq = prepare_sequence(sentence, word_to_ix)

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, tags, word_seq)
        
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        

        #check sample train loss
        if global_step % valid_train_step == 0:  
            model.eval()
            index = 0
            loss = 0
            for sent, tag in training_data:
                index += 1
                sent, w_seq = prepare_sequence(sent, word_to_ix)
                part_loss = model.neg_log_likelihood(sent, tag, w_seq)
                loss += part_loss.data
                if index == sample:
                    break
            print("epoch("+str(epoch)+"), global step("+str(global_step)+") approx train loss: "+str(loss[0]/sample))
 
        #valid part 
        p,r,f = -1,-1,-1
        if global_step % valid_step == 0:
            model.eval()
            predicts = []
            gold = []
            for sent, tag in deving_data:
                precheck_sent, w_seq = prepare_sequence(sent, word_to_ix)
                result = model(precheck_sent, w_seq)
                predicts.append(result)
                gold.append(tag)
            p, r, f = evaluation(gold, predicts)
            print("epoch("+str(epoch)+"), step("+str(global_step)+") valid result --- Pre="+str(p)+ " , Rec="+str(r)+" , F1="+str(f))
            ############@@@@
            if f > cur_best:
                cur_best = f
                print('\nNote!: saving best model to directory:  ' + 'model/ckpt-'+str(epoch)+'-'+str(global_step)+'-'+str(f)+'.pkl')
                torch.save(model.state_dict(), 'model/ckpt-'+str(epoch)+'-'+str(global_step)+'-'+str(f)+'.pkl') 
                predicts = []
                gold = []
                for sent, tag in testing_data:
                    precheck_sent, w_seq = prepare_sequence(sent, word_to_ix)
                    result = model(precheck_sent, w_seq)
                    predicts.append(result)
                    gold.append(tag)
                p, r, f = evaluation(gold, predicts)
                print("Best model on test set result --- Pre="+str(p)+ " , Rec="+str(r)+" , F1="+str(f)+"\n")
                #model.load_state_dict(torch.load('net_params.pkl'))
    time_end=time.time()              
    cost_time = time_end-time_start  
    print('epoch('+str(epoch)+') done, spend '+str(cost_time)+' seconds.')

        
print('\nDone training process.')



#predicts = []
#gold = []
#for sent, tag in testing_data:
#    model.eval()
#    precheck_sent = prepare_sequence(sent, word_to_ix)
#    result = model(precheck_sent)
#    predicts.append(result)
#    gold.append(tag)
#p, r, f = evaluation(gold, predicts)
#print("testing set result --- precision, recall, f1: "+str(p)+" "+str(r)+" "+str(f))

        
        

