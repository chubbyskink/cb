#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools


# In[2]:


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

print("---------------------------------------------")
print("---------------------------------------------")
print("---------------------------------------------")
print("---------------------------------------------")

print("I DON'T WANT TO RUN THIS FILE")

print("---------------------------------------------")
print("---------------------------------------------")
print("---------------------------------------------")
print("---------------------------------------------")



# In[3]:


lines_filepath = "/Users/gnarlygav/Desktop/folders/active_projects/ai/chatbot/archive/movie_lines.txt"
conv_filepath = "/Users/gnarlygav/Desktop/folders/active_projects/ai/chatbot/archive/movie_conversations.txt"

# Visualize some lines
with open(lines_filepath, 'r', encoding='iso-8859-1') as file:
    lines = file.readlines()

for line in lines[:8]:
    print(line.strip())


# In[4]:


line_fields = ["lineID", "characterID", "movieID", "character", "text"]
lines = {}
with open(lines_filepath, "r", encoding="iso-8859-1") as f:
    for line in f:
        values = line.split(" +++$+++ ")
        # Extract fields
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
        lines[lineObj["lineID"]] = lineObj


# In[6]:


# Group fields from loadlines into conversations based on 
conv_fields = ["characterID", "character2ID", "movieID", "utteranceIDs"]
conversations = []

with open(conv_filepath, 'r', encoding='iso-8859-1') as f:
    for line in f:
        values = line.split(" +++$+++ ")
        # Extract fields
        convObj = {}
        for i, field in enumerate(conv_fields):
            convObj[field] = values[i]
        
        # convert string result to split
        lineIds = eval(convObj["utteranceIDs"])
        #reassemble
        convObj["lines"] = []
        for lineId in lineIds:
            try:
                convObj["lines"].append(lines[lineId])
            except KeyError:
                print("line id error: " + lineId)
                pass
        conversations.append(convObj)


# In[7]:


qa_pairs = []
for conversation in conversations:
    # iterate
    for i in range(len(conversation["lines"]) - 1):
        inputLine = conversation["lines"][i]["text"].strip()
        targetLine = conversation["lines"][i + 1]["text"].strip()
        #filter wrong samples
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])


# In[8]:


datafile = "/Users/gnarlygav/Desktop/folders/active_projects/ai/chatbot/archive/formatted_movie_lines.txt"
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

print('\nWriting new file...')
with open(datafile, 'w', encoding="utf-8") as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)
        
print("done writing!")


# In[9]:


# visualize
datafile = "/Users/gnarlygav/Desktop/folders/active_projects/ai/chatbot/archive/formatted_movie_lines.txt"
with open(datafile, 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines[:8]:
    print(line.strip())


# In[20]:


PAD_token = 0 # padding short sentences
SOS_token = 1 # start of sentence token <START>
EOS_token = 2 # end of sentence token <END>

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}        
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}        
        self.num_words = 3 # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
            
    # Remove words below certain count threshold
    def trim(self, min_count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        print("keep words {} / {} = {:.4f}".format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))
        
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}        
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}        
        self.num_words = 3 # Count SOS, EOS, PAD
        
        for word in keep_words:
            self.addWord(word)
    


# In[11]:


def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# In[12]:


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# In[13]:


normalizeString("aa123aa!s's   dd?")


# # DEFINE THE CLASS

# In[28]:


datafile = "/Users/gnarlygav/Desktop/folders/active_projects/ai/chatbot/archive/formatted_movie_lines.txt"

print("Reading file...")
lines = open(datafile, encoding="utf-8").read().strip().split("\n")

pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines]
print("Done reading")

voc = Vocabulary("GrowVocab")


# In[29]:


MAX_LENGTH = 10
def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# In[30]:


pairs[1]


# In[31]:


pairs = [pair for pair in pairs if len(pair) > 1]
print("There are {} pairs/conversations".format(len(pairs)))
pairs = filterPairs(pairs)
print("There are {} pairs/conversations".format(len(pairs)))


# In[32]:


for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
    
print("Counted words: ", voc.num_words)
for pair in pairs[:10]:
    print(pair)
      


# In[33]:


MIN_COUNT = 3

def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_s = pair[0]
        output_s = pair[1]
        keep_i = True
        keep_o = True
        # Check input sentence
        for word in input_s.split(" "):
            if word not in voc.word2index:
                keep_i = False
                break
        # Check output sentence
        for word in output_s.split(" "):
            if word not in voc.word2index:
                keep_o = False
                break
                
        if keep_i and keep_o:
            keep_pairs.append(pair)
            
    print("Trimmed from {} pairs to {}, {:4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

pairs = trimRareWords(voc, pairs, MIN_COUNT)


# In[34]:


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


# In[39]:


pairs[1][0]


# In[36]:


indexesFromSentence(voc, pairs[1][0])


# In[40]:


# samples for testing
inp = []
out = []
i = 0
for pair in pairs[:10]:
    inp.append(pair[0])
    out.append(pair[1])
    
print(inp)
print(len(inp))
indexes = [indexesFromSentence(voc, s) for s in inp]
indexes
    


# In[47]:


a = [[3, 4, 2],
 [7, 8, 9, 10, 4, 11, 12, 13, 2],
 [16, 4, 2],
 [8, 31, 22, 6, 2],
 [33, 34, 4, 4, 4, 2],
 [35, 36, 37, 38, 7, 39, 40, 41, 4, 2],
 [42, 2],
 [47, 7, 48, 40, 45, 49, 6, 2],
 [50, 51, 52, 6, 2],
 [58, 2]]

list(itertools.zip_longest(*a, fillvalue=0))


# In[45]:


def zeroPadding(l, fillvalue = 0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# In[43]:


# Matches our MAX_LENGTH
leng = [len(ind) for ind in indexes]
max(leng)


# In[49]:


test_result = zeroPadding(indexes)
print(len(test_result))
test_result


# In[52]:


def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
                
    return m
    


# In[53]:


binary_result = binaryMatrix(test_result)
binary_result


# In[55]:


# returns padded  input sequense tensor and tensor of lengths for the sequences in the batch
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# In[87]:


# returns padded target seq tensor, padding mask and max target len
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    
    mask = binaryMatrix(padList)
    mask = torch.tensor(mask, dtype=bool)
    
    padVar = torch.LongTensor(padList)
    
    return padVar, mask, max_target_len


# In[88]:


# returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    # sort by desc len
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    
    inp, lengths = inputVar(input_batch, voc)
    
    # assert len(inp) == lengths[0]
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
    


# In[89]:


# example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)



# # Encoder
# 
# 1. convert word indexes to embeddings
# 2. pack paded batch of sequences for RNN module
# 3. forward pass through GRU (gated recurrent unit)
# 4. unpack padding
# 5. return output of final hidden state
# 

# In[75]:


class EncoderRnn(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRnn, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.n_layers = n_layers
        
        # init GRU
        # input size and hidden size are both set to hidden size
        # because our input size is a word embedding with number
        # of features == hidden size
        
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        
    def forward(self, input_seq, input_lengths, hidden=None):
        # input_seq = batch of input sentences; shape=(max_length, batch_size)
        # input_lengths = list of sentence lengths corresponding to each sentence in the batch
        # hidden state, of shape: (n_layers x num_directions, batch_size, hidden_size)
        
        # convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        
        # pack padded batch of sequenes for rnn module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        # forward pass through gpu
        outputs, hidden = self.gru(packed, hidden)
        
        # unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        return outputs, hidden
        # outputs: the output features h_t from the last layer of the GRU, for each timestep (sum of bidirectional outputs)
        # outputs shape = (max_length, batch_size, hidden_size)
        # hidden: hidden state for the last timestep of shape = (n_layers x num_directions, batch_size, hidden_size)
        


# # Luong attn layer

# In[83]:


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        # hidden of shape (1, batch_size, hidden_size)
        # encoder_outputs of shape (max_length, batch_size, hidden_size)
        
        # calculate attn weights
        attn_energies = self.dot_score(hidden, encoder_outputs) # (max_length, batch_size)
        
        # transpose max_length and batch_size dim
        attn_energies = attn_energies.t()
        
        # return softmax normalized probability scores with added dim
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
        

        


# In[84]:


class LuongAttnDecoderRnn(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers = 1, dropout = 0.1):
        super(LuongAttnDecoderRnn, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)
        
    def forward(self, input_step, last_hidden, encoder_outputs):
        # input step: one timestep (one word) of input sequence batch; shap = (1, batch_size)
        # last_hidden: final hidden layer of GRU; shape=(n_layers x num_directinos, batch_size, hidden_size)
        # encoder_outputs: encoder models output; shape=(max_length, batch_size, hidden_size)
        # NOTE - ran one word at a time
        
        # get embedding
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # rnn_output shape = (num_layers * num_directions, batch, hidden_size)
        
        # calculate attention weights from curent GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # multiply attn weights to encoder outputs to get new wieghted sum context vextor
        # (batch_size, 1, max_length) bmm with (batch_size, max_length, hidden) = (batch_size,1, hidden)
        
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # concat weighted vector and gru output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # predict next word using luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        
        # return output of final hidden state
        return output, hidden
    
        # output: softmax normalized tensor giving probability of each word being correct next word in the decoded sequence
        # shape=(batch_size, voc.num_words)
        # hidden: final hidden state of GRU
        # shape = (n_layers x num_directions, batch_size, hidden_size)
        
        


# # Training code
# 

# In[124]:


def maskNLLLoss(decoder_out, target, mask):
    nTotal = mask.sum()
    target = target.view(-1, 1)
    # decoder out shape = (batch_size, vocab_size) target_size = (batch_size, 1)
    gathered_tensor = torch.gather(decoder_out, 1, target)
    #calculate negative log likelihood loss
    crossEntropy = -torch.log(gathered_tensor)
    
    # select non-zero elements
    loss = crossEntropy.masked_select(mask)
    
    # calculate mean of the loss
    
    loss = loss.mean()
    loss = loss.to(device)
    
    return loss, nTotal.item()
    


# In[85]:


# visualize what's happening

dec_o = torch.rand(5,7)
dec_o = F.softmax(dec_o, dim=1)
tar = torch.tensor([2,1,5,4,0], dtype=torch.long)
tar = tar.view(-1,1)
mask = torch.tensor([1,0,1,1,0], dtype=torch.bool)
print(dec_o)
print(tar)

gath_ten = torch.gather(dec_o, 1, tar) # softmax scores
print(gath_ten)
print(gath_ten.shape)
crossEntropy = -torch.log(gath_ten)

print("cross entropy")
print(crossEntropy)

mask = mask.unsqueeze(1)
loss = crossEntropy.masked_select(mask)

print("loss:")
print(loss)
print(loss.shape)

print("sum of mask elements - how many we consider:", mask.sum())
print("mean of loss:", loss.mean())
print("mean of cross entropy loss without masking:", crossEntropy.mean())


# In[125]:


# visualize what's happening for one itt. ONLY for vis

small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_var shape:", input_variable.shape)
print("lengths shape:", lengths.shape)
print("target_var shape:", target_variable.shape)
print("mask shape:", mask.shape)
print("max_target_length:", max_target_len)

# define params
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
attn_model = 'dot'
embedding = nn.Embedding(voc.num_words, hidden_size)

# define encoder/decoder
encoder = EncoderRnn(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRnn(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
encoder = encoder.to(device)
decoder = decoder.to(device)

# ensure train mode
encoder.train()
decoder.train()

# init with learning rate
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)
encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

input_variable = input_variable.to(device)
lengths = lengths.to(device)
target_variable = target_variable.to(device)
mask = mask.to(device)

loss = 0
print_losses = []
n_totals = 0

encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
print("Encoder outputs shape:", encoder_outputs.shape)
print("Last encoder hidden shape:", encoder_hidden.shape)

decoder_input = torch.LongTensor([[SOS_token for _ in range(small_batch_size)]])
decoder_input = decoder_input.to(device)
print("Initial decoder input shape:", decoder_input.shape)
print(decoder_input)

# set initial decoder hidden state to the encoders final hidden state
decoder_hidden = encoder_hidden[:decoder_n_layers] ## DOUBLE CHECK
print("INIT DECODER HIDDEN SHAPE:", decoder_hidden.shape)
print("\n")
print("------------------------------------------------")
print("every timestep of GRU")
print("------------------------------------------------")
print("\n")


for t in range(max_target_len):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
    print("Decoder output shape:", decoder_output.shape)
    print("Decoder hidden shape:", decoder_hidden.shape)
    # teacher forcing
    decoder_input = target_variable[t].view(1, -1)
    print("Target variable at current timestep before reshaping:", target_variable[t])
    print("Target variable at current timestep shape before reshaping:", target_variable[t].shape)
    print("The decoder input shape (reshape the target var):", decoder_input.shape)
    # calculate and accumulate loss
    print("The mask at the current timestep:", mask[t])
    print("The mask at the current timestep shape:", mask[t].shape)
    mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
    print("Mask loss:", mask_loss)
    print("Total:", nTotal)
    loss += mask_loss
    print_losses.append(mask_loss.item() * nTotal)
    print(print_losses)
    n_totals += nTotal
    print(n_totals)
    encoder_optimizer.step()
    decoder_optimizer.step()
    returned_loss = sum(print_losses) / n_totals
    
    print("returned loss:", returned_loss) 
    print("\n")
    print("-----------------------One timestep done-------------------------")
    print("\n")







# In[126]:


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for RNN packing should always be on the CPU
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# In[120]:


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


# In[93]:


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# In[94]:


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# In[95]:


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000


# In[127]:


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Vocabulary(corpus_name)
    return voc, pairs

# Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using the ``filterPair`` condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

corpus_name = "movie-corpus"
corpus = os.path.join("data", corpus_name)

# Load/Assemble voc and pairs
save_dir = "/Volumes/go_ssd/ai"
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

loadFilename = os.path.join(save_dir, model_name, corpus_name,
                    '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                    '{}_checkpoint.tar'.format(checkpoint_iter))


# In[128]:


loadFilename = None

# Load model if a ``loadFilename`` is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRnn(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRnn(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# In[129]:


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have CUDA, configure CUDA to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)


# In[ ]:


# Set dropout layers to ``eval`` mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)

