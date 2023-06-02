#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata

PAD_token = 0 # padding short sentences
SOS_token = 1 # start of sentence token <START>
EOS_token = 2 # end of sentence token <END>
MAX_LENGTH = 10

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


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

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


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


# # # In[95]:


# # Configure models
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# # # Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 4000

# # # Turn a Unicode string to plain ASCII, thanks to
# # # https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# # # Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

datafile = os.path.join("/Volumes/go_ssd/ai/cb_model/movie-corpus/2-2_500/4000_checkpoint.tar")
voc = Vocabulary()

loadFilename = datafile

# Load model if a ``loadFilename`` is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    print("Loading file:", loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    print("----- SUCCESSFULLY LOADED MODEL -----")


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

# Set dropout layers to ``eval`` mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)

"""
> hello?
Bot: hello .
> where am I?
Bot: you re in a hospital .
> who are you?
Bot: i m a lawyer .
> how are you doing?
Bot: i m fine .
> are you my friend?
Bot: no .
> you're under arrest
Bot: i m trying to help you !
> i'm just kidding
Bot: i m sorry .
> where are you from?
Bot: san francisco .
> it's time for me to leave
Bot: i know .
> goodbye
Bot: goodbye .
"""