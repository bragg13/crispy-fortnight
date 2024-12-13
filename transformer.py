# %%
import os
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from tqdm import tqdm

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EXPERIMENT 1:
EMB_DIM = 128
N_LAYERS = 1
N_HEADS = 8
FORWARD_DIM = 512
DROPOUT = 0.05
LEARNING_RATE = 7e-4
BATCH_SIZE = 64
GRAD_CLIP = 1
MAX_LEN = 128 # ????
# Optimizer: AdamW
mask=None

# %%
# Task 0: DataLoader and Preprocessing
class TasksData(Dataset):
    def __init__(self, data_dir, file, transform=None):
        self.data_dir = data_dir
        self.file = file
        text_file = os.path.join(data_dir, file)

        data_dict = {"src": [], "tgt": []}

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                src = line.split('OUT:')[0]
                src = src.split('IN:')[1].strip()
                tgt = line.split('OUT:')[1].strip()

                data_dict['src'].append(src)
                data_dict['tgt'].append(tgt)

        self.data = pd.DataFrame(data_dict)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data['src'].iloc[idx]
        tgt = self.data['tgt'].iloc[idx]
        return src, tgt

def create_vocab(dataset):
    vocab = set()

    for sample in dataset:
        vocab.update(sample.split())
    return vocab

# %%
# creating datasets
train_data = TasksData(data_dir='./data/Experiment-3', file='tasks_train_addprim_turn_left.txt')
test_data = TasksData(data_dir='./data/Experiment-3', file='tasks_test_addprim_turn_left.txt')

#creating source and target vocab
src_train_data = [src for src, tgt in train_data]
vocab_train_src = create_vocab(src_train_data)

tgt_train_data = [tgt for src, tgt in train_data]
vocab_train_tgt = create_vocab(tgt_train_data)

# we need to do word2idx to map the words to indexes. Bc the input for nn.Embedding has to be numbers
# since nn.Embdding has different weights in input andoutput embedding the same index will not be encoded to the same vector
word2idx_src = {w: idx + 1 for (idx, w) in enumerate(vocab_train_src)}
word2idx_src['<PAD>'] = 0

word2idx_tgt= {w: idx + 1 for (idx, w) in enumerate(vocab_train_tgt)}
word2idx_tgt['<PAD>'] = 0

# We need Vocabulary size without padding
# word2idx
# padding
#vocabulary and word2idx

def custom_collate_fn(batch):
    #input: batch of sentences
    # tokenize, word2idx, pad
    padded_src = pad_sequence([torch.tensor([word2idx_src[w] for w in src.split()]) for src, tgt in batch], batch_first=True, padding_value=0).to(device)
    padded_tgt = pad_sequence([torch.tensor([word2idx_tgt[w] for w in tgt.split()]) for src, tgt in batch], batch_first=True, padding_value=0).to(device)

    return padded_src, padded_tgt

# %%
# create dataloaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
# next(iter(train_loader))


# ---


# %%
# Task 1: MultiHead Attention
"""
Here we will build the famous 2017 Transformer Encoder-Decoder from the Paper [Attention is All You Need](https://arxiv.org/abs/1706.03762).
We will start by implementing Multi-Head Attention, which concatenates multiple single scaled dot-product attention (SDPA) modules along the number of attention heads we desire. However, as concatenation implies sequential procedures, we will directly implement multi-head attention as a tensor operation on `nn.Linear()` layers by dividing them into `num_heads` subparts and calculating SDPA on each of them. By doing this, we entirely avoided sequential calculations.

In order to have trainable parameters, we can conveniently build all modules using torch's `nn.Module` functionality.

* Our module's `__init__()` method takes in the embedding dimension `emb_dim` of our transformer, as well as the number of heads `num_heads`.
    * It stores the `head_dim = emb_dim // num_heads`
* We create 4 linear layers
    * The linear layers for query, key, and value each have `(emb_dim, num_heads * head_dim)` size
    * The output linear layer needs to take the `num_heads * head_dim` as input size, and outputs the original model embedding dimension `emb_dim`
* The `forward()` method of this module takes in `query`, `key`, `value`, and an optional `mask`, and performs the calculations of the following formula:
    * Remember that our input at this stage has dimensions `(batch_size, seq_len, emb_dim)`
    * We pass `query`, `key`, `value` through their respective linear layers
    * Then, we perform the multi-head splitting of the linearly projected outputs
        * each projection's hidden dimension has to be reshaped to fit the `num_heads` and `head_dim` structure (in that order)
        * Hint: Both `batch_size` and `seq_len` shouldn't be changed
    * Afterwards, we perform the matrix multiplication step of queries with their transposed keys, visualized by the $QK^T$ in the above formula
    * Hint: The output shape after this step should be `(batch_size, num_heads, num_query_seq, num_key_seq)`
    * Call this output `key_out`
    * After this step, add in the optional step to mask the `key_out` tensor. We provided this code snipped, just include it at this step in the forward pass
    * Following this, we perform the softmax step on the result of the division from `key_out` with the square root of our `head_dim`
        * Make sure to apply softmax to the correct dimension
    * Now we need just need to matrix multiply this result with the values (which were passed through their respective linear layer earlier)
    * The output shape of this operation is `(batch_size, seq_len, num_heads, head_dim)`
    * Reshape it to fit the input shape of our output linear layer
    * Pass it through the ouput linear layer
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads

        self.q_w = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.k_w = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.v_w = nn.Linear(emb_dim, num_heads * self.head_dim)

        self.out = nn.Linear(num_heads * self.head_dim, emb_dim)

    # We changed the 3 inputs query, key, and value to just 1 input X
    def forward(self, query, key, value, mask=None):
        # TODO
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # stacking?
        # batch, n_heads, seq_len, emb_dim
        #X = torch.stack([X]*self.num_heads, dim=2)

        ##print(X.shape)

        # Is query key mask not all just the same? datamatrix X?
        # masking the padding here so we dont learn useless reprensetations?
        Q = self.q_w(query).to(device).reshape(batch_size, query.shape[1], self.num_heads, self.head_dim)
        K = self.k_w(key).to(device).reshape(batch_size, key.shape[1], self.num_heads, self.head_dim)
        V = self.v_w(value).to(device).reshape(batch_size, value.shape[1], self.num_heads, self.head_dim)

        d_k = self.q_w.weight.shape[1]


        Q_permute = Q.permute(0, 2, 1, 3)
        K_transposed = K.T.permute(3, 1, 0, 2)

        key_out = torch.matmul(Q_permute, K_transposed) / np.sqrt(d_k) # This is now a torch.Tensor object
        # #print(attention_score)
        ##print(key_out.shape)
        # TODO: masking
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -torch.inf)

        key_out = nn.functional.softmax(key_out, dim=-1)

        V_permuted = V.permute(0, 2, 1, 3)
        attention_output = torch.matmul(key_out, V_permuted).permute(0, 2, 1, 3)  # to have the output matrix shape (self.out)
        attention_output = attention_output.contiguous().view(batch_size, seq_len, -1)
        #print("Attention output shape:")
        #print(attention_output.shape)

        # We assume that this is correct,
        output = self.out(attention_output)
        #print(output.shape)

        return Q, K, V, output

# %%
## Task 1.2: Transformer Blocks
"""
We will now create Transformer Blocks out of our `MultiHeadAttention` module, combined with Feedforward-Networks.

* To create the blocks, our module takes as input in it's `__init__` method:
    * the embedding dimension `emb_dim`, the number of heads `num_heads`, a dropout rate `dropout`, and the dimension of the hidden layer in the feedforward network, often called `forward_dim`
    * in the `__init__` method, we further need two `nn.Layernorm` objects with an epsilon parameter `eps=1e-6`
    * then, still in the `__init__` method, we set up the feedforward network
    * we build it by creating an `nn.Sequential` module and filling it with:
        * a linear layer projecting the input onto the `forward_dim`
        * running it through `nn.ReLU`
        * and projecting the `forward_dim` back to the embedding dimension with another linear layer
* the `forward()` method takes `query`, `key`, `value` and the `mask`
    * first, we run `query`, `key`, `value`, and the `mask` through multi-head attention
    * secondly, we build a skip-connection by adding the `query` back to the output of multi-head attention
        * dropout is applied to the sum, followed by our first layer norm
    * third, the output is put through our FFN
    * fourth, we build another skip-connection by adding the input of the FFN onto the output of the FFN
    * apply dropout to the result of the skip-connection, apply normalization on the dropped-out result, and return it
"""
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim, eps=1e-6):
        super().__init__()

        # TODO
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim # MLP hidden layer dimension
        self.eps = eps

        # step 1
        self.multiheadAttention = MultiHeadAttention(
          emb_dim=self.emb_dim,
          num_heads=self.num_heads)
        self.norm1 = nn.LayerNorm(emb_dim, eps=eps)

        self.norm2 = nn.LayerNorm(emb_dim, eps=eps)  # for some reason, we need this separate (it stores the normalized values)

        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )

        self.dropout = nn.Dropout(dropout)

    # def forward(self, query, key, value, mask):
    def forward(self, query, key, value, mask):
        x = query
        with torch.no_grad():
            query, key, value, output = self.multiheadAttention(query, key, value, mask)

        # Add X_in to output
        pre_norm_1 = self.dropout(x + output)
        checkpoint_1 = self.norm1(pre_norm_1)

        out_2 = self.ffnn(checkpoint_1)
        pre_norm_2 = checkpoint_1 + out_2
        block_out = self.norm2(pre_norm_2)

        ##print(f"transformer block out {block_out.shape}")
        return block_out

# %%
## Task 1.3 Encoder
"""
This already convenes the encoder side of the transformer. We now just need to incorporate it into an appropriate format so that it can take input sequences, move them to the GPU, etc. To achieve this, we create another module called `Encoder`.
* The `Encoder` takes as input in its `__init__` method:
    * the (source) vocabulary size `vocab_size`, embedding dimension `emb_dim`, number of layers `num_layers`, number of heads `num_heads`, feedforward-dimension `forward_dim`, the dropout rate `dropout`, and the maximum sequence length `max_len`
    * Note that the preprocessing, in this case the truncation of sequences to the maximum allowed length, is handled in the data loading process that we performed in the first exercise while loading the sequences. Here, we define the model architecture that (usually) dictates the necessary preprocessing steps.
    * We then define
        * the token level embeddings with dimensions `vocab_size x emb_dim`
        * positional encodings with the sinusoidal approach (function is given below)
            * You need to create an additional `nn.Embedding` layer and load in the sinusoid table with the `.from_pretrained` method
            * Freeze these embeddings
        * a dropout layer
        * and, lastly, instantiate `num_layers` many `TransformerBlock` modules inside an `nn.ModuleList`
* In the `forward()` method, we take in the batched sequence inputs, as well as a mask
    * Then, we create the input to the positional encodings by defining a matrix which represents the index of each token in the sequence
        * Move the positions to the device on which the batched sequences are located
        * Make sure to shift each index by `+1` (and the `max_len` in the creation of the sinusoidal table, too)
        * This is done because index `0` is usually reserved for special tokens like `[PAD]`, which don't need a positional encoding.
    * We then run our input through the embeddings, the above create positions are run through the positional encodings, and both results are summed up
    * Apply dropout to the summed up result
    * This will be our `query`, `key`, and `value` input that runs `num_layers` times through our encoder module list
    * Return the last output
"""
def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()
        # TODO: check padding_idx
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim).to(device)
        self.positional_encoding = nn.Embedding.from_pretrained(get_sinusoid_table(max_len+1, emb_dim), freeze=True).to(device)
        self.dropout = nn.Dropout(dropout)
        self.t_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout, forward_dim).to(device) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        # x = x.to(device)
        # embedding layer

        token_embeddings = self.token_embeddings(x)
        ##print(f"token embeddings shape {token_embeddings.shape}")

        ##print(self.positional_encoding)
        ##print(f"x shape {x.shape}")

        # positional embedding layer (?) - no need to shift probably
        pos_embedding_input = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1) + 1
        pos_embedding_input = pos_embedding_input.to(device)
        #print(f"pos_embedding_input shape {pos_embedding_input.shape}")
        pos_embeddings = self.positional_encoding(pos_embedding_input)
        #print(f"pos embeddings shape {pos_embeddings.shape}")

        embeddings = token_embeddings + pos_embeddings

        #print(f"embeddings shape {embeddings.shape}")
        z = self.dropout(embeddings)
        # z.to(device)
        #print(f"embeddings (after dropout) shape {z.shape}")

        # pass through trasnformer blocks
        for i, block in enumerate(self.t_blocks):
            z = block.forward(query=z, key=z, value=z, mask=mask)
            #print(f"z shape {z.shape}")

        #print(f'last output: {z.shape}')
        return z

# %%
# Task 1.4: Decoder Blocks
"""
Now to the decoder part!

A `DecoderBlock` looks very similar to our previous `TransformerBlock`, but slightly extends the functionality because at its second stage, it receives inputs from both the encoder and its first stage (look closely at the input arrows in the picture!)

* To build one, the module's `__init__` method takes as input:
    * the embedding dimension `emb_dim`, number of heads `num_heads`, a feedforward dimension `forward_dim`, and a `dropout` rate
    * It then initializes:
        * an `nn.LayerNorm` with `eps=1e-6`, the `MultiHeadAttention` module, a `TransformerBlock`, and the dropout rate
* The decoder block's `forward()` method takes:
    * the batched sequence input, `value`, `key`, a source mask, and a target mask
    * First, we compute *self-attention* representations of the input (i.e., the input serves as `query`, `key`, and `value`), and takes the *target mask* for the mask parameter
        * This is the input that is symbolized by the arrow coming from the bottom of the image
    * Secondly, we use a skip-connection by summing up the above self-attention result with the original input (again, apply dropout here and normalize the result)
        * This output is our new `query`
    * We now run this above created `query` as the query-input through a `TransformerBlock`, where the `value` and `key` arguments for the `TransformerBlock` come from the `Encoder` output
        * This is called *cross-attention*
        * Include the source mask as the `mask` argument in the `TransformerBlock`
        * return the output of the `TransformerBlock`
"""
class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim

        self.layerNorm = nn.LayerNorm(emb_dim,eps=1e-6)
        self.multiheadAttention = MultiHeadAttention(
          emb_dim=self.emb_dim,
          num_heads=self.num_heads
        )
        # .to(device)

        self.dropout = nn.Dropout(dropout)

        self.transformerBlock = TransformerBlock(emb_dim, num_heads, dropout, forward_dim) #.to(device)

    def forward(self, x, value, key, src_mask, tgt_mask):
        # TODO
        # x is the input to decoder
        Q, K, V, output = self.multiheadAttention(x, x, x, mask=tgt_mask) #output is the context correcting vectors, final product of the multihead self-attention
        #print(f"output shape {output.shape}")

        norm_skip = self.layerNorm(self.dropout(x + output))

        decoder_output = self.transformerBlock(norm_skip, key, value, mask=src_mask)

        return decoder_output

# %%
# Task 1.5: Decoder
"""
As we could see from the large overview of the transformer architecture, this is already most of what is happening on the decoder side. Similar to our `Encoder`, we now must enable the `DecoderBlock` to take external input and embed its own sequences. We will do this in the `Decoder` module below.

The `Decoder`'s `__init__` method
* takes as input:
    * the (target) vocabulary size `vocab_size`, embedding dimension `emb_dim`, number of layers `num_layers`, number of heads `num_heads`, the hidden dimensionality `forward_dim` of the feedforward module, as well as the maximum sequence length `max_len`

    * We then initialize:
        * token embeddings, a dropout layer, and `num_layers` many `DecoderBlocks` inside another `nn.ModuleList`
        * We also need positional encodings, but here we don't use sinusoidal embeddings, but instead something called *relative positional encodings*, which capture the relative position between the decoder input tokens and the output tokens at each decoding step
            * They are trainable, and are implemented by another `nn.Embedding` layer, but with dimensions `max_len x emb_dim`
        * lastly, we need a linear output layer which maps the embedding dimension back to the vocabulary size

* The modules `forward()` pass then takes as input the batched sequence input, the encoder output, and a source and target mask
    * The decoder then:
        * processes the sequences through our normal embeddings
        * creates inputs to the relative positional encodings by again creating a matrix of position indices from each token in the sequence (no `+1` shifting this time because we train each position relative to the current encoded sequence position output)
        * The inputs again need to be moved to the batched sequence input's device
        * runs these positions through the relative positional encodings, and sums them up with the token embeddings
            * apply dropout on the sum
        * the sum will be the input to the `num_layers` decoder block
        * loop through all layers by passing the previous output as input through the next layer
        * the last output will put through the linear output layer and returned
"""
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim).to(device)
        self.relative_pos_encondings = nn.Embedding(max_len, emb_dim).to(device)
        self.dropout = nn.Dropout(dropout)
        self.d_blocks = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads, forward_dim, dropout).to(device) for _ in range(num_layers)
        ])
        self.output = nn.Linear(emb_dim, vocab_size)

        self.emb_dim = emb_dim
        self.max_len = max_len



    def forward(self, x, encoder_out, src_mask, tgt_mask):
        embeddings = self.token_embeddings(x)

        pos_embedding_input = torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1) + 1
        pos_embedding_input = pos_embedding_input.to(device)
        #print(f"pos_embedding_input shape {pos_embedding_input.shape}")

        pos_embeddings = self.relative_pos_encondings(pos_embedding_input)

        embeddings = embeddings + pos_embeddings
        z = self.dropout(embeddings)

        # pass through trasnformer blocks
        for i, block in enumerate(self.d_blocks):
            #print(f"decoder block {i}")
            #print(f"z shape {z.shape}")
            z = block.forward(z, value=encoder_out, key=encoder_out, src_mask=src_mask, tgt_mask=tgt_mask)

        final_output = self.output(z)

        return final_output

# %%
# Task 1.6: Transformer
"""
Now, we just need to put everything together into one `Transformer`.

* Gather all necessary arguments to initialize one `Encoder` and one `Decoder` in the `__init__` method
* Additionally, we also need to include a source and a target padding index
* For simplicity, we provide both mask creation functions

During the `forward()` pass, we:
* take in our batched source and target sequences
* call both `create_mask` functions on the respective source and target sequence
* encode the sequence using the initialized encoder and the source mask
* input the original target sequences as input into the decoder, together with the encoder output and both masks
* return the output of the decoder

That's it - you made it!

If you want to test the general functionality of your Transformer, we provide a test for you below. If the asserted shape is returned, you are on the right track.
"""
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(
            src_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )

    # padding mask
    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(device)
        return src_mask

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        #print('masks ok')

        encoder_out = self.encoder(src, src_mask).to(device)
        # print('encoder ok')
        decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        #print('decoder ok')

        return decoder_out


# ---



# %%
# Initialize the model, optimizer, and loss function
model = Transformer(
    src_vocab_size=len(word2idx_src),
    tgt_vocab_size=len(word2idx_tgt),
    src_pad_idx=word2idx_src['<PAD>'],
    tgt_pad_idx=word2idx_tgt['<PAD>'],
    emb_dim=EMB_DIM,
    num_layers=N_LAYERS,
    num_heads=N_HEADS,
    forward_dim=FORWARD_DIM,
    dropout=DROPOUT,
    max_len=MAX_LEN,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx_tgt['<PAD>'])

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for step, (src, tgt) in (pbar := tqdm(enumerate(train_loader), total=len(train_loader))):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        output = model(src, tgt[:, :-1])
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output, tgt)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

# %%


# %%

# %%
# # general test case
# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# multiheadAttention = Transformer(
#     src_vocab_size=200,
#     tgt_vocab_size=220,
#     src_pad_idx=0,
#     tgt_pad_idx=0,
# ).to(device)

# # source input: batch size 4, sequence length of 75
# src_in = torch.randint(0, 200, (4, 75)).to(device)

# # target input: batch size 4, sequence length of 80
# tgt_in = torch.randint(0, 220, (4, 80)).to(device)

# # expected output shape of the model
# expected_out_shape = torch.Size([4, 80, 220])

# with torch.no_grad():
#     out = multiheadAttention(src_in, tgt_in)

# assert out.shape == expected_out_shape, f"wrong output shape, expected: {expected_out_shape}"
