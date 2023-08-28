#changed to kmeans
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# import utils
from sklearn.metrics.pairwise import cosine_similarity
import pickle

NUM_PIXELS = 256
#keywordsN = 5
#autograd.set_detect_anomaly(True)

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


# add global attention and local attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size, att_type, block_length):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.att_type = att_type
        self.block_length = block_length

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None, attn_visual=False):
        # print("q:", q.shape, q)
        # print("k:", k.shape, k)
        # print("v:", v.shape, v)
        orig_q_size = q.size()
        # print("orig_q_size:", orig_q_size, q)

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        attn_dist = None

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        # print("q shape:", q.shape)
        # print("v shape:", v.shape)
        # print("k shape:", k.shape)

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)

        if self.att_type == "global":
            x = torch.matmul(q, k)  # [b, h, q_len, k_len]
            # print("x shape:", x.shape)
            # print("mask shape:", mask.shape)
            # print("mask.unsqueeze(1) shape:", mask.unsqueeze(1).shape)
            #print("torch.einsum shape:", torch.einsum("...kd,...qd->...qk", k.transpose(2, 3), q).shape)
            # print("x device:", x.device)
            # print("mask device:", mask.to(x.device).device)
            # print("x shape:", x.shape)
            # print("attention distribution before mask:", x)
            # print("mask:", mask.shape, mask)
            x.masked_fill_(mask.unsqueeze(1), -1e9)  # mask when mask val is 0
            # print("attention distribution after mask:", x.shape, x)
            if attn_visual == True:
                attn_dist = x
            x = torch.softmax(x, dim=3)
            # print("attn_dist:", attn_dist)
            x = self.att_dropout(x)
            x = x.matmul(v)  # [b, h, q_len, attn]
        elif self.att_type == "local_1d":
            #             blen = self.block_length
            #             pad = (0, 0, 0, (-q.shape[2]) % self.block_length)
            #             q = F.pad(q, pad)
            q_len = q.shape[2]
            blen = self.block_length
            k = k.transpose(2, 3)
            pad = (0, 0, 0, (-q_len) % self.block_length)  # Append to multiple of block length
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            # dot product
            x_first = torch.matmul(q[:,:,:blen,:], k[:,:,:blen,:].transpose(2, 3))  # [b, h, q_len, k_len]
            # print("x shape:", x_first.shape)
            # print("attention distribution before mask:", x_first)
            # print("first mask:", mask.shape, mask)
            x_first.masked_fill_(mask.unsqueeze(1), -1e9)  # mask when mask val is 0
            # print("attention distribution after mask:", x_first.shape, x_first)
            # print("attn_visual:", attn_visual)
            if attn_visual == True:
                attn_dist = x_first
            x_first = torch.softmax(x_first, dim=3)
            x_first = self.att_dropout(x_first)
            x_first = x_first.matmul(v[:,:,:blen,:])  # [b, h, q_len, attn]
            if q.shape[2] <= blen:
                if attn_visual == True:
                    # print("<blen attn_dist:", attn_dist.shape, attn_dist)
                    attn_dist = attn_dist[:,:,:q_len,:q_len]
                    # print("<blen attn_dist:", attn_dist.shape, attn_dist)
                x = x_first[:,:,:q_len,:]  # previous dot product as global attention
            else:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3])
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3])  # b, head_size, k_len, nblock, blen
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3])
                local_k = torch.cat([k[:, :, :-1], k[:, :, 1:]], 3)  # [batch, nheads, (nblocks - 1), blen * 2, depth]
                local_v = torch.cat([v[:, :, :-1], v[:, :, 1:]], 3)
                tail_q = q[:, :, 1:]
                x_tail = torch.matmul(tail_q, local_k.permute([0, 1, 2, 4, 3]))  # [b, h, q_len, k_len]
                # print("x shape:", x_tail.shape)
                # print("attention distribution before mask:", x_tail)
                ones = torch.ones(blen, 2*blen, dtype=torch.uint8, device=q.device)
                mask = torch.triu(ones, diagonal=blen+1).unsqueeze(0)  #tril
                mask = mask.type(torch.BoolTensor).to(q.device)
                # print("tail mask:", mask.shape, mask)
                x_tail.masked_fill_(mask.unsqueeze(1), -1e9)  # mask when mask val is 0
                # print("attention distribution after mask:", x_tail.shape, x_tail)
                # print("x_tail:", x_tail.shape, x_tail)

                if attn_visual == True:
                    # print(">blen attn_dist first:", attn_dist.shape, attn_dist)
                    pad_first = (0, (-self.block_length) % q_len)  # Append to multiple of block length
                    attn_dist = F.pad(attn_dist, pad_first)
                    # print(">blen attn_dist first:", attn_dist.shape, attn_dist)
                    attn_dist_cat = torch.zeros(attn_dist.shape[0], attn_dist.shape[1], q_len, q_len, dtype=attn_dist.dtype, device=attn_dist.device)
                    attn_dist_cat[:,:,:self.block_length,:] = attn_dist
                    # print("attn_dist_cat:", attn_dist_cat.shape, attn_dist_cat[:,:,:self.block_length+2])

                    for block_i in range(x_tail.shape[2]-1):
                        attn_dist = x_tail[:,:,block_i]
                        pad_each = (self.block_length*block_i, (-self.block_length*(block_i+2)) % q_len)  # Append to multiple of block length
                        attn_dist = F.pad(attn_dist, pad_each)
                        # print("attn_dist:", block_i, attn_dist.shape, attn_dist)
                        attn_dist_cat[:, :, self.block_length*(block_i+1):self.block_length*(block_i+2), :] = attn_dist
                        # print(self.block_length*(block_i+1)+2)
                        # print("attn_dist_cat:", attn_dist_cat.shape, attn_dist_cat[:,:,:self.block_length*(block_i+1)+2])

                    attn_dist = x_tail[:, :, -1, :q_len%self.block_length, :self.block_length+q_len%self.block_length]
                    pad_tail = (self.block_length * (x_tail.shape[2]-1), 0)  # Append to multiple of block length
                    attn_dist = F.pad(attn_dist, pad_tail)
                    # print("attn_dist:", attn_dist.shape, attn_dist)
                    attn_dist_cat[:, :, self.block_length * x_tail.shape[2]:, :] = attn_dist
                    # torch.set_printoptions(profile="full")
                    # print("attn_dist_cat:", attn_dist_cat.shape, attn_dist_cat)
                          #attn_dist_cat[:, :, :self.block_length * x_tail.shape[2] + 2])
                    attn_dist = attn_dist_cat.masked_fill(attn_dist_cat == 0, -1e9)

                x_tail = torch.softmax(x_tail, dim=4)
                x_tail = self.att_dropout(x_tail)
                x_tail = x_tail.matmul(local_v)
                x_tail = x_tail.view(x_tail.shape[0], x_tail.shape[1], -1, x_tail.shape[4])
                x = torch.cat([x_first, x_tail], 2)
                x = x[:, :, :q_len, :]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)
        #print("x:", x.shape, x)
        #print("output of attention:", x.shape, x)

        assert x.size() == orig_q_size
        return x, attn_dist


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, head_size, att_type, block_length):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate, 1, "global", 0)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask, attn_visual):  # pylint: disable=arguments-differ
        # print("----------x:", x.shape)
        y = self.self_attention_norm(x)
        # print("----------y:", y.shape)
        y, attn_dist = self.self_attention(y, y, y, mask, None, attn_visual)
        # print("attn_dist:", attn_dist)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, attn_dist


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, head_size, att_type, block_length):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate, head_size, att_type, block_length)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate, 1, "global", 0)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, self_mask, i_mask, cache, attn_visual):
        y = self.self_attention_norm(x)
        y, attn_dist1 = self.self_attention(y, y, y, self_mask, None, attn_visual)
        y = self.self_attention_dropout(y)
        x = x + y

        attn_dist2 = None
        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y, attn_dist2 = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                       cache, attn_visual)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, attn_dist1, attn_dist2


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, head_size, att_type, block_length, img_size):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate, head_size, att_type, block_length)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.img_size = img_size

    def forward(self, inputs, mask, attn_visual, output_file):
        # print("inputs:", inputs.shape)
        encoder_output = inputs
        attn_dist_layers = []
        for enc_layer in self.layers:
            encoder_output, attn_dist = enc_layer(encoder_output, mask, attn_visual)
            # print("attn_visual:", attn_visual)
            # print("attn_dist:", attn_dist.shape, attn_dist)
            if attn_dist is not None:
                attn_dist = torch.softmax(attn_dist, dim=3)
            attn_dist_layers.append(attn_dist)
            # print("attn_dist:", attn_dist)

        if attn_visual == True:
            attn_dist_layers = torch.cat(attn_dist_layers)
            # np.savetxt('./attention_matrix_encoder.txt', attn_dist_layers.numpy())
            np.save(f'./results/ImageSize{self.img_size}/attention_matrix/attention_matrix_encoder{output_file}.npy', attn_dist_layers.cpu().numpy())
        return self.last_norm(encoder_output)


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, head_size, att_type, block_length, img_size):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate, head_size, att_type, block_length)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.img_size = img_size

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache, attn_visual, output_file):
        decoder_output = targets
        attn_dist1_layers = []
        attn_dist2_layers = []
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output, attn_dist1, attn_dist2 = dec_layer(decoder_output, enc_output,
                                       t_self_mask, i_mask, layer_cache, attn_visual)
            if attn_dist1 is not None:
                attn_dist1 = torch.softmax(attn_dist1, dim=3)
            if attn_dist2 is not None:
                attn_dist2 = torch.softmax(attn_dist2.transpose(2,3), dim=3)#.transpose(2,3)
            # print("attn_dist2:", attn_dist2.shape, attn_dist2)
            attn_dist1_layers.append(attn_dist1)
            attn_dist2_layers.append(attn_dist2)

        if attn_visual == True:
            attn_dist1_layers = torch.cat(attn_dist1_layers)
            attn_dist2_layers = torch.cat(attn_dist2_layers)
            # print("attn_dist2_layers:", attn_dist2_layers.shape, attn_dist2_layers)
            # np.savetxt('./attention_matrix_self_decoder.txt', attn_dist1_layers.numpy())
            # np.savetxt('./attention_matrix_encoder_decoder.txt', attn_dist2_layers.numpy())
            np.save(f'./results/ImageSize{self.img_size}/attention_matrix/attention_matrix_self_decoder{output_file}.npy', attn_dist1_layers.cpu().numpy())
            np.save(f'./results/ImageSize{self.img_size}/attention_matrix/attention_matrix_encoder_decoder{output_file}.npy', attn_dist2_layers.cpu().numpy())

        return self.last_norm(decoder_output)


class Transformer(nn.Module):
    def __init__(self,  # i_vocab_size, t_vocab_size,
                 dataset="Logo-2K+",
                 charvocab_size=100,
                 clusterN = 5,
                 n_layers=1,  # 6
                 hidden_size=64,  # 512
                 filter_size=128,  # 2048
                 dropout_rate=0.1,
                 head_size=1,
                 att_type="global",
                 block_length=0,
                 # share_target_embedding=False,  # True
                 has_inputs=True,
                 src_pad_idx=None,
                 trg_pad_idx=None,
                 distr="cat",
                 channels=3,
                 img_size=32):
        super(Transformer, self).__init__()

        self.dataset = dataset
        self.charvocab_size = charvocab_size
        self.clusterN = clusterN
        self.hidden_size = hidden_size
        self.att_type = att_type
        self.block_length = block_length
        self.emb_scale = hidden_size ** 0.5
        self.has_inputs = has_inputs
        #self.input_pad_idx = input_pad_idx
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.distr = distr
        self.channels = channels
        self.img_size = img_size

        #         self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        #         nn.init.normal_(self.t_vocab_embedding.weight, mean=0,
        #                         std=hidden_size**-0.5)
        #         self.t_emb_dropout = nn.Dropout(dropout_rate)
        self.t_embedding = nn.Embedding(NUM_PIXELS * self.channels, self.hidden_size)
        # nn.init.normal_(self.t_embedding.weight, mean=0,
        #                 std=hidden_size**-0.5)
        # self.t_emb_dropout = nn.Dropout(dropout_rate)
        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, n_layers, head_size, att_type, block_length, img_size)

        if has_inputs:
            #             if not share_target_embedding:
            #                 self.i_vocab_embedding = nn.Embedding(i_vocab_size,
            #                                                       hidden_size)
            #                 nn.init.normal_(self.i_vocab_embedding.weight, mean=0,
            #                                 std=hidden_size**-0.5)
            #             else:
            #                 self.i_vocab_embedding = self.t_vocab_embedding

            #             self.i_emb_dropout = nn.Dropout(dropout_rate)
            self.i_embedding = nn.Embedding(self.charvocab_size, self.hidden_size)
            self.encoder = Encoder(hidden_size, filter_size,
                                   dropout_rate, n_layers, head_size, att_type, block_length, img_size)
        if self.distr == "cat":  # Categorical
            self.output_layer = nn.Linear(hidden_size, NUM_PIXELS, bias=True)
            initialize_weight(self.output_layer)
        elif self.distr == "dmol":  # Discretized mixture of logistic, for ordinal valued inputs
            pass
        self.cluster_pred_layer = nn.Linear(self.img_size*self.img_size*self.channels*NUM_PIXELS, self.clusterN, bias=True)
        initialize_weight(self.cluster_pred_layer)

    def forward(self, inputs, targets, attn_visual, output_file):
        # print("inputs:", inputs.shape, inputs)
        enc_output, i_mask = None, None
        if self.has_inputs:
            # i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)
            # inputs already embedded, should have been [b, #kw] to create mask
            # i_mask = self.create_pad_mask(inputs, self.src_pad_idx, device=inputs.device)
            # i_mask = torch.ones(inputs.size()[0], inputs.size()[1], dtype=torch.uint8, device=inputs.device).unsqueeze(1)
            # i_mask = i_mask.type(torch.BoolTensor).to(inputs.device)
            # i_mask = self.create_pad_mask(inputs, self.src_pad_idx, device=inputs.device) #inputs: numerified inputs
            i_mask = (inputs == (torch.ones(inputs.shape).to(device=inputs.device)*self.src_pad_idx)).unsqueeze(-2) # for True/False value mask, mask when value is True
            # print("inputs device:", inputs.device)
            # print("i_mask device:", i_mask.device)
            # print("i_mask:", i_mask)
            # print("inputs:", inputs.shape)
            enc_output = self.encode(inputs, i_mask, attn_visual=False, output_file="None")

        # Reshape and Convert to indexes, and use separate embeddings for different channels
        targets = targets.view(targets.shape[0], targets.shape[1], targets.shape[2] * targets.shape[
            3])  # Flatten channels into width, target size: b, h, w, c->b, h, w*c
        # targets = (targets * (NUM_PIXELS - 1)).long()  # only if it is standardized?
        channel_addition = (torch.tensor([0, 1, 2]) * NUM_PIXELS).to(targets.device).repeat(targets.shape[2] // 3).view(
            1, 1, -1)
        targets_channel = targets + channel_addition

        target_size = targets_channel.size()[1] * targets_channel.size()[2]  # targets_channel.size(): b, h, w*c
        # t_mask = utils.create_pad_mask(targets, self.trg_pad_idx)
        # t_mask just used for masking its own embedding
        t_mask = self.create_pad_mask(targets_channel.view(targets_channel.shape[0], targets_channel.shape[1], targets_channel.shape[2]),
                                      self.trg_pad_idx, device=targets.device)
        # print("t_mask:", t_mask.shape, t_mask)
        # t_self_mask = utils.create_trg_self_mask(target_size,
        #                                          device=targets.device)
        if self.att_type == "global":
            t_self_mask = self.create_trg_self_mask(target_size,
                                                device=targets.device)
        elif self.att_type == "local_1d":
            t_self_mask = self.create_trg_self_mask(self.block_length,
                                                    device=targets.device)
        # print("t_self_mask:", t_self_mask)
        # return self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)
        dec_output = self.decode(targets_channel, enc_output, i_mask, t_self_mask, t_mask, attn_visual=attn_visual, output_file=output_file)
        # cluster_pred_output = self.pred(dec_output.view(dec_output.shape[0], -1))
        with open(f"./{self.dataset}/ImageSize{self.img_size}/{self.clusterN}Clusters/kmeans_model.pkl", "rb") as f:
            kmeans_model = pickle.load(f)
        cluster_pred_output = torch.LongTensor(kmeans_model.predict(torch.argmax(dec_output, dim=-1).view(dec_output.shape[0], -1).cpu().numpy())).to(inputs.device)
        # print("cluster_pred_output:", cluster_pred_output.shape, cluster_pred_output)
        cluster_pred_output = F.one_hot(cluster_pred_output, num_classes=self.clusterN).type(torch.FloatTensor).to(inputs.device)
        # print("cluster_pred_output:", cluster_pred_output.shape, cluster_pred_output)

        return dec_output, cluster_pred_output

    def encode(self, inputs, i_mask, attn_visual, output_file):
        # Input embedding
        # print("encoder inputs before embedding:", inputs.shape, inputs)
        input_embedded = self.i_embedding(inputs) #inputs: numerified inputs
        #print("encoder inputs after embedding:", input_embedded.shape, input_embedded)
        # print("i_mask:", i_mask.shape, i_mask)
        # print("i_mask.squeeze(1):", i_mask.squeeze(1))
        # print("i_mask.squeeze(1).unsqueeze(-1):", i_mask.squeeze(1).unsqueeze(-1))
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)
        # print("input_embedded:", input_embedded.shape, input_embedded)
        # print("i_mask:", i_mask)
        # print("i_mask.squeeze(1).unsqueeze(-1):", i_mask.squeeze(1).unsqueeze(-1))
        # input_embedded.masked_fill_(i_mask, 0)  #(i_mask.squeeze(1).unsqueeze(-1), 0)
        input_embedded *= self.emb_scale
        input_embedded = self.add_position_encoding(input_embedded)
        # input_embedded = self.i_emb_dropout(input_embedded)

        return self.encoder(input_embedded, i_mask, attn_visual, output_file)

    def decode(self, targets_channel, enc_output, i_mask, t_self_mask, t_mask,
               cache=None, attn_visual=False, output_file="None"):
        # target embedding: target size: b, h, w*c->b, h, w*c, d
        # print("targets type:", type(targets))
        # print("targets:", targets)
        # print("encoder outputs:", enc_output.shape, enc_output)
        # print("decoder inputs before embedding:", targets_channel.shape, targets_channel)
        target_embedded = self.t_embedding(targets_channel)  # self.t_vocab_embedding(targets)
        # print("decoder inputs after embedding:", target_embedded.shape, target_embedded)
        target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)
        # print("decoder inputs after mask:", target_embedded.shape, target_embedded)

        # print("target_embedded shape:", target_embedded.shape)
        target_shape = target_embedded.shape
        target_embedded = target_embedded.view(target_shape[0], target_shape[1] * target_shape[2], target_shape[3])
        #target_embedded.masked_fill_(t_mask, 0) #(t_mask.squeeze(1).unsqueeze(-1), 0)

        # Shifting
        # print("target_embedded shape", target_embedded.shape)
        # print("target_embedded", target_embedded)
        target_embedded = target_embedded[:, :-1, :]
        # print("after shift target_embedded", target_embedded)
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))
        # print("after pad target_embedded", target_embedded)

        target_embedded = target_embedded.view(target_shape)  # b, h, w*c, d
        target_embedded *= self.emb_scale
        target_embedded = self.add_position_encoding(target_embedded)

        target_shape = target_embedded.shape
        target_embedded = target_embedded.view(target_shape[0], -1, target_shape[3])  # b, h*w*c, d
        # target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(target_embedded, enc_output, i_mask,
                                      t_self_mask, cache, attn_visual, output_file)  # size=[b, pixelN, d]
        # linear: can use reverse embedding or linear
        output = self.output_layer(decoder_output)  # torch.matmul(decoder_output,
        #                       self.t_embedding.weight.transpose(0, 1))
        output = output.view(target_shape[:3] + (-1,))  # b, h, w*c, NP
        output = output.view(output.shape[0], output.shape[1], output.shape[2] // self.channels, self.channels,
                             output.shape[3])  # b, h, w, c, NP
        # print("decoder outputs:", output.shape, output)
        return output

    def pred(self, dec_output):
        #print("dec_output shape:", dec_output.shape)
        return self.cluster_pred_layer(dec_output)

    def add_position_encoding(self, x):
        # For positional encoding
        num_dims = len(x.shape) - 2  # 2 corresponds to batch and hidden_size dimensions
        num_timescales = self.hidden_size // (num_dims * 2)
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32, device=x.device) *
            -log_timescale_increment)
        #self.register_buffer('inv_timescales', inv_timescales)

        # timing signal
        for dim in range(num_dims):
            max_length = x.size()[dim + 1]  # add 1 to exclude batch dim
            position = torch.arange(max_length, dtype=torch.float32,
                                    device=x.device)
            scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                               dim=1)
            prepad = dim * 2 * num_timescales
            postpad = self.hidden_size - (dim + 1) * 2 * num_timescales
            signal = F.pad(signal, (prepad, postpad))
            for _ in range(1 + dim):
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-2)
            # print("x shape:", x.shape)
            # print("signal shape:", signal.shape)
            # print("dim:", dim, "; signal:", signal)
            x += signal
            # signal = signal.view(1, max_length, self.hidden_size)
        return x

    def create_pad_mask(self, t, pad, device=None):
        # print("t:", t)
        # print("pad:", pad)
        # print("t == pad:", t == pad)
        #mask = (t == pad).unsqueeze(-2)
        mask = torch.zeros(t.size(), dtype=torch.uint8, device=device)  # not ones here, because we use bool
        #print("create mask shape:", mask.shape)
        mask = mask.type(torch.BoolTensor).to(device)
        return mask.unsqueeze(1)

    def create_trg_self_mask(self, target_len, device=None):
        # Prevent leftward information flow in self-attention.
        ones = torch.ones(target_len, target_len, dtype=torch.uint8, device=device)
        t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0) #tril
        t_self_mask = t_self_mask.type(torch.BoolTensor).to(device)
        return t_self_mask

    def img_ce_loss(self, preds, targets):
        if self.distr == "dmol":
            return torch.zeros((1,))
        elif self.distr == "cat":
            # targets = (targets * (NUM_PIXELS - 1)).long()
            # print("preds:", preds.shape, preds[0][0][0][0])  # b, h, w, c, 256
            preds_argmax = torch.argmax(preds, dim=-1)

            # print("preds_argmax:", preds_argmax.shape, preds_argmax[0][0][0])#, preds_argmax)
            # print("targets shape:", targets.shape, targets[0][0][0])  # b, h, w, c
            # channel_dist = torch.cdist(preds_argmax.type(torch.FloatTensor), targets.type(torch.FloatTensor), p=2)
            channel_dist = (preds_argmax - targets).pow(2).sum(-1).sqrt() / NUM_PIXELS
            # print("channel_dist shape:", channel_dist.shape, channel_dist[0][0])  # b, h, w, c
            # channel_dist = torch.nn.functional.relu(channel_dist - 0.1)
            epsilon = 0.1
            zero = torch.zeros(channel_dist.shape, requires_grad=True).to(preds.device)
            channel_dist = torch.where(channel_dist > epsilon, channel_dist, zero)
            loss = channel_dist.view(channel_dist.shape[0], -1).sum(1)
            # print("channel_dist shape:", channel_dist.shape, channel_dist)  # b, h, w, c
            # print("loss", loss.shape, loss)

            # print("max targets:", targets.max())
            #if targets.max() > 256:
            #    print(targets)
            # channel_addition = (torch.tensor([0, 1, 2]) * NUM_PIXELS).to(targets.device).repeat(
            #     targets.shape[2] // 3).view(1, 1, -1)
            # targets -= channel_addition
            #print("after max targets:", targets.max())
            #print("after targets:", targets)
            # print("preds.permute(0, 4, 1, 2, 3):", preds.permute(0, 4, 1, 2, 3))
            # print("targets:", targets)
            '''img_ce = F.cross_entropy(preds.permute(0, 4, 1, 2, 3), targets, reduction='none')
            # print("img_ce shape", img_ce.shape)  # b, h, w, c
            print("img_ce", img_ce)
            # print("img_ce.view(img_ce.shape[0], -1):", img_ce.view(img_ce.shape[0], -1))
            img_ce = img_ce.view(img_ce.shape[0], -1).sum(1)
            loss = img_ce / (np.log(2.) * (self.img_size ** 2 * self.channels))  # loss_weight * img_ce - (1 - loss_weight) * kw_dist
            print("loss", loss.shape, loss)'''
            # print("img_ce", img_ce)
            # img_ce = img_ce.mean(0)
            # print("after img_ce shape", img_ce)
            # print("keywords shape", keywords.shape)
            # print("keywords reshape shape", keywords.view(keywords.shape[0], keywords.shape[1]//self.hidden_size, self.hidden_size).shape)
            # print("inputs shape", inputs.shape)
            # kw_dist = self.kw_distance(keywords, inputs)
            # print("kw_dist", kw_dist)
            #print("loss_weight * img_ce + (1 - loss_weight) * kw_dist:", loss_weight * img_ce + (1 - loss_weight) * kw_dist)
            return loss

    def cluster_ce_loss(self, preds, clusters):
        cluster_ce = F.cross_entropy(preds, clusters, reduction='none')
        #print("cluster_ce:", cluster_ce)
        return cluster_ce

    # Assumes targets have been rescaled to [-1., 1.]
    def loss(self, img_ce, cluster_ce, loss_weight):
        return loss_weight * img_ce + (1 - loss_weight) * cluster_ce

    def pixel_accuracy(self, preds, targets):
        if self.distr == "cat":
            # targets = (targets * (NUM_PIXELS - 1)).long()
            argmax_preds = torch.argmax(preds, dim=-1)
            acc = torch.eq(argmax_preds, targets).float().sum() / np.prod(argmax_preds.shape)
            return acc
        else:
            # Computing accuracy for dmol is more computationally intensive, so we skip it
            return torch.zeros((1,))

    def cluster_accuracy(self, preds, clusters):
        argmax_preds = torch.argmax(preds, dim=-1)
        acc = torch.eq(argmax_preds, clusters).float().sum() / np.prod(argmax_preds.shape)
        return acc

    def kw_distance(self, keywords, inputs):
        # print("keywords shape:", keywords.shape)
        # print("inputs shape:", inputs.shape)
        # print("torch.mean(keywords, 1):", torch.mean(keywords, 1))
        #print("cosine_similarity(torch.mean(keywords, 1), torch.mean(inputs, 1)):", cosine_similarity(torch.mean(keywords, 1), torch.mean(inputs, 1)))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # print("cos(torch.mean(keywords, 1), torch.mean(inputs, 1)):", cos(torch.mean(keywords, 1), torch.mean(inputs, 1)))
        # print("keywords:", keywords)
        # print("inputs:", inputs)
        keywords = keywords.view(keywords.shape[0], keywords.shape[1] // self.hidden_size, self.hidden_size)
        return cos(torch.mean(keywords, 1), torch.mean(inputs, 1))

class Generator(nn.Module):
    def __init__(self, inferencing_model):
        super(Generator, self).__init__()
        self.transformer = inferencing_model

    def __call__(self, input_data, device, initial_pixel):

        input_data = input_data.unsqueeze(0)  # b, charN
        # print("input_data:", input_data.shape, input_data)
        #print("----------keywords:", keywords)
        # print("self.transformer weight:", self.transformer.state_dict()["keyword_pred_layer.weight"])

        i_mask = torch.zeros(input_data.shape[0],input_data.shape[1], dtype=torch.uint8, device=device)#.unsqueeze(1)
        i_mask = i_mask.type(torch.BoolTensor).to(device)
        #i_mask = self.transformer.create_pad_mask([keywords.shape[0],keywords.shape[1]], self.transformer.trg_pad_idx, device=device).unsqueeze(1)
        t_self_mask, t_mask = None, None
        enc_output = self.transformer.encode(input_data, i_mask, attn_visual=False, output_file="None")
        img_pred, cluster_pred = self.decode(enc_output, i_mask, t_self_mask, t_mask, device, initial_pixel, attn_visual=False, output_file="None")
        return img_pred, cluster_pred


    '''def decode(self, enc_output, i_mask, t_self_mask, t_mask,
               device, initial_pixel, cache=None, attn_visual=False):
        outputs = []
        img_pred = []
        decoder_input = torch.zeros(self.transformer.hidden_size).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # print("decoder_input:", decoder_input.shape, decoder_input)
        # initial_pixel = initial_pixel.view(initial_pixel.shape[0], initial_pixel.shape[1]*initial_pixel.shape[2])  # h, w*c
        # initial_pixel = initial_pixel.unsqueeze(0) # b, h, w*c
        # channel_addition = (torch.tensor([0, 1, 2]) * NUM_PIXELS).to(initial_pixel.device).repeat(initial_pixel.shape[2] // 3).view(
        #     1, 1, -1)
        # initial_pixel_channel = initial_pixel # + channel_addition, first channel, no need for addition
        # initial_pixel_embedded = self.transformer.t_embedding(initial_pixel_channel)
        # decoder_input = initial_pixel_embedded
        # print("new decoder_input:", decoder_input.shape, decoder_input)

        #print("--------decoder_input:", deco
        # der_input.shape)
        # decoder
        for i in range(self.transformer.img_size**2 * self.transformer.channels):
            # height = math.ceil(math.ceil(decoder_input.shape[2]/self.transformer.channels)/self.transformer.img_size)
            pad_number = self.transformer.img_size*self.transformer.img_size*self.transformer.channels-decoder_input.shape[2]
            # print("height:", height)
            # print("pad_number:", pad_number)
            #decoder_input_pos = F.pad(decoder_input, (0, 0, 1, 0))
            decoder_input_pos = torch.cat([decoder_input, torch.zeros(pad_number,self.transformer.hidden_size).to(device).unsqueeze(0).unsqueeze(0)], dim=2)
            decoder_input_pos = decoder_input_pos.view(decoder_input_pos.shape[0], self.transformer.img_size, self.transformer.img_size*self.transformer.channels, decoder_input_pos.shape[-1])
            # print("decoder_input_pos:", decoder_input_pos.shape, decoder_input_pos)
            decoder_input_pos = decoder_input_pos*self.transformer.emb_scale
            # print("decoder_input scale:", decoder_input_pos.shape, decoder_input_pos)
            decoder_input_pos = self.transformer.add_position_encoding(decoder_input_pos)
            # print("decoder_input pos:", decoder_input_pos.shape, decoder_input_pos)
            decoder_input_pos = decoder_input_pos.view(decoder_input_pos.shape[0], -1,
                                                        decoder_input_pos.shape[3])

            t_mask0 = torch.zeros(decoder_input_pos.shape[0], decoder_input.shape[2], dtype=torch.uint8, device=device).unsqueeze(1)
            t_mask1 = torch.ones(decoder_input_pos.shape[0], pad_number, dtype=torch.uint8,
                                 device=device).unsqueeze(1)
            # print("t_mask0:", t_mask0.shape, t_mask0)
            # print("t_mask1:", t_mask1.shape, t_mask1)
            t_mask = torch.cat([t_mask0, t_mask1], dim=2)
            t_mask = t_mask.type(torch.BoolTensor).to(device)

            # print("t_mask:", t_mask.shape, t_mask)
            # print("t_mask.squeeze(1).unsqueeze(-1):", t_mask.squeeze(1).unsqueeze(-1).shape, t_mask.squeeze(1).unsqueeze(-1))
            decoder_input_pos.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)
            # print("decoder input after mask:", decoder_input_pos.shape, decoder_input_pos)
            if self.transformer.att_type == "global":
                # t_self_mask = torch.ones(decoder_input.shape[0], decoder_input.shape[1], dtype=torch.uint8, device=device).unsqueeze(1)
                # t_self_mask = t_self_mask.type(torch.BoolTensor).to(device)
                t_self_mask = self.transformer.create_trg_self_mask(decoder_input_pos.shape[1], device=device)
                # print("t_self_mask:", t_self_mask)
            elif self.transformer.att_type == "local_1d":
                # t_self_mask = torch.ones(decoder_input.shape[0], self.transformer.block_length, dtype=torch.uint8, device=device).unsqueeze(1)
                # t_self_mask = t_self_mask.type(torch.BoolTensor).to(device)
                t_self_mask = self.transformer.create_trg_self_mask(self.transformer.block_length, device=device)
                # print("t_self_mask:", t_self_mask)
            # print("t_mask:", t_mask)
            # print("t_self_mask:", t_self_mask)
            decoder_output = self.transformer.decoder(decoder_input_pos, enc_output, i_mask,
                                      t_self_mask, cache)  # size=[b, pixelN, d]
            # print("decoder_output:", decoder_output.shape, decoder_output)
            #decoder_input = decoder_output #torch.cat([decoder_input, decoder_output], dim=1)

            decoder_output = decoder_output[0][decoder_input.shape[2]-1].unsqueeze(0).unsqueeze(0)
            # print("one decoder_output:", decoder_output.shape, decoder_output)
            # decoder_input = torch.cat([decoder_input, decoder_output], dim=1)
            # print("decoder_input:", decoder_input.shape)
            output = self.transformer.output_layer(decoder_output)  # hidden_size->256
            # print("output:", output.shape, output)  # b, 1, 256
            outputs.append(output)
            # print("torch.softmax(output.squeeze(0).squeeze(0)):", torch.softmax(output, dim=2).squeeze(0).squeeze(0))
            pred_pixel_channel = torch.multinomial(torch.softmax(output, dim=2).squeeze(0).squeeze(0), 1).squeeze(0) #torch.argmax(output)
            # print("pred_pixel_channel:", pred_pixel_channel)
            pred_pixel_embedded = self.transformer.t_embedding(pred_pixel_channel.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            decoder_input = torch.cat([decoder_input, pred_pixel_embedded], dim=2)
            # print("decoder_input:", decoder_input.shape, decoder_input)

            img_pred.append(pred_pixel_channel)
            # print("output:", output.shape)
            #print("outputs:", outputs)
            # print("img_pred:", img_pred)

            # if i>3:
            #     print(kfsk)

        outputs = torch.cat(outputs)
        #print("outputs:", outputs.shape)
        outputs = outputs.view(self.transformer.img_size*self.transformer.img_size*self.transformer.channels*NUM_PIXELS)  # b, h, w, c
        img_pred = torch.IntTensor(np.array(img_pred))
        #print("pixels:", pixels)
        img_pred = img_pred.view(self.transformer.img_size, self.transformer.img_size, self.transformer.channels)  # b, h, w, c
        # print("img_pred:", img_pred)
        # cluster_pred = self.transformer.cluster_pred_layer(outputs.unsqueeze(0))[0]  #kwN, hidden_size
        
        with open(f"./{self.dataset}/ImageSize{self.img_size}/kmeans_model.pkl", "rb") as f:
            kmeans_model = pickle.load(f)
        img_pred = img_pred.unsqueeze(0)
        cluster_pred_output = torch.LongTensor(kmeans_model.predict(img_pred.view(img_pred.shape[0], -1))).to(device)
        cluster_pred_output = F.one_hot(cluster_pred_output, num_classes=self.transformer.clusterN).type(torch.FloatTensor).to(device)

        return img_pred.squeeze(0), cluster_pred_output'''

    def decode(self, enc_output, i_mask, t_self_mask, t_mask,
               device, initial_pixel, cache=None, attn_visual=False, output_file="None"):
        outputs = []
        img_pred = []
        decoder_input = torch.zeros(self.transformer.hidden_size).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # print("decoder_input:", decoder_input.shape, decoder_input)
        # initial_pixel = initial_pixel.view(initial_pixel.shape[0], initial_pixel.shape[1]*initial_pixel.shape[2])  # h, w*c
        # initial_pixel = initial_pixel.unsqueeze(0) # b, h, w*c
        # channel_addition = (torch.tensor([0, 1, 2]) * NUM_PIXELS).to(initial_pixel.device).repeat(initial_pixel.shape[2] // 3).view(
        #     1, 1, -1)
        # initial_pixel_channel = initial_pixel # + channel_addition, first channel, no need for addition
        # initial_pixel_embedded = self.transformer.t_embedding(initial_pixel_channel)
        # decoder_input = initial_pixel_embedded
        # print("new decoder_input:", decoder_input.shape, decoder_input)

        #print("--------decoder_input:", deco
        # der_input.shape)
        # decoder
        for i in range(self.transformer.img_size**2 * self.transformer.channels):
            # height = math.ceil(math.ceil(decoder_input.shape[2]/self.transformer.channels)/self.transformer.img_size)
            pad_number = self.transformer.img_size*self.transformer.img_size*self.transformer.channels-decoder_input.shape[2]
            # print("height:", height)
            # print("pad_number:", pad_number)
            #decoder_input_pos = F.pad(decoder_input, (0, 0, 1, 0))
            decoder_input_pos = torch.cat([decoder_input, torch.zeros(pad_number,self.transformer.hidden_size).to(device).unsqueeze(0).unsqueeze(0)], dim=2)
            decoder_input_pos = decoder_input_pos.view(decoder_input_pos.shape[0], self.transformer.img_size, self.transformer.img_size*self.transformer.channels, decoder_input_pos.shape[-1])
            # print("decoder_input_pos:", decoder_input_pos.shape, decoder_input_pos)
            decoder_input_pos = decoder_input_pos*self.transformer.emb_scale
            # print("decoder_input scale:", decoder_input_pos.shape, decoder_input_pos)
            decoder_input_pos = self.transformer.add_position_encoding(decoder_input_pos)
            # print("decoder_input pos:", decoder_input_pos.shape, decoder_input_pos)
            decoder_input_pos = decoder_input_pos.view(decoder_input_pos.shape[0], -1,
                                                        decoder_input_pos.shape[3])

            t_mask0 = torch.zeros(decoder_input_pos.shape[0], decoder_input.shape[2], dtype=torch.uint8, device=device).unsqueeze(1)
            t_mask1 = torch.ones(decoder_input_pos.shape[0], pad_number, dtype=torch.uint8,
                                 device=device).unsqueeze(1)
            # print("t_mask0:", t_mask0.shape, t_mask0)
            # print("t_mask1:", t_mask1.shape, t_mask1)
            t_mask = torch.cat([t_mask0, t_mask1], dim=2)
            t_mask = t_mask.type(torch.BoolTensor).to(device)

            # print("t_mask:", t_mask.shape, t_mask)
            # print("t_mask.squeeze(1).unsqueeze(-1):", t_mask.squeeze(1).unsqueeze(-1).shape, t_mask.squeeze(1).unsqueeze(-1))
            decoder_input_pos.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)
            # print("decoder input after mask:", decoder_input_pos.shape, decoder_input_pos)
            if self.transformer.att_type == "global":
                # t_self_mask = torch.ones(decoder_input.shape[0], decoder_input.shape[1], dtype=torch.uint8, device=device).unsqueeze(1)
                # t_self_mask = t_self_mask.type(torch.BoolTensor).to(device)
                t_self_mask = self.transformer.create_trg_self_mask(decoder_input_pos.shape[1], device=device)
                # print("t_self_mask:", t_self_mask)
            elif self.transformer.att_type == "local_1d":
                # t_self_mask = torch.ones(decoder_input.shape[0], self.transformer.block_length, dtype=torch.uint8, device=device).unsqueeze(1)
                # t_self_mask = t_self_mask.type(torch.BoolTensor).to(device)
                t_self_mask = self.transformer.create_trg_self_mask(self.transformer.block_length, device=device)
                # print("t_self_mask:", t_self_mask)
            # print("t_mask:", t_mask)
            # print("t_self_mask:", t_self_mask)
            decoder_output = self.transformer.decoder(decoder_input_pos, enc_output, i_mask,
                                      t_self_mask, cache, attn_visual, output_file)  # size=[b, pixelN, d]
            # print("decoder_output:", decoder_output.shape, decoder_output)
            #decoder_input = decoder_output #torch.cat([decoder_input, decoder_output], dim=1)

            decoder_output = decoder_output.squeeze(0)[decoder_input.shape[2]-1].unsqueeze(0).unsqueeze(0)
            # print("one decoder_output:", decoder_output.shape, decoder_output)
            # print("decoder_input:", decoder_input.shape)
            decoder_input = torch.cat([decoder_input, decoder_output.unsqueeze(0)], dim=2)
            # print("decoder_input:", decoder_input.shape)
            output = self.transformer.output_layer(decoder_output)  # hidden_size->256
            # print("output:", output.shape, output)  # b, 1, 256
            outputs.append(output)
            # print("torch.softmax(output.squeeze(0).squeeze(0)):", torch.softmax(output, dim=2).squeeze(0).squeeze(0))
            pred_pixel_channel = torch.argmax(output) #torch.multinomial(torch.softmax(output, dim=2).squeeze(0).squeeze(0), 1).squeeze(0) #torch.argmax(output)
            # print("pred_pixel_channel:", pred_pixel_channel)
            # pred_pixel_embedded = self.transformer.t_embedding(pred_pixel_channel.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            # decoder_input = torch.cat([decoder_input, pred_pixel_embedded], dim=2)
            # print("decoder_input:", decoder_input.shape, decoder_input)

            img_pred.append(pred_pixel_channel)
            # print("output:", output.shape)
            #print("outputs:", outputs)
            # print("img_pred:", img_pred)

            # if i>3:
            #     print(kfsk)

        outputs = torch.cat(outputs)
        #print("outputs:", outputs.shape)
        outputs = outputs.view(self.transformer.img_size*self.transformer.img_size*self.transformer.channels*NUM_PIXELS)  # b, h, w, c
        img_pred = torch.IntTensor(img_pred).to(device)
        # print("img_pred:", img_pred)
        img_pred = img_pred.view(self.transformer.img_size, self.transformer.img_size, self.transformer.channels)  # b, h, w, c
        # print("img_pred:", img_pred)
        # cluster_pred = self.transformer.cluster_pred_layer(outputs.unsqueeze(0))[0]  #kwN, hidden_size

        with open(f"./{self.transformer.dataset}/ImageSize{self.transformer.img_size}/{self.transformer.clusterN}Clusters/kmeans_model.pkl", "rb") as f:
            kmeans_model = pickle.load(f)
        img_pred = img_pred.unsqueeze(0)
        cluster_pred_output = torch.LongTensor(kmeans_model.predict(img_pred.view(img_pred.shape[0], -1).cpu().numpy())).to(device)
        cluster_pred_output = F.one_hot(cluster_pred_output, num_classes=self.transformer.clusterN).type(torch.FloatTensor).to(device)

        return img_pred.squeeze(0), cluster_pred_output
