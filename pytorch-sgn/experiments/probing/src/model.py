import torch
from torch import nn
import torch.nn.functional as F

class ListMax(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(ListMax, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.LSTM = nn.LSTM(300, hidden_dim, 1, batch_first=True, bidirectional=True) # Need Implement
        self.FC = nn.Linear(hidden_dim*2, 5) # Need Implement

    def forward(self, input):

        x = input.transpose(0,1) # bz x 5
        embeds = self.embedding(x) # bz x 5 x embed_dim
        output, (hn, cn) = self.LSTM(embeds) # 2 x bz x hidden_dim
        DIR, BZ, H = hn.size()
        hidden = hn.transpose(0, 1).reshape(BZ,  H * DIR) # bz x (2 * hidden_dim)
        logits = self.FC(hidden) # 2 x bz x 5

        return logits.squeeze() # bz x 5


class CNN(nn.Module):
    def __init__(self,  out_channels=16, pretrained=None ):
        super(CNN, self).__init__()

        self.output_size = 5
        self.embedding_dim = 300
        self.kernel_heights = [3, 4, 5]
        self.in_channels = 1
        self.out_channels = out_channels

        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[0], self.embedding_dim))
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[1], self.embedding_dim))
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[2], self.embedding_dim))

        self.dropout = nn.Dropout()
        self.label = nn.Linear(len(self.kernel_heights) * self.out_channels, self.output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input_seqs):
        input = self.embedding(input_seqs).transpose(0,1)
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)

        return logits

import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




class TransformerModel(nn.Module):

    def __init__(self, pretrained, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(300, dropout)
        encoder_layers = TransformerEncoderLayer(300, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = nn.Embedding.from_pretrained(pretrained)


        self.ninp = 300
        self.decoder = nn.Linear(300, 5)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = output.mean(dim=0)
        return output


class MLP(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.FC1 = nn.Linear(2*300, hidden_dim) # Need Implement
        self.FC2 = nn.Linear(hidden_dim, 1) # Need Implement
        self.RELU = nn.ReLU()

    def forward(self, input):
        x = input.transpose(0,1) # bz x 2
        BZ, SEQ = x.size()
        embeds = self.embedding(x) # bz x 2 x embed_dim
        embeds = embeds.view(BZ, -1) # bz x (2*embed_dim)
        out = self.FC1(embeds)
        out = self.RELU(out)
        out = self.FC2(out)

        return out.squeeze() # bz x 5


class MLP3Diff(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(MLP3Diff, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.FC1 = nn.Linear(2*300, hidden_dim) # Need Implement
        self.FC2 = nn.Linear(hidden_dim, 50) # Need Implement
        self.FC3 = nn.Linear(50, 1) # Need Implement
        self.RELU = nn.ReLU()

    def forward(self, input):
        x = input.transpose(0,1) # bz x 2
        BZ, SEQ = x.size()
        embeds = self.embedding(x) # bz x 2 x embed_dim
        embeds = embeds.view(BZ, -1) # bz x (2*embed_dim)
        out = self.FC1(embeds)
        out = self.RELU(out)
        out = self.FC2(out)
        out = self.RELU(out)
        out = self.FC3(out)

        return out.squeeze() # bz x 5

class BiLinearDiff1(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(BiLinearDiff1, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.biliear = nn.Bilinear(300, 300, 1)

    def forward(self, input):
        x = input.transpose(0,1) # bz x 2
        embeds = self.embedding(x) # bz x 2 x embed_dim
        feature0 = embeds[:, 0, :]
        feature1 = embeds[:, 1, :]
        out = self.biliear(feature0, feature1)

        return out.squeeze() # bz x 5

class BiLinearDiffH(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(BiLinearDiffH, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.biliear = nn.Bilinear(300, 300, hidden_dim)
        self.FC = nn.Linear(hidden_dim, 1)
        self.RELU = nn.ReLU()

    def forward(self, input):
        x = input.transpose(0,1) # bz x 2
        embeds = self.embedding(x) # bz x 2 x embed_dim
        feature0 = embeds[:, 0, :]
        feature1 = embeds[:, 1, :]
        out = self.biliear(feature0, feature1) # bz x hidden
        out = self.RELU(out)
        out = self.FC(out)

        return out.squeeze() # bz x 5

class MLP1(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(MLP1, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.FC1 = nn.Linear(300, hidden_dim) # Need Implement
        self.FC2 = nn.Linear(hidden_dim, 50) # Need Implement
        self.FC3 = nn.Linear(50, 1) # Need Implement

        self.RELU = nn.ReLU()


    def forward(self, input):


        embeds = self.embedding(input) # bz x embed_dim
        out = self.FC1(embeds)
        out = self.RELU(out)
        out = self.FC2(out)
        out = self.RELU(out)
        out = self.FC3(out)

        return out.squeeze() # bz x 5

class MLP0(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(MLP0, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.FC1 = nn.Linear(300, hidden_dim) # Need Implement
        self.FC2 = nn.Linear(hidden_dim, 1) # Need Implement
        self.RELU = nn.ReLU()


    def forward(self, input):
        embeds = self.embedding(input) # bz x embed_dim
        out = self.FC1(embeds)
        out = self.RELU(out)
        out = self.FC2(out)

        return out.squeeze() # bz x 5



class MLP00(nn.Module):

    def __init__(self, hidden_dim, pretrained):
        super(MLP00, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.FC1 = nn.Linear(300, 1) # Need Implement


    def forward(self, input):
        embeds = self.embedding(input) # bz x embed_dim
        out = self.FC1(embeds)

        return out.squeeze() # bz x 5
