from torch import nn
import torch
from torch.nn import functional as F

class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, config, pretrained_embed=None):
        super(BiGRU, self).__init__()
        self.config = config
        self.embedding_dim = embedding_dim # input feature number

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embed is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embeddings.requires_grad_(False) # Fix Embedding, No fine-tune

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=config.rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2*config.rnn_hidden_size, 8)

    def forward(self, input_seqs, input_lengths):
        embeds = self.embeddings(input_seqs) # B x L x D
        # RNN type
        packed = nn.utils.rnn.pack_padded_sequence(input=embeds,
                                                   lengths=input_lengths,
                                                   enforce_sorted=False,
                                                   batch_first=True)
        _, hidden = self.gru(packed) # dir x B x hidden_size
        dir, B, hidden_size = hidden.size()
        hidden = hidden.transpose(0,1).reshape(B, hidden_size*dir) # B x (hidden_size*dir)

        output = self.dropout(hidden)
        output = self.relu(output)
        output = self.fc(output)

        return output # B x cls(8)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, config, pretrained_embed=None, out_channels=16):
        super(CNN, self).__init__()
        self.batch_size = config.batch_sz
        self.output_size = 8
        self.embedding_dim = embedding_dim
        self.kernel_heights = [3, 4, 5]
        self.in_channels = 1
        self.out_channels = out_channels

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embed is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embeddings.requires_grad_(False) # Fix Embedding, No fine-tune


        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[0], self.embedding_dim))
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[1], self.embedding_dim))
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[2], self.embedding_dim))

        self.dropout = nn.Dropout(config.dropout_rate)
        self.label = nn.Linear(len(self.kernel_heights)*self.out_channels, self.output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
        
        return max_out

    def forward(self, input_seqs, input_lengths):
        input = self.embeddings(input_seqs)
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