from torch import nn
from torch.nn import functional as F
from nn_layers import convolutional_module


class Encoder(nn.Module):
    """This is the encoder part of tacotron2. It includes a stack of three 1d convolutional layers
    followed by batch normalization and ReLU activations, and a bidirectional LSTM layer.
    These part encodes sequences of input characters."""
    def __init__(self, encoder_params):
        super(Encoder, self).__init__()

        stack_of_convolutions = []
        for _ in range(encoder_params['encoder_convs']):
            conv_layer = nn.Sequential(convolutional_module(encoder_params['symbols_embedding_length'],
                                                            encoder_params['symbols_embedding_length'],
                                                            kernel_size=encoder_params['conv_kernel_size'],
                                                            stride=encoder_params['conv_stride'],
                                                            padding=int((encoder_params['conv_kernel_size'] - 1) / 2),
                                                            dilation=encoder_params['conv_dilation'],
                                                            w_init_gain=encoder_params['w_init_gain']),
                                       nn.BatchNorm1d(encoder_params['symbols_embedding_length']))
            stack_of_convolutions.append(conv_layer)
        self.stack_conv = nn.ModuleList(stack_of_convolutions)

        self.bi_lstm = nn.LSTM(encoder_params['symbols_embedding_length'],
                               int(encoder_params['symbols_embedding_length'] / 2), 1, batch_first=True,
                               bidirectional=True)

    def forward(self, input_sequences, input_lengths):
            for conv in self.stack_conv:
                input_sequences = F.dropout(F.relu(conv(input_sequences)), 0.5, self.training)

            input_sequences = input_sequences.transpose(1, 2)
            # After convolution filters, is the original sequence length the same? CHECK IT OUT
            input_lengths = input_lengths.cpu().numpy()
            # Returns a packed sequence object with variable-length sequences before passing through BiLSTM layer
            input_sequences = nn.utils.rnn.pack_padded_sequence(input_sequences, input_lengths, batch_first=True)
            self.bi_lstm.flatten_parameters()
            outputs, _ = self.bi_lstm(input_sequences)
            # Pads again the tensor back to normal format before packing
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            return outputs  # [N, Max_seq_length, E_length]

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs
