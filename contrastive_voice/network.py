
import torch

class TDNN(torch.nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = torch.nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = torch.nn.ReLU()
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = torch.nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = torch.nn.functional.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layers_size = layers_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, layers_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        '''
            inputs: batch * dim * len
            output: batch * dim
        '''
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[2]
        assert self.input_size == inputs.shape[1], 'inputs dim should match input_size'
        if hidden is None:
            hidden=(torch.zeros(self.layers_size, batch_size, self.hidden_size), 
                    torch.zeros(self.layers_size, batch_size, self.hidden_size))
        inputs = inputs.view(seq_len, batch_size, self.input_size)
        output, hidden = self.lstm(inputs, hidden)
        # last output fc
        output = self.linear(output[-1].view(batch_size, -1))
        output = self.softmax(output)
        return output, hidden
