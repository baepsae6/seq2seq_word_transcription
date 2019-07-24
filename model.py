import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self,input_dim, enc_embed_dim, hidden_dim, n_layers):
        super(Encoder,self).__init__()
        self.input_dim     = input_dim
        self.enc_embed_dim = enc_embed_dim
        self.hidden_dim    = hidden_dim
        self.n_layers      = n_layers
        
        self.embedding = nn.Embedding(input_dim, enc_embed_dim)
        self.LSTM      = nn.LSTM(enc_embed_dim, hidden_dim, n_layers, batch_first=True)
    
    def forward(self,input_seq):
        # input_seq = (32, seq_len) 
        # embedded = (32, seq_len, 256)
        embedded                = self.embedding(input_seq)
        outputs, (hidden, cell) = self.LSTM(embedded) 
        return hidden, cell

    
class Decoder(nn.Module):
    def __init__(self, output_dim, dec_embed_dim, hidden_dim, n_layers):
        super(Decoder,self).__init__()
        self.output_dim    = output_dim
        self.dec_embed_dim = dec_embed_dim
        self.hidden_dim    = hidden_dim
        self.n_layers      = n_layers

        self.embedding = nn.Embedding(output_dim, dec_embed_dim)
        self.LSTM      = nn.LSTM(dec_embed_dim, hidden_dim, n_layers, batch_first=True)
        self.out       = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, word, hidden, cell):
        # word       = word.unsqueeze(1)
        #  embedded  = (32, seq_len, 256)
        #     output = (1, 1, 64)
        # prediction = (1, 56)
        embedded               = self.embedding(word) 
        output, (hidden, cell) = self.LSTM(embedded, (hidden, cell))
        prediction             = self.out(output.squeeze(1)) 
        return prediction, hidden, cell
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
       
    def forward(self, word, transcription, teacher_forcing_ratio = 0.5):  
        hidden, cell         = self.encoder(word)
        output, hidden, cell = self.decoder(transcription, hidden, cell)
        return output
    
        # without teacher force:
        # batch_size = transcription.shape[0]
        # max_len    = transcription.shape[1]
        # transcription_vocab_size = self.decoder.output_dim
        # inp = transciption[:, 0]
        # tensor to store decoder outputs
        # outputs = torch.zeros(max_len, batch_size, transcription_vocab_size)
        # for t in range(1, max_len):
        # output, hidden = self.decoder(inp, hidden)         
        # outputs[t] = output
        # teacher_force = random.random() < teacher_forcing_ratio
        # top1 = output.max(1)[1]
        # inp = (transcription[:,t] if teacher_force else top1)

    def predict(self, word):
        # word = (1, seq_len)
        word         = word.unsqueeze(0) 
        hidden, cell = self.encoder(word)
        
        # predicted_transcription = (1, 1) where first dim is <sos> token
        predicted_transcription = torch.LongTensor([2]).to(device) 
        predicted_transcription = predicted_transcription.unsqueeze(0)
        preds = []
        for _ in range(20):
            # output, hidden = (1, 1, 256)
            output, hidden, cell = self.decoder(predicted_transcription, hidden, cell)
            output               = torch.argmax(output)
            predicted_char       = output.item()
            
            if predicted_char == 3:
                break
            preds.append(predicted_char)
            predicted_transcription = torch.LongTensor([predicted_char]).unsqueeze(0).to(device)
        return preds