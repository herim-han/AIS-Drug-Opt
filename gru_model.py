from torch import nn

import torch
import math
from data import  sos_id, eos_id, pad_id

import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, n_layer=4, dropout =0.1, lr = 1e-4, kl_factor=0.03):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(vocab_size, seq_len, d_model, n_head, n_layer, dropout)
        self.decoder = Decoder(vocab_size, seq_len, d_model, n_head, n_layer, dropout)

        self.loss_f  = nn.NLLLoss(ignore_index=0)
        self.lr = lr

        self.n_layer = n_layer
        self.seq_len = seq_len
        self.list_test_output=[]

        self.kl_loss = lambda mu, logvar:  -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())* kl_factor
    def training_step(self, batch, batch_idx):
        src, trg, prop = batch
        src_inp, mu, logvar, properties = self.encoder(src, src ==pad_id , prop)
        loss2 = self.kl_loss(mu, logvar)

        feature = self.decoder.sampling(mu, logvar)
        feature = torch.cat((feature, properties), 1)
        output = self.decoder( src_inp, feature )

        loss1 = self.loss_f(output[:,:-1].reshape(-1, output.size(-1)) , trg[:,1:].reshape(-1) )
        loss = loss1+loss2
        self.log( "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size =  src.size(0), sync_dist=True)
        self.log( "train_loss1", loss1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size =  src.size(0), sync_dist=True)
        self.log( "train_loss2", loss2, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size =  src.size(0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg, prop = batch
        src_inp, mu, logvar, properties = self.encoder(src, src ==0, prop )

        feature = self.decoder.sampling(mu, logvar)
        feature = torch.cat((feature, properties), 1)
        # generate probability
        output = self.decoder(src_inp, feature)   # (B, L, d_model)
        # generate tokens 
        val_tokens = self.generate_tokens(feature, seq_len=src_inp.size(1) )

        loss1 = self.loss_f(output[:,:-1].reshape(-1, output.size(-1)) , trg[:,1:].reshape(-1) )
        loss2= self.kl_loss(mu, logvar)
        loss = loss1+loss2
        self.log( "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0), sync_dist=True )
        self.log( "valid_loss1", loss1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0) , sync_dist=True)
        self.log( "valid_loss2", loss2, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0), sync_dist=True )
        
        trg = trg*torch.cumprod( trg!=eos_id, dim=-1)
        val_tokens = val_tokens*torch.cumprod( val_tokens !=eos_id, dim=-1)
        acc = torch.all(trg == val_tokens, dim=-1).sum() / src.size(0)

        self.log( "valid_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0), sync_dist=True )

    def predict_step(self, batch, batch_idx):
        feature = batch[0]

        mu, logvar = torch.split(feature, feature.size(-1)//2, dim=-1 )
        feature = self.decoder.sampling(mu, logvar)

        mw_mean = torch.tensor([372.53]).to(feature.device)
        mw_std = torch.tensor([88.89]).to(feature.device)
        mw_prop = torch.tensor([400]).to(feature.device)

        properties = torch.stack([(mw_prop-mw_mean)/mw_std for i in range(feature.size(0))]).to(torch.float32)
        feature = torch.cat((feature, properties), dim=-1)
        return self.generate_tokens(feature)

    def generate_tokens(self, feature, seq_len=0):
        if len(feature.size())==2:
            feature = feature.unsqueeze(0).repeat(self.n_layer, 1, 1)
        elif len(feature.size())==3:
            pass
        else:
            assert RuntimeError ('Size of feature should be (batch_size, d_model) or (n_layer, batch_size, d_model) ')  
        seq_len = self.seq_len if seq_len==0 else seq_len
        output = torch.tensor( [sos_id], dtype=torch.long, device=feature.device).view(1,1).repeat(feature.size(1),1) # (B, 1) 

        for _ in range(1, seq_len):
            embed_tokens = self.encoder.src_embedding(output) # (B, L) => (B, L, d_model)
            embed_tokens = self.encoder.positional_encoder(embed_tokens) # (B, L, d_model) => (B, L, d_model)
                            # decode_single (B, vocab_size)
            new_token    = self.decoder.decode_single(embed_tokens, feature ).topk(1, dim=-1)[1]
            output       = torch.cat( [ output, new_token.view(-1, 1)], dim=1)

        mask = torch.cat ( [ torch.ones((output.size(0), 1), device=output.device, dtype=output.dtype ), 
                             torch.cumprod( output!=eos_id, dim=-1)[:,:-1] ], dim = -1 ) 

        output = output*mask
        #torch.cumprod( output!=eos_id, dim=-1)
        return output
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class Encoder(pl.LightningModule):
    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, n_layer=4, dropout =0.1):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.src_embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(seq_len, d_model)
        self.encoder = nn.TransformerEncoder( nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True), n_layer)

        self.readout1 = nn.Linear(d_model+1, d_model)#added prop layer
        self.readout2 = nn.Linear(d_model+1, d_model)#added prop layer

    def forward(self, src, e_mask=None, prop=None):
        src_inp = self.src_embedding(src) # (B, L) => (B, L, d_model)
        src_inp = self.positional_encoder(src_inp) # (B, L, d_model) => (B, L, d_model)
        e_output = self.encoder(src_inp, src_key_padding_mask=e_mask) # (B, L, d_model)
        mw_mean = torch.tensor([372.53]).to(src_inp.device)
        mw_std = torch.tensor([88.89]).to(src_inp.device)
        properties = ((prop-mw_mean)/mw_std).clone().detach().unsqueeze(-1).to(torch.float32)
        
        mu     = self.readout1(torch.cat((e_output[:,0], properties), 1))
        logvar = self.readout2(torch.cat((e_output[:,0], properties), 1))
        return src_inp, mu, logvar, properties # (B, n_layer, d_model)

    def predict_step(self, batch, batch_idx):
        src, e_mask, prop = batch
        src_inp, mu, logvar, properties = self.forward(src, e_mask, prop)
        return torch.cat([mu, logvar], dim=-1)

class Decoder(pl.LightningModule):
    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, n_layer=4, dropout =0.1):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        #self.decoder = nn.GRU( input_size=d_model, hidden_size=d_model, num_layers=n_layer, dropout=dropout, batch_first=True)
        self.decoder = nn.GRU( input_size=d_model, hidden_size=d_model+1, num_layers=n_layer, dropout=dropout, batch_first=True)
        self.output_linear = nn.Linear(d_model+1, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.d_model = d_model
        self.n_layer = n_layer

    def forward(self, src_inp, feature):
        if len(feature.size())==2:
            feature = feature.unsqueeze(0).repeat(self.n_layer, 1, 1)
        elif len(feature.size())==3:
            pass
        else:
            assert RuntimeError ('Size of feature should be (batch_size, d_model) or (n_layer, batch_size, d_model) ') 
        d_output, _ = self.decoder(src_inp, feature) # (B, L, d_model+1)
        output = self.softmax( self.output_linear(d_output) ) # (B, L, d_model+1) => # (B, L, trg_vocab_size)
        return output

    def decode_single(self, src_inp, feature ):
        # h (n_layer, B, d_model)
        assert feature.size(0) == self.n_layer

        d_output = self.decoder(src_inp, feature)[0] # (B, L, d_model)
        d_output =self.softmax( self.output_linear(d_output[:,-1]) ) # (B, d_model) => # (B, trg_vocab_size) 
        return d_output

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)        

class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        # Make initial positional encoding matrix with 0
        self.positional_encoding= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    self.positional_encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    self.positional_encoding[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        self.positional_encoding = self.positional_encoding.unsqueeze(0) # (1, L, d_model)
        #self.positional_encoding = self.positional_encoding.to(device=device).requires_grad_(False)
        self.positional_encoding = torch.nn.parameter.Parameter(self.positional_encoding, requires_grad=False)
        self.d_model = d_model

    def forward(self, x):
        #print(x.size(1), self.positional_encoding.size())
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding[:,:x.size(1)] # (B, L, d_model)
        #.to(x.device) # (B, L, d_model)
        #x = x + self.positional_encoding # (B, L, d_model)

        return x
