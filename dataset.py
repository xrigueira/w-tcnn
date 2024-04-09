import os
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    
    """
    Dataset class used for transformer models.
    ----------
    Arguments:
    data: tensor, the entire train, validation or test data sequence before any slicing. 
    If univariate, data.size() will be [number of samples, number of variables] where the 
    number of variables will be equal to 1 + the number of exogenous variables. Number of 
    exogenous variables would be 0 if univariate.

    indices: a list of tuples. Each tuple has two elements:
        1) the start index of a sub-sequence
        2) the end index of a sub-sequence. 
        The sub-sequence is split into src, tgt and tgt_y later.  

    encoder_sequence_len: int, the desired length of the input sequence given to the the first layer of
        the transformer model.

    tgt_sequence_len: int, the desired length of the target sequence (the output of the model)

    tgt_idx: The index position of the target variable in data. Data is a 2D tensor
    """
    
    def __init__(self, data: torch.tensor, indices: list, encoder_sequence_len: int, decoder_sequence_len: int, tgt_sequence_len: int) -> None:
        super().__init__()

        self.data = data
        self.indices = indices
        self.encoder_sequence_len = encoder_sequence_len
        self.decoder_sequence_len = decoder_sequence_len
        self.tgt_sequence_len = tgt_sequence_len

        # print("From get_src_tgt: data size = {}".format(data.size()))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) tgt (the decoder input)
        3) tgt_y (the target)
        4) src_p (the encoder input for plotting)
        5) tgt_p (the target for plotting)
        """
        
        # Get the first element of the i'th tuple in the list self.indices
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]
        
        # print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, tgt, tgt_y, tgt_p = self._get_src_tgt(sequence=sequence, encoder_sequence_len=self.encoder_sequence_len,
            decoder_sequence_len=self.decoder_sequence_len, tgt_sequence_len=self.tgt_sequence_len)
        
        src_p = src # Used for plotting

        src = self._crush_src(src=src)

        return src, tgt, tgt_y, src_p, tgt_p
    
    def _get_src_tgt(self, sequence: torch.Tensor, encoder_sequence_len: int, decoder_sequence_len: int, 
                    tgt_sequence_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), tgt (decoder input) and tgt_y (the target)
        sequences from a sequence. 
        ----------
        Arguments:
        sequence: tensor, a 1D tensor of length n where  n = encoder input length + target 
            sequence length 
        encoder_sequence_len: int, the desired length of the input to the transformer encoder
        tgt_sequence_len: int, the desired length of the target sequence (the one against 
            which the model output is compared)

        Return: 
        src: tensor, 1D, used as input to the transformer model
        tgt: tensor, 1D, used as input to the transformer model
        tgt_y: tensor, 1D, the target sequence against which the model output is compared
            when computing loss. 
        """
        
        assert len(sequence) == encoder_sequence_len + tgt_sequence_len, "Sequence length does not equal (input length + target length)"
        
        # encoder input. Selects the last all variables excluding the last one (-1), which is the target
        src = sequence[:encoder_sequence_len, :-1]
        
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of tgt_y except the last (i.e. it must be shifted right by 1)
        tgt = sequence[encoder_sequence_len-1:len(sequence)-1, :-1] # Selects only target variable
        
        assert len(tgt) == tgt_sequence_len, "Length of tgt does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        tgt_y = sequence[-tgt_sequence_len:, -1:] # Select the target variable one step ahead
        
        assert len(tgt_y) == tgt_sequence_len, "Length of tgt_y does not match target sequence length"

        # the target sequence corresponding to the encoder input (src) in the same range for plotting purposes
        tgt_p = sequence[:encoder_sequence_len, -1:] # Select the target variable in the same range as the encoder input

        return src, tgt, tgt_y.squeeze(-1), tgt_p # change size from [batch_size, tgt_seq_len, num_features] to [batch_size, tgt_seq_len] 
        # tgt_y.squeeze(-1) is reverted in the test function with tgt_y.squeeze(2). Left as it is for now.
    
    def _crush_src(self, src):
        
        """Summarize the values of src.
        Finner density when we get closer to the prection point.
        and coarse when we are far away.
        ---------
        Arguments:
        src (pytorch.Tensor): input to the model.
        
        Returns:
        src (pytroch.Tensor): summarized input to the model.
        """
        
        num_years = 3 # Number of years with yearly average
        num_months = 8 # Number of months with monthly average
        num_weeks = 3*4 # Number of weeks with weakly average (#months * #weeks in a month)
        
        # Define start and end indixes
        end_idx_years = num_years * 365
        start_idx_months, end_idx_months = end_idx_years, num_years * 365 + num_months * 30
        start_idx_weeks, end_idx_weeks = end_idx_months, num_years * 365 + num_months * 30 + num_weeks * 7
        
        # Reshape the tensor to represent years, months, weeks, and days
        years_data = src[:end_idx_years].view(num_years, 365, -1).mean(dim=1)
        months_data = src[start_idx_months:end_idx_months].view(num_months, 30, -1).mean(dim=1)
        weeks_data = src[start_idx_weeks:end_idx_weeks].view(num_weeks, 7, -1).mean(dim=1)
        last_month_data = src[-30:].squeeze(0)  # Keep the last month as is
        
        # Concatenate the reduced data to create the final tensor
        src = torch.cat([years_data, months_data, weeks_data, last_month_data], dim=0)

        return src