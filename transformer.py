import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Any, Union, Callable, Tuple
from torch import Tensor

class PositionalEncoder(nn.Module):
    
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    
    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    
    def __init__(self, dropout: float, max_seq_len: int, d_model: int, batch_first: bool=True) -> None:
        super().__init__()
        self.d_model = d_model # The dimension of the output of the sub-layers in the model
        self.dropout = nn.Dropout(p=dropout) 
        self.batch_first = batch_first
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # PE(pos,2i)   = sin(pos/(10000^(2i/d_model)))
        # PE(pos,2i+1) = cos(pos/(10000^(2i/d_model)))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
                        [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)
    
class customTransformerEncoderLayer(nn.Module):
    
    """
    This class implements a custom encoder layer made up of self-attention and feedforward network.
    
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. It has been modified to return
    the attention weights.
    ----------
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    
    Returns:
        x: encoder output.
        sa_weights: self-attention weights.
    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # Define objects to store custom weights
        self._sa_weights = None

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = nn._get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu
    
    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        # why_not_sparsity_fast_path = ''
        # if not src.dim() == 3:
        #     why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        # elif self.training:
        #     why_not_sparsity_fast_path = "training is enabled"
        # elif not self.self_attn.batch_first :
        #     why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        # elif not self.self_attn._qkv_same_embed_dim :
        #     why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        # elif not self.activation_relu_or_gelu:
        #     why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        # elif not (self.norm1.eps == self.norm2.eps):
        #     why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        # elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
        #     why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        # elif self.self_attn.num_heads % 2 == 1:
        #     why_not_sparsity_fast_path = "num_head is odd"
        # elif torch.is_autocast_enabled():
        #     why_not_sparsity_fast_path = "autocast is enabled"
        # if not why_not_sparsity_fast_path:
        #     tensor_args = (
        #         src,
        #         self.self_attn.in_proj_weight,
        #         self.self_attn.in_proj_bias,
        #         self.self_attn.out_proj.weight,
        #         self.self_attn.out_proj.bias,
        #         self.norm1.weight,
        #         self.norm1.bias,
        #         self.norm2.weight,
        #         self.norm2.bias,
        #         self.linear1.weight,
        #         self.linear1.bias,
        #         self.linear2.weight,
        #         self.linear2.bias,
        #     )

        #     # We have to use list comprehensions below because TorchScript does not support
        #     # generator expressions.
        #     _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
        #     if torch.overrides.has_torch_function(tensor_args):
        #         why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
        #     elif not all((x.device.type in _supported_device_type) for x in tensor_args):
        #         why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
        #                                     f"{_supported_device_type}")
        #     elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
        #         why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
        #                                     "input/output projection weights or biases requires_grad")

        #     if not why_not_sparsity_fast_path:
        #         merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
        #         return torch._transformer_encoder_layer_fwd(
        #             src,
        #             self.self_attn.embed_dim,
        #             self.self_attn.num_heads,
        #             self.self_attn.in_proj_weight,
        #             self.self_attn.in_proj_bias,
        #             self.self_attn.out_proj.weight,
        #             self.self_attn.out_proj.bias,
        #             self.activation_relu_or_gelu == 2,
        #             self.norm_first,
        #             self.norm1.eps,
        #             self.norm1.weight,
        #             self.norm1.bias,
        #             self.norm2.weight,
        #             self.norm2.bias,
        #             self.linear1.weight,
        #             self.linear1.bias,
        #             self.linear2.weight,
        #             self.linear2.bias,
        #             merged_mask,
        #             mask_type,
        #         )


        x = src
        if self.norm_first:
            x = self.norm1(x)
            tmp_x_sa, self._sa_weights = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + tmp_x_sa
            x = x + self._ff_block(self.norm2(x))
        else:
            tmp_x_sa, self._sa_weights = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self.norm1(x + tmp_x_sa)
            x = self.norm2(x + self._ff_block(x))
        
        return x, self._sa_weights

    # self-attention block
    def _sa_block(self, x: Tensor,
                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, self._sa_weights = self.self_attn(x, x, x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        need_weights=True, is_causal=is_causal)
        return self.dropout1(x), self._sa_weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class customTransformerDecoderLayer(nn.Module):
    
    """
    This class implements a custom encoder layer made up of self-attn, multi-head-attn and feedforward network..

    This custom decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. It has been modified to return
    the attention weights.
    ----------
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Returns:
        x: decoder output.
        sa_weights: self-attention weights.
        mha_weights: multihead-attention weights.
    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # Define objects to store custom weights
        self._sa_weights = None
        self._mha_weights = None

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = nn._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = self.norm1(x)
            tmp_x_sa, self._sa_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + tmp_x_sa
            x = self.norm2(x)
            temp_x_mha, self._mha_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + temp_x_mha
            x = x + self._ff_block(self.norm3(x))
        else:
            tmp_x_sa, self._sa_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = self.norm1(x + tmp_x_sa)
            temp_x_mha, self._mha_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = self.norm2(x + temp_x_mha)
            x = self.norm3(x + self._ff_block(x))

        return x, self._sa_weights, self._mha_weights

    # self-attention block
    def _sa_block(self, x: Tensor,
                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        x, self._sa_weights = self.self_attn(x, x, x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        is_causal=is_causal,
                        need_weights=True)
        return self.dropout1(x), self._sa_weights

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        x, self._mha_weights = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=True)
        return self.dropout2(x), self._mha_weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class TimeSeriesTransformer(nn.Module):
    
    """
    This class implements a transformer model that can be used for times series
    forecasting.
    """

    # d_embed from Tianfang's transformer is d_model from my case
    def __init__(self, input_size: int, decoder_sequence_len: int, batch_first: bool,
                d_model: int, n_encoder_layers: int, n_decoder_layers: int, n_heads: int,
                dropout_encoder: float, dropout_decoder: float, dropout_pos_encoder: float,
                dim_feedforward_encoder: int, dim_feedforward_decoder: int,
                num_src_features: int, num_predicted_features: int):
        super(TimeSeriesTransformer, self).__init__()
        
        """
        Arguments:
        input_size (int): number of input variables. 1 if univariate.
        decoder_sequence_len (int): the length of the input sequence fed to the decoder
        d_model (int): all sub-layers in the model produce outputs of dimension d_model
        n_encoder_layers (int): number of stacked encoder layers in the encoder
        n_decoder_layers (int): number of stacked encoder layers in the decoder
        n_heads (int): the number of attention heads (aka parallel attention layers)
        dropout_encoder (float): the dropout rate of the encoder
        dropout_decoder (float): the dropout rate of the decoder
        dropout_pos_encoder (float): the dropout rate of the positional encoder
        dim_feedforward_encoder (int): number of neurons in the linear layer of the encoder
        dim_feedforward_decoder (int): number of neurons in the linear layer of the decoder
        num_predicted_features (int) : the number of features you want to predict. Most of the time, 
        this will be 1 because we're only forecasting FCR-N prices in DK2, but in we wanted to also 
        predict FCR-D with the same model, num_predicted_features should be 2.
        """
        
        self.model_tpye = 'Transformer'

        self.decoder_sequence_len = decoder_sequence_len
        
        # print("input_size is: {}".format(input_size))
        # print("d_model is: {}".format(d_model))
        
        # Create the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=d_model)
        
        self.decoder_input_layer = nn.Linear(in_features=num_src_features, out_features=d_model)
        
        self.linear_mapping = nn.Linear(in_features=d_model, out_features=num_predicted_features)
        
        # Create the positional encoder dropout, max_seq_len, d_model, batch_first
        self.positional_encoding_layer = PositionalEncoder(dropout=dropout_pos_encoder, max_seq_len=5000, d_model=d_model, batch_first=batch_first)
        
        # The encoder layer used in the paper is identical to the one used by Vaswani et al (2017) 
        # on which the PyTorch transformer model is based
        encoder_layer = customTransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward_encoder,
                                                dropout=dropout_encoder, batch_first=batch_first)

        # Stack the encoder layers in nn.TransformerEncoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        
        # Define the decoder layer
        decoder_layer = customTransformerDecoderLayer(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward_decoder,
                                                dropout=dropout_decoder,  batch_first=batch_first)
        
        
        # Stack the decoder layers in nn.TransformerDecoder
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)

        # Initialize the weights
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, memory_mask: Tensor=None, tgt_mask: Tensor=None) -> Tensor:
        
        """
        Arguments:
        src: the output sequence of the encoder. Shape: (S, E) for unbatched input, (N, S, E) if 
            batch_first=True, where S is the source of the sequence length, N is the batch size, and E is the number of 
            features (1 if univariate).
        tgt: the sequence to the decoder. Shape (T, E) for unbatched input, or (N, T, E) if batch_first=True, 
            where T is the target sequence length, N is the batch size, and E is the number of features (1 if univariate).
        src_mask: the mask for the src sequence to prevent the model from using data points from the target sequence.
        tgt_mask: the mask fro the tgt sequence to prevent the model from using data points from the target sequence.
        
        Returns:
        Tensor of shape: [target_sequence_length, batch_size, num_predicted_features]
        """
        
        # print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))
        
        # Pass through the input layer right before the encoder
        # src shape: [batch_size, src length, d_model] regardless of number of input features
        src = self.encoder_input_layer(src)
        # print("From model.forward(): Size of src after input layer: {}".format(src.size()))
        
        
        # Pass through the positional encoding layer            
        # src shape: [batch_size, src length, d_model] regardless of number of input features
        # print(src.shape) # Maybe I just have to use a squeeze on source to add a dimension of 1 for the batch size. Probably have to do it with tgt and maybe tgt_y too
        src = self.positional_encoding_layer(src)
        # print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))
        
        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded which they are 
        # not in this time series use case, because all my  input sequences are naturally of 
        # the same length. 
        # src shape: [batch_size, encoder_sequence_len, d_model]
        src, sa_weights_encoder = self.encoder(src=src, mask=src_mask)
        # print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        # Pass decoder input through decoder input layer
        # tgt shape: [target sequence length, batch_size, d_model] regardless of number of input features
        # print(f"From model.forward(): Size of tgt before the decoder = {tgt.size()}")
        decoder_output = self.decoder_input_layer(tgt)
        # print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))
        
        # if src_mask is not None:
        #     print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        # if tgt_mask is not None:
        #     print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))
        
        # Pass through the positional encoding layer            
        # src shape: [batch_size, tgt length, d_model] regardless of number of input features
        decoder_output = self.positional_encoding_layer(decoder_output)

        # Pass through the decoder
        # Output shape: [batch_size, target seq len, d_model]
        decoder_output, sa_weights, mha_weights = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))
        
        # Pass through the linear mapping
        # shape [batch_size, target seq len]
        decoder_output = self.linear_mapping(decoder_output)
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))
        
        return decoder_output, sa_weights_encoder, sa_weights, mha_weights