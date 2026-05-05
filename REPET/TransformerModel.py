import sys
from typing import Any, Mapping
try:
    from smallerModels.FixedPredFromTrapez import PredFromTrapez
    import UpdateVectors as UV
except:
    from REPET.smallerModels.FixedPredFromTrapez import PredFromTrapez
    import REPET.UpdateVectors as UV
try:
    from Logger import *
    import Utils
    import Constants
except:
    sys.path.insert(0,"..")
    from Logger import *
    import Utils
    import Constants

import torch
from torch import nn, Tensor
import math
        
class PositionalEncoding(nn.Module):
    """
    Positional encoding module for the Transformer model.

    Parameters:
    - nbInstrPerSample:     the number of G-code instructions per sample
    - nbValuesPerInstr:        the vector size for each G-code instruction (should be G-code vector size + printer vector size)
    - device:                a torch.device describing the device to send the module to (cpu or gpu)
    - floatType             a torch.dtype describing the wanted float type for the module (float16, float32 or float64)
    """
    
    def __init__(self, nbInstrPerSample: int, nbValuesPerInstr: int, device: torch.device, floatType: torch.dtype) -> None:
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.encoding = self._generate_encoding(nbInstrPerSample, nbValuesPerInstr, floatType)
    
    def _generate_encoding(self, nbInstrPerSample: int, nbValuesPerInstr: int, floatType: torch.dtype) -> Tensor:
        encoding = torch.zeros(nbValuesPerInstr, nbInstrPerSample, dtype=floatType, device=self.device)
        position = torch.arange(0, nbValuesPerInstr, dtype=floatType, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nbInstrPerSample, 2, dtype=floatType, device=self.device) * -(math.log(10000.0) / nbInstrPerSample))
        encoding[:, 0::2] = torch.sin(position * div_term)
        if nbInstrPerSample % 2 != 0:
            encoding[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.transpose(0, 1).unsqueeze(0)
    
    def forward(self, source: Tensor) -> Tensor:
        return source + self.encoding[:, :source.size(1)]

class TransformerModel(nn.Module):
    """
    Transformer model class.

    Parameters:
    - nbInstrPerSample:         the number of G-code instructions per sample
    - input_vector_size:        input vector size for each G-code instruction
    - label_vector_size:        label vector size for each G-code instruction
    - left_offset:              output prediction window left offset, aka number of not-predicted instructions on the left side of the input window
    - right_offset:             output prediction window right offset, aka number representing until which instruction (excluded) the models predicts
    - device:                   a torch.device descriibing the device to send the model to (cpu or gpu)
    - embed_dim:                internal embedding size for the transformer layer(s)
    - nb_head:                  number of attention head(s) for the transformer layer(s)
    - feedforward_dim:          the size of the linear feed forward layers in the transformer layer(s)
    - nlayers:                  number of transformer layer(s)
    - dropout:                  dropout rate
    - floatType:                a torch.dtype describing the wanted float precision for the model
    - log:                      a Logger (see Logger.py in the parent folder), None by default (new logger without file writing)
    - generator:                a torch.Generator, used to control the randomness of the weights initialization, None by default
    - init_weights:             a boolean, True by default. If true, will initialize the trainable weights with a random uniform distribution
    - trained_on_log_target:    a boolean, False by default. If true, it means the model is/was trained on the log(target + 1) instead of the target. Used to inverse the output for inference

    Note:
    For `left_offset` and `right_offset`, the values must follow these two rules:
    `0 <= left_offset < right_offset <= nbInstrPerSample`
    and
    `right_offset - left_offset >= nbInstrPerSample - right_offset + left_offset`
    """
    
    def __init__(self,
                nbInstrPerSample: int,
                input_vector_size: int,
                label_vector_size: int,
                leftOffset: int,
                rightOffset: int,
                device: torch.device,
                *,
                embed_dim: int = Constants.DEFAULT_EMBED_DIM,
                nb_head: int = Constants.DEFAULT_NB_HEAD,
                feedforward_dim: int = Constants.DEFAULT_FEEDF_DIM,
                nlayers: int = Constants.DEFAULT_LAYER_COUNT,
                dropout: float = Constants.DEFAULT_DROPOUT_RATE,
                floatType: torch.dtype = torch.float32 if Constants.DEFAULT_FLOAT_TYPE == 32 else torch.float64,
                log: Logger = None,
                generator: torch.Generator = None,
                init_weights: bool = True,
                trained_on_log_target: bool = False,
                simple_rule_head: bool = False
            ) -> None:
        super(TransformerModel, self).__init__()
        assert(0 <= leftOffset < rightOffset <= nbInstrPerSample)
        assert(rightOffset - leftOffset >= nbInstrPerSample - rightOffset + leftOffset) # Ensures there are more predicted instructions that ommited ones
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.') from e
        
        # Local variables
        if log == None:
            self.log = Logger(writesLog=False)
        else:
            self.log = log
        self.nbInstrPerSample = nbInstrPerSample
        self.input_vector_size = input_vector_size
        self.label_vector_size = label_vector_size
        self.left_offset = leftOffset
        self.right_offset = rightOffset
        self.input_size = (nbInstrPerSample, input_vector_size)
        self.output_size = (self.right_offset - self.left_offset, label_vector_size)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.generator = generator
        self.device = device
        self.floatType = floatType
        self.trained_on_log_target = trained_on_log_target
        self.simple_rule_head = simple_rule_head

        if self.simple_rule_head:
            self.trapeze_head = PredFromTrapez(
                7,
                7,
                self.nbInstrPerSample - 1,
                device,
                floatType=self.floatType,
                log=self.log,
                is_prediction_head=True
            )
            if not self.label_vector_size == 1:
                log.error(f"If a transformer model has a simple rule prediction head, it must be trained to predict only the start velocity (got {self.label_vector_size}).", 1)
            if not self.left_offset == 0:
                log.error(f"If a transformer model has a simple rule prediction head, it must not have a left offset (got {self.left_offset}).", 1)
            if not self.right_offset == self.nbInstrPerSample - 1:
                log.error(f"If a transformer model has a simple rule prediction head, it must have a right offset of 1 (got {self.right_offset}).", 1)
            try:
                self.target_speed_index = UV.SELECTED_INPUTS.index(6) if UV.SELECTED_INPUTS is not None else 6
                self.extr_len_index = UV.SELECTED_INPUTS.index(7) if UV.SELECTED_INPUTS is not None else 7
                if UV.addDeltaCoordsAndAngle == UV.UPDATING_FUNCTIONS[-1]:
                    self.delta_x_index = -5
                    self.delta_y_index = -4
                    self.delta_z_index = -3
                    self.seg_len_index = -2
                else:
                    self.log.error(f"For a Transformer model to have a simple rule prediction head, the vector updating function `addDeltaCoordsAndAngle` must be last in the `UPDATING_FUNCTIONS` list in `UpdateVectors.py` script.", 1)
            except ValueError:
                self.log.error(f"This error should not happen", 1)
        else:
            self.trapeze_head = None
            self.target_speed_index = None
            self.delta_x_index = None
            self.delta_y_index = None
            self.delta_z_index = None
            self.seg_len_index = None

        # Architecture related variables
        self.embed_dim = embed_dim
        self.nb_head = nb_head
        self.feedforward_dim = feedforward_dim
        self.nlayers = nlayers
        self.dropout = dropout
        
        ### Modules ###
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.nbInstrPerSample, self.input_vector_size, self.device, self.floatType)
        # Normalization layer
        self.normalizer = nn.BatchNorm1d(self.input_vector_size, device=self.device, dtype=self.floatType)
        # Base transformer layer
        encoder_layers = TransformerEncoderLayer(self.embed_dim,
                                                 self.nb_head,
                                                 self.feedforward_dim,
                                                 self.dropout,
                                                 device=self.device,
                                                 dtype=self.floatType,
                                                 batch_first=True
                                                 )
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers, enable_nested_tensor=False).to(self.device)
        # Linear layer for input dim accommodation (used on each instruction): from [input_vector_size] to [embed_dim]
        self.encoder = nn.Linear(self.input_vector_size, self.embed_dim, device=self.device, dtype=self.floatType)
        # Linear layer for output dim accommodation: from [embed_dim] to [label_vector_size]
        self.decoder = nn.Linear(self.embed_dim, self.label_vector_size, device=self.device, dtype=self.floatType)

        if init_weights:
            self.init_weights()

    def summary(self) -> None:
        """
        Prints a summary of the model's architecture.
        """
        self.log.emptyLines()
        self.log.info("Model's architecture: Transformer")
        string = f" {'=' * 90}\n\t"
        string += f"Input shapes:               [batch_size, {self.nbInstrPerSample}, {self.input_vector_size}] (G-code vectors)\n\t"
        string += f"Output shape:               [batch_size, {self.right_offset - self.left_offset}, {self.label_vector_size}]\n\t"
        string += f"Transformer encoder:        {self.nlayers} layer(s) of embeding dimension [{self.embed_dim}]\n\t"
        string += f"Transformer layer(s):       each with {self.nb_head} attention heads, and a feedforward dimension of {self.feedforward_dim}\n\t"
        string += f"Log-normalization:          {self.trained_on_log_target}\n\t"
        string += f"Uses trapeze pred head:     {self.simple_rule_head}\n\t"
        string += f"Total number of parameters: {sum(p.numel() for p in self.parameters())}\n\t"
        string += f"{'=' * 90}\n"
        self.log.info(string)
        self.log.emptyLines()

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        base_dict: dict[str, Any] = super().state_dict(*args, **kwargs)
        additionnalData = {}
        additionnalData["modelType"] = self.model_type
        additionnalData["instrPerSample"] = self.nbInstrPerSample
        additionnalData["inputVectorSize"] = self.input_vector_size
        additionnalData["labelVectorSize"] = self.label_vector_size
        additionnalData["leftOffset"] = self.left_offset
        additionnalData["rightOffset"] = self.right_offset
        additionnalData["embedDim"] = self.embed_dim
        additionnalData["headCount"] = self.nb_head
        additionnalData["fForwardDim"] = self.feedforward_dim
        additionnalData["nbLayer"] = self.nlayers
        additionnalData["trainedOnLogTarget"] = self.trained_on_log_target
        base_dict.update(additionnalData)
        return base_dict
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.model_type = state_dict["modelType"]
        self.nbInstrPerSample = state_dict["instrPerSample"]
        self.input_vector_size = state_dict["inputVectorSize"]
        self.label_vector_size = state_dict["labelVectorSize"]
        self.left_offset = state_dict["leftOffset"]
        self.right_offset = state_dict["rightOffset"]
        self.embed_dim = state_dict["embedDim"]
        self.nb_head = state_dict["headCount"]
        self.feedforward_dim = state_dict["fForwardDim"]
        self.nlayers = state_dict["nbLayer"]
        self.trained_on_log_target = state_dict["trainedOnLogTarget"]
        del state_dict["modelType"], state_dict["instrPerSample"], state_dict["inputVectorSize"], state_dict["labelVectorSize"], state_dict["leftOffset"]
        del state_dict["rightOffset"], state_dict["embedDim"], state_dict["headCount"], state_dict["fForwardDim"], state_dict["nbLayer"]
        del state_dict["trainedOnLogTarget"]
        return super().load_state_dict(state_dict, strict, assign)
    
    def getFilename(self):
        filename = f"{self.model_type}_sampleSize-{self.nbInstrPerSample}_iVecSize-{self.input_vector_size}_lVecSize-{self.label_vector_size}"
        filename += f"_lOffset-{self.left_offset}_rOffset-{self.right_offset}_embedDim-{self.embed_dim}_headCount-{self.nb_head}"
        filename += f"_fForwardDim-{self.feedforward_dim}_nbLayer-{self.nlayers}"
        return filename

    def _generate_square_subsequent_mask(self, sz: int) -> None:
        """
        Mask generation.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self) -> None:
        """
        Model layers' weights initialization
        """
        initrange = 0.5
        nn.init.uniform_(self.encoder.bias, -initrange, initrange, generator=self.generator)
        nn.init.uniform_(self.encoder.weight, -initrange, initrange, generator=self.generator)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange, generator=self.generator)
        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -initrange, initrange, generator=self.generator)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


    def forward(self, src: Tensor, has_mask=False) -> Tensor:
        """
        Forward pass.

        Input shapes:     src=[batch_size, nbInstrPerSample, input_vector_size]

        Output shape:    [batch_size, nbInstrPerSample, label_vector_size]
        """

        if self.simple_rule_head:
            target_speed = src[..., self.target_speed_index].clone().unsqueeze(-1)
            extr_len = src[..., self.extr_len_index].abs().clone().unsqueeze(-1)
            delta_x = src[..., self.delta_x_index].clone().unsqueeze(-1)
            delta_y = src[..., self.delta_y_index].clone().unsqueeze(-1)
            delta_z = src[..., self.delta_z_index].clone().unsqueeze(-1)
            seg_len = src[..., self.seg_len_index].clone().unsqueeze(-1)

        if has_mask: # Generate the mask if required, this is in prevision of some future pre-training, as it is not used yet
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # Input normalizing layer
        src = self.normalizer(src.permute(0, 2, 1)).permute(0, 2, 1)

        # Positional encoding and transformer encoding
        src = self.pos_encoder(src)

        src = self.encoder(src)

        output: Tensor = self.transformer_encoder(src, self.src_mask)

        # Decoding
        output = self.decoder(output)

        if self.simple_rule_head:
            if self.trained_on_log_target:
                output = torch.exp(output) - Constants.DEFAULT_LOG_NORM_EPSILON
            justE = (seg_len < 1e-12).logical_and(delta_x < 1e-12).logical_and(delta_y < 1e-12).logical_and(delta_z < 1e-12)
            b_idx, t_idx, _ = justE.nonzero(as_tuple=True)
            seg_len[b_idx, t_idx, :] = extr_len[b_idx, t_idx, :]
            simpleRuleInput = torch.cat(
                (
                    target_speed[:, :self.nbInstrPerSample - 1, :],
                    output[:, :self.nbInstrPerSample - 1, :],
                    output[:, 1:, :],
                    delta_x[:, :self.nbInstrPerSample - 1, :],
                    delta_y[:, :self.nbInstrPerSample - 1, :],
                    delta_z[:, :self.nbInstrPerSample - 1, :],
                    seg_len[:, :self.nbInstrPerSample - 1, :]
                ), dim=-1)
            actual_out = self.trapeze_head(simpleRuleInput)
            return actual_out

        return output[:, self.left_offset:self.right_offset, :]
        
def main() -> None:
    log = Logger(writesLog=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device used: {device}\n")
    
    linear = torch.tensor([
        [float(f"{i}.{j}") for j in range(Constants.DEFAULT_INPUT_SIZE)] for i in range(Constants.DEFAULT_SAMPLE_SIZE)
        ], dtype=torch.float32).unsqueeze(0).to(device)
    log.info(f"Base tensor: {linear}\n    Base tensor shape: {linear.shape}")
    
    model = TransformerModel(Constants.DEFAULT_SAMPLE_SIZE, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 10, 90, device,\
                            embed_dim=32, nlayers=1, floatType=torch.float32, log=log)
    model.summary()
    with torch.no_grad():
        o = model(linear)
    log.info(f"Result tensor: {o}\n       Result shape: {o.shape}")
    
if __name__ == "__main__":
    main()
        
"""
End of file
"""