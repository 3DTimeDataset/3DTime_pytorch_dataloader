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

class BiLSTMModel(nn.Module):
    """
    Bi-LSTM model class.

    Parameters:
    - nbInstrPerSample:        the number of G-code instructions per sample
    - nbValuesPerInstr:        vector size for each G-code instruction
    - printerVectorSize:    size of the printer vector
    - device:                a torch.device descriibing the device to send the model to (cpu or gpu)
    - normalizing_tensor:    a torch.Tensor containing the max values of the concatenated input tensors, used to normalize the input data.
    - embed_dim:            internal hidden size
    - nlayers:                number of Bi-LSTM layer(s)
    - dropout:                dropout rate
    - floatType:            a torch.dtype describing the wanted float precision for the model
    - log:                    a Logger (see Logger.py in the parent folder), None by default (new logger without file writing)
    - generator:            a torch.Generator, used to control the randomness of the weights initialization, None by default
    - init_weights:            a boolean, True by default. If true, will initialize the trainable weights with a random uniform distribution
    - trained_on_log_target:    a boolean, False by default. If true, it means the model is/was trained on the log(target + 1) instead of the target. Used to inverse the output for inference
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
                nlayers: int = Constants.DEFAULT_LAYER_COUNT,
                dropout: float = Constants.DEFAULT_DROPOUT_RATE,
                floatType: torch.dtype = torch.float32 if Constants.DEFAULT_FLOAT_TYPE == 32 else torch.float64,
                log: Logger = None,
                generator: torch.Generator = None,
                init_weights: bool = True,
                trained_on_log_target: bool = False,
                simple_rule_head: bool = False
            ) -> None:
        super(BiLSTMModel, self).__init__()
        assert(0 <= leftOffset < rightOffset <= nbInstrPerSample)
        assert(rightOffset - leftOffset >= nbInstrPerSample - rightOffset + leftOffset) # Ensures there are more predicted instructions that ommited ones
        
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
        self.model_type = 'BiLSTM'
        self.generator = generator
        self.trained_on_log_target = trained_on_log_target
        self.simple_rule_head = simple_rule_head

        self.device = device
        self.floatType = floatType

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
                log.error(f"If a model has a simple rule prediction head, it must be trained to predict only the start velocity (got {self.label_vector_size}).", 1)
            if not self.left_offset == 0:
                log.error(f"If a model has a simple rule prediction head, it must not have a left offset (got {self.left_offset}).", 1)
            if not self.right_offset == self.nbInstrPerSample - 1:
                log.error(f"If a model has a simple rule prediction head, it must have a right offset of 1 (got {self.right_offset}).", 1)
            try:
                self.target_speed_index = UV.SELECTED_INPUTS.index(6) if UV.SELECTED_INPUTS is not None else 6
                self.extr_len_index = UV.SELECTED_INPUTS.index(7) if UV.SELECTED_INPUTS is not None else 7
                if UV.addDeltaCoordsAndAngle == UV.UPDATING_FUNCTIONS[-1]:
                    self.delta_x_index = -5
                    self.delta_y_index = -4
                    self.delta_z_index = -3
                    self.seg_len_index = -2
                else:
                    self.log.error(f"For a model to have a simple rule prediction head, the vector updating function `addDeltaCoordsAndAngle` must be last in the `UPDATING_FUNCTIONS` list in `UpdateVectors.py` script.", 1)
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
        self.nlayers = nlayers
        self.dropout = dropout
        
        ### Modules ###
        # Normalizing layer
        self.normalizer = nn.BatchNorm1d(self.input_vector_size, device=self.device, dtype=self.floatType)
        # LSTM module
        self.rnn = nn.LSTM(self.input_vector_size, self.embed_dim, self.nlayers, \
                    batch_first=True, dropout=self.dropout, bidirectional=True, device=device, dtype=floatType)
        # Decoder
        self.decoder = nn.Linear(2 * self.embed_dim, self.label_vector_size, device=device, dtype=floatType)

        if init_weights:
            self.init_weights()

    def summary(self) -> None:
        """
        Prints a summary of the model's architecture.
        """
        self.log.emptyLines()
        self.log.info("Model architecture: BiLSTM")
        string = f" {'=' * 90}\n\t"
        string += f"Input shapes:               [batch_size, {self.nbInstrPerSample}, {self.input_vector_size}] (G-code vectors)\n\t"
        string += f"Output shape:               [batch_size, {self.right_offset - self.left_offset}, {self.label_vector_size}]\n\t"
        string += f"Bi-LSTM layers:             {self.nlayers} layer(s) of hidden size [{self.embed_dim}]\n\t"
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
        self.nlayers = state_dict["nbLayer"]
        self.trained_on_log_target = state_dict["trainedOnLogTarget"]
        del state_dict["modelType"], state_dict["instrPerSample"], state_dict["inputVectorSize"], state_dict["labelVectorSize"], state_dict["leftOffset"]
        del state_dict["rightOffset"], state_dict["embedDim"], state_dict["nbLayer"], state_dict["trainedOnLogTarget"]
        return super().load_state_dict(state_dict, strict, assign)

    def getFilename(self):
        filename = f"{self.model_type}_sampleSize-{self.nbInstrPerSample}_iVecSize-{self.input_vector_size}_lVecSize-{self.label_vector_size}"
        filename += f"_lOffset-{self.left_offset}_rOffset-{self.right_offset}_embedDim-{self.embed_dim}_nbLayer-{self.nlayers}"
        return filename

    def init_weights(self) -> None:
        """
        Model's params initialization
        """
        initrange = math.sqrt(1.0 / self.embed_dim)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange, generator=self.generator)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data, generator=self.generator)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.

        Input shapes:     src=[batch_size, nbInstrPerSample, nbValuesPerInstr]
        """

        if self.simple_rule_head:
            target_speed = src[..., self.target_speed_index].clone().unsqueeze(-1)
            extr_len = src[..., self.extr_len_index].abs().clone().unsqueeze(-1)
            delta_x = src[..., self.delta_x_index].clone().unsqueeze(-1)
            delta_y = src[..., self.delta_y_index].clone().unsqueeze(-1)
            delta_z = src[..., self.delta_z_index].clone().unsqueeze(-1)
            seg_len = src[..., self.seg_len_index].clone().unsqueeze(-1)

        # Input normalizing layer
        src = self.normalizer(src.permute(0, 2, 1)).permute(0, 2, 1)
        # LSTM layers
        output, _ = self.rnn(src)
        # Decoder : [2*embedDim] -> [1]
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
    nbInstrPerSample = Constants.DEFAULT_SAMPLE_SIZE
    nbValuesPerInstr = Constants.DEFAULT_VALUES_PER_VECTOR
    printerVectorSize = Constants.DEFAULT_PRINTER_VECTOR_SIZE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device used: {device}\n")
    
    linear = torch.tensor([[float(f"{i}.{j}") for j in range(nbValuesPerInstr)] for i in range(nbInstrPerSample)], dtype=torch.float32).unsqueeze(0).to(device)
    printerData = torch.tensor([float(i)/10.0 for i in range(printerVectorSize)], dtype=torch.float32).unsqueeze(0).to(device)
    log.info(f"Base tensor: {linear}\n")
    log.info(f"Printer tensor: {printerData}\n")
    
    model = BiLSTMModel(nbInstrPerSample, nbValuesPerInstr, printerVectorSize, device,\
                        embed_dim=32, nlayers=2, floatType=torch.float32, log=log)
    model.summary()
    with torch.no_grad():
        o = model(linear, printerData)
    log.info(f"Result tensor: {o}\n       Result shape: {o.shape}")
    
if __name__ == "__main__":
    main()
        
"""
End of file
"""