import sys
from typing import Any, Mapping
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

class FullyConnectedModel(nn.Module):
    """
    Transformer model class.

    Parameters:
    - nbInstrPerSample:         the number of G-code instructions per sample
    - input_vector_size:        input vector size for each G-code instruction
    - label_vector_size:        label vector size for each G-code instruction
    - device:                   a torch.device descriibing the device to send the model to (cpu or gpu)
    - subWindowSize:            an int describing how many input instructions are used to predict one single instruction
    - predictedInstr:           an int describing which instruction is predicted for each given sub-window
    - layer_sizes:              a list of ints describing the shapes of each linear layers, first value must be equal to the number of instr
                                per sub-window times the number of values per instr, and the final value must be equal to the number of labels
                                per instr. Each layer `i` input and output dims are described as: `layer_sizes[i]` and `layer_sizes[i + 1]`
    - dropout:                  dropout rate
    - floatType:                a torch.dtype describing the wanted float precision for the model
    - log:                      a Logger (see Logger.py in the parent folder), None by default (new logger without file writing)
    - generator:                a torch.Generator, used to control the randomness of the weights initialization, None by default
    - init_weights:             a boolean, True by default. If true, will initialize the trainable weights with a random uniform distribution
    - trained_on_log_target:    a boolean, False by default. If true, it means the model is/was trained on the log(target + 1) instead of the target. Used to inverse the output for inference
    """
    
    def __init__(self,
                nbInstrPerSample: int,
                input_vector_size: int,
                label_vector_size: int,
                device: torch.device,
                *,
                subWindowSize: int = Constants.DEFAULT_FCC_SUBWINDOW,
                predictedInstr: int = Constants.DEFAULT_FCC_PREDICTED_INSTR,
                layer_sizes: list[int] = Constants.DEFAULT_FCC_LAYERS, 
                dropout: float = Constants.DEFAULT_DROPOUT_RATE,
                floatType: torch.dtype = torch.float32 if Constants.DEFAULT_FLOAT_TYPE == 32 else torch.float64,
                log: Logger = None,
                generator: torch.Generator = None,
                init_weights: bool = True,
                trained_on_log_target: bool = False
            ) -> None:
        super(FullyConnectedModel, self).__init__()
        assert(subWindowSize < nbInstrPerSample)
        assert(predictedInstr < subWindowSize)
        assert(layer_sizes[0] == subWindowSize * input_vector_size)
        assert(layer_sizes[-1] == label_vector_size)
        
        # Local variables
        if log == None:
            self.log = Logger(writesLog=False)
        else:
            self.log = log
        self.nbInstrPerSample = nbInstrPerSample
        self.input_vector_size = input_vector_size
        self.label_vector_size = label_vector_size
        self.left_offset = predictedInstr
        self.right_offset = nbInstrPerSample - (subWindowSize - predictedInstr) + 1
        self.input_size = (nbInstrPerSample, input_vector_size)
        self.output_size = (self.right_offset - self.left_offset, label_vector_size)
        self.model_type = 'FullyConnected'
        self.generator = generator
        self.device = device
        self.floatType = floatType
        self.trained_on_log_target = trained_on_log_target

        assert(self.right_offset - self.left_offset >= self.nbInstrPerSample - self.right_offset + self.left_offset)

        # Architecture related variables
        self.sub_window_size = subWindowSize
        self.predicted_instr = predictedInstr
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        
        ### Modules ###
        # Normalization layer
        self.normalizer = nn.BatchNorm1d(self.input_vector_size, device=self.device, dtype=self.floatType)
        self.layers = nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], device=self.device, dtype=self.floatType))
        self.relu = nn.ReLU()
        self.dp_layer = nn.Dropout(self.dropout)

        if init_weights:
            self.init_weights()

    def summary(self) -> None:
        """
        Prints a summary of the model's architecture.
        """
        self.log.emptyLines()
        self.log.info("Model's architecture: Fully Connected")
        string = f" {'=' * 90}\n\t"
        string += f"Input shapes:               [batch_size, {self.nbInstrPerSample}, {self.input_vector_size}] (G-code vectors)\n\t"
        string += f"Output shape:               [batch_size, {self.right_offset - self.left_offset}, {self.label_vector_size}]\n\t"
        string += f"Sub window format:          {self.sub_window_size} input instructions, predicting n°{self.predicted_instr}\n\t"
        for i in range(len(self.layer_sizes) - 1):
            string += f"Layer {i + 1}:                    Input length of [{self.layer_sizes[i]}] and output length of [{self.layer_sizes[i + 1]}]\n\t"
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
        additionnalData["subWindowSize"] = self.sub_window_size
        additionnalData["predictedInstr"] = self.predicted_instr
        additionnalData["layerSizes"] = self.layer_sizes
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
        self.sub_window_size = state_dict["subWindowSize"]
        self.predicted_instr = state_dict["predictedInstr"]
        self.layer_sizes = state_dict["layerSizes"]
        self.trained_on_log_target = state_dict["trainedOnLogTarget"]
        del state_dict["modelType"], state_dict["instrPerSample"], state_dict["inputVectorSize"], state_dict["labelVectorSize"], state_dict["leftOffset"]
        del state_dict["rightOffset"], state_dict["subWindowSize"], state_dict["predictedInstr"], state_dict["layerSizes"]
        del state_dict["trainedOnLogTarget"]
        return super().load_state_dict(state_dict, strict, assign)
    
    def getFilename(self):
        filename = f"{self.model_type}_sampleSize-{self.nbInstrPerSample}_iVecSize-{self.input_vector_size}_lVecSize-{self.label_vector_size}"
        filename += f"_lOffset-{self.left_offset}_rOffset-{self.right_offset}_subWindowSize-{self.sub_window_size}_predictedInstr-{self.predicted_instr}"
        filename += f"_layerSizes-{'+'.join(map(str, self.layer_sizes))}"
        return filename

    def init_weights(self) -> None:
        """
        Model layers' weights initialization
        """
        initrange = 0.5
        for l in self.layers:
            nn.init.uniform_(l.weight, -initrange, initrange, generator=self.generator)
            nn.init.zeros_(l.bias)

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.

        Input shapes:     src=[batch_size, nbInstrPerSample, input_vector_size]

        Output shape:    [batch_size, nbInstrPerSample, label_vector_size]
        """

        # Input normalizing layer
        src = self.normalizer(src.permute(0, 2, 1)).permute(0, 2, 1)

        nb_windows = self.nbInstrPerSample - self.sub_window_size + 1
        batch_size = src.size()[0]
        src = src.unfold(dimension=1, size=self.sub_window_size, step=1)
        src = src.contiguous().view(batch_size, nb_windows, -1)
        src = self.dp_layer(src)
        for i, l in enumerate(self.layers):
            if i != 0 and i != len(self.layers) - 1:
                src = self.relu(src)
            src = l(src)
        return src
        
def main() -> None:
    log = Logger(writesLog=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device used: {device}\n")
    
    linear = torch.tensor([
        [float(f"{i}.{j}") for j in range(Constants.DEFAULT_INPUT_SIZE)] for i in range(Constants.DEFAULT_SAMPLE_SIZE)
        ], dtype=torch.float32).unsqueeze(0).to(device)
    log.info(f"Base tensor: {linear}\n    Base tensor shape: {linear.shape}")
    
    model = FullyConnectedModel(Constants.DEFAULT_SAMPLE_SIZE, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, device,\
                                floatType=torch.float32, log=log)
    model.summary()
    with torch.no_grad():
        o = model(linear)
    log.info(f"Result tensor: {o}\n       Result shape: {o.shape}")
    
if __name__ == "__main__":
    main()
        
"""
End of file
"""