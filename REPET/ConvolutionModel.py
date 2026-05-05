import sys
from typing import Any
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

class ConvolutionModel(nn.Module):
    """
    Convolution model class.

    Parameters:
    - nbInstrPerSample:         the number of G-code instructions per sample
    - input_vector_size:        input vector size for each G-code instruction
    - label_vector_size:        label vector size for each G-code instruction
    - device:                   a torch.device descriibing the device to send the model to (cpu or gpu)
    - kernels_size:             list of ints, containing the kernel sizes for each layer, hence also indicating the number of layers
    - channels:                 list of ints, containing the number of channels for each layer, length must match with 'kernels_size'
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
                channels: list[int] = Constants.DEFAULT_CHANNELS,
                kernels_size: list[int] = Constants.DEFAULT_KERNELS,
                dropout: float = Constants.DEFAULT_DROPOUT_RATE,
                floatType: torch.dtype = torch.float32 if Constants.DEFAULT_FLOAT_TYPE == 32 else torch.float64,
                log: Logger = None,
                generator: torch.Generator = None,
                init_weights: bool = True,
                trained_on_log_target: bool = False
            ) -> None:
        super(ConvolutionModel, self).__init__()
        assert(len(kernels_size) == len(channels))
        
        # Local variables
        if log == None:
            self.log = Logger(writesLog=False)
        else:
            self.log = log
        self.nbInstrPerSample = nbInstrPerSample
        self.input_vector_size = input_vector_size
        self.label_vector_size = label_vector_size
        self.output_size = nbInstrPerSample
        for kernel in kernels_size:
            self.output_size -= (kernel - 1)
        self.model_type = 'Convolution'
        self.generator = generator
        self.trained_on_log_target = trained_on_log_target

        # Architecture related variables
        self.kernels_size = kernels_size
        self.channels = channels
        self.dropout = dropout
        self.device = device
        self.floatType = floatType
        
        ### Modules ###
        self.normalizer = nn.BatchNorm1d(input_vector_size, device=self.device, dtype=floatType)
        # Convolution layers
        self.layers = nn.Sequential()
        input_channel = self.input_vector_size
        for kernel_len, channel in zip(self.kernels_size, self.channels):
            self.layers.append(nn.Conv1d(input_channel, channel, kernel_len, stride=1, padding=0, device=device, dtype=floatType))
            input_channel = channel
        if input_channel != self.label_vector_size:
            self.log.error(f"The last Conv1D layer of the Convolution model must have {self.label_vector_size} output channel.", 1)
        self.relu = nn.ReLU()
        if init_weights:
            self.init_weights()

    def summary(self) -> None:
        """
        Prints a summary of the model's architecture.
        """
        self.log.emptyLines()
        self.log.info("Model's architecture: Convolution")
        string = f" {'=' * 90}\n\t"
        string += f"Input shapes:               [batch_size, {self.nbInstrPerSample}, {self.input_vector_size}] (G-code vectors)\n\t"
        string += f"Output shape:               [batch_size, {self.output_size}, {self.label_vector_size}]\n\t"
        for i, (kernel, channel) in enumerate(zip(self.kernels_size, self.channels)):
            string += f"Conv layer {i + 1}:            {channel} channel(s) with a kernel of length {kernel}\n\t"
        string += f"Total number of parameters:    {sum(p.numel() for p in self.parameters())}\n\t"
        string += f"{'=' * 90}\n"
        self.log.info(string)
        self.log.emptyLines()
    
    def state_dict(self, *args, **kwargs):
        base_dict: dict[str, Any] = super().state_dict(*args, **kwargs)
        additionnalData = {}
        additionnalData["modelType"] = self.model_type
        additionnalData["instrPerSample"] = self.nbInstrPerSample
        additionnalData["inputVectorSize"] = self.input_vector_size
        additionnalData["labelVectorSize"] = self.label_vector_size
        additionnalData["leftOffset"] = self.left_offset
        additionnalData["rightOffset"] = self.right_offset
        additionnalData["kernelsSize"] = self.kernels_size
        additionnalData["channels"] = self.channels
        additionnalData["trainedOnLogTarget"] = self.trained_on_log_target
        base_dict.update(additionnalData)
        return base_dict
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        raise NotImplementedError(f"TODO")
        return super().load_state_dict(state_dict, strict, assign)
    
    def getFilename(self):
        raise NotImplementedError
        filename = f"Convolution_instrPerSample-{self.nbInstrPerSample}_inputVectorSize-{self.input_vector_size}_labelVectorSize-"
        filename += f"{self.label_vector_size}_kernelLengths-{'+'.join(map(str, args.k))}_nbChannels-{'+'.join(map(str, args.c))}"

    def init_weights(self) -> None:
        """
        Model's params initialization
        """
        initrange = 1.0
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -initrange, initrange, generator=self.generator)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.

        Input shapes:     src=[batch_size, nbInstrPerSample, input_vector_size]

        Output shape:    [batch_size, output_size, label_vector_size]
        """
        # Input normalizing layer
        output = self.normalizer(src.permute(0, 2, 1))
        # Convolutions
        for i, layer in enumerate(self.layers):
            if i > 0:
                output = self.relu(output)
            output: Tensor = layer(output)
        return output.permute(0, 2, 1)
        
def main() -> None:
    log = Logger(writesLog=False)
    nbInstrPerSample = Constants.DEFAULT_SAMPLE_SIZE
    nbValuesPerInstr = Constants.DEFAULT_VALUES_PER_VECTOR
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device used: {device}\n")
    
    linear = torch.tensor([[float(f"{i}.{j}") for j in range(nbValuesPerInstr)] for i in range(nbInstrPerSample)],
                          dtype=torch.float32).unsqueeze(0).to(device)
    log.info(f"Base tensor: {linear}\n")
    
    model = ConvolutionModel(nbInstrPerSample, nbValuesPerInstr, device, \
                            channels=[32, 12, 1], kernels_size=[5, 5, 23], floatType=torch.float32, log=log)
    model.summary()
    with torch.no_grad():
        o = model(linear)
    log.info(f"Result tensor: {o}\n       Result shape: {o.shape}")
    
if __name__ == "__main__":
    main()
        
"""
End of file
"""