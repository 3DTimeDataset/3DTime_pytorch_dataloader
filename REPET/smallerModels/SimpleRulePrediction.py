import sys
from typing import Any, Mapping
try:
    from Logger import *
    import Utils
    import Constants
    import UpdateVectors as UV
except:
    try:
        sys.path.insert(0,"..")
        from Logger import *
        import Utils
        import Constants
        import UpdateVectors as UV
    except:
        sys.path.insert(0,"../..")
        from Logger import *
        import Utils
        import Constants
        import REPET.UpdateVectors as UV

import torch
from torch import nn, Tensor

class SimpleRuleModel(nn.Module):
    """
    Simple rule model class.

    Parameters:
    - input_vector_size:        input vector size for each G-code instruction
    - label_vector_size:        label vector size for each G-code instruction
    - device:                   a torch.device descriibing the device to send the model to (cpu or gpu)
    - floatType:                a torch.dtype describing the wanted float precision for the model
    - log:                      a Logger (see Logger.py in the parent folder), None by default (new logger without file writing)
    """
    
    def __init__(self,
                input_vector_size: int,
                label_vector_size: int,
                device: torch.device,
                *,
                floatType: torch.dtype = torch.float32 if Constants.DEFAULT_FLOAT_TYPE == 32 else torch.float64,
                log: Logger = None,
            ) -> None:
        super(SimpleRuleModel, self).__init__()
        
        # Local variables
        if log is None:
            self.log: Logger = Logger(writesLog=False)
        else:
            self.log: Logger = log
        self.nbInstrPerSample = 1
        self.input_vector_size = input_vector_size
        self.label_vector_size = label_vector_size
        self.left_offset = 0
        self.right_offset = 1
        self.input_size = (1, input_vector_size)
        self.output_size = (1, label_vector_size)
        self.model_type = 'SimpleRule'
        self.device = device
        self.floatType = floatType
        self.trained_on_log_target = False

        if UV.SELECTED_LABELS is not None:
            self.log.error(f"The simple rule model requires the full instruction label vector (aka `SELECTED_LABELS` in `UpdateVectors.py` must be `None`).", 1)

        for x in [3, 6, 8, 10]:
            if x not in UV.SELECTED_INPUTS:
                self.log.error(f"For the simple rule to work, the instruction input vectors must contain the following information: segment length, target speed, acceleration, and minimum cruise ratio.", 1)
        __indexes = [UV.SELECTED_INPUTS.index(x) for x in [3, 6, 8, 10]] if UV.SELECTED_INPUTS is not None else [3, 6, 8, 10]
        self.__segLen_index = __indexes[0]
        self.__targetSpeed_index = __indexes[1]
        self.__accel_index = __indexes[2]
        self.__mcr_index = __indexes[3]

    def summary(self) -> None:
        """
        Prints a summary of the model's architecture.
        """
        self.log.emptyLines()
        self.log.info("Model's architecture: Simple Rule")
        string = f" {'=' * 90}\n\t"
        string += f"Input shapes:               [batch_size, {self.nbInstrPerSample}, {self.input_vector_size}] (G-code vector)\n\t"
        string += f"Output shape:               [batch_size, {self.right_offset - self.left_offset}, {self.label_vector_size}]\n\t"
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
        self.trained_on_log_target = state_dict["trainedOnLogTarget"]
        del state_dict["modelType"], state_dict["instrPerSample"], state_dict["inputVectorSize"], state_dict["labelVectorSize"], state_dict["leftOffset"], state_dict["rightOffset"]
        del state_dict["trainedOnLogTarget"]
        return super().load_state_dict(state_dict, strict, assign)
    
    def getFilename(self):
        filename = f"{self.model_type}_iVecSize-{self.input_vector_size}_lVecSize-{self.label_vector_size}"
        return filename

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.

        Input shapes:     src=[batch_size, nbInstrPerSample, input_vector_size]

        Output shape:    [batch_size, nbInstrPerSample, label_vector_size]
        """
        batch_size = src.shape[0]
        if batch_size > 1:
            src = src.squeeze()
        result = torch.zeros((batch_size, 7), device=src.device, dtype=src.dtype)
        d = src[..., self.__accel_index]# * (1 - src[..., self.__mcr_index])
        t_a = src[..., self.__targetSpeed_index] / (src[..., self.__accel_index] * 60 + 1e-3)
        t_d = src[..., self.__targetSpeed_index] / (d * 60 + 1e-3)
        d_a = 1/2 * t_a * (src[..., self.__targetSpeed_index] / 60)
        d_d = 1/2 * t_d * (src[..., self.__targetSpeed_index] / 60)
        d_c = src[..., self.__segLen_index] - d_a - d_d
        is_d_c_ok = (d_c >= (src[..., self.__segLen_index] * src[..., self.__mcr_index]))
        result[is_d_c_ok, 5] = d_c[is_d_c_ok] / (src[is_d_c_ok, self.__targetSpeed_index] / 60 + 1e-3)
        result[is_d_c_ok, 4] = t_a[is_d_c_ok]
        result[is_d_c_ok, 6] = t_d[is_d_c_ok]
        result[is_d_c_ok, 0] = t_a[is_d_c_ok] + t_d[is_d_c_ok] + result[is_d_c_ok, 5]
        result[is_d_c_ok, 2] = (src[is_d_c_ok, self.__targetSpeed_index] / 60)
        is_d_c_ok = torch.logical_not(is_d_c_ok)
        new_F = torch.sqrt((2 * src[is_d_c_ok, self.__segLen_index] * (1 - src[is_d_c_ok, self.__mcr_index])) / (1/(src[is_d_c_ok, self.__accel_index] + 1e-3) + 1/(d[is_d_c_ok] + 1e-3)) + 1e-3)
        n_d_c = src[is_d_c_ok, self.__segLen_index] * src[is_d_c_ok, self.__mcr_index]
        n_t_a = new_F / (src[is_d_c_ok, self.__accel_index] + 1e-3)
        n_t_c = n_d_c / (new_F + 1e-3)
        n_t_d = new_F / (d[is_d_c_ok] + 1e-3)
        result[is_d_c_ok, 0] = n_t_a + n_t_c + n_t_d
        result[..., 1] = 0
        result[is_d_c_ok, 2] = new_F
        result[..., 3] = 0
        result[is_d_c_ok, 4] = n_t_a
        result[is_d_c_ok, 5] = n_t_c
        result[is_d_c_ok, 6] = n_t_d
        return result.unsqueeze(1)
        
def main() -> None:
    log = Logger(writesLog=False, verbose_level=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device used: {device}\n")

    linear = torch.tensor([
        [float(f"0.{b}{v}") for v in range(Constants.DEFAULT_INPUT_SIZE)] for b in range(Constants.DEFAULT_BATCH_SIZE)
    ], device=device)
    linear.unsqueeze(1)

    log.info(f"Base tensor: {linear}\n    Base tensor shape: {linear.shape}")
    
    model = SimpleRuleModel(Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, device, floatType=torch.float32, log=log)
    model.summary()
    with torch.no_grad():
        o = model(linear)
    log.info(f"Result tensor: {o}\n       Result shape: {o.shape}")
    
if __name__ == "__main__":
    main()
        
"""
End of file
"""