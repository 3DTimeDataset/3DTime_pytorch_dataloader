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
    try:
        sys.path.insert(0,"..")
        from Logger import *
        import Utils
        import Constants
    except:
        sys.path.insert(0,"../..")
        from Logger import *
        import Utils
        import Constants

import torch
from torch import nn, Tensor

class ConstantMeanModel(nn.Module):
    """
    Linear regression model class.

    Parameters:
    - input_vector_size:        input vector size for each G-code instruction
    - label_vector_size:        label vector size for each G-code instruction
    - device:                   a torch.device descriibing the device to send the model to (cpu or gpu)
    - dropout:                  dropout rate
    - floatType:                a torch.dtype describing the wanted float precision for the model
    - log:                      a Logger (see Logger.py in the parent folder), None by default (new logger without file writing)
    - generator:                a torch.Generator, used to control the randomness of the weights initialization, None by default
    - init_weights:             a boolean, True by default. If true, will initialize the trainable weights with a random uniform distribution
    """
    
    def __init__(self,
                input_vector_size: int,
                label_vector_size: int,
                device: torch.device,
                *,
                floatType: torch.dtype = torch.float32 if Constants.DEFAULT_FLOAT_TYPE == 32 else torch.float64,
                log: Logger = None,
                simple_rule_head: bool = False
            ) -> None:
        super(ConstantMeanModel, self).__init__()
        
        # Local variables
        if log == None:
            self.log = Logger(writesLog=False)
        else:
            self.log = log
        self.nbInstrPerSample = 100
        self.input_vector_size = input_vector_size
        self.label_vector_size = label_vector_size
        self.left_offset = 0
        self.right_offset = 100
        self.input_size = (100, input_vector_size)
        self.output_size = (100, label_vector_size)
        self.model_type = 'ConstantMean'
        self.device = device
        self.floatType = floatType
        self.trained_on_log_target = False
        self.simple_rule_head = simple_rule_head

        if all([v in UV.SELECTED_INPUTS for v in [6, 7]]) and UV.addDeltaCoordsAndAngle == UV.UPDATING_FUNCTIONS[-1] and UV.SELECTED_LABELS == [1]:
            self.right_offset = 99
            self.output_size = (99, label_vector_size)

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

        assert(self.right_offset - self.left_offset >= self.nbInstrPerSample - self.right_offset + self.left_offset)

        self.out = torch.tensor([Constants.DEFAULT_LABEL_MEAN_VALUES[i] for i in UV.SELECTED_LABELS], device=self.device, dtype=self.floatType)

    def summary(self) -> None:
        """
        Prints a summary of the model's architecture.
        """
        self.log.emptyLines()
        self.log.info("Model's architecture: Constant Mean Prediction")
        string = f" {'=' * 90}\n\t"
        string += f"Input shapes:               [batch_size, {self.nbInstrPerSample}, {self.input_vector_size}] (G-code vectors)\n\t"
        string += f"Output shape:               [batch_size, {self.right_offset - self.left_offset}, {self.label_vector_size}]\n\t"
        string += f"Uses trapeze pred head:     {self.simple_rule_head}\n\t"
        string += f"{'=' * 90}\n"
        self.log.info(string)
        self.log.emptyLines()

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        additionnalData = {}
        additionnalData["modelType"] = self.model_type
        additionnalData["instrPerSample"] = self.nbInstrPerSample
        additionnalData["inputVectorSize"] = self.input_vector_size
        additionnalData["labelVectorSize"] = self.label_vector_size
        additionnalData["leftOffset"] = self.left_offset
        additionnalData["rightOffset"] = self.right_offset
        additionnalData["trainedOnLogTarget"] = self.trained_on_log_target
        return additionnalData 
    
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

        output = self.out.unsqueeze(0).unsqueeze(1).expand(src.shape[0], src.shape[1], -1)

        if self.simple_rule_head:
            target_speed = src[..., self.target_speed_index].clone().unsqueeze(-1)
            extr_len = src[..., self.extr_len_index].abs().clone().unsqueeze(-1)
            delta_x = src[..., self.delta_x_index].clone().unsqueeze(-1)
            delta_y = src[..., self.delta_y_index].clone().unsqueeze(-1)
            delta_z = src[..., self.delta_z_index].clone().unsqueeze(-1)
            seg_len = src[..., self.seg_len_index].clone().unsqueeze(-1)
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

        if src.abs().sum() < 1e-8:
            return torch.zeros((src.shape[0], self.nbInstrPerSample, self.label_vector_size))
        return output
        
def main() -> None:
    log = Logger(writesLog=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device used: {device}\n")

    linear = torch.tensor([
        [float(f"{b}.{v}") for v in range(Constants.DEFAULT_INPUT_SIZE)] for b in range(Constants.DEFAULT_BATCH_SIZE)
    ], device=device)
    linear.unsqueeze(1)

    log.info(f"Base tensor: {linear}\n    Base tensor shape: {linear.shape}")
    
    model = ConstantMeanModel(Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, device, floatType=torch.float32, log=log)
    model.summary()
    with torch.no_grad():
        o = model(linear)
    log.info(f"Result tensor: {o}\n       Result shape: {o.shape}")
    
if __name__ == "__main__":
    main()
        
"""
End of file
"""