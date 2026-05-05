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
from math import pi

class PredFromTrapez(nn.Module):
    """
    TODO:

    Note: requires the "giveTrapezValues" function in the updated vectors, and no other values

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
                nbInstrPerSample: int,
                device: torch.device,
                *,
                floatType: torch.dtype = torch.float32 if Constants.DEFAULT_FLOAT_TYPE == 32 else torch.float64,
                log: Logger = None,
                is_prediction_head: bool = False
            ) -> None:
        super(PredFromTrapez, self).__init__()
        
        # Local variables
        if log is None:
            self.log: Logger = Logger(writesLog=False)
        else:
            self.log: Logger = log
        self.nbInstrPerSample = nbInstrPerSample
        self.input_vector_size = input_vector_size
        if self.input_vector_size != 7:
            log.error(f"The simple rule trapeze prediction model must have an input vector size of 7 (target speed, start vel, end vel, delta x, delta y, delta z, seg len)", 1)
        self.label_vector_size = label_vector_size
        if self.label_vector_size != 7:
            log.error(f"The simple rule trapeze prediction model must have a label vector size of 7 (all default label values)", 1)
        self.left_offset = 0
        self.right_offset = self.nbInstrPerSample
        self.input_size = (self.nbInstrPerSample, input_vector_size)
        self.output_size = (self.nbInstrPerSample, label_vector_size)
        self.model_type = 'FromTrapez'
        self.device = device
        self.floatType = floatType
        self.trained_on_log_target = False
        self.is_prediction_head = is_prediction_head

        self.nb_below_zero = 0
        self.nb_above_target_speed = 0
        self.nb_endVel_too_high = 0
        self.nb_startVel_too_high = 0
        self.tot_instr = 0

        # Printer configuration values
        self.def_accel = Constants.DEFAULT_XY_ACCEL
        self.def_decel = Constants.DEFAULT_XY_DECEL
        self.z_accel = Constants.DEFAULT_Z_ACCEL
        self.z_decel = Constants.DEFAULT_Z_DECEL
        if Constants.DEFAULT_E_ONLY_ACCEL is None:
            # If no E only acceleration is provided, process it based on the nozzle and filament diameters
            # This formula was taken from the Klipper source code
            self.e_accel = self.def_accel * ( (4 * Constants.DEFAULT_NOZZLE_DIAMETER**2) / (pi * (0.5 * Constants.DEFAULT_FILAMENT_DIAMETER)**2) )
            self.e_decel = self.e_accel
        else:
            self.e_accel = Constants.DEFAULT_E_ONLY_ACCEL
            self.e_decel = Constants.DEFAULT_E_ONLY_ACCEL
        self.mcr = Constants.DEFAULT_MINIMUM_CRUISE_RATIO

        if not (0 <= self.mcr <= 1):
            self.log.error(f"The minimum cruise ratio must be within 0 and 1. Got value: {self.mcr}.", 1)

        if not self.is_prediction_head and UV.SELECTED_LABELS is not None:
            self.log.error(f"The prediction from trapez model requires the full instruction label vector (aka `SELECTED_LABELS` in `UpdateVectors.py` must be `None`).", 1)
        if not self.is_prediction_head and (UV.giveTrapezValues not in UV.UPDATING_FUNCTIONS or (len(UV.UPDATING_FUNCTIONS) > 1)):
            self.log.error(f"The prediction from trapez model requires the `giveTrapezValues` function in the `UPDATING_FUNCTION` list, and no other function.", 1)

    def summary(self) -> None:
        """
        Prints a summary of the model's architecture.
        """
        self.log.emptyLines()
        self.log.info("Model's architecture: Trapez processing from speeds")
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
        filename = f"{self.model_type}"
        return filename

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.

        Input shapes:     src=[batch_size, nbInstrPerSample, input_vector_size]

        Output shape:    [batch_size, nbInstrPerSample, label_vector_size]

        Input vector values: target speed, start vel, end vel, delta x, delta y, delta z, seg len
        """

        # Result tensor, containing all 7 label values
        result = torch.zeros((src.shape[0], *self.output_size), device=src.device, dtype=src.dtype)

        # Placeholder tensors for the accel decel and mcr, used for processing
        mcr = torch.full((src.shape[0], src.shape[1]), self.mcr, device=self.device, dtype=src.dtype)
        accel = torch.full((src.shape[0], src.shape[1]), self.def_accel, device=self.device, dtype=src.dtype)
        decel = torch.full((src.shape[0], src.shape[1]), self.def_decel, device=self.device, dtype=src.dtype)

        # Normalize speeds to mm/s instead of mm/min
        speeds = src[..., 0] / 60

        # Updates the accels, decels, and mcr based on the 3 distinct types of instructions
        # E only moves get special accel and decel
        # Moves with Z get special accel and decel
        # E only or Z only get a mcr of 0
        contains_z_movement = (src[..., 5] > 1e-7)
        e_only_move = (src[..., 3:6] < 1e-7).all(dim=-1)
        z_only_move = (src[..., 3:5] < 1e-7).all(dim=-1).logical_and(src[..., 5] > 1e-7)
        b_idx, t_idx = contains_z_movement.nonzero(as_tuple=True)
        accel[b_idx, t_idx] = self.z_accel
        decel[b_idx, t_idx] = self.z_decel
        b_idx, t_idx = e_only_move.nonzero(as_tuple=True)
        accel[b_idx, t_idx] = self.e_accel
        decel[b_idx, t_idx] = self.e_decel
        b_idx, t_idx = e_only_move.logical_or(z_only_move).nonzero(as_tuple=True)
        mcr[b_idx, t_idx] = 0.0

        # Some checks on the start and end velocities
        if self.is_prediction_head:
            below_zero = (src[..., 1] < 0)
            above_target_speed = (src[..., 1] > speeds)
            endVel_too_high = (src[..., 2] > src[..., 1] + accel * src[..., 6])
            startVel_too_high = (src[..., 1] > src[..., 2] + decel * src[..., 6])
            self.nb_below_zero += below_zero.sum().item()
            self.nb_above_target_speed += above_target_speed.sum().item()
            self.nb_startVel_too_high = startVel_too_high.sum().item()
            self.nb_endVel_too_high = endVel_too_high.sum().item()
            b_idx, t_idx = below_zero.nonzero(as_tuple=True)
            src[b_idx, t_idx, 1] = 0
            end_below_zero = (src[..., 2] < 0)
            b_idx, t_idx = end_below_zero.nonzero(as_tuple=True)
            src[b_idx, t_idx, 2] = 0
            end_above_target_speed = (src[..., 2] > speeds)
            b_idx, t_idx = end_above_target_speed.nonzero(as_tuple=True)
            src[b_idx, t_idx, 2] = speeds[b_idx, t_idx]
            b_idx, t_idx = above_target_speed.nonzero(as_tuple=True)
            src[b_idx, t_idx, 1] = speeds[b_idx, t_idx]
            b_idx, t_idx = startVel_too_high.nonzero(as_tuple=True)
            src[b_idx, t_idx, 1] = src[b_idx, t_idx, 2] + decel[b_idx, t_idx] * src[b_idx, t_idx, 6]
            b_idx, t_idx = endVel_too_high.nonzero(as_tuple=True)
            src[b_idx, t_idx, 2] = src[b_idx, t_idx, 1] + accel[b_idx, t_idx] * src[b_idx, t_idx, 6]
            self.tot_instr += accel.numel()

        # Pre-process some values that are used several times
        f_2 = speeds**2
        vin_2 = src[..., 1]**2
        vout_2 = src[..., 2]**2
        inv_2a = 1 / (2 * accel + 1e-8)
        inv_2d = 1 / (2 * decel + 1e-8)

        # Process acceleration and deceleration distances
        d_a = inv_2a * (f_2 - vin_2)
        d_d = inv_2d * (f_2 - vout_2)

        # Process first guess of the cruise distance
        d_c = src[..., -1] - d_a - d_d

        # Check for each instruction if the cruise distance is good (aka geq than total distance (known) times mcr)
        is_dc_not_ok = (d_c < src[..., -1] * mcr)
        b_idx, t_idx = is_dc_not_ok.nonzero(as_tuple=True)
        d_c[b_idx, t_idx] = src[b_idx, t_idx, -1] * mcr[b_idx, t_idx] # Update actual cruise distance for instructions that do not satisfy this condition

        """
        Process the new cruise speed F' for instructions with updated cruise distance. Formula is:

        \begin{equation}
            F' = sqrt{ \frac{ D \times \left( 1 - \text{mcr} \right) + \frac{v_{\text{in}}^2}{2a} + \frac{v_{\text{out}}^2}{2d} }{ \frac{1}{2a} + \frac{1}{2d} } }
        \end{equation}

        \noindent with $D$ the total instruction segment length (known) in mm, $mcr$ the Minimum Cruise Ratio (known) between 0 and 1, $v_{\text{in}}$ the instruction start velocity (processed) in mm/s, $v_{\text{out}}$ the instruction end velocity (processed) in mm/s, $a$ the acceleration (known, but different between X and Y, Z, and E only) in mm/s², and $a$ the (absolute) deceleration (known, but different between X and Y, Z, and E only) in mm/s².

        Note that in practice, the acceleration and deceleration are equal in most cases.
        """
        speeds[b_idx, t_idx] = torch.sqrt(
            (src[b_idx, t_idx, -1] * (1 - mcr[b_idx, t_idx]) + (vin_2[b_idx, t_idx] / (2 * accel[b_idx, t_idx])) + (vout_2[b_idx, t_idx] / (2 * decel[b_idx, t_idx]))) / (inv_2a[b_idx, t_idx] + inv_2d[b_idx, t_idx])
        )

        # Process the acceleration, cruise, deceleration, and total durations
        t_a = (speeds - src[..., 1]) / (accel + 1e-8)

        # Third case scenario: cruise velocity is smaller than start velocity
        negative_t_a = t_a < 0
        b_idx, t_idx = negative_t_a.nonzero(as_tuple=True)
        speeds[b_idx, t_idx] = src[b_idx, t_idx, 1] # Force cruise velocity equal to start velocity
        # In that scenario, d_c = D - d_d
        d_c[b_idx, t_idx] = src[b_idx, t_idx, -1] - inv_2d[b_idx, t_idx] * (speeds[b_idx, t_idx]**2 - vout_2[b_idx, t_idx])
        t_a[b_idx, t_idx] = 0

        t_d = (speeds - src[..., 2]) / (decel + 1e-8)

        # Fourth case scenario: cruise velocity is smaller than end velocity
        negative_t_d = t_d < 0
        b_idx, t_idx = negative_t_d.nonzero(as_tuple=True)
        speeds[b_idx, t_idx] = src[b_idx, t_idx, 2] # Force cruise velocity equal to end velocity
        # In that scenario, d_c = D - d_a
        d_c[b_idx, t_idx] = src[b_idx, t_idx, -1] - inv_2a[b_idx, t_idx] * (speeds[b_idx, t_idx]**2 - vin_2[b_idx, t_idx])
        t_d[b_idx, t_idx] = 0
        t_a[b_idx, t_idx] = (speeds[b_idx, t_idx] - src[b_idx, t_idx, 1]) / (accel[b_idx, t_idx] + 1e-8)

        # Final times processings
        t_c = d_c / (speeds + 1e-8)
        t_t = t_a + t_c + t_d

        # Put the results in the result vector
        result[..., 0] = t_t[:, self.left_offset:self.right_offset]
        result[..., 1] = src[:, self.left_offset:self.right_offset, 1]
        result[..., 2] = speeds[:, self.left_offset:self.right_offset]
        result[..., 3] = src[:, self.left_offset:self.right_offset, 2]
        result[..., 4] = t_a[:, self.left_offset:self.right_offset]
        result[..., 5] = t_c[:, self.left_offset:self.right_offset]
        result[..., 6] = t_d[:, self.left_offset:self.right_offset]

        return result
    
    def __del__(self) -> None:
        if self.tot_instr >0 and self.is_prediction_head:
            self.log.info(f"Trapeze prediction head: found a total of {self.nb_below_zero} ({self.nb_below_zero * 100 / self.tot_instr} %) instructions with a predicted start velocity below zero, {self.nb_above_target_speed} ({self.nb_above_target_speed * 100 / self.tot_instr} %) instructions with a predicted start velocity above the target speed, {self.nb_endVel_too_high} ({self.nb_endVel_too_high * 100 / self.tot_instr} %) with an end velocity too high compared to the start velocity and instruction length, and {self.nb_startVel_too_high} ({self.nb_startVel_too_high * 100 / self.tot_instr} %) with a start velocity too high compared to the end velocity and instruction length.")
        
def main() -> None:
    log = Logger(writesLog=False, verbose_level=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device used: {device}\n")

    linear = torch.rand((Constants.DEFAULT_BATCH_SIZE, Constants.DEFAULT_SAMPLE_SIZE, UV.TRUE_INPUT_VECTOR_SIZE), device=device)

    log.info(f"Base tensor: {linear}\n    Base tensor shape: {linear.shape}")
    
    model = PredFromTrapez(Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, Constants.DEFAULT_SAMPLE_SIZE - 1, device, floatType=torch.float32, log=log)
    model.summary()
    with torch.no_grad():
        o = model(linear)
    log.info(f"Result tensor: {o}\n       Result shape: {o.shape}")
    
if __name__ == "__main__":
    main()
        
"""
End of file
"""