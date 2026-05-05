"""
This is the only script to change in order to update the input and label vectors.

The main three variables to change are:
- `SELECTED_INPUTS`: lists the base input vectors indexes to retain for the models (default is None, meaning all values are retained)
- `SELECTED_LABELS`: lists the base label vectors indexes to retain for the models (default is None, meaning all values are retained)
- `UPDATING_FUNCTIONS`: lists all the functions to call to add values to the input vectors.
    + For each function, the input parameters must be:
        + Base, unmasked, input tensor (of shape `[batch_size, sequence_length, base_input_vector_size]`)
        + Base, unmasked, label tensor (of shape `[batch_size, sequence_length, base_label_vector_size]`)
        + The dataset's input sequence left offset
        + The dataset's input sequence right offset
    + Each function must return a tuple of two tensor:
        + Both of shape `[batch_size, sequence_length, whatever]`
        + Both can have different `whatever` size
        + Both can be set to `None`, meaning they will be ignored
        + They represent the values to add to the vectors, respectivelly to the input and label vectors, per G-code instruction
    + Two examples of such function are given bellow

Note:

By default, the extracted vectors contain the following values:
- Input vector:
    + Start X coordinates
    + Start Y coordinates
    + Start Z coordinates
    + End X coordinates
    + End Y coordinates
    + End Z coordinates
    + Actual target speed (minimum between the `F` in the G-code, and the printer configuration set maximum velocity, in `mm/min`)
    + Extrusion distance (in `mm`)
    + Segment type, as a bitField encoded in float format (see functions `decodeMoveTypeFromBitField` and `tensorMaskForMoveType` in Utils.py to decode this bitfield):
- Label vector:
    + Total instruction time (in `s`)
    + Instruction start velocity (in `mm/s`)
    + Instruction actual cruise velocity (in `mm/s`)
    + Instruction end velocity (in `mm/s`)
    + Instruction acceleration time (in `s`)
    + Instruction cruise time (in `s`)
    + Instruction deceleration time (in `s`)
"""

import sys
try:
    import Constants
    import Logger
except:
    sys.path.insert(0,"..")
    import Constants
    import Logger
import torch
from typing import Callable, Any, Tuple
log = Logger.Logger(writesLog=False, verbose_level=0, prints_thread_id=False)

##############################################################################################################################################################
"""
Put the vector updating functions here
"""

def addDeltaCoordsAndAngle(input: torch.Tensor, label: torch.Tensor, l_offset: int, r_offset: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    result = torch.zeros((input.shape[0], input.shape[1], 5), device=input.device, dtype=input.dtype)

    # ΔX, ΔY, ΔZ
    result[..., 0] = (input[..., 0] - input[..., 3]) # Delta X
    result[..., 1] = (input[..., 1] - input[..., 4]) # Delta Y
    result[..., 2] = (input[..., 2] - input[..., 5]) # Delta Z

    # Segment length (3D)
    result[..., 3] = torch.sqrt(result[..., 0]**2 + result[..., 1]**2 + result[..., 2]**2)

    # Current 2D vector
    curr = result[..., :2]
    prev = torch.nn.functional.pad(curr[:, :-1], (0, 0, 1, 0))  # [B, S-1, 2] → pad front → [B, S, 2]

    # Angle computation
    dot = (prev * curr).sum(dim=-1) # \vec{v}_i \cdot \vec{v}_{i-1}
    norm_prod = (prev.norm(dim=-1) * curr.norm(dim=-1)).clamp(min=1e-6) # \|\vec{v}_i\| \cdot \|\vec{v}_{i-1}\|
    cos_angle = (dot / norm_prod).clamp(-1.0, 1.0)
    angles = torch.acos(cos_angle)

    # Set angle = 0 where vectors are near-zero
    mask = (prev.norm(dim=-1) < 1e-6) | (curr.norm(dim=-1) < 1e-6)
    angles[mask] = 0.0
    result[..., 4] = angles

    # Update the delta values to absolute since their relative values are not needed anymore
    result[..., :3] = result[..., :3].abs()

    return result, None

def giveTrapezValues(input: torch.Tensor, label: torch.Tensor, l_offset: int, r_offset: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Prodides the required values for the fixed prediction from trapez model.

    NOTE: This function is used for "cheating", as it copies the the start and end velocities from the label tensor as part of the input tensor

    This function is hence only used for testing, validation, and stats.

    Output tensor: start vel, end vel, delta x, delta y, delta z, seg len (6 values)

    If the instruction is a prime or a retract move, the seg len is the extrusion distance
    """
    result = torch.zeros((input.shape[0], input.shape[1], 6), device=input.device, dtype=input.dtype)

    result[:, l_offset:r_offset, 0] = label[..., 1]
    result[:, l_offset:r_offset, 1] = label[..., 3]

    # Segment length
    result[:, l_offset:r_offset, 2] = torch.abs(input[:, l_offset:r_offset, 0] - input[:, l_offset:r_offset, 3]) # Delta X
    result[:, l_offset:r_offset, 3] = torch.abs(input[:, l_offset:r_offset, 1] - input[:, l_offset:r_offset, 4]) # Delta Y
    result[:, l_offset:r_offset, 4] = torch.abs(input[:, l_offset:r_offset, 2] - input[:, l_offset:r_offset, 5]) # Delta Z
    result[:, l_offset:r_offset, 5] = torch.sqrt(result[:, l_offset:r_offset, 2]**2 + result[:, l_offset:r_offset, 3]**2 + result[:, l_offset:r_offset, 4]**2) # Seg len

    is_len_zero = (result[:, l_offset:r_offset, 5] < 1e-8) # If the len is 0, it means that instruction is a E only move
    b_idx, t_idx = is_len_zero.nonzero(as_tuple=True)

    result[b_idx, t_idx, 5] = torch.abs(input[b_idx, t_idx, 7])

    return result, None


##############################################################################################################################################################
"""
Update the next three variables as needed
"""


# NOTE: Change this by your needs

#SELECTED_INPUTS: list[int] | None = [6] # Fixed pred from trapez vectors mode
#SELECTED_INPUTS: list[int] | None = [6, 7] # Upgraded vectors mode
SELECTED_INPUTS: list[int] | None = None # Default vectors mode
"""
List of the base input vector values to retain, must be either None (all values are retained, or a set of indexes within `[0, DEFAULT_INPUT_SIZE[`)

Example:
`SELECTED_INPUTS = [3, 5, 6, 7]`
"""



# NOTE: Change this by your needs
SELECTED_LABELS: list[int] | None = None
#SELECTED_LABELS: list[int] | None = [1]
"""
List of the base label vector values to retain, must be either None (all values are retained, or a set of indexes within `[0, DEFAULT_LABEL_SIZE[`)

Example:
`SELECTED_LABELS = [0, 4, 5, 6]`
"""



# NOTE: Change this by your needs
UPDATING_FUNCTIONS: list[Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor | None, torch.Tensor | None]]] = [addDeltaCoordsAndAngle] # Upgraded vectors mode
#UPDATING_FUNCTIONS: list[Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor | None, torch.Tensor | None]]] = [] # Default vectors mode
#UPDATING_FUNCTIONS: list[Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor | None, torch.Tensor | None]]] = [giveTrapezValues] # Prediction from Trapez mode
"""
List of the functions to call for input vector updates, each must take as input the base (masked) input tensor, and return a tensor of the new values to add.

Example:
`UPDATING_FUNCTIONS = [addDeltaCoordsAndAngle]`
"""


##############################################################################################################################################################
"""
Do not change anything here
"""

if SELECTED_INPUTS is not None: # Ensure the variable is OK
    assert(all(0 <= i < Constants.DEFAULT_INPUT_SIZE for i in SELECTED_INPUTS))
    assert(all(x1 < x2 for x1, x2 in zip(SELECTED_INPUTS[:-1], SELECTED_INPUTS[1:])))

if SELECTED_LABELS is not None: # Ensure the variable is OK
    assert(all(0 <= i < Constants.DEFAULT_LABEL_SIZE for i in SELECTED_LABELS))
    assert(all(x1 < x2 for x1, x2 in zip(SELECTED_LABELS[:-1], SELECTED_LABELS[1:])))

# Ensure all given vector updating functions are correct
__test_input = torch.randn(2, Constants.DEFAULT_SAMPLE_SIZE, Constants.DEFAULT_INPUT_SIZE)
__test_label = torch.randn(2, Constants.DEFAULT_RIGHT_OFFSET - Constants.DEFAULT_LEFT_OFFSET, Constants.DEFAULT_LABEL_SIZE)
__nb_added_values_input = 0
__nb_added_values_label = 0
for func in UPDATING_FUNCTIONS:
    try:
        i, l = func(__test_input, __test_label, Constants.DEFAULT_LEFT_OFFSET, Constants.DEFAULT_RIGHT_OFFSET)
    except BaseException as e:
        log.error(
            f"For the given input vector extension function '{func.__name__}', there was an exception at runtime.", 1, e
            )
    try:
        if i is None:
            pass
        elif len(i.shape) == 3:
            assert(i.shape[:2] == (2, Constants.DEFAULT_SAMPLE_SIZE))
            __nb_added_values_input += i.shape[2]
        elif len(i.shape) == 2:
            assert(i.shape == (2, Constants.DEFAULT_SAMPLE_SIZE))
            __nb_added_values_input += 1
        else:
            raise AssertionError
    except AssertionError as e:
        log.error(
            f"For the given input vector extension function '{func.__name__}', the result input tensor is not if the correct shape: {i.shape}.", 1, e
            )
    except BaseException as e:
        log.error(
            f"For the given input vector extension function '{func.__name__}', failed to test result input shape ({i.shape}). \
Are you sure the result tensor is correct?", 1, e
            )
    try:
        if l is None:
            pass
        elif len(l.shape) == 3:
            assert(l.shape[:2] == (2, Constants.DEFAULT_RIGHT_OFFSET - Constants.DEFAULT_LEFT_OFFSET))
            __nb_added_values_label += l.shape[2]
        elif len(l.shape) == 2:
            assert(l.shape == (2, Constants.DEFAULT_RIGHT_OFFSET - Constants.DEFAULT_LEFT_OFFSET))
            __nb_added_values_label += 1
        else:
            raise AssertionError
    except AssertionError as e:
        log.error(
            f"For the given input vector extension function '{func.__name__}', the result label tensor is not if the correct shape: {l.shape}.", 1, e
            )
    except BaseException as e:
        log.error(
            f"For the given input vector extension function '{func.__name__}', failed to test result label shape ({l.shape}). \
Are you sure the result tensor is correct?", 1, e
            )
    del i, l
del __test_input, __test_label
# These two variables are only used for dataset and models initialization for other scripts, DO NOT CHANGE THEM
TRUE_INPUT_VECTOR_SIZE = Constants.DEFAULT_INPUT_SIZE if SELECTED_INPUTS is None else len(SELECTED_INPUTS)
TRUE_INPUT_VECTOR_SIZE += __nb_added_values_input
TRUE_LABEL_VECTOR_SIZE = Constants.DEFAULT_LABEL_SIZE if SELECTED_LABELS is None else len(SELECTED_LABELS)
TRUE_LABEL_VECTOR_SIZE += __nb_added_values_label