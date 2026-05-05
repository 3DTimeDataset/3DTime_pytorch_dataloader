import random
from torch import nn

# Default values

DEFAULT_BATCH_SIZE = 256
"""
Default batch size for ML trainings.
"""
DEFAULT_SAMPLE_SIZE = 100
"""
Default number of instruction as ML models input.
"""
DEFAULT_LEFT_OFFSET = 0
"""
Default prediction window left offset.

Set so that:

`0 <= left_offset < right_offset <= sample_size`

And:

`right_offset - left_offset >= sample_size - right_offset + left_offset`
"""
DEFAULT_RIGHT_OFFSET = DEFAULT_SAMPLE_SIZE - DEFAULT_LEFT_OFFSET
"""
Default prediction window right offset.

Set so that:

`0 <= left_offset < right_offset <= sample_size`

And:

`right_offset - left_offset >= sample_size - right_offset + left_offset`
"""
DEFAULT_NB_EPOCH = 60
"""
Default number of ML training epochs.
"""
DEFAULT_STOP_CONDITION = 60
"""
Default ML training stop condiction (number of epochs without progress needed to stop the training early).
"""
DEFAULT_MODEL_FOLDER = "./models"
"""
Default output folder, to save the models' parameters after training.
"""
DEFAULT_LEARNING_RATE = 0.0001 # Use 0.0001 for small and medium model, use 0.00005 for large model
"""
Default ML training learning rate.
"""
DEFAULT_DROPOUT_RATE = 0.0
"""
Default ML training dropout rate for required layers.
"""
DEFAULT_EMBED_DIM = 32
"""
Default Transformer model's embedding dimension, or Recurrent (BiLSTM) model's hidden size.
"""
DEFAULT_NB_HEAD = 1
"""
Default Transformer model's number of attention head.
"""
DEFAULT_LAYER_COUNT = 1
"""
Default number of Transformer or Recurrent model's layer.
"""
DEFAULT_FEEDF_DIM = 256
"""
Default Transformer model's feed forward layers size.
"""
DEFAULT_FLOAT_TYPE = 32
"""
Default Pytorch float type, must be either 16 (float16), 32 (float32), or 64 (float64).

NOTE: Pytorch's Transformer normalization layers currently have a bug, making them unable to support float16 (last checked: Pytorch 2.2.0),\
	hence the float16 type is not supported.
"""
DEFAULT_KERNELS = [5, 5, 23]
"""
Default Convolution model's 1D kernel sizes.

NOTE: The number of output instruction for the Convolution models depend on those values:

` nb_output_instr = nb_input_instr - ( \\sum{ i }{ \\text{DEFAULT_KERNELS}_i -1 } )`
"""
DEFAULT_CHANNELS = [32, 16, 7]
"""
Default Convolution model's layers number of output channels (first layer's number of input channels is the number of values per G-code instruction).

NOTE: The last layer must have exactly one output channel.
"""
DEFAULT_VALUES_PER_VECTOR = 16
"""
Default number of values in each G-code instruction extracted vector, depends on the data extraction Gato3D pass, and the Dataset implementation:
- ` gato/apps/timingDataGetter/timingDataGetterPass.h `
- ` REPET/VectorDataset.py `
"""
DEFAULT_INPUT_SIZE = 9
"""
Default number of values for each instruction input vector (aka full instruction vector without labels), depends on the data extraction Gato3D pass.
"""
DEFAULT_LABEL_SIZE = 7
"""
Default number of values for each instruction label vector (aka full instruction vector without inputs), depends on the data extraction Gato3D pass.
"""

DEFAULT_FCC_SUBWINDOW = 25
"""
Default fully connected models prediction sub-window.
"""
DEFAULT_FCC_PREDICTED_INSTR = 15
"""
Default index of the single predicted instruction per sub-window for the fully connected models.
"""
DEFAULT_FCC_LAYERS = [DEFAULT_FCC_SUBWINDOW * DEFAULT_INPUT_SIZE, DEFAULT_FCC_SUBWINDOW * 2, DEFAULT_FCC_SUBWINDOW, DEFAULT_FCC_SUBWINDOW // 2, DEFAULT_LABEL_SIZE]
"""
Default fully connected models linear layer input and output shapes.
"""

RANDOM_SEED = 12345 # Seed for pseudo random
"""
Manual seed for the python random library, and the pytorch generators.
"""
random.seed(RANDOM_SEED)

TRANSFORMER_DICT = {
	"instrPerSample": DEFAULT_SAMPLE_SIZE,
	"inputVectorSize": DEFAULT_INPUT_SIZE,
	"labelVectorSize": DEFAULT_LABEL_SIZE,
	"leftOffset": DEFAULT_LEFT_OFFSET,
	"rightOffset": DEFAULT_RIGHT_OFFSET,
	"embedDim": DEFAULT_EMBED_DIM,
	"headCount": DEFAULT_NB_HEAD,
	"fForwardDim": DEFAULT_FEEDF_DIM,
	"nbLayer": DEFAULT_LAYER_COUNT
}
"""
Transformer model's required architecture parameters, and their default values. Used for model parsing from file.
"""
LSTM_DICT = {
	"instrPerSample": DEFAULT_SAMPLE_SIZE,
	"inputVectorSize": DEFAULT_INPUT_SIZE,
	"labelVectorSize": DEFAULT_LABEL_SIZE,
	"leftOffset": DEFAULT_LEFT_OFFSET,
	"rightOffset": DEFAULT_RIGHT_OFFSET,
	"hiddenSize": DEFAULT_EMBED_DIM,
	"nbLayer": DEFAULT_LAYER_COUNT
}
"""
Recurrent model's required architecture parameters, and their default values. Used for model parsing from file.
"""
CONVOLUTION_DICT = {
	"instrPerSample": DEFAULT_SAMPLE_SIZE,
	"inputVectorSize": DEFAULT_INPUT_SIZE,
	"labelVectorSize": DEFAULT_LABEL_SIZE,
	"leftOffset": DEFAULT_LEFT_OFFSET,
	"rightOffset": DEFAULT_RIGHT_OFFSET,
	"kernelLengths": DEFAULT_KERNELS,
	"nbChannels": DEFAULT_CHANNELS
}
"""
Convolution model's required architecture parameters, and their default values. Used for model parsing from file.
"""
FCC_DICT = {
    "instrPerSample": DEFAULT_SAMPLE_SIZE,
    "inputVectorSize": DEFAULT_INPUT_SIZE,
    "labelVectorSize": DEFAULT_LABEL_SIZE,
    "subWindowSize": DEFAULT_FCC_SUBWINDOW,
    "predictedInstr": DEFAULT_FCC_PREDICTED_INSTR,
    "layerSizes": DEFAULT_FCC_LAYERS
}
"""
Fully Connected model's required architecture parameters, and their default values. Used for model parsing from file.
"""
LINEAR_DICT = {
    "instrPerSample": 1,
    "inputVectorSize": DEFAULT_INPUT_SIZE,
    "labelVectorSize": DEFAULT_LABEL_SIZE,
    "hiddenSizes": DEFAULT_FCC_LAYERS
}
"""
Linear Regression model's required architecture parameters, and their default values. Used for model parsing from file.
"""

DEFAULT_LABEL_MEAN_VALUES = [
    0.0545955, # Arithmetic mean value of the total instruction print durations
    27.526998, # Arithmetic mean value of the instruction start velocities
    45.991583, # Arithmetic mean value of the instruction cruise velocities
    27.527000, # Arithmetic mean value of the instruction end velocities
    0.0054306, # Arithmetic mean value of the instruction acceleration durations
    0.0437357, # Arithmetic mean value of the instruction cruise durations
    0.0054292  # Arithmetic mean value of the instruction deceleration durations
]
"""
Training dataset's mean label values, used only for the constant mean prediction model (as a reference for other models)
"""

MEASURED_MINIMUM_INSTR_TOTAL_TIME = 8.333e-6
"""
Measured smallest instruction (by total duration in seconds) in the full dataset, equivalent to 8.333 µs.
"""

MEASURED_MAXIMUM_INSTR_TOTAL_TIME = 20.009
"""
Measured largest instruction (by total duration in seconds) in the full dataset.
"""

CONST_MEAN_DICT = {
    "instrPerSample": 1,
    "inputVectorSize": DEFAULT_INPUT_SIZE,
    "labelVectorSize": DEFAULT_LABEL_SIZE,
    "labelMeanValues": DEFAULT_LABEL_MEAN_VALUES
}
"""
Constant mean model's required architecture parameters, and their default values. Used for model parsing from file.
"""

#    "unknown",
MOVE_TYPES = [
    "wallInner",
    "wallOuter",
    "skin",
    "infill",
    "skirt",
    "supportMaterial",
    "supportInterface",
    "print",
    "wall",
    "support",
    "bridge",
    "thin",
    "overhang",
    "gap",
    "sparse",
    "internal",
    "retract",
    "prime",
    "travel",
    "zlift",
    "combing"
]
"""
List of the different move types, as given by Gato3D, in the file `gato3d/include/Common/MoveType.h`.
"""

# Loss function
DEFAULT_LOSS_TYPE = "mse"
"""
Default loss function type, either 'logcosh', 'mse', 'huber', or 'smape'.
"""

DEFAULT_KENDALL_MULTI_TASK = False
"""
Default loss function Kendall's multi task loss weight optimization behavior.
"""

DEFAULT_LOSS_LOG_NORM = False
"""
Default loss function log normalization of the output and predicted values before loss behavior.
"""

DEFAULT_LOSS_EPOCH_ALPHA = 0.0
"""
Default loss task weight by epoch alpha.
"""

DEFAULT_LOSS_EPOCH_BETA = 0.13
"""
Default loss task weight by epoch beta.
"""

DEFAULT_LOG_NORM_EPSILON = 1e-3
"""
Constant added to control the epsilon value of the log-normalization process
Set to 1 for default ICML paper submission
Set to 1e-3 for log-norm correction
"""

STARTVELLOSS_LAMBDA1 = 10
STARTVELLOSS_LAMBDA2 = 10
STARTVELLOSS_LAMBDA3 = 10
STARTVELLOSS_LAMBDA4 = 10

DEFAULT_DIVISION_BY_ZERO_EPSILON = 1e-8
"""
Constant added to some divisions here and there to prevent divisions by zero.
"""

DEFAULT_XY_ACCEL = 4000
"""
Default printer acceleration along the X and Y axes.
"""

DEFAULT_XY_DECEL = 4000
"""
Default printer deceleration along the X and Y axes.
"""

DEFAULT_Z_ACCEL = 350
"""
Default printer acceleration along the X and Y axes.
"""

DEFAULT_Z_DECEL = 350
"""
Default printer deceleration along the X and Y axes.
"""

DEFAULT_NOZZLE_DIAMETER = 0.4
"""
Default printer nozzle diameter. Used only if the default E only acceleration is not provided.
"""

DEFAULT_FILAMENT_DIAMETER = 1.75
"""
Default filament diameter. Used only if the default E only acceleration is not provided.
"""

DEFAULT_E_ONLY_ACCEL = None
"""
Default acceleration for E only moves.
"""

DEFAULT_MINIMUM_CRUISE_RATIO = 0.5
"""
Default minimum cruise ratio.
"""