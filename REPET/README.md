# REPET folder: ML models and tools

This folder is organized as follows:
- `TrainModel.py`: main script for training the ML models
- `testModel.py`: main script for testing the ML models
- `readModelDetails.py`: used to read a trained model information
- `UpdateVectors.py`: used to modify at runtime the G-code instruction vectors for other scripts, details are below

The first three scripts are the only scripts you should need to execute at some point. They all have a `--help` or `-h` flag to further guide how to use them.

Other files are indirectly used by those main scripts, but there is no need to change them.

## Runtim vector update

The only script to change in order to update the input and label vectors is the `UpdateVector.py` script. If you use a pre-trained model, you might get an error message if the latter script is not changed accordingly, hence the utility of the `readModelDetails.py` script.

The main three variables to change are:
- `SELECTED_INPUTS`: lists the base input vectors indexes to retain for the models (default is None, meaning all values are retained)
- `SELECTED_LABELS`: lists the base label vectors indexes to retain for the models (default is None, meaning all values are retained)
- `UPDATING_FUNCTIONS`: lists all the functions to call to add values to the input vectors.
    + For each function, the input parameters must be:
        + Base, unmasked, input feature tensor (of shape `[BATCH_SIZE, WINDOW_SIZE, Constants.DEFAULT_INPUT_SIZE]`)
        + Base, unmasked, label tensor (of shape `[BATCH_SIZE, RIGHT_OFFSET - LEFT_OFFSET, Constants.DEFAULT_LABEL_SIZE]`)
        + The dataset's input sequence left offset
        + The dataset's input sequence right offset
    + Each function must return a tuple of two tensor:
        + The input feature appending vector, of shape `[BATCH_SIZE, WINDOW_SIZE, whatever_you_want]`
        + The label appending vector, of shape `[BATCH_SIZE, RIGHT_OFFSET - LEFT_OFFSET, whatever_you_want]`
        + Both can have different `whatever_you_want` size
        + Both can be set to `None`, meaning nothing will be added
        + They represent the values to add to the vectors, respectivelly to the input and label vectors, per G-code instruction
    + Two examples of such function are given in the script (including the one used for the paper experiments, called `addDeltaCoordsAndAngle`)

Note:

By default, the extracted vectors contain the following values:
- Input feature vector:
    + Start X coordinates
    + Start Y coordinates
    + Start Z coordinates
    + End X coordinates
    + End Y coordinates
    + End Z coordinates
    + Actual target speed (WARNING: in `mm/min`, not in `mm/s`)
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
