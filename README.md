[comment]: <> (This README file was generated on 2026-05-05 by Niels Cobat.)

[comment]: <> (Last updated: [2026-05-06].)
 
# 3DTime Dataset

We present 3DTime, the first public large-scale dataset for predicting the duration of 3D printing instructions. It comprises 99,005 multivariate time series, each representing a sequence of G-code instructions annotated with execution durations, totaling more than 12 years of print. The dataset introduces several modeling challenges: long sequences (on average 79,934 instructions, up to 27 million), multi-input multi-target annotations, and strong contextual dependencies, where instruction durations depend on both past and future operations. These properties make 3DTime a relevant benchmark for long-context and sequence-to-sequence time-series modeling.

## Dataset download access

Full dataset download link (NOTE FOR REVIEWERS: this DOI link will be made accessible upon acceptance, in the meantime, the OpenReview submission contains an equivalent private link):

> Masked for review

Small, easy to manipulate, version of the dataset:

> https://huggingface.co/datasets/3DTimeDataset/3DTime

## Code base

This repository contains the code base of the 3DTime dataset, currently under review for NeurIPS Datasets and Benchmarks 2026. It includes files used to generate the dataset, scripts to train and test the presented ML models, and statistical scripts for various data analysis.

It is organized as follows:
- `datasetGeneration`: includes files used for the dataset generation, such as slicing and annotation scripts
- `REPET`: contains all ML related scripts, such as training scripts, model descriptions, and model testing scripts
- `stats`: contains numerous statistical analysis scripts, both for the dataset itself, and for result files of ML scripts
- `templates`: placeholder files used for the dataset generation. Also includes the list of default slicing parameters, and the printer configuration files
- The `environment.yml` file contains all python dependencies required for all scripts
- Other scripts at the root: common python functions, classes, and constants, used by other scripts

## How to use the dataset for a new script

For most use cases, the already existing scripts can be used. However, if you want to use the dataset in any other way, we still recommend to use our custom made PyTorch dataset reader, for efficient use, as it was specifically developped for the peculiar data format and size. Each sub-folder contains its own README.md for more details.

To use it, you can simply create a new python script at the root of this repository, and use the following:

```python
# Basic imports for common functions, classes and constants
import Logger
import Utils
import Constants
import torch

# Custom Pytorch Dataset loader script
import REPET.VectorDataset as vd

# The following script can be used to modify the G-code instruction vector content at runtime (it must be imported anyway)
import REPET.UpdateVectors as UV

# Dataset initialization (which might take up to a few minutes for the full dataset)
dataset = vd.VectorDataset(
    DATASET_PATH_TO_CHANGE, # Path to the dataset binary files (note: does not work with raw G-code files)
    WINDOW_SIZE, # Number of G-code instructions to load per sample, for example in the paper, we used window
                 # sizes of 100 and 1,000

    Constants.DEFAULT_INPUT_SIZE, # Do not change, this represents the number of input features per instruction
    Constants.DEFAULT_LABEL_SIZE, # Do not change, this represents the number of label features per instruction

    # For these two values, we recommend to set them at 0 and {WINDOW_SIZE} respectivelly for easier usage.
    # They are used to reduce the target tensor sequence size compared to the input. See the file
    # "REPET/VectorDataset.py" for documentation
    LEFT_OFFSET,
    RIGHT_OFFSET,

    torch.device('cuda' if torch.cuda.is_available() else 'cpu'), # PyTorch device specification

    random_seed=None, # Set to None if you do not want to randomize the dataset file reading order

    # These three values are used to (potentially) update the instruction vectors at runtime, we do not recommend
    # to change them, but instead to modify the "REPET/UpdateVectors.py" script directly
    masked_inputs=UV.SELECTED_INPUTS,
    masked_labels=UV.SELECTED_LABELS,
    append_input_vectors=UV.UPDATING_FUNCTIONS
)

# You can then use the dataset with a pytorch dataloader:
import torch.utils.data as td
dataloader = td.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# If you want to use a shuffled dataloader (for example to train an ML model), we HIGHLY recommend to use the following:
# dataloader = td.DataLoader(dataset, num_workers=0, batch_sampler=Utils.BatchSampler(len(dataset), BATCH_SIZE, RANDOM_SEED))

# You can then iterate the dataloader:
for i, (features, target) in enumerate(dataloader):
    # The shape of both "features" and "target" PyTorch tensors will be respectivelly:
    # features.shape = [BATCH_SIZE, WINDOW_SIZE, UV.TRUE_INPUT_VECTOR_SIZE]
    # target.shape = [BATCH_SIZE, RIGHT_OFFSET - LEFT_OFFSET, UV.TRUE_LABEL_VECTOR_SIZE]
    ...

```

If, instead of using the full dataset, you want to iterate over individual G-code files, we recommend the following:

```python
import os
for root, _, files in os.walk(DATASET_PATH_TO_CHANGE):
    for file in files:
        if file.endswith(".dat"):
            dataset = vd.VectorDataset(
                os.path.join(root, file),
                ... # Other class flags, as shown above
            )
            ... # Whatever you want to do with a single G-code file data
```

Or alternatively:

```python
from glob import glob
path = f"{DATASET_PATH_TO_CHANGE}/**/*.dat"
filenames = glob(path, recursive=True)
for file in filenames:
    dataset = vd.VectorDataset(
        file,
        ... # Other class flags, as shown above
    )
    ... # Whatever you want to do with a single G-code file data
```
 
## Contact email:

> niels.cobat@inria.fr

> romaric.gaudel@irisa.fr

> damien.hardy@irisa.fr
   
## Description of sources and methods used to collect and generate data:

This dataset uses the Slice-100k dataset as its source for 3D meshes [Jignasu et al. (2024). Slice-100K: A Multimodal Dataset for Extrusion-based 3D Printing. arXiv.org, abs/2407.04180. https://doi.org/10.48550/ARXIV.2407.04180].

For 3D mesh re-slicing, we use the Prusa Slicer software in CLI mode (version 2.9.2): https://www.prusa3d.com/p/prusaslicer/

To annotate the G-code files with detailed print duration and speed profile, we use the TimeKlip software [Bedell et al. TimeKlip: firmware-based print time estimation and its applications. Int J Adv Manuf Technol (2026). https://doi.org/10.1007/s00170-026-18038-0].
 
