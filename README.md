[comment]: <> (This README file was generated on 2026-05-05 by Niels Cobat.)

[comment]: <> (Last updated: [2026-05-05].)
 
# GENERAL INFORMATION
 
## Dataset name: 3DTime

We present 3DTime, the first public large-scale dataset for predicting the duration of 3D printing instructions. It comprises 99,005 multivariate time series, each representing a sequence of G-code instructions annotated with execution durations, totaling more than 12 years of print. The dataset introduces several modeling challenges: long sequences (on average 79,934 instructions, up to 27 million), multi-input multi-target annotations, and strong contextual dependencies, where instruction durations depend on both past and future operations. These properties make 3DTime a relevant benchmark for long-context and sequence-to-sequence time-series modeling.
 
## DOI:

> doi:10.57745/QR5GGS
 
## Contact email:

> niels.cobat@inria.fr

> romaric.gaudel@irisa.fr

> damien.hardy@irisa.fr

## Usefull links

Smaller, easier to use, version of the dataset:

> https://huggingface.co/datasets/3DTimeDataset/3DTime

Source code, including for dataset generation, and for efficient PyTorch data loading:

> https://github.com/3DTimeDataset/3DTime_pytorch_dataloader
 
# METHODOLOGICAL INFORMATION 
  
## Description of sources and methods used to collect and generate data:

This dataset uses the Slice-100k dataset as its source for 3D meshes [Jignasu et al. (2024). Slice-100K: A Multimodal Dataset for Extrusion-based 3D Printing. arXiv.org, abs/2407.04180. https://doi.org/10.48550/ARXIV.2407.04180].

For 3D mesh re-slicing, we use the Prusa Slicer software in CLI mode (version 2.9.2): https://www.prusa3d.com/p/prusaslicer/

To annotate the G-code files with detailed print duration and speed profile, we use the TimeKlip software [Bedell et al. TimeKlip: firmware-based print time estimation and its applications. Int J Adv Manuf Technol (2026). https://doi.org/10.1007/s00170-026-18038-0].
 
## Methods for processing the data: 

In order to use the dataset, we recommend users to use our custom made PyTorch dataloader (https://github.com/3DTimeDataset/3DTime_pytorch_dataloader), as it was specifically designed and optimized for the peculiar shape and size of the dataset.
 
## File hierarchy:

 The dataset is divided in three main folders:
 - `gcodes`: contains the textual annotated G-code files, groupped by sub-sets of at most 5,000 files, each compressed
 - `binary`: contains the binary representation of all G-code files, groupped by sub-sets of at most 5,000 files, only usable with the custom PyTorch dataloader presented above 
 - `metadata`: contains the metadata of the full dataset