[comment]: <> (This README file was generated on 2026-05-05 by Niels Cobat.)

[comment]: <> (Last updated: [2026-05-05].)
 
# GENERAL INFORMATION
 
## Dataset name: 3DTime

We present 3DTime, the first public large-scale dataset for predicting the duration of 3D printing instructions. It comprises 99,005 multivariate time series, each representing a sequence of G-code instructions annotated with execution durations, totaling more than 12 years of print. The dataset introduces several modeling challenges: long sequences (on average 79,934 instructions, up to 27 million), multi-input multi-target annotations, and strong contextual dependencies, where instruction durations depend on both past and future operations. These properties make 3DTime a relevant benchmark for long-context and sequence-to-sequence time-series modeling.

## Code base

This depository contains the code base of the 3DTime dataset, currently under review for NeurIPS Datasets and Benchmarks 2026. It includes files used to generate the dataset, scripts to train and test the presented ML models, and statistical scripts for various data analysis.

It is organized as follows:
- `datasetGeneration`: includes files used for the dataset generation, such as slicing and annotation scripts
- `REPET`: contains all ML related scripts, such as training scripts, model descriptions, and model testing scripts
- `stats`: contains numerous statistical analysis scripts, both for the dataset itself, and for result files of ML scripts
- `templates`: placeholder files used for the dataset generation. Also includes the list of default slicing parameters, and the printer configuration files
- Other scripts at the root: common python functions, classes, and constants, used by other scripts
 
## Contact email:

> niels.cobat@inria.fr

> romaric.gaudel@irisa.fr

> damien.hardy@irisa.fr

## Usefull links

Smaller, easier to use, version of the dataset:

> https://huggingface.co/datasets/3DTimeDataset/3DTime

Full dataset access:

> https://doi.org/10.57745/QR5GGS
   
## Description of sources and methods used to collect and generate data:

This dataset uses the Slice-100k dataset as its source for 3D meshes [Jignasu et al. (2024). Slice-100K: A Multimodal Dataset for Extrusion-based 3D Printing. arXiv.org, abs/2407.04180. https://doi.org/10.48550/ARXIV.2407.04180].

For 3D mesh re-slicing, we use the Prusa Slicer software in CLI mode (version 2.9.2): https://www.prusa3d.com/p/prusaslicer/

To annotate the G-code files with detailed print duration and speed profile, we use the TimeKlip software [Bedell et al. TimeKlip: firmware-based print time estimation and its applications. Int J Adv Manuf Technol (2026). https://doi.org/10.1007/s00170-026-18038-0].
 
