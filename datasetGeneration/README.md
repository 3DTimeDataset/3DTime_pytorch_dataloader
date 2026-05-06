# Dataset generation folder

This folder simply contains the scripts used for slicing and annotation.

The `sliceModels.py` contains a `--help` or `-h` flag that explains its usage. However, it expects the Prusa Slicer software CLI to be installed locally.

The `annotateGcodes.sh` is used to annotate and compress (due to the resulting much larger G-code files) the raw G-code files. It requires the TimeKlip software installed locally.
