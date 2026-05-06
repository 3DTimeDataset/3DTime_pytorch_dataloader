# Statistical scripts

This folder contains the following scripts, designed specifically for the data format of other scripts in this repository:
- `analyseSequenceLengths.py`: used to process some statistics about the dataset G-code sequence length (defined in the paper as the number of instructions between two moments with the printer nozzle reaching a velocity of 0 mm/s).
- `drawLossCurves.py`: used to draw the loss curves of trained ML models
- `exploreDatasetValues.py`: used to obtain dataset frequency histograms (such as the paper Figure 3)
- `makeBoxPlot.py`: used to obtain result box plots (such as the paper Figure 6)
- `makePlot.py`: used to point plot data (such as the paper appendix Figure 26)
- `nbInfillInstrPerInfillTypeDensity.py`: used to count the number of instruction move types per infill pattern or density (such as paper appendix Figures 10 to 14)
- `plotActualVsTargetSpeed.py`: used to plot the printer actual vs target G-code speed (such as the paper Figure 2)
- `plotHeatmap.py`: used to obtain dataset frequency heatmaps (such as the paper Figures 4 and 5)
- `segtypeAnalysis.py`: used to process some statistics per instruction move type
- `statisticalDistributionTest.py`: used to perform the Lilliefors test (mentioned in the paper Appendix C.2)

Each script has a `--help` or `-h` flag for further usage details.

## Used commands for main paper body

Here is the list of all commands used for the figures of the main paper body:

- Figure 2:
```bash
python plotActualVsTargetSpeed.py -n 5 -i 15700 $DATASET_PATH/binary/test/binary_21/thing-1816918-file-2836593.dat
```

- Figure 3 (note: variable `SELECTED_LABELS` in `REPET.UpdateVectors.py` must be set either to `None` or `[0]`):
```bash
python exploreDatasetValues.py -v -r 100 -s 0.002 -t "" -x "Instruction Print Durations (s)" $DATASET_PATH/binary/
```

- Figure 4 (note: variable `SELECTED_LABELS` in `REPET.UpdateVectors.py` must be set either to `None` or `[0]`, `SELECTED_INPUTS` must be set to `[6, 7]`, and `UPDATING_FUNCTIONS` must be set to `[addDeltaCoordsAndAngle]`):
```bash
python plotHeatmap.py -v -r 100 --input_x 5 --label_y 0 --x_spacing 0.12 --y_spacing 0.002 -x "Instruction segment length (mm)" -y "Instruction print duration (s)" -t "" $DATASET_PATH/binary/
```

- Figure 5 (note: variable `SELECTED_LABELS` in `REPET.UpdateVectors.py` must be set to `None`, `SELECTED_INPUTS` must be set to `[6, 7]`, and `UPDATING_FUNCTIONS` must be set to `[addDeltaCoordsAndAngle]`):
```bash
python plotHeatmap.py -v -r 100 --input_x 0 --label_y 2 --x_spacing 3.0 --y_spacing 3.0 -x "Instruction target speed (mm/s)" -y "Instruction cruise speed (mm/s)" -t "" $DATASET_PATH/binary/
```

- Figure 6:
```bash
python makeBoxPlot.py -f $DATASET_PATH/metadata/results/biLSTM_justStartVel_1000w_test.csv -x "Percentage error (%)"
```
