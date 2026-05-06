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
