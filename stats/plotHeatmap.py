import sys
import os
import argparse
import time
import datetime

import matplotlib.colors as colors
import numpy as np
import torch
import torch.utils.data as td
from bisect import bisect_left
sys.path.insert(0,"..")
import Logger
import Utils
import Constants
from REPET.VectorDataset import VectorDataset
import REPET.UpdateVectors as UV
import matplotlib.pyplot as plt
log = Logger.Logger()

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    # Flags
    parser.add_argument('--save_fig', default=None, type=str, required=False, \
                         help=f"Add this flag to save the figure as .pgf instead of displaying it, flag value indicates the name for the figure file.")
    parser.add_argument('-v', '--verbose', required=False, action='count', default=0, \
                     help=f"Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this argument is specified")
    parser.add_argument('-q', '--quiet', required=False, action='count', default=0, \
                     help=f"Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time this argument is specified")
    parser.add_argument('--metadata', required=False, default=None, type=str, help=f"Path to the metadata file corresponding to the given dataset, used only together with the '--column' and '--values' flags. When used together, this script will only consider files that match the given value in the given column from the given metadata file.")
    parser.add_argument('--column', required=False, default=None, type=str, help=f"Used only with the '--metadata' and the 'values' flags, see the help message of '--metadata'.")
    parser.add_argument('--values', required=False, default=None, nargs='+', type=str, help=f"Used only with the '--metadata' and the 'column' flags, see the help message of '--metadata'. You can add several values after the flag (e.g. \"--values\")")
    parser.add_argument('--justsegtypes', required=False, default=None, nargs='+', type=str, help=f"Add this flag to limit this script to instructions of a certain move type, must be one of {', '.join([f'\"{v}\"' for v in Constants.MOVE_TYPES])}")
    parser.add_argument('--applyandtomovetypes', required=False, default=False, action='store_true', help=f"Add this flag to apply the limitation to move types as an 'AND' instead of an 'OR'. For examle, if the '--justsegtypes' flag is set to 'infill travel', the default behavior will display the heatmap for instructions that are either infill or travel. If you add this flag, the heatmap will instead be displayed for instructions that are both infill and travel.")
    
    # Optional arguments
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('-r', required=False, default=15, type=int, metavar="nb_ranges", \
                        help=f"number of bars for the graph, default is 15")
    
    parser.add_argument('--x_spacing', required=False, default=0.1, type=float, metavar="x_spacing", \
                        help=f"spacing value between the X axis ranges. For example, default value is 0.1, \
meaning that the ranges will be [0.0, 0.1, 0.2, 0.3, ..., 1.5] with the default value of '-r'.")
    parser.add_argument('--y_spacing', required=False, default=0.1, type=float, metavar="y_spacing", \
                        help=f"spacing value between the Y axis ranges. For example, default value is 0.1, \
meaning that the ranges will be [0.0, 0.1, 0.2, 0.3, ..., 1.5] with the default value of '-r'.")
    
    # Required arguments
    parser.add_argument('-x', required=True, type=str, metavar="x_axis_label", help=f"x axis legend")
    parser.add_argument('-y', required=True, type=str, metavar="y_axis_label", help=f"y axis legend")
    parser.add_argument('-t', required=True, type=str, metavar="graph_title", help=f"graphic title")
    parser.add_argument('--input_x', required=False, default=None, type=int, metavar="input_x", help=f"Input vector index to use for the X axis, if not provided, '--label_x' must be. An error is raised if both are provided.")
    parser.add_argument('--label_x', required=False, default=None, type=int, metavar="label_x", help=f"Label vector index to use for the X axis, if not provided, '--input_x' must be. An error is raised if both are provided.")
    parser.add_argument('--input_y', required=False, default=None, type=int, metavar="input_y", help=f"Input vector index to use for the Y axis, if not provided, '--label_y' must be. An error is raised if both are provided.")
    parser.add_argument('--label_y', required=False, default=None, type=int, metavar="label_y", help=f"Label vector index to use for the Y axis, if not provided, '--input_y' must be. An error is raised if both are provided.")
    
    parser.add_argument('dataset_path', metavar="dataset_path", type=str, help=f"path to dataset to explore")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if not os.path.isdir(args.dataset_path):
        log.error("The given path does not exist.", 1)
    if args.b < 0:
        log.error(f"Invalid batch size: {args.b}", 1)
    if args.r < 0:
        log.error("Number of ranges must be positive.", 1)

    if args.x_spacing < 0:
        log.error(f"The ranges spacing along the X axis cannot be negative.", 1)
    if args.y_spacing < 0:
        log.error(f"The ranges spacing along the Y axis cannot be negative.", 1)

    if args.label_x is None and args.input_x is None:
        log.error(f"Either an input vector index or a label vector index must be specified for the X axis.", 1)
    if args.label_y is None and args.input_y is None:
        log.error(f"Either an input vector index or a label vector index must be specified for the Y axis.", 1)
    if args.label_x is not None and args.input_x is not None:
        log.error(f"Both input vector index and a label vector index cannot be specified together for the X axis.", 1)
    if args.label_y is not None and args.input_y is not None:
        log.error(f"Both input vector index and a label vector index cannot be specified together for the Y axis.", 1)

    if args.input_x is not None and not (0 <= args.input_x < UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"The given X axis input vector index is out of range.", 1)
    if args.label_x is not None and not (0 <= args.label_x < UV.TRUE_LABEL_VECTOR_SIZE):
        log.error(f"The given X axis label vector index is out of range.", 1)
    if args.input_y is not None and not (0 <= args.input_x < UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"The given Y axis input vector index is out of range.", 1)
    if args.label_y is not None and not (0 <= args.label_y < UV.TRUE_LABEL_VECTOR_SIZE):
        log.error(f"The given Y axis input vector index is out of range.", 1)

    if args.metadata is not None and (not os.path.isfile(args.metadata) or not args.metadata.endswith(".csv")):
        log.error(f"The given metadata file does not exist or is not a CSV file: {args.metadata}.", 1)
    if (args.metadata is None or args.column is None or args.values is None) and not (args.metadata is None and args.column is None and args.values is None):
        log.error(f"If one of the 3 following flags is provided, the other two must also be: '--metadata', '--column', and '--values'.", 1)
    if args.justsegtypes is not None:
        if any([st not in Constants.MOVE_TYPES for st in args.justsegtypes]):
            log.error(f"One of the given segment move type {args.justsegtypes} is not part of the accepted values. Must be one of: {', '.join([f'\"{v}\"' for v in Constants.MOVE_TYPES])}", 1)
    return args

def main() -> None:
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    if args.verbose >= 0 and args.quiet == 0:
        verbose_level = args.verbose
    elif args.verbose == 0 and args.quiet > 0:
        verbose_level = - args.quiet
    else:
        log.error(f"Cannot execute the program in both quiet and verbose mode.", 1)
    log = Logger.Logger(verbose_level=verbose_level)

    if args.metadata is not None:
        import pandas as pd
        metadata = pd.read_csv(args.metadata)
        if args.column not in metadata.columns:
            log.error(f"Could not find a column with name '{args.column}' in CSV file {args.metadata}", 1)
    else:
        metadata = None

    if args.justsegtypes is not None and (UV.SELECTED_INPUTS is not None) and (8 not in UV.SELECTED_INPUTS):
        log.error(f"This script requires the input vectors to contain the information of the segment type for each instruction. Aka either `UV.SELECTED_INPUTS` in `UpdateVectors.py` is set to `None`, or it must contain the value `8`. Currently it is set to {UV.SELECTED_INPUTS}.", 1)

    x_index = args.input_x if args.input_x is not None else args.label_x
    y_index = args.input_y if args.input_y is not None else args.label_y

    x_is_target_speed = False # Condition controling x axis values division by 60, to convert from mm/min to mm/s, only for the target speed
    y_is_cruise_speed = False
    total_isntrs = 0
    total_instrs_reaches_target_speed = 0

    # This condition basically checks if the x axis value is the target speed
    if args.input_x is not None and (UV.SELECTED_INPUTS is None and args.input_x == 6 or args.input_x == 0 and UV.SELECTED_INPUTS.__contains__(6) and UV.SELECTED_INPUTS.index(6) == args.input_x):
        x_is_target_speed = True
        if args.label_y is not None and args.label_y == 2:
            y_is_cruise_speed = True

    x_ranges = [float(i * args.x_spacing) for i in range(args.r + 1)]
    y_ranges = [float(i * args.y_spacing) for i in range(args.r + 1)]
    frequencies = [[0 for _ in range(len(x_ranges))] for _ in range(len(y_ranges))]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    dataset = VectorDataset(args.dataset_path, 500, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 500, device, log=log,
                            masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS,
                            metadata=args.metadata, column=args.column, values=args.values)
    dt_time = time.time() - start_time

    dataset.summary()

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started iterating ...")

    dl = td.DataLoader(dataset, batch_size=args.b, shuffle=False, num_workers=0)
    nb_batch = len(dl)

    start_time = time.time()
    log.emptyLines(verbose_level=1)
    i = 0
    for i, batch in enumerate(dl):
        inputs, labels = batch
        combined = torch.cat([inputs, labels], dim=-1)
        valid_mask = combined.abs().sum(dim=-1) > 1e-8
        inputs = inputs[valid_mask]
        labels = labels[valid_mask]
        if args.justsegtypes is not None:
            type_mask = Utils.tensorMaskForMoveType(inputs, args.justsegtypes, not args.applyandtomovetypes)
            inputs = inputs[type_mask]
            labels = labels[type_mask]
        else:
            # zmask = inputs[..., -3] > 0.0001
            # inputs = inputs[zmask]
            # labels = labels[zmask]
            pass
        x_tensor = inputs if args.input_x is not None else labels
        y_tensor = inputs if args.input_y is not None else labels
        x_vals: torch.Tensor = x_tensor[..., x_index]
        y_vals: torch.Tensor = y_tensor[..., y_index]
        if x_is_target_speed:
            x_vals = x_vals / 60
            if y_is_cruise_speed:
                total_isntrs += len(x_vals)
                total_instrs_reaches_target_speed += ((x_vals - y_vals).abs() < 1.0).sum().item()
        x_vals = x_vals.tolist()
        y_vals = y_vals.tolist()
        for x_value, y_value in zip(x_vals, y_vals):
            x_idx = min(bisect_left(x_ranges, x_value), args.r)
            y_idx = min(bisect_left(y_ranges, y_value), args.r)
            frequencies[y_idx][x_idx] += 1
        if i % 100 == 0:
            log.eraseLine(verbose_level=1)
            log.debug(f"Batch [{i}/{nb_batch}]")
        if i == 1000:
            temp_time = time.time() - start_time
            batch_time = temp_time / 1000
            tot_time = batch_time * nb_batch
            log.eraseLine(verbose_level=1)
            log.info(f"Estimated finishing time: {datetime.datetime.fromtimestamp(tot_time + start_time).strftime("%Y-%m-%d %H:%M:%S")}")
            log.emptyLines(verbose_level=1)
        i += 1
    elapsed_time = time.time() - start_time
    log.eraseLine(verbose_level=1)

    log.info(f"Iterating done, took {Utils.format_time(elapsed_time)}.")
    if x_is_target_speed and y_is_cruise_speed:
        log.info(f"Total number of instructions: {total_isntrs}.")
        log.info(f"Number of instructions that reach their target speed: {total_instrs_reaches_target_speed}.")
        log.info(f"Percentage of instructions with a cruise speed within a 1 mm/s difference of their target speed: {total_instrs_reaches_target_speed * 100 / total_isntrs:.2f}%.")

    heatmap_data = np.array(frequencies[:-1][:-1])
    x_min, x_max = x_ranges[0], x_ranges[-1]
    y_min, y_max = y_ranges[0], y_ranges[-1]

    plt.figure(figsize=(4.8, 3.6))
    im = plt.imshow(heatmap_data, origin='lower', aspect='auto', cmap='coolwarm', extent=[x_min, x_max, y_min, y_max], norm=colors.LogNorm(vmin=1, vmax=heatmap_data.max()))
    plt.colorbar(im, label='Instruction Count')
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(args.t)
    plt.grid(True)
    plt.tight_layout(pad=0)
    if args.save_fig is not None:
        filename = Utils.get_next_versioned_filename(f"./figures/{args.save_fig}.pdf")
        plt.savefig(filename, bbox_inches="tight")
        log.info(f"Figure saved at: {filename}")
    plt.show()

if __name__ == "__main__":
    main()
    
"""
End of file.
"""