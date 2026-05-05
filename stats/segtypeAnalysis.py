"""
Goal of this script: statistical analysis of the full datasets, based on the segment type

Hence the resulting graph is the seg type on the x axis, and the following possible stats on the y axis:
- Number of instruction
- Total distance (plot)
- Total extrusion distance (plot)
- Label value display (boxplot)
- Input value display (boxplot)
"""

import sys
import os
import argparse
import time
import datetime

import torch
import torch.utils.data as td
import numpy as np
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
    parser.add_argument('--ylinear', action='store_true', default=True, help=f"(True by default) makes Y axis scale linear")
    parser.add_argument('--ylogscale', dest='ylinear', action='store_false', help=f"makes Y axis scale logarithmic")
    parser.add_argument('--fliers', action='store_true', default=True, help=f"(True by default) Add to show flier values")
    parser.add_argument('--nofliers', dest='fliers', action='store_false', help=f"Add to remove flier values")
    
    # Optional arguments
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('-u', required=False, default="s", type=str, metavar="stat_unit", \
                        help=f"measurement unit for the mean value printing (default is \"s\")")
    parser.add_argument('-l', required=False, default=0, type=int, metavar="label_col_num", \
                        help=f"Label vector column index (from 0 to {UV.TRUE_LABEL_VECTOR_SIZE} excluded, default is 0).")
    parser.add_argument('-i', required=False, default=None, type=int, metavar="input_col_num", \
                        help=f"Input vector column index (from 0 to {UV.TRUE_INPUT_VECTOR_SIZE} excluded, default is None). If provided, the stats will be done on this column, and the `-l` flag will be ignored.")
    parser.add_argument('--nbinstr', required=False, action='store_true', default=False, help=f"Add this flag to count the number of instruction per segment type as a plot instead of the default input/label value boxplot. Will override any `-l` and `-i` flags.")
    parser.add_argument('--seglen', required=False, action='store_true', default=False, help=f"Add this flag to plot the total segment length per segment type instead of the default input/label value boxplot. Will override any `-l` and `-i` flags, and the `--nbinstr` flag.")
    parser.add_argument('--extrlen', required=False, action='store_true', default=False, help=f"Add this flag to plot the total extrusion length per segment type instead of the default input/label value boxplot. Will override any `-l` and `-i` flags, along with the `--nbinstr` and `--seglen` flags.")
    parser.add_argument('-y', required=False, default="Label print time (s)", type=str, metavar="x_axis_label", \
                        help=f"x axis legend (default is \"Label print time (s)\")")
    parser.add_argument('-t', required=False, default="", type=str, metavar="graph_title", \
                        help=f"graphic title (default is \"\")")
    
    # Required arguments
    parser.add_argument('dataset_path', metavar="dataset_path", type=str, help=f"path to dataset to explore")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if (not os.path.isdir(args.dataset_path)) and (not os.path.isfile(args.dataset_path)):
        log.error("The given path does not exist.", 1)
    if args.b < 0:
        log.error(f"Invalid batch size: {args.b}", 1)
    if not (0 <= args.l < UV.TRUE_LABEL_VECTOR_SIZE):
        log.error(f"Invalid label vector index value: {args.l}, should be between 0 and {UV.TRUE_LABEL_VECTOR_SIZE} excluded.", 1)
    if args.i is not None and not (0 <= args.i < UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"Invalid input vector index value: {args.i}, should be between 0 and {UV.TRUE_INPUT_VECTOR_SIZE} excluded.", 1)
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

    if (UV.SELECTED_INPUTS is not None) and (8 not in UV.SELECTED_INPUTS):
        log.error(f"This script requires the input vectors to contain the information of the segment type for each instruction. Aka either `UV.SELECTED_INPUTS` in `UpdateVectors.py` is set to `None`, or it must contain the value `8`. Currently it is set to {UV.SELECTED_INPUTS}.", 1)

    # Verification of the vector values: do they contain the asked data?
    if args.extrlen: # Stat on extrusion length
        try:
            data_index = 7 if UV.SELECTED_INPUTS is None else UV.SELECTED_INPUTS.index(7)
        except ValueError:
            log.error(f"For this script to displau the extrusion distance per segment type, `UV.SELECTED_INPUTS` in `UpdateVectors.py` must either be set to `None` or contain the value `7`. Currently: {UV.SELECTED_INPUTS}", 1)
    elif args.seglen: # Stat on segment length
        if UV.UPDATING_FUNCTIONS[-1] is not UV.addDeltaCoordsAndAngle:
            log.error(f"For this script to display the segment lengths per segment types, the last updating function of `UV.UPDATING_FUNCTIONS` in `UpdateVectors.py` must be `addDeltaCoordsAndAngle`. Currently: {UV.UPDATING_FUNCTIONS}", 1)
        data_index = -2
    elif args.nbinstr: # Stat on the number of instr
        data_index = 0
    elif args.i is not None: # Stat on the input vector value
        data_index = args.i
    else: # Stat on the label vector value
        data_index = args.l

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    dataset = VectorDataset(args.dataset_path, 500, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 500, device, log=log,
                            masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)
    dt_time = time.time() - start_time

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started iterating ...")

    dl = td.DataLoader(dataset, batch_size=args.b, shuffle=False, num_workers=0)
    nb_batch = len(dl)

    log.alert(f"This script will take an estimated maximum of {Utils.format_memory_usage(len(dataset) * 500 * 4)} of CPU memory!")

    interesting_move_types: list[str] = ["wallInner", "wallOuter", "skin", "skirt", "infill", "support", "travel", "prime", "retract"]

    segTypeValues: dict[str, list[float]] = {}
    for st in interesting_move_types:
        segTypeValues[st] = []

    index_of_seg_type = 8 if UV.SELECTED_INPUTS is None else UV.SELECTED_INPUTS.index(8)

    start_time = time.time()
    log.emptyLines(verbose_level=1)
    for i, (inputs, labels) in enumerate(dl):
        combined = torch.cat([inputs, labels], dim=-1)
        valid_mask = combined.abs().sum(dim=-1) > 1e-8
        inputs: torch.Tensor = inputs[valid_mask]
        labels: torch.Tensor = labels[valid_mask]
        if args.extrlen or args.seglen or args.nbinstr or args.i is not None: # Stat on extrusion length
            data = inputs[..., data_index]
        else: # Stat on the label vector value
            data = labels[..., data_index]
        for s in interesting_move_types:
            type_mask = Utils.tensorMaskForMoveType(inputs, s)
            if args.extrlen or args.seglen:
                segTypeValues[s].append(data[type_mask].abs().sum().item())
            elif args.nbinstr:
                segTypeValues[s].append(torch.sum(type_mask).item())
            else:
                segTypeValues[s].extend(data[type_mask].flatten().tolist())
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
    elapsed_time = time.time() - start_time
    log.eraseLine(verbose_level=1)

    log.info(f"Iterating done, took {Utils.format_time(elapsed_time)}. Building figure (this might take a while)...")

    # Figure plotting

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(bottom=0.1, hspace=0.5)

    all_total = 0
    if args.seglen or args.extrlen or args.nbinstr:
        result: dict[str, float] = {}
        for key in interesting_move_types:
            result[key] = torch.tensor(segTypeValues[key], device=device, dtype=torch.float32).sum().item()
            log.info(f"Total value for {key}: {result[key]}")
            all_total += result[key]
        ax.bar([p for p in Utils.frange(0.5, len(interesting_move_types) + 0.1, 1.0)], [result[key] for key in interesting_move_types], [0.5 for _ in range(len(interesting_move_types))])
        log.info(f"Total for all types of instructions: {all_total}")
    else:
        meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='red')
        mean_legend = plt.Line2D([0], [0], color='w', marker='D', markerfacecolor='red', markersize=10, label='Mean')
        for i, st in enumerate(interesting_move_types):
            ax.boxplot(segTypeValues[st], positions=[0.5 + i], tick_labels=[""], widths=[0.4], showfliers=args.fliers, showmeans=True,
                        meanprops=meanprops, patch_artist=True, boxprops=dict(facecolor="C1"), medianprops=dict(color="black"))
        ax.legend(handles=[mean_legend])
    ax.set_xlabel("Segment type")
    ax.set_ylabel(args.y)
    ax.set_xticks([p for p in Utils.frange(0.5, len(interesting_move_types) - 0.4, 1.0)])
    ax.set_xticklabels(interesting_move_types, rotation=45)
    if not args.ylinear:
        plt.yscale("log")
    ax.set_title(args.t)
    ax.grid(True)
    fig.tight_layout(pad=0)
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