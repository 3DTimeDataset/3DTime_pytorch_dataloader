import sys
import os
import argparse
import time
import datetime

import torch
import torch.utils.data as td
import numpy as np
from bisect import bisect_left
sys.path.insert(0,"..")
import Logger
import Utils
import Constants
from REPET.VectorDataset import VectorDataset
from REPET.TransformerModel import TransformerModel
import REPET.UpdateVectors as UV
import matplotlib.pyplot as plt
log = Logger.Logger()

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    # Flags

    # "Everyday" flags
    parser.add_argument('--save_fig', default=None, type=str, required=False, \
                         help=f"Add this flag to save the figure as .pgf instead of displaying it, flag value indicates the name for the figure file.")
    parser.add_argument('-v', '--verbose', required=False, action='count', default=0, \
                     help=f"Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this argument is specified")
    parser.add_argument('-q', '--quiet', required=False, action='count', default=0, \
                     help=f"Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time this argument is specified")
    
    # Data formatting and selection flags
    # Data unit flags
    parser.add_argument('--istime', action='store_true', default=True, help=f"(True by default) formats the printed mean value as a time")
    parser.add_argument('--nottime', dest='istime', action='store_false', help=f"add this flag to cancel the mean printing as a time")
    # Data pre-processing flags
    parser.add_argument('--ylinear', action='store_true', default=True, help=f"(True by default) makes Y axis scale linear")
    parser.add_argument('--ylogscale', dest='ylinear', action='store_false', help=f"makes Y axis scale logarithmic")
    parser.add_argument('--lindata', action='store_true', default=True, help=f"(True by default) no data transformation is performed")
    parser.add_argument('--logdata', dest='lindata', action='store_false', help=f"Apply a log-transformation to the data before plotting.")
    parser.add_argument('--excludezero', required=False, default=False, action='store_true', help=f"Add this flag to ignore the zeros.")
    # Metadata filtering flags
    parser.add_argument('--metadata', required=False, default=None, type=str, help=f"Path to the metadata file corresponding to the given dataset, used only together with the '--column' and '--values' flags. When used together, this script will only consider files that match the given value in the given column from the given metadata file.")
    parser.add_argument('--column', required=False, default=None, type=str, help=f"Used only with the '--metadata' and the 'values' flags, see the help message of '--metadata'.")
    parser.add_argument('--values', required=False, default=None, nargs='+', type=str, help=f"Used only with the '--metadata' and the 'column' flags, see the help message of '--metadata'. You can add several values after the flag (e.g. \"--values\")")
    parser.add_argument('--justsegtypes', required=False, default=None, nargs='+', type=str, help=f"Add this flag to limit this script to instructions of a certain move type, must be one of {', '.join([f'\"{v}\"' for v in Constants.MOVE_TYPES])}")
    parser.add_argument('--applyandtomovetypes', required=False, action='store_true', default=False, help=f"Add this flag to apply the limitation to move types as an 'AND' instead of an 'OR'. For examle, if the '--justsegtypes' flag is set to 'infill travel', the default behavior will display the heatmap for instructions that are either infill or travel. If you add this flag, the heatmap will instead be displayed for instructions that are both infill and travel.")
    # Value specific filtering flags
    parser.add_argument('--min', required=False, default=0.0, type=float, metavar="min_value", help=f"starting minimum value of the histogram.")
    parser.add_argument('--max', required=False, default=None, type=float, metavar="max_value", help=f"ending maximum value of the histogram. If provided, the minimum value will be ignored, and processed by the number of bars and their spacing.")
    
    # Histogram format flags
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('-u', required=False, default="s", type=str, metavar="stat_unit", \
                        help=f"measurement unit for the mean value printing (default is \"s\")")
    parser.add_argument('-r', required=False, default=200, type=int, metavar="nb_ranges", \
                        help=f"number of bars for the graph, default is 200")
    parser.add_argument('-s', required=False, default=0.1, type=float, metavar="bar_range_spacing", \
                        help=f"spacing value between the bar plot ranges. For example, default value is 0.1, \
meaning that the ranges will be [0.0, 0.1, 0.2, 0.3, ..., 20.0]. If the X axis scale is set to log, this value is instead used to process the maximum value of the ranges, with `max_value = nb_ranges * bar_range_spacing`, and the ranges will be processed based on this maximum value and the number of ranges given with '-r'.")
    
    # Value selection flags: what to actually plot
    parser.add_argument('-l', required=False, default=0, type=int, metavar="label_col_num", \
                        help=f"Label vector column index (from 0 to {UV.TRUE_LABEL_VECTOR_SIZE} excluded, default is 0).")
    parser.add_argument('-i', required=False, default=None, type=int, metavar="input_col_num", \
                        help=f"Input vector column index (from 0 to {UV.TRUE_INPUT_VECTOR_SIZE} excluded, default is None). If provided, the stats will be done on this column, and the label vector colum index will be ignored.")
    
    # Optional prediction model flags
    parser.add_argument('-m', required=False, default=None, type=str, metavar="model_path", \
                        help=f"path to a trained model, unused if the `-p` flag is not specified.")
    parser.add_argument('-p', required=False, default=None, type=int, metavar="pred_col_num", \
                        help=f"Prediction vector column index. If provided, the label and input column indexes will be ignored.")
    parser.add_argument('--testrawpred', action='store_true', default=False, help=f"Add this flag to prevent any prediction model post processing")
    
    # Figure legend and title flags
    parser.add_argument('-x', required=False, default="Label print time (s)", type=str, metavar="x_axis_label", \
                        help=f"x axis legend (default is \"Label print time (s)\")")
    parser.add_argument('-t', required=False, default="", type=str, metavar="graph_title", \
                        help=f"graphic title (default is empty)")
    
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
    if args.s < 0:
        log.error("Bar plot ranges spacing must be positive.", 1)
    if args.metadata is not None and (not os.path.isfile(args.metadata) or not args.metadata.endswith(".csv")):
        log.error(f"The given metadata file does not exist or is not a CSV file: {args.metadata}.", 1)
    if (args.metadata is None or args.column is None or args.values is None) and not (args.metadata is None and args.column is None and args.values is None):
        log.error(f"If one of the 3 following flags is provided, the other two must also be: '--metadata', '--column', and '--values'.", 1)
    if args.justsegtypes is not None:
        if any([st not in Constants.MOVE_TYPES for st in args.justsegtypes]):
            log.error(f"One of the given segment move type {args.justsegtypes} is not part of the accepted values. Must be one of: {', '.join([f'\"{v}\"' for v in Constants.MOVE_TYPES])}", 1)
    if args.m is not None and not os.path.isfile(args.m):
        log.error(f"The provided model path does not exist.", 1)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.p is not None:
        if args.m is not None:
            model, _, _, _ = Utils.parseModelArchitecture(args.m, device, torch.float32, 0.0, 1, log, not args.testrawpred)
            model.summary()
            model.eval()
        else:
            log.error(f"If the `-p` flag is provided, the `-m` flag must be provided too.", 1)
        if hasattr(model, "simple_rule_head") and model.simple_rule_head:
            if not (0 <= args.p < Constants.DEFAULT_LABEL_SIZE):
                log.error(f"The provided prediction column index must be between 0 (included) and {Constants.DEFAULT_LABEL_SIZE} (excluded) for models with a trapeze prediction head.", 1) 
        else:
            if not (0 <= args.p < model.output_size[1]):
                log.error(f"The provided prediction column index must be between 0 (included) and {model.output_size[1]} (excluded) for this trained model.", 1)
    else:
        model = None

    if args.metadata is not None:
        import pandas as pd
        metadata = pd.read_csv(args.metadata)
        if args.column not in metadata.columns:
            log.error(f"Could not find a column with name '{args.column}' in CSV file {args.metadata}", 1)
    else:
        metadata = None

    if args.justsegtypes is not None and ((UV.SELECTED_INPUTS is not None) and (8 not in UV.SELECTED_INPUTS)):
        log.error(f"This script requires the input vectors to contain the information of the segment type for each instruction. Aka either `UV.SELECTED_INPUTS` in `UpdateVectors.py` is set to `None`, or it must contain the value `8`. Currently it is set to {UV.SELECTED_INPUTS}.", 1)

    start_time = time.time()
    if model is None:
        dataset = VectorDataset(args.dataset_path, 500, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 500, device, log=log,
                            masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS,
                            metadata=args.metadata, column=args.column, values=args.values)
    else:
        dataset = VectorDataset(args.dataset_path, model.input_size[0], Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, model.left_offset, model.right_offset, device, log=log, masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS, metadata=args.metadata, column=args.column, values=args.values)
    dt_time = time.time() - start_time

    dataset.summary()

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started iterating ...")

    dl = td.DataLoader(dataset, batch_size=args.b, shuffle=False, num_workers=0)
    nb_batch = len(dl)

    if args.max is None:
        actual_min: float = args.min
        actual_max: float = args.min + args.r * args.s
    else:
        actual_min: float = args.max - args.r * args.s
        actual_max: float = args.max

    if args.lindata:
        ranges = [v for v in Utils.frange(actual_min, actual_max, args.s)]
    else:
        ranges = np.linspace(np.log(actual_min + Constants.DEFAULT_LOG_NORM_EPSILON), np.log(actual_max + Constants.DEFAULT_LOG_NORM_EPSILON), args.r).tolist()
    ranges_gpu = torch.tensor(ranges, device=device, dtype=dataset.floatType)
    upper_range = actual_max
    frequencies = torch.zeros(len(ranges) + 1, device=device, dtype=torch.int64)
    nb_above_max = 0
    nb_below_min = 0
    null_vector_discards = 0
    segtype_discards = 0
    excludezero_discards = 0
    negative_discards = 0
    sums = torch.zeros(len(dl), device=device)
    nb_instr = 0
    min_value, max_value = torch.inf, -torch.inf

    epsilon = torch.scalar_tensor(Constants.DEFAULT_LOG_NORM_EPSILON, device=device, dtype=dataset.floatType)

    start_time = time.time()
    log.emptyLines(verbose_level=1)
    for i, batch in enumerate(dl):
        inputs: torch.Tensor = batch[0]
        labels: torch.Tensor = batch[1]

        if args.p is not None:
            with torch.no_grad():
                preds = model(inputs)
            if model.trained_on_log_target and (hasattr(model, "simple_rule_head") and model.simple_rule_head):
                preds = torch.exp(preds) - Constants.DEFAULT_LOG_NORM_EPSILON

        # Sort out null vectors filled with zeros for end of samples
        if model is not None:
            valid_mask = (inputs[:, model.left_offset:model.right_offset, :].abs().sum(dim=-1) + labels.abs().sum(dim=-1)) > 1e-8
            b_idx, t_idx = valid_mask.nonzero(as_tuple=True)
            inputs = inputs[b_idx, t_idx + model.left_offset, :]
        else:
            valid_mask = (inputs.abs().sum(dim=-1) + labels.abs().sum(dim=-1)) > 1e-8
            b_idx, t_idx = valid_mask.nonzero(as_tuple=True)
            inputs = inputs[b_idx, t_idx, :]
        labels = labels[b_idx, t_idx, :]
        if args.p is not None:
            preds = preds[b_idx, t_idx, :]
        null_vector_discards += valid_mask.logical_not().nonzero().numel()

        # Sort out vectors of instruction types not matching the given flags
        if args.justsegtypes is not None:
            type_mask = Utils.tensorMaskForMoveType(inputs, args.justsegtypes, not args.applyandtomovetypes)
            inputs = inputs[type_mask]
            labels = labels[type_mask]
            if args.p is not None:
                preds = preds[type_mask]
            segtype_discards += type_mask.logical_not().nonzero().numel()
        else:
            # zmask = (inputs[..., -3] > 1e-4) & (inputs[..., -5:-3] < 1e-4).all(dim=-1)
            # inputs = inputs[zmask]
            # labels = labels[zmask]
            pass

        # Some very model specific code
        # Replaces the predicted start velocities with the amount of off-prediction (only above target speed F)
        # isAbove = preds[..., 0] > (inputs[..., 0] / 60)
        # preds = preds[isAbove, 0] - (inputs[isAbove, 0] / 60)
        # if preds.numel() == 0:
        #     continue

        try:
            if args.i is None and args.p is None:
                data: torch.Tensor = labels[..., args.l]
            elif args.p is None:
                data: torch.Tensor = inputs[..., args.i]
            else:
                data: torch.Tensor = preds[..., args.p]
        except BaseException as e:
            e.add_note(f"inputs shape: {inputs.shape}, labels shape: {labels.shape}")
            if args.p is not None:
                e.add_note(f"preds shape: {preds.shape}")
            raise e

        below_min = data < actual_min
        data = data[below_min.logical_not()]
        nb_below_min += below_min.nonzero().numel()
        above_max = data > upper_range
        data = data[above_max.logical_not()]
        nb_above_max += above_max.nonzero().numel()

        if args.excludezero:
            equalzero = -1e-12 < data < 1e-12
            data = data[equalzero.logical_not()]
            excludezero_discards += equalzero.nonzero().numel()
        if not args.lindata:
            negatives = data < 1e-12
            data = torch.log(data[data >= 1e-12] + epsilon)
            negative_discards += negatives.nonzero().numel()
        data = data.squeeze().contiguous()
        nb_instr += data.numel()

        if data.numel() != 0:
            # Analysis of all retained values
            sums[i] = data.sum()
            local_min = data.min().item()
            local_max = data.max().item()
            if local_min < min_value:
                min_value = local_min
            elif local_max > max_value:
                max_value = local_max
            idxs = torch.bucketize(data, ranges_gpu, right=False).flatten()
            frequencies += torch.bincount(idxs, minlength=len(ranges) + 1)

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
    total = torch.sum(sums).item()
    if nb_instr != 0:
        mean = total / nb_instr
    else:
        log.eraseLine(verbose_level=1)
        log.alert(f"Found no instruction that matched all constraints! See logs below for detailed information on said constraints.")
        log.emptyLines(verbose_level=1)
        mean = np.nan
        nb_instr = np.nan
    log.eraseLine(verbose_level=1)

    log.info(f"Iterating done, took {Utils.format_time(elapsed_time)}.")
    log.info(f"Total number of instructions found: {nb_instr}")
    log.info(f"{nb_above_max} ({nb_above_max * 100 / nb_instr:.2f}%) instructions are above the maximum of {upper_range} {args.u} of the plot.")
    log.info(f"{nb_below_min} ({nb_below_min * 100 / nb_instr:.2f}%) instructions are below the minimum of {actual_min} {args.u} of the plot.")
    log.info(f"{null_vector_discards} vectors were discarded as placeholder vectors (end of G-code fillers).")
    if args.justsegtypes is not None:
        log.info(f"{segtype_discards} instructions were discarded because not of the correct segment type.")
    if args.excludezero:
        log.info(f"{excludezero_discards} instructions were removed because of a zero value.")
    if not args.lindata:
        log.info(f"{negative_discards} instructions were removed because of a negative value (for log-transformation).")
    if args.istime:
        log.info(f"Total value is {Utils.format_time(total)} (unformatted: {total} {args.u})")
        log.info(f"Mean value is {Utils.format_time(mean)} (unformatted: {mean} {args.u}).")
        log.info(f"Minimum value is {Utils.format_time(min_value)} (unformatted: {min_value}), and maximum value is {Utils.format_time(max_value)} (unformatted: {max_value}).")
    else:
        log.info(f"Total value is {total} {args.u}")
        log.info(f"Mean value is {mean} {args.u}.")
        log.info(f"Minimum value is {min_value}, and maximum value is {max_value}.")

    widths = ranges[1] - ranges[0]
    bar_positions = [r + widths/2 for r in ranges]

    #log.emptyLines()
    #log.title(f"Number of instructions per range:\n{'\n'.join([f'- [{ranges[i]} ; {ranges[i + 1]}[ = {frequencies[i].item()}' for i in range(len(ranges) - 1)])}\n")

    plt.figure(figsize=(4.5, 3))
    plt.bar(bar_positions, frequencies.cpu().numpy()[:-1], width=widths, edgecolor='black', align='center')
    if args.lindata:
        plt.xlabel(args.x)
    else:
        plt.xlabel(f"Log-transformation of {args.x}")
    plt.ylabel('Frequency')
    if not args.ylinear:
        plt.yscale("log")
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