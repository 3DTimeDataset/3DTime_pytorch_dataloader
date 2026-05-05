import sys
import os
import argparse
import time

import numpy as np
import torch
import torch.utils.data as td
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
    
    # Optional arguments
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('--x_label', required=False, default=0, type=int, metavar="x_label_col", \
                        help=f"Label vector column index to use for the X axis (from 0 to {UV.TRUE_LABEL_VECTOR_SIZE} excluded, default is 0).")
    parser.add_argument('--x_input', required=False, default=None, type=int, metavar="x_input_col", \
                        help=f"Input vector column index to use for the X axis (from 0 to {UV.TRUE_INPUT_VECTOR_SIZE} excluded, default is None). If provided, the stats will be done on this column, and the label vector colum index will be ignored.")
    parser.add_argument('--y_label', required=False, default=0, type=int, metavar="y_label_col", \
                        help=f"Label vector column index to use for the Y axis (from 0 to {UV.TRUE_LABEL_VECTOR_SIZE} excluded, default is 0).")
    parser.add_argument('--y_input', required=False, default=None, type=int, metavar="y_input_col", \
                        help=f"Input vector column index to use for the Y axis (from 0 to {UV.TRUE_INPUT_VECTOR_SIZE} excluded, default is None). If provided, the stats will be done on this column, and the label vector colum index will be ignored.")
    parser.add_argument('-m', required=False, default=50, type=int, metavar="max_lag", \
                        help=f"Maximum lag used to process the cross-correlation of the given value, default is 50.")
    parser.add_argument('-n', required=False, default=1000000, type=int, metavar="nb_sample", \
                        help=f"Number of samples to use for the cross-correlation processing, default is 1,000,000.")
    parser.add_argument('-t', required=False, default="Cross-correlation of input segment length and label total time", type=str, metavar="graph_title", \
                        help=f"graphic title (default is \"Cross-correlation of input segment length and label total time\")")
    
    # Required arguments
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
    if not (0 <= args.x_label < UV.TRUE_LABEL_VECTOR_SIZE):
        log.error(f"Invalid label vector index value for the X axis: {args.c}, should be between 0 and {UV.TRUE_LABEL_VECTOR_SIZE} excluded.", 1)
    if args.x_input is not None and not (0 <= args.x_input < UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"Invalid input vector index value for the X axis: {args.i}, should be between 0 and {UV.TRUE_INPUT_VECTOR_SIZE} excluded.", 1)
    if not (0 <= args.y_label < UV.TRUE_LABEL_VECTOR_SIZE):
        log.error(f"Invalid label vector index value for the Y axis: {args.c}, should be between 0 and {UV.TRUE_LABEL_VECTOR_SIZE} excluded.", 1)
    if args.y_input is not None and not (0 <= args.y_input < UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"Invalid input vector index value for the Y axis: {args.i}, should be between 0 and {UV.TRUE_INPUT_VECTOR_SIZE} excluded.", 1)
    if args.m <= 0:
        log.error(f"The maximum lag must be strictly positive.", 1)
    if args.n <= 0:
        log.error(f"The number of samples to use must be strictly positive.", 1)
    return args

def compute_cross_correlation_dataloader(
        dataloader: td.DataLoader,
        index_x: int,
        index_y: int,
        is_label_x: bool = False,
        is_label_y: bool = False,
        max_lag: int = 50,
        max_samples: int = 100000
    ) -> np.ndarray:
    """
    Computes cross-correlation between two scalar features.
    """
    x_vals, y_vals = [], []
    total = 0

    for i, (inputs, labels) in enumerate(dataloader):
        x_tensor = labels if is_label_x else inputs
        y_tensor = labels if is_label_y else inputs

        x = x_tensor[..., index_x]
        y = y_tensor[..., index_y]
        mask = (x_tensor.abs().sum(dim=-1) > 1e-8) & (y_tensor.abs().sum(dim=-1) > 1e-8)

        x_vals.append(x[mask].flatten().cpu().numpy())
        y_vals.append(y[mask].flatten().cpu().numpy())

        total += mask.sum().item()
        if total >= max_samples:
            break
        if i % 10 == 0:
            log.eraseLine(verbose_level=1)
            log.debug(f"Batch [{i}/{len(dataloader)}]")
    log.eraseLine(verbose_level=1)
    log.info(f"Iterating done, started computing.")

    x_arr = np.concatenate(x_vals)[:max_samples]
    y_arr = np.concatenate(y_vals)[:max_samples]
    x_arr -= x_arr.mean()
    y_arr -= y_arr.mean()

    corr = np.correlate(x_arr, y_arr, mode='full')
    corr = corr[corr.size // 2:]
    corr /= np.sqrt((x_arr**2).sum() * (y_arr**2).sum())
    return corr[:max_lag + 1]

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

    start_time = time.time()
    dataset = VectorDataset(args.dataset_path, 500, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 500, device, log=log,
                            masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)
    dt_time = time.time() - start_time

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started processing auto-correlation...")

    dl = td.DataLoader(dataset, batch_size=args.b, shuffle=False, num_workers=0)

    start_time = time.time()
    log.emptyLines(verbose_level=1)
    crosscorr = compute_cross_correlation_dataloader(
        dl,
        args.x_input if args.x_input is not None else args.x_label,
        args.y_input if args.y_input is not None else args.y_label,
        args.x_input is None,
        args.y_input is None,
        args.m,
        args.n
    )
    elapsed_time = time.time() - start_time
    log.eraseLine(verbose_level=1)

    log.info(f"Computing done, total time is {Utils.format_time(elapsed_time)}.")

    plt.figure(figsize=(4.5, 3.375))
    lags = range(len(crosscorr))
    plt.plot(lags, crosscorr, marker='o')
    plt.title(args.t)
    plt.xlabel("Lag (instructions)")
    plt.ylabel("Cross-correlation")
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