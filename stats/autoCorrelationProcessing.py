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
log = Logger.Logger(verbose_level=1)

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    # Flags
    parser.add_argument('--save_fig', default=None, type=str, required=False, \
                         help=f"Add this flag to save the figure as .pdf instead of displaying it, flag value indicates the name for the figure file.")
    parser.add_argument('-v', '--verbose', required=False, action='count', default=0, \
                     help=f"Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this argument is specified")
    parser.add_argument('-q', '--quiet', required=False, action='count', default=0, \
                     help=f"Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time this argument is specified")
    
    # Optional arguments
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('-l', required=False, default=0, type=int, metavar="label_col_num", \
                        help=f"Label vector column index to use (from 0 to {UV.TRUE_LABEL_VECTOR_SIZE} excluded, default is 0). If set to -1, will process the auto-correlation of the average speed of instructions.")
    parser.add_argument('-i', required=False, default=None, type=int, metavar="input_col_num", \
                        help=f"Input vector column index (from 0 to {UV.TRUE_INPUT_VECTOR_SIZE} excluded, default is None). If provided, the stats will be done on this column, and the label vector colum index will be ignored.")
    parser.add_argument('-m', required=False, default=50, type=int, metavar="max_lag", \
                        help=f"Maximum lag used to process the auto-correlation of the given value, default is 50.")
    parser.add_argument('-n', required=False, default=1000000, type=int, metavar="nb_sample", \
                        help=f"Number of samples to use for the auto-correlation processing, default is 1,000,000.")
    parser.add_argument('-t', required=False, default="Auto-correlation of label instruction total times", type=str, metavar="graph_title", \
                        help=f"graphic title (default is \"Auto-correlation of label instruction total times\")")
    
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
    if not (0 <= args.l < UV.TRUE_LABEL_VECTOR_SIZE) and args.l != -1:
        log.error(f"Invalid label vector index value: {args.c}, should be between 0 and {UV.TRUE_LABEL_VECTOR_SIZE} excluded.", 1)
    if args.i is not None and not (0 <= args.i < UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"Invalid input vector index value: {args.i}, should be between 0 and {UV.TRUE_INPUT_VECTOR_SIZE} excluded.", 1)
    if args.m <= 0:
        log.error(f"The maximum lag must be strictly positive.", 1)
    if args.n <= 0:
        log.error(f"The number of samples to use must be strictly positive.", 1)
    return args

def compute_autocorrelation_dataloader(
        dataloader: td.DataLoader,
        index: int,
        is_label: bool = False,
        max_lag: int = 50,
        max_samples: int = 100000
    ) -> np.ndarray:
    """
    Computes 1D autocorrelation for a single feature from the dataset.
    - index: index in input or label vector
    - is_label: True if index belongs to label vector
    """
    values = []
    total = 0

    if index == -1:
        if UV.SELECTED_LABELS is not None and 0 not in UV.SELECTED_LABELS:
            log.error(f"Asked to compute the auto-correlation of the average speed per instruction, but the instruction print time does not seem to be included in the label vectors, check REPET.UpdateVectors.py.", 1)
        if len(UV.UPDATING_FUNCTIONS) != 1 and not UV.UPDATING_FUNCTIONS.__contains__(UV.addDeltaCoordsAndAngle):
            log.error(f"In order to process the auto-correlation of the average instruction speed, this script requires (and limits) the appending vector function called `UpdateVectors.addDeltaCoordsAndAngle()`, see UpdateVectors.py", 1)
        seg_len_index = UV.TRUE_INPUT_VECTOR_SIZE - 2

    for i, (inputs, labels) in enumerate(dataloader):
        tensor: torch.Tensor = labels if is_label else inputs
        mask = tensor.abs().sum(dim=-1) > 1e-8
        if index == -1:
            data = (inputs[..., seg_len_index] / (labels[..., 0] + 1e-6))[mask]
        else:
            data = tensor[..., index][mask]
        selected = data.flatten().cpu().numpy()
        values.append(selected)
        total += len(selected)
        if total >= max_samples:
            break
        if i % 10 == 0:
            log.eraseLine(verbose_level=1)
            log.debug(f"Batch [{i}/{len(dataloader)}]")
    log.eraseLine(verbose_level=1)
    log.info(f"Iterating done, started computing.")

    signal = np.concatenate(values)[:max_samples]
    signal -= signal.mean()
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr /= autocorr[0]  # normalize
    autocorr = autocorr[:max_lag + 1]
    return autocorr

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
    dataset = VectorDataset(args.dataset_path, 100, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 100, device, log=log, random_seed=12345,
                            masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)
    dt_time = time.time() - start_time

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started processing auto-correlation...")

    dl = td.DataLoader(dataset, num_workers=0, batch_sampler=Utils.BatchSampler(len(dataset), args.b, Constants.RANDOM_SEED))

    start_time = time.time()
    log.emptyLines(verbose_level=1)
    index = args.i if args.i is not None else args.l
    is_label = args.i is None
    if args.l == -1:
        index = -1
        is_label = True
    autocorr = compute_autocorrelation_dataloader(dl, index, is_label, args.m, args.n)
    elapsed_time = time.time() - start_time
    log.eraseLine(verbose_level=1)

    log.info(f"Computing done, total time is {Utils.format_time(elapsed_time)}.")

    plt.figure(figsize=(4.5,2.25))
    lags = np.arange(len(autocorr))
    plt.plot(lags, autocorr, marker='.', label="Auto-Correlation")
    plt.ylim(0, 1)
    plt.title(args.t)
    plt.xlabel("Lag (instructions)")
    plt.ylabel("Autocorrelation")
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