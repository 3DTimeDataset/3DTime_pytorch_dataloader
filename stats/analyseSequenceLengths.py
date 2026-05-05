import sys
import os
import argparse
import time
import datetime

import torch
import torch.utils.data as td
sys.path.insert(0,"..")
import Logger
import math
import Utils
import Constants
from REPET.VectorDataset import VectorDataset
log = Logger.Logger()

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    # Flags
    parser.add_argument('-v', '--verbose', required=False, action='count', default=0, \
                     help=f"Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this argument is specified")
    parser.add_argument('-q', '--quiet', required=False, action='count', default=0, \
                     help=f"Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time this argument is specified")
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    
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

    start_time = time.time()
    dataset = VectorDataset(args.dataset_path, 500, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 500, device, log=log)
    dt_time = time.time() - start_time

    dataset.summary()

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started iterating ...")

    dl = td.DataLoader(dataset, batch_size=args.b, shuffle=False, num_workers=0)
    nb_batch = len(dl)

    min_seq_len = 0
    max_seq_len = 0
    sdtdev_seq_len = 0

    current_len = 0  # carries across batches

    count = 0
    mean = 0.0
    M2 = 0.0

    min_seq_len = float("inf")
    max_seq_len = 0

    start_time = time.time()
    log.emptyLines(verbose_level=1)
    for i, batch in enumerate(dl):
        labels: torch.Tensor = batch[1]

        valid_mask = labels[..., 0] > 1e-8
        seq_list = []
        for b in range(labels.size(0)):
            valid_len = valid_mask[b].sum()
            seq_list.append(labels[b, :valid_len])

        seq = torch.cat(seq_list, dim=0)

        is_sep = seq[:, 1] < 1e-8
        sep_idx = torch.nonzero(is_sep, as_tuple=False).flatten()

        if sep_idx.numel() > 0:
            first = sep_idx[0].item() + 1
            lengths = []
            lengths.append(current_len + first)
            if sep_idx.numel() > 1:
                diffs = (sep_idx[1:] - sep_idx[:-1]).tolist()
                lengths.extend(diffs)
            current_len = (seq.size(0) - 1 - sep_idx[-1].item())
        else:
            current_len += seq.size(0)
            lengths = []

        for L in lengths:
            if L <= 0:
                continue
            count += 1
            delta = L - mean
            mean += delta / count
            M2 += delta * (L - mean)

            min_seq_len = min(min_seq_len, L)
            max_seq_len = max(max_seq_len, L)
        
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

    # handle trailing sequence (no closing separator)
    if current_len > 0:
        L = current_len
        count += 1
        delta = L - mean
        mean += delta / count
        M2 += delta * (L - mean)

        min_seq_len = min(min_seq_len, L)
        max_seq_len = max(max_seq_len, L)

    variance = M2 / (count - 1) if count > 1 else 0.0
    sdtdev_seq_len = math.sqrt(variance)

    log.eraseLine(verbose_level=1)

    log.info(f"Iterating done, took {Utils.format_time(elapsed_time)}.")

    log.info(f"Minimum sequence length: {min_seq_len}")
    log.info(f"Maximum sequence length: {max_seq_len}")
    log.info(f"Mean sequence length: {mean}")
    log.info(f"Sandart deviation of sequence length: {sdtdev_seq_len}")

if __name__ == "__main__":
    main()
    
"""
End of file.
"""