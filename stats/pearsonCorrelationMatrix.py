import sys
import os
import argparse
import time

import seaborn as sns
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
                         help=f"Add this flag to save the figure as .pgf instead of displaying it, flag value indicates the name for the figure file.")
    parser.add_argument('-v', '--verbose', required=False, action='count', default=0, \
                     help=f"Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this argument is specified")
    parser.add_argument('-q', '--quiet', required=False, action='count', default=0, \
                     help=f"Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time this argument is specified")
    
    parser.add_argument('--legends', required=False, type=str, default=None, help=f"Legends of the values of the input vectors, based on UpdateVectors.py.")
    
    # Optional arguments
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('-n', required=False, default=1_000_000, type=int, metavar="nb_sample", \
                        help=f"Number of samples to use for the cross-correlation processing, default is 1,000,000.")
    parser.add_argument('-s', required=False, default=10, type=int, metavar="nb_batch", \
                        help=f"Number sub-matrices to process, for faster computation. The final matrix will mean all of those sub-matrices. Default is 10.")
    parser.add_argument('-t', required=False, default="", type=str, metavar="graph_title", \
                        help=f"graphic title (default is None)")
    
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
    if args.n <= 0:
        log.error(f"The number of samples to use must be strictly positive.", 1)
    if args.legends is not None and (len(args.legends.split(",")) != UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"The given legend list is not of the correct length: got a length of {len(args.legends.split(","))} but expected {UV.TRUE_INPUT_VECTOR_SIZE}", 1)
    return args

def compute_input_label_pearson(dataloader: td.DataLoader, max_samples: int = 100_000, n_batches: int = 10) -> np.ndarray:
    """
    Computes Pearson correlation matrix between input and label vectors.
    Y-axis: inputs, X-axis: labels.
    """

    stats = []
    for _ in range(n_batches):
        stats.append({
            "sx": None, "sy": None,
            "sxx": None, "syy": None,
            "sxy": None,
            "n": 0
        })

    for i, (inputs, labels) in enumerate(dataloader):
        # inputs: [B, T, in_dim], labels: [B, T, out_dim]
        concat = torch.cat([inputs, labels], dim=-1)
        mask = concat.abs().sum(dim=-1) > 1e-8 # [B, T]

        X = inputs.view(-1, inputs.shape[-1])[mask.view(-1)]
        Y = labels.view(-1, labels.shape[-1])[mask.view(-1)]
        if X.numel() == 0:
            continue

        # randomly assign samples to batches
        assign = torch.randint(0, n_batches, (len(X),), device=concat.device)

        for k in range(n_batches):
            idx = assign == k
            if not idx.any():
                continue
            Xk = X[idx]
            Yk = Y[idx]
            nk = len(Xk)

            if stats[k]["n"] + nk > max_samples:
                continue
            if stats[k]["sx"] is None:
                in_dim = Xk.shape[1]
                out_dim = Yk.shape[1]
                stats[k]["sx"]  = torch.zeros(in_dim, device=concat.device)
                stats[k]["sy"]  = torch.zeros(out_dim, device=concat.device)
                stats[k]["sxx"] = torch.zeros(in_dim, device=concat.device)
                stats[k]["syy"] = torch.zeros(out_dim, device=concat.device)
                stats[k]["sxy"] = torch.zeros(in_dim, out_dim, device=concat.device)
            stats[k]["sx"]  += Xk.sum(0)
            stats[k]["sy"]  += Yk.sum(0)
            stats[k]["sxx"] += (Xk ** 2).sum(0)
            stats[k]["syy"] += (Yk ** 2).sum(0)
            stats[k]["sxy"] += Xk.T @ Yk
            stats[k]["n"]   += nk
        if i % 100 == 0:
            log.eraseLine(verbose_level=1)
            log.debug(f"Batch [{i}/{len(dataloader)}]")
    
    corrs = []
    for s in stats:
        n = s["n"]
        if n == 0:
            continue

        mx = s["sx"] / n
        my = s["sy"] / n

        cov = s["sxy"] / n - mx[:, None] * my[None, :]
        vx  = s["sxx"] / n - mx ** 2
        vy  = s["syy"] / n - my ** 2

        corr = cov / torch.sqrt(vx[:, None] * vy[None, :] + 1e-12)
        corrs.append(corr)
    return torch.stack(corrs).mean(0)

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

    if args.legends is not None:
        legends = args.legends.split(",")
    else:
        legends = [str(i + 1) for i in range(UV.TRUE_INPUT_VECTOR_SIZE)]

    xlegends = ["Total duration", "Start velocity", "Cruise velocity", "End velocity", "Accel. duration", "Cruise duration", "Decel. duration"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    dataset = VectorDataset(args.dataset_path, 100, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 100, device, log=log, random_seed=897453,
                            masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)
    dt_time = time.time() - start_time

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started processing auto-correlation...")

    dl = td.DataLoader(dataset, num_workers=0, batch_sampler=Utils.BatchSampler(len(dataset), args.b, 165423))

    start_time = time.time()
    log.emptyLines(verbose_level=1)
    corr_matrix: torch.Tensor = compute_input_label_pearson(dl, args.n)
    elapsed_time = time.time() - start_time
    log.eraseLine(verbose_level=1)

    log.info(f"Computing done, total time is {Utils.format_time(elapsed_time)}.")

    plt.figure(figsize=(4.8, 3.6))
    corr_np = corr_matrix.detach().cpu().numpy()
    ax = sns.heatmap(corr_np, annot=True, xticklabels=xlegends, yticklabels=legends, cmap='coolwarm', center=0, fmt=".2f", cbar=True, vmin=-1.0, vmax=1.0)
    ax.set_aspect("equal")
    plt.xticks(rotation=45, ha='right')
    plt.title(args.t)
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