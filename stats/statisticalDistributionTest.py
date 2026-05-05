import sys
import os
import argparse
import time
import datetime

import torch
import torch.utils.data as td
sys.path.insert(0,"..")
import Logger
import Utils
import Constants
from REPET.VectorDataset import VectorDataset
import REPET.UpdateVectors as UV
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
    
    # Optional arguments
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('-l', required=False, default=0, type=int, metavar="label_col_num", \
                        help=f"Label vector column index to use for this test (from 0 to {UV.TRUE_LABEL_VECTOR_SIZE} excluded, default is 0).")
    parser.add_argument('-i', required=False, default=None, type=int, metavar="input_col_num", \
                        help=f"Input vector column index to use for this test (from 0 to {UV.TRUE_INPUT_VECTOR_SIZE} excluded, default is None). If provided, the stats will be done on this column, and the label vector colum index will be ignored.")
    parser.add_argument('--logdistr', required=False, default=False, action='store_true', help=f"Add this flag to test for log normal distribution instead of normal distribution.")

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
        log.error(f"Invalid label vector index value: {args.c}, should be between 0 and {UV.TRUE_LABEL_VECTOR_SIZE} excluded.", 1)
    if args.i is not None and not (0 <= args.i < UV.TRUE_INPUT_VECTOR_SIZE):
        log.error(f"Invalid input vector index value: {args.i}, should be between 0 and {UV.TRUE_INPUT_VECTOR_SIZE} excluded.", 1)
    return args

@torch.no_grad()
def ksLillieforsTest(
        dataloader: td.DataLoader,
        device: torch.device,
        index: int = 0,
        isInput: bool = False,
        logNorm: bool = False,
        maxSamples: int = 1_000_000
    ) -> dict[str, float | int | bool]:
    """
    Performs a Lilliefors test (Lilliefors, 1967), based on the Kolmogorov-Smirnov test, to test the null hypothesis that the data comes from a normaly distributed population. The Lilliefors variation is used here, because the null hypothesis does not specify *which* normal distribution, i.e. it does not specify the expected mean and variance of the distribution. This function can also test the log-normality of the data with a simple transformation.

    Also, this function performs a Q-Q analysis, by processing the Q-Q plot.
    
    :param dataloader: PyTorch dataloader of the dataset to test
    :type dataloader: td.DataLoader
    :param device: PyTorch device to use
    :type device: torch.device
    :param index: Index, either in the input or the label tensor, to measure for this test
    :type index: int
    :param isInput: Indicates wether the `index` parameter corresponds to a value in the input, or label vector
    :type isInput: bool
    :param logNorm: Indicates wether this function should test the normality, or the log-normality of the data
    :type logNorm: bool
    :param maxSamples: Maximum number of samples to use for processing
    :type maxSamples: int
    :return: Returns a python dict that contains the following values. "D": the KS score result, "n_total": the total number of instruction used for processing, "n_used": the number of bins used, "n_zero": the number of reported zeros in the data (not counted for a log-normality test, are counted for a normality test) which might invalidate the log-normality, "log_normal": a copy of the `logNorm` parameter, to indicate which type of distribution was tested, "qq_slope": beta value of the processed Q-Q plot, "qq_intercept": alpha value of the processed Q-Q plot, "qq_rmse": global deviation of the processed Q-Q plot, "qq_tail_max": maximum tail deviation from the processed Q-Q plot
    :rtype: dict[str, float | int | bool]
    """
    global log

    n = 0
    mean = torch.tensor(0.0, device=device)
    m2 = torch.tensor(0.0, device=device)
    nb_zeros = torch.tensor(0, device=device, dtype=torch.int64)
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=device))

    reservoir = torch.empty(maxSamples, device=device)
    filled = 0

    log.info(f"Dataset pass: processing mean and standard deviation...")
    start_time = time.time()
    log.emptyLines(verbose_level=1)
    for i, batch in enumerate(dataloader):
        # Vector selection
        if isInput:
            y, _ = batch # y is of shape [batch_size, seq_len, input_len]
        else:
            _, y = batch # y is of shape [batch_size, seq_len, label_len]
        valid_mask = y.abs().sum(dim=-1) > 1e-8
        y: torch.Tensor = y[valid_mask][..., index]
        zero_mask = y.abs() < 1e-12
        nb_zeros += zero_mask.sum().to(dtype=torch.int64)

        # If log-normality test, apply log after zero mask
        if logNorm:
            y = y[~zero_mask]
            y = torch.log(y)

        is_not_nan = y.isnan().logical_not().logical_and(y.isneginf().logical_not())
        y = y[is_not_nan]
        k = torch.sum(is_not_nan).item()
        if k ==0:
            continue

        # Process mean and variance
        batch_mean = y.mean()
        batch_var = y.var(unbiased=False)
        delta = batch_mean - mean
        new_n = n + k

        mean = mean + delta * k / new_n
        m2 = m2 + batch_var * k + delta**2 * n * k / new_n
        old_n = n
        n = new_n

        # Fill reservoirs
        if filled < maxSamples:
            take = min(maxSamples - filled, k)
            reservoir[filled:filled + take] = y[:take]
            filled += take
            y = y[take:]
            k = y.numel()
        if k > 0:
            offsets = torch.arange(1, k + 1, device=device)
            u = torch.rand(k, device=device)
            j = torch.floor(u * (old_n + offsets)).to(torch.long)
            mask = j < maxSamples
            reservoir[j[mask]] = y[mask]

        # Logger things
        if i % 100 == 0:
            log.eraseLine(verbose_level=1)
            log.debug(f"Batch [{i}/{len(dataloader)}]")
        if i == 1000:
            temp_time = time.time() - start_time
            batch_time = temp_time / 1000
            tot_time = batch_time * len(dataloader)
            log.eraseLine(verbose_level=1)
            log.info(f"Estimated finishing time for the first pass: {datetime.datetime.fromtimestamp(tot_time + start_time).strftime("%Y-%m-%d %H:%M:%S")}")
            log.emptyLines(verbose_level=1)
    
    elapsed_time = time.time() - start_time
    log.eraseLine(verbose_level=1)
    log.info(f"Dataset pass done, took {Utils.format_time(elapsed_time)}.")

    if filled < 20:
        log.error(f"Not enough data was given to this script to accurately perform the test.", 1)
    
    log.info(f"Started normalizing the data...")
    second_time = time.time()
    std = torch.sqrt(m2 / (n - 1))

    z = (reservoir[:filled] - mean) / std

    log.info(f"Done, took {Utils.format_time(time.time() - second_time)}. Started sorting the values...")
    third_time = time.time()
    z_sorted, _ = torch.sort(z)
    log.info(f"Done, took {Utils.format_time(time.time() - third_time)}. Started processing distributions...")

    # Q-Q plot processing shenanigans
    nq = z_sorted.numel()
    eps = 1.0 / (nq + 1)
    p = (torch.arange(1, nq + 1, device=device) - 0.5) / nq
    p = torch.clamp(p, eps, 1.0 - eps)
    q = sqrt2 * torch.erfinv(2.0 * p - 1.0)
    z_mean = z_sorted.mean()
    q_mean = q.mean()
    cov = torch.mean((z_sorted - z_mean) * (q - q_mean))
    var_q = torch.mean((q - q_mean) ** 2)
    # Actual Q-Q plot values
    beta = cov / var_q # slope
    alpha = z_mean - beta * q_mean # intercept
    # Global deviation RMSE
    residuals = z_sorted - (beta * q + alpha)
    qq_rmse = torch.sqrt(torch.mean(residuals**2))
    # Tail deviation: 1% on each side
    tail = max(1, int(0.01 * nq))
    tail_left = torch.max(torch.abs(residuals[:tail]))
    tail_right = torch.max(torch.abs(residuals[-tail:]))
    qq_tail_max = torch.maximum(tail_left, tail_right)

    # Lilliefors test final processing
    ecdf = torch.arange(1, filled + 1, device=device) / filled
    tcdf = 0.5 * (1.0 + torch.erf(z_sorted / sqrt2))
    D = torch.max(torch.abs(ecdf - tcdf))

    log.info(f"Finished. Total time is {Utils.format_time(time.time() - start_time)}.")

    """
    How to interprete those values (this is a rule of thumb, in practice it is quite impossible to draw an actual conclusion from those values due to a very large "n_total").

    KS score:
    - D < 0.01: good fit
    - 0.01 < D < 0.03: approximate fit
    - D > 0.03: questionnable fit, but could be biased for extreme n_total values (a large D value is more likely for large datasets)
    - n_zero >> 0: not that important for normality test, but indicates a strong bias for zero-values if log-normality test (as zeros are discarded for the latter)

    Q-Q plot:
    - qq_slope ~= 1: correct scale
    - qq_intercept ~= 0: correct centering
    - qq_rmse small: good global fit
    - qq_tail_max small: tails look OK
    """
    return {
        "D": float(D),
        "n_total": n,
        "n_used": filled,
        "n_zero": int(nb_zeros.item()),
        "log_normal": logNorm,
        "qq_slope": float(beta),
        "qq_intercept": float(alpha),
        "qq_rmse": float(qq_rmse),
        "qq_tail_max": float(qq_tail_max),
    }

def main() -> None:
    global log
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

    log.info(f"Dataset built, took {Utils.format_time(dt_time)}, started iterating ...")

    dl = td.DataLoader(dataset, batch_size=args.b, shuffle=False, num_workers=0)

    result = ksLillieforsTest(dl, device, args.l if args.i is None else args.i, args.i is not None, args.logdistr, 10_000_000)

    log.info(f"Results: KS score = {result["D"]:.6f} for a {'log-normal' if result["log_normal"] else 'normal'} distribution, used instructions = {result["n_total"]}, used reservoirs = {result["n_used"]}. number of zeros in the data: {result["n_zero"]}")

    log.info(f"Q-Q plot values: slope = {result["qq_slope"]}, intercept = {result["qq_intercept"]}, global deviation (RMSE) = {result["qq_rmse"]}, and tail deviation = {result["qq_tail_max"]}")

if __name__ == "__main__":
    main()
    
"""
End of file.
"""