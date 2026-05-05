import sys
import os
import argparse
import time
import datetime

import torch
import torch.utils.data as td
import numpy as np
import pandas as pd
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
    parser.add_argument('--ylinear', action='store_true', default=True, help=f"(True by default) makes Y axis scale linear")
    parser.add_argument('--ylogscale', dest='ylinear', action='store_false', help=f"makes Y axis scale logarithmic")

    parser.add_argument('--infilltype', action='store_true', default=False, required=False, help=f"Add this flag to display the number of infill instruction per infill type instead of per range of infill densities.")
    parser.add_argument('--justinfill', action='store_true', default=False, required=False, help=f"Add this flag to only display the number of infill instructions, and not the other print moves, travel moves, and prime and retract moves.")
    parser.add_argument('--normbyvalue', default=None, required=False, type=float, help=f"Add this flag and its corresponding value to multiply the counts in each bar by the provided value.")
    
    # Optional arguments
    parser.add_argument('-b', required=False, default=Constants.DEFAULT_BATCH_SIZE, type=int, metavar="batch_size", \
                        help=f"dataloader batch size, default is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('-t', required=False, default="", type=str, metavar="graph_title", \
                        help=f"graphic title (default is \"\")")
    
    # Required arguments
    parser.add_argument('dataset_path', metavar="dataset_path", type=str, help=f"path to dataset to explore")
    parser.add_argument('data_csv_path', metavar="data_csv_path", type=str, help=f"Path to the CSV file containing the file names and the infill density information.")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if (not os.path.isdir(args.dataset_path)) and (not os.path.isfile(args.dataset_path)):
        log.error("The given path does not exist.", 1)
    if not os.path.isfile(args.data_csv_path):
        log.error("The given CSV file does not exist.", 1)
    if args.b < 0:
        log.error(f"Invalid batch size: {args.b}", 1)
    if args.normbyvalue is not None and args.normbyvalue == 0.0:
        log.error(f"The normalization factor cannot be 0")
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

    if args.normbyvalue is None:
        args.normbyvalue = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (UV.SELECTED_INPUTS is not None) and (8 not in UV.SELECTED_INPUTS):
        log.error(f"This script requires the input vectors to contain the information of the segment type for each instruction. Aka either `UV.SELECTED_INPUTS` in `UpdateVectors.py` is set to `None`, or it must contain the value `8`. Currently it is set to {UV.SELECTED_INPUTS}.", 1)

    csv_file = pd.read_csv(args.data_csv_path)

    file_paths = [os.path.join(root, file) for root, _, files in os.walk(args.dataset_path) for file in files if file.endswith(".dat")]
    if len(file_paths) > csv_file.shape[0]:
        log.warning(f"Found a superior number of DAT files in the provided folder than number of rows in the provided CSV file.")
    elif len(file_paths) < csv_file.shape[0]:
        log.warning(f"Found an inferior number of DAT files in the provided folder than number of rows in the provided CSV file.")
    log.info(f"Found {len(file_paths)} inside the given dataset path.")

    if "Infill type" not in csv_file.columns and not args.infilltype:
        log.error(f"For this script to display data per infill type, the column \"Infill type\" must be present in the CSV file.", 1)
    unique_infill_types: list[str] = csv_file["Infill type"].unique().tolist()
    log.info(f"Unique infill types: {', '.join(unique_infill_types)}")

    if not args.infilltype:
        infill_counts: list[int] = [0 for _ in range(101)]
        other_prints_counts: list[int] = [0 for _ in range(101)]
        travel_counts: list[int] = [0 for _ in range(101)]
        pr_counts: list[int] = [0 for _ in range(101)]
    else:
        infill_counts: list[int] = [0 for _ in range(len(unique_infill_types))]
        other_prints_counts: list[int] = [0 for _ in range(len(unique_infill_types))]
        travel_counts: list[int] = [0 for _ in range(len(unique_infill_types))]
        pr_counts: list[int] = [0 for _ in range(len(unique_infill_types))]

    nb_files_per_range: list[int] = [0 for _ in range(len(infill_counts))]

    index_of_seg_type = 8 if UV.SELECTED_INPUTS is None else UV.SELECTED_INPUTS.index(8)
    
    log.emptyLines(verbose_level=1)
    for i, file in enumerate(file_paths):
        log.eraseLine(verbose_level=1)
        log.debug(f"File {i + 1}/{len(file_paths)}: {file}")
        if "Binary file name" not in csv_file.columns:
            log.eraseLine(verbose_level=1)
            log.error('Missing column "Binary file name" in CSV.', 1)
        else:
            row = csv_file[csv_file["Binary file name"] == os.path.split(file)[1]]
            if row.empty:
                log.eraseLine(verbose_level=1)
                log.alert(f"Could not find file {file} in CSV file, skipping.")
                log.emptyLines(verbose_level=1)
                continue
        dataset = VectorDataset(file, 500, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, 0, 500, device=device, floatType=torch.float32, log=log, random_seed=None, masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)
        dl = td.DataLoader(dataset, batch_size=args.b, shuffle=False, num_workers=0)

        infill_sum = torch.zeros(len(dl), device=device, dtype=torch.int32)
        other_prints_sum = torch.zeros(len(dl), device=device, dtype=torch.int32)
        travel_sum = torch.zeros(len(dl), device=device, dtype=torch.int32)
        pr_sum = torch.zeros(len(dl), device=device, dtype=torch.int32)
        for i, (inputs, labels) in enumerate(dl):
            combined = torch.cat([inputs, labels], dim=-1)
            valid_mask = combined.abs().sum(dim=-1) > 1e-8
            inputs: torch.Tensor = inputs[valid_mask]
            labels: torch.Tensor = labels[valid_mask]
            segTypes = Utils.decodeMoveTypeFromBitField(inputs[..., index_of_seg_type])
            infill_sum[i] = torch.sum(segTypes["infill"]).item()
            other_prints_sum[i] = torch.sum(torch.concat([segTypes[key] for key in ["skin", "skirt", "wall", "support"]])).item()
            travel_sum[i] = torch.sum(torch.concat([segTypes[key] for key in ["travel", "combing", "zlift"]])).item()
            pr_sum[i] = torch.sum(torch.concat([segTypes[key] for key in ["prime", "retract"]])).item()
        nb_infill_instr = infill_sum.sum().item()
        nb_other_print_instr = other_prints_sum.sum().item()
        nb_travel_instr = travel_sum.sum().item()
        nb_pr_instr = pr_sum.sum().item()
        if not args.infilltype:
            infill_counts[row["Infill density (%)"].item()] += nb_infill_instr
            other_prints_counts[row["Infill density (%)"].item()] += nb_other_print_instr
            travel_counts[row["Infill density (%)"].item()] += nb_travel_instr
            pr_counts[row["Infill density (%)"].item()] += nb_pr_instr
            nb_files_per_range[row["Infill density (%)"].item()] += 1
        else:
            infill_counts[unique_infill_types.index(row["Infill type"].item())] += nb_infill_instr
            other_prints_counts[unique_infill_types.index(row["Infill type"].item())] += nb_other_print_instr
            travel_counts[unique_infill_types.index(row["Infill type"].item())] += nb_travel_instr
            pr_counts[unique_infill_types.index(row["Infill type"].item())] += nb_pr_instr
            nb_files_per_range[unique_infill_types.index(row["Infill type"].item())] += 1
    log.eraseLine(verbose_level=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bin_width = 5.0
    if not args.infilltype:
        bins = np.arange(0, 100 + bin_width, bin_width)
        densities = range(101)
        infill_bars, _ = np.histogram(densities, bins=bins, weights=[c * args.normbyvalue for c in infill_counts])
        other_print_bars, _ = np.histogram(densities, bins=bins, weights=[c * args.normbyvalue for c in other_prints_counts])
        travel_bars, _ = np.histogram(densities, bins=bins, weights=[c * args.normbyvalue for c in travel_counts])
        pr_bars, _ = np.histogram(densities, bins=bins, weights=[c * args.normbyvalue for c in pr_counts])
        nb_files = np.histogram(densities, bins=bins, weights=nb_files_per_range)[0]
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        if not args.justinfill:
            ax.bar(bin_centers, other_print_bars, width=bin_width * 0.9, edgecolor="black", align="center", label="Non-infill print moves")
            ax.bar(bin_centers, travel_bars, width=bin_width * 0.9, edgecolor="black", align="center", bottom=other_print_bars, label="Travel moves")
            ax.bar(bin_centers, pr_bars, width=bin_width * 0.9, edgecolor="black", align="center", bottom=[other_print_bars[i] + travel_bars[i] for i in range(len(travel_bars))], label="Prime and retract moves")
            ax.bar(bin_centers, infill_bars, width=bin_width * 0.9, edgecolor="black", align="center", bottom=[other_print_bars[i] + travel_bars[i] + pr_bars[i] for i in range(len(travel_bars))], label="Infill print moves")
        else:
            ax.bar(bin_centers, infill_bars, width=bin_width * 0.9, edgecolor="black", align="center")
        ax.set_xlabel("Infill density (%)")
        ax.set_xlim(0, 100)
        ranges = [f"[{a}%, {b}%[" for a, b in zip(bins[:-1], bins[1:])]
        log.info(f"Number of infill instruction per range of 5% infill density:\n{'\n'.join([f'{r}: {n} for {f} files. If 5% generation: {n * 0.05:.2f} ; if 95%: {n * 0.95:.2f}' for r, n, f in zip(ranges, infill_bars, nb_files)])}")
    else:
        valid_indexes: list[bool] = [c != 0 for c in nb_files_per_range]
        actual_infills = [val for i, val in enumerate(unique_infill_types) if valid_indexes[i]]
        if not args.justinfill:
            ax.bar(actual_infills, [val * args.normbyvalue for i, val in enumerate(other_prints_counts) if valid_indexes[i]], label="Non-infill print moves")
            ax.bar(actual_infills, [val * args.normbyvalue for i, val in enumerate(travel_counts) if valid_indexes[i]], bottom=[val * args.normbyvalue for i, val in enumerate(other_prints_counts) if valid_indexes[i]], label="Travel moves")
            ax.bar(actual_infills, [val * args.normbyvalue for i, val in enumerate(pr_counts) if valid_indexes[i]], bottom=[[val * args.normbyvalue for i, val in enumerate(other_prints_counts) if valid_indexes[i]][i] + [val * args.normbyvalue * args.normbyvalue for i, val in enumerate(travel_counts) if valid_indexes[i]][i] for i in range(len(valid_indexes))], label="Prime and retract moves")
            ax.bar(actual_infills, [val * args.normbyvalue for i, val in enumerate(infill_counts) if valid_indexes[i]], bottom=[[val * args.normbyvalue for i, val in enumerate(other_prints_counts) if valid_indexes[i]][i] + [val * args.normbyvalue for i, val in enumerate(travel_counts) if valid_indexes[i]][i] + [val * args.normbyvalue for i, val in enumerate(pr_counts) if valid_indexes[i]][i] for i in range(len(valid_indexes))], label="Infill print moves")
        else:
            ax.bar(actual_infills, [val * args.normbyvalue for i, val in enumerate(infill_counts) if valid_indexes[i]])
        ax.set_xlabel("Infill patterns")
        ax.set_xticks([i for i in range(len(actual_infills))])
        ax.set_xticklabels(actual_infills, rotation=45, ha='right')
        log.info(f"Number of infill instruction per infill type:\n{'\n'.join([f'{t}: {n} for {f} files' for t, n, f in zip(actual_infills, [val * args.normbyvalue for i, val in enumerate(infill_counts) if valid_indexes[i]], [val for i, val in enumerate(nb_files_per_range) if valid_indexes[i]])])}")
        log.info(f"Number of non-infill print instruction per infill type:\n{'\n'.join([f'{t}: {n} for {f} files' for t, n, f in zip(actual_infills, [val * args.normbyvalue for i, val in enumerate(other_prints_counts) if valid_indexes[i]], [val for i, val in enumerate(nb_files_per_range) if valid_indexes[i]])])}")
        log.info(f"Number of travel instruction per infill type:\n{'\n'.join([f'{t}: {n} for {f} files' for t, n, f in zip(actual_infills, [val * args.normbyvalue for i, val in enumerate(travel_counts) if valid_indexes[i]], [val for i, val in enumerate(nb_files_per_range) if valid_indexes[i]])])}")
        log.info(f"Number of prime and retract instruction per infill type:\n{'\n'.join([f'{t}: {n} for {f} files' for t, n, f in zip(actual_infills, [val * args.normbyvalue for i, val in enumerate(pr_counts) if valid_indexes[i]], [val for i, val in enumerate(nb_files_per_range) if valid_indexes[i]])])}")
    if not args.ylinear:
        plt.yscale("log")
    ax.set_ylabel("Number of instructions")
    ax.set_title(args.t)
    if not args.justinfill:
        ax.legend()
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