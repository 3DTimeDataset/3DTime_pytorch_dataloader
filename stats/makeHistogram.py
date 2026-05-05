import sys

try:
    import Logger, Utils
except:
    sys.path.insert(0,"..")
    import Logger, Utils
log = Logger.Logger()
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    parser.add_argument('--save_fig', default=None, type=str, required=False, \
                         help=f"Add this flag to save the figure as .pgf instead of displaying it, flag value indicates the name for the figure file.")

    parser.add_argument('--title', type=str, required=False, default="", help=f"Title of the graph")
    parser.add_argument('--xname', type=str, required=False, default=None, help=f"X axis name, default is the corresponding column name")

    parser.add_argument('--linear', action='store_true', default=True, help=f"(True by default) makes Y axis scale linear")
    parser.add_argument('--logscale', dest='linear', action='store_false', help=f"makes Y axis scale logarithmic")

    parser.add_argument('-x', metavar="x_axis_stat", type=str, help=f"main stat for the histogram")
    parser.add_argument('-n', metavar="nb_bars", required=False, type=int, default=50, help=f"number of bars to plot for the histogram")
    parser.add_argument('-f', metavar="file_path", action='append', type=str, help=f"path(s) to the data file(s), either a file or a directory. Files must end either in '.csv', '.xls', or '.xlsx', and must contain the same column names")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if args.f is None:
        log.error(f"No input file was given.", 1)
    if args.n < 1:
        log.error(f"The number of bars in the histogram cannot be lower than 1, provided: {args.n}", 1)
    for file in args.f:
        if not (os.path.isfile(file) or os.path.isdir(file)):
            log.warning(f"The specified argument {file} does not exist, it will not be considered")
            args.f.remove(file)
        if os.path.isfile(file) and not file.endswith((".csv", ".xls", ".xlsx")):
            log.warning(f"The specified file {file} is not of a supported file format ('.csv', '.xls', or '.xlsx'), it will not be considered")
            args.f.remove(file)
    if len(args.f) == 0:
        log.error(f"No given input file exists.", 1)
    return args

def main() -> None:
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    csv_files = []
    excel_files = []
    for f in args.f:
        if os.path.isfile(f) and f.endswith(".csv"):
            csv_files.append(f)
        elif os.path.isfile(f) and f.endswith((".xls", "xlsx")):
            excel_files.append(f)
        elif os.path.isdir(f):
            csv_files.extend(glob.glob(os.path.join(f, "*.csv")))
            excel_files.extend(glob.glob(os.path.join(f, "*.xlsx")))
            excel_files.extend(glob.glob(os.path.join(f, "*.xls")))
    all_dataframes = [pd.read_csv(f) for f in csv_files]
    all_dataframes.extend([pd.read_excel(f) for f in excel_files])
    csv = pd.concat(all_dataframes, ignore_index=True)

    main_stat = args.x

    try:
        _ = csv[main_stat]
    except KeyError:
        log.error(f"Could not find a column named '{main_stat}' in the provided data file(s).", 1)

    nb_rows = csv.shape[0]
    log.info(f"Total number of rows: {nb_rows}")

    csv = csv[csv[main_stat].notna()]

    if csv.shape[0] != nb_rows:
        log.warning(f"Dropped {nb_rows - csv.shape[0]} rows because of NaN values.")

    #csv = csv[csv["Segment Type"] != "UNKNOWN"]
    
    log.info(f"Total value of {main_stat}: {csv[main_stat].sum()}, with a mean of {csv[main_stat].mean()}")
    log.info(f"Mean absolute value of {main_stat}: {csv[main_stat].abs().mean()}")
    minValue = csv[main_stat].min()
    maxValue = csv[main_stat].max()
    log.info(f"Min value of {main_stat}: {minValue}, and max value: {maxValue}")

    bin_width = (maxValue - minValue) / args.n
    values = csv[main_stat].to_numpy()
    bins = np.arange(minValue, maxValue + bin_width, bin_width)
    bars, _ = np.histogram(values, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    fig, axs = plt.subplots(figsize=(6, 4))
    axs.bar(bin_centers, bars, width=bin_width * 0.9, edgecolor="black", align="center")
    axs.set_xlabel(main_stat if args.xname is None else args.xname)
    axs.set_ylabel("Frequency count")
    axs.grid(True)
    axs.set_title(args.title)
    if not args.linear:
        plt.yscale("log")
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