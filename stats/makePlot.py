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
import matplotlib.pyplot as plt
import numpy as np

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    parser.add_argument('--save_fig', default=None, type=str, required=False, \
                         help=f"Add this flag to save the figure as .pgf instead of displaying it, flag value indicates the name for the figure file.")

    parser.add_argument('--title', type=str, required=False, default="", help=f"Title of the graph")
    parser.add_argument('--xname', type=str, required=False, default=None, help=f"X axis name, default is the corresponding column name")
    parser.add_argument('--yname', type=str, required=False, default=None, help=f"Y axis name, default is the corresponding column name")

    parser.add_argument('--linear', action='store_true', default=True, help=f"(True by default) makes X axis scale linear")
    parser.add_argument('--logscale', dest='linear', action='store_false', help=f"makes X axis scale logarithmic")

    parser.add_argument('--exclude', required=False, default=None, type=str, help=f"Used only with the 'e-values' flags, used to exclude the rows that contain one of the specified values on the column given by this flag.")
    parser.add_argument('--e-values', required=False, default=None, nargs='+', type=str, help=f"Used only with the 'exclude' flags, see the help message of '--exclude'. You can add several values after the flag (e.g. \"--e-values 10 11 12\")")
    parser.add_argument('--include', required=False, default=None, type=str, help=f"Used only with the 'i-values' flags, used to include only the rows that contain one of the specified values on the column given by this flag.")
    parser.add_argument('--i-values', required=False, default=None, nargs='+', type=str, help=f"Used only with the 'include' flags, see the help message of '--include'. You can add several values after the flag (e.g. \"--i-values 10 11 12\")")

    parser.add_argument('-x', metavar="x_axis_stat", type=str, help=f"x axis data")
    parser.add_argument('-y', metavar="y_axis_stat", action='append', type=str, help=f"y axis data, this flag can be present one or two times for plotting several columns on the y axis")
    parser.add_argument('-f', metavar="file_path", action='append', type=str, help=f"path(s) to the data file(s), either a file or a directory. Files must end either in '.csv', '.xls', or '.xlsx', and must contain the same column names")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if args.f is None:
        log.error(f"No input file was given.", 1)
    for file in args.f:
        if not (os.path.isfile(file) or os.path.isdir(file)):
            log.warning(f"The specified argument {file} does not exist, it will not be considered")
            args.f.remove(file)
        if os.path.isfile(file) and not file.endswith((".csv", ".xls", ".xlsx")):
            log.warning(f"The specified file {file} is not of a supported file format ('.csv', '.xls', or '.xlsx'), it will not be considered")
            args.f.remove(file)
    if len(args.f) == 0:
        log.error(f"No given input file exists.", 1)
    if not (0 < len(args.y) < 3):
        log.error(f"Can only plot one or two columns on the Y axis", 1)
    if (args.exclude is None or args.e_values is None) and not (args.exclude is None and args.e_values is None):
        log.error(f"If one of the 2 following flags is provided, the other one must also be: '--exclude', and '--e-values'.", 1)
    if (args.include is None or args.i_values is None) and not (args.include is None and args.i_values is None):
        log.error(f"If one of the 2 following flags is provided, the other one must also be: '--include', and '--i-values'.", 1)
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


    if args.exclude is not None:
        if args.exclude not in csv.columns:
            log.error(f"Could not find a column with name '{args.exclude}' in CSV file(s)", 1)
        __valueType = csv[args.exclude].dtype
        values = np.array(args.e_values, dtype=__valueType)
        csv = csv[~csv[args.exclude].isin(values)]
    if args.include is not None:
        if args.include not in csv.columns:
            log.error(f"Could not find a column with name '{args.include}' in CSV file(s)", 1)
        __valueType = csv[args.include].dtype
        values = np.array(args.i_values, dtype=__valueType)
        csv = csv[csv[args.include].isin(values)]

    try:
        _ = csv[args.x]
    except KeyError:
        log.error(f"Could not find a column named '{args.x}' in the provided data file(s).", 1)
    for y in args.y:
        try:
            _ = csv[y]
        except KeyError:
            log.error(f"Could not find a column named '{y}' in the provided data file(s).", 1)

    nb_rows = csv.shape[0]
    log.info(f"Total number of rows: {nb_rows}")

    csv = csv[csv[args.x].notna()]

    if csv.shape[0] != nb_rows:
        log.warning(f"Dropped {nb_rows - csv.shape[0]} rows because of NaN values.")

    #fig, axs = plt.subplots(figsize=(4.5, 3.375))
    fig, axs = plt.subplots(figsize=(5.5, 4))
    fig.subplots_adjust(bottom=0.1, hspace=0.5)

    lines = []

    l1 = axs.scatter(
        csv[args.x], csv[args.y[0]],
        color="tab:blue", label=args.y[0],
        marker="."
    )
    axs.set_ylabel(args.y[0] if args.yname is None else args.yname)
    axs.set_xlabel(args.x if args.xname is None else args.xname)
    lines.append(l1)
    axs.grid(True)
    axs.set_title(args.title)
    if len(args.y) == 2:
        ax2 = axs.twinx()
        l2 = ax2.scatter(
            csv[args.x], csv[args.y[1]],
            color="tab:orange", label=args.y[1],
            marker="x"
        )
        ax2.set_ylabel(args.y[1])
        lines.append(l2)
    if not args.linear:
        axs.set_xscale("log")
        #axs.set_yscale("log")
        if len(args.y) == 2:
            ax2.set_yscale("log")
    axs.legend(lines, [l.get_label() for l in lines], loc="best")
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