import sys
try:
    from Logger import *
    import Utils
except:
    sys.path.insert(0,"..")
    from Logger import *
    import Utils
log = Logger(writesLog=False)
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--save_fig', default=None, type=str, required=False, \
                         help=f"Add this flag to save the figure as .pgf instead of displaying it, flag value indicates the name for the figure file.")
    parser.add_argument('-c', metavar="column_name", action='append', type=str, help=f"column name in the CSV file. User can set up to 7 '-c' flags")
    parser.add_argument('-f', required=True, metavar="csv_file_path", type=str, help=f"path to the CSV file")

    parser.add_argument('--linear', action='store_true', default=True, help=f"(True by default) makes y axis scale linear")
    parser.add_argument('--logscale', dest='linear', action='store_false', help=f"makes y axis scale logarithmic")
    parser.add_argument('--title', type=str, metavar="graph_title", default=None, help=f"title displayed for the graph")
    parser.add_argument('--train', action='store_true', default=False, help=f"add this flag to draw all train losses (separate)")
    parser.add_argument('--valid', action='store_true', default=False, help=f"add this flag to draw all validation losses (separate)")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if not (os.path.isfile(args.f)):
        log.error(f"The given file {args.f} does not exist.", 1)
    if args.c != None and len(args.c) > 7:
        log.error(f"Can only parse up to 7 columns (got {len(args.c)}).", 1)
    return args

def main() -> None:
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    csv = pd.read_csv(args.f)
        
    indexes = list(range(1, len(csv.index) + 1))

    columns = csv.columns
    train_columns = [c for c in columns if c.startswith("Train")][1:]
    eval_columns = [c for c in columns if c.startswith("Eval")][1:]

    color_index = 0
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    if args.train:
        args.c = [
            "Train value 1 loss",
            "Train value 2 loss",
            "Train value 3 loss",
            "Train value 4 loss",
            "Train value 5 loss",
            "Train value 6 loss",
            "Train value 7 loss"
            ]
        args.c = [args.c[i] for i in range(len(train_columns))]
        if args.title is None:
            args.title = "Individual train loss values for each epoch"
        labels = [
            "Total time train loss",
            "Start velocity train loss",
            "Cruise velocity train loss",
            "End velocity train loss",
            "Acceleration train time loss",
            "Cruise time train loss",
            "Deceleration train time loss"
        ]
        labels = [labels[i] for i in range(len(train_columns))]
    elif args.valid:
        args.c = [
            "Eval value 1 loss",
            "Eval value 2 loss",
            "Eval value 3 loss",
            "Eval value 4 loss",
            "Eval value 5 loss",
            "Eval value 6 loss",
            "Eval value 7 loss"
            ]
        args.c = [args.c[i] for i in range(len(eval_columns))]
        if args.title is None:
            args.title = "Individual validation loss values for each epoch"
        labels = [
            "Total time validation loss",
            "Start velocity validation loss",
            "Cruise velocity validation loss",
            "End velocity validation loss",
            "Acceleration validation time loss",
            "Cruise time validation loss",
            "Deceleration validation time loss"
        ]
        labels = [labels[i] for i in range(len(eval_columns))]
    elif args.c == None:
        args.c = ["Train loss", "Eval loss"]
        labels = args.c
    else:
        labels = args.c
    plt.figure(figsize=(4.5, 3.375))
    for c, l in zip(args.c, labels):
        plt.plot(indexes, csv[c], linestyle='-', color=colors[color_index], label=l)
        color_index += 1
    plt.title(f"{args.title if args.title != None else "Train and validation loss for each epoch"}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    if not args.linear:
        plt.yscale("log")
    plt.legend()
    plt.grid(linestyle = '--', linewidth = 0.5)
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