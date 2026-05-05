import sys
import UpdateVectors as UV
try:
    import Logger
    import Utils
except:
    sys.path.insert(0,"..")
    import Logger
    import Utils
log = Logger.Logger(writesLog=False, verbose_level=1, prints_thread_id=False)

import torch
import os
import argparse

def arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_path", type=str, help=f"Path to the saved model file to read")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if not os.path.isfile(args.model_path):
        log.error(f"The given path is not a file: {args.model_path}", 1)
    return args

def main() -> None:
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    device = torch.device("cpu")

    model, lossFunction, prev_epoch, optimizer = Utils.parseModelArchitecture(args.model_path, device, torch.float32, 0.0, 0, log)

    model.summary()

    log.info(lossFunction.summary())
    log.emptyLines()

    log.info(f"Last training epoch: {prev_epoch}")
    log.emptyLines()

    log.info(f"Selected inputs: {UV.SELECTED_INPUTS}, selected labels: {UV.SELECTED_LABELS}")
    log.info(f"Updating functions: {', '.join([f.__name__ for f in UV.UPDATING_FUNCTIONS])}")

if __name__ == "__main__":
    main()

"""
Enf of file.
"""