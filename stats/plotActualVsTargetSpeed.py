import sys
import os
import argparse

import torch
from numpy import nan
sys.path.insert(0,"..")
import Logger
import Utils
import Constants
from REPET.VectorDataset import VectorDataset
import REPET.UpdateVectors as UV
from REPET.TransformerModel import TransformerModel
import matplotlib.pyplot as plt
log = Logger.Logger(verbose_level=1)

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """    
    # Optional arguments
    parser.add_argument('--save_fig', default=None, type=str, required=False, \
                         help=f"Add this flag to save the figure as .pgf instead of displaying it, flag value indicates the name for the figure file.")
    parser.add_argument('-n', required=False, default=30, type=int, metavar="nb_instr", \
                        help=f"number of instructions to plot (length of the sequence).")
    parser.add_argument('-i', required=False, default=0, type=int, metavar="start_index", help=f"Index of the instruction to use as the start of the plot, min is 0, max depends on the given file (error raised).")
    parser.add_argument('-m', required=False, default=None, type=str, metavar="model_file", \
                        help=f"path to a trained model path, used to draw its prediction graph along the others. If provided, the dataloader will be initialized with the model's parameters (such as sequence length, left and right offset), and the vector updates in `UpdateVectors.py` must match the updates used during the training of said model.")
    
    # Required arguments
    parser.add_argument('dataset_path', metavar="dataset_path", type=str, help=f"path to dataset to explore, MUST BE A SINGLE FILE")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if not os.path.isfile(args.dataset_path):
        log.error(f"The given file does not exist: {args.dataset_path}", 1)
    if args.n < 0:
        log.error(f"Invalid number of instructions: {args.n}", 1)
    if args.i < 0:
        log.error(f"Invalid first instruction index: {args.i}", 1)
    if args.m is not None and (not os.path.isfile(args.m)):
        log.error(f"The given model file path does not exist: {args.m}", 1)
    return args

def main() -> None:
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    try:
        target_speed_index = UV.SELECTED_INPUTS.index(6) if UV.SELECTED_INPUTS is not None else 6
    except ValueError:
        log.error(f"In order to use this script, the instruction target value must be present in the input vectors (aka `SELECTED_INPUTS` in `UpdateVectors.py` must contain the index `6`).", 1)

    device = torch.device('cpu')

    if args.m is not None:
        model, _, _, _ = Utils.parseModelArchitecture(args.m, device, torch.float32, 0.0, 1, log, True)
        model.summary()
    else:
        model = None

    if UV.SELECTED_LABELS is not None:
        log.error(f"In order to use this script, all label values must be retained (aka `SELECTED_LABELS` in `UpdateVectors.py` must be `None`)", 1)

    if model is None:
        dataset = VectorDataset(args.dataset_path, Constants.DEFAULT_SAMPLE_SIZE, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, Constants.DEFAULT_LEFT_OFFSET, Constants.DEFAULT_RIGHT_OFFSET, device, log=log, masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)
    else:
        dataset = VectorDataset(args.dataset_path, model.input_size[0], Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, model.left_offset, model.right_offset, device, log=log, masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)

    dataset.summary()

    if args.i + args.n >= len(dataset) * (dataset.right_offset - dataset.left_offset):
        log.error(f"The given arguments are too large for the file: start index = {args.i}, length of instruction sequence = {args.n} (sum is {args.i + args.n}), and length of the file {len(dataset) * (dataset.right_offset - dataset.left_offset)}", 1)

    sample_id = args.i // (dataset.right_offset - dataset.left_offset)
    start_id_in_sample = args.i % (dataset.right_offset - dataset.left_offset)

    nb_instr = 0
    concat_inputs = torch.zeros((args.n, UV.TRUE_INPUT_VECTOR_SIZE), device=device, dtype=torch.float32)
    concat_predictions = torch.zeros((args.n, UV.TRUE_LABEL_VECTOR_SIZE), device=device, dtype=torch.float32)
    concat_labels = torch.zeros((args.n, UV.TRUE_LABEL_VECTOR_SIZE), device=device, dtype=torch.float32)
    i = 0

    while nb_instr < args.n:
        sample = dataset[sample_id]
        inputs = sample[0]
        labels = sample[1]
        if model is not None:
            prediction: torch.Tensor = model(sample[0].unsqueeze(0)).squeeze()
            if model.trained_on_log_target and not (hasattr(model, "simple_rule_head") and model.simple_rule_head):
                prediction = torch.exp(prediction) - Constants.DEFAULT_LOG_NORM_EPSILON
            if model.nbInstrPerSample == 1:
                inputs = inputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
                prediction = prediction.unsqueeze(0)
        if nb_instr == 0:
            nb_instr += (dataset.right_offset - dataset.left_offset) - start_id_in_sample
            if nb_instr >= args.n:
                concat_inputs = inputs[(dataset.left_offset + start_id_in_sample):(dataset.left_offset + start_id_in_sample + args.n), :]
                concat_labels = labels[start_id_in_sample:(start_id_in_sample + args.n), :]
                if model is not None:
                    concat_predictions = prediction[start_id_in_sample:(start_id_in_sample + args.n), :]
            else:
                concat_inputs[:nb_instr, :] = inputs[(dataset.left_offset + start_id_in_sample):dataset.right_offset, :]
                concat_labels[:nb_instr, :] = labels[start_id_in_sample:, :]
                if model is not None:
                    concat_predictions[:nb_instr, :] = prediction[start_id_in_sample:, :]
        else:
            nb_instr += (dataset.right_offset - dataset.left_offset)
            if nb_instr >= args.n:
                needed = args.n - i
                concat_inputs[i:, :] = inputs[dataset.left_offset : dataset.left_offset + needed, :]
                concat_labels[i:, :] = labels[:needed, :]
                if model is not None:
                    concat_predictions[i:, :] = prediction[:needed, :]
            else:
                concat_inputs[i:nb_instr, :] = inputs[dataset.left_offset:dataset.right_offset, :]
                concat_labels[i:nb_instr, :] = labels
                if model is not None:
                    concat_predictions[i:nb_instr, :] = prediction
        i = nb_instr
        sample_id += 1

    model_times = []
    model_speeds = []
    label_times = []
    label_speeds = []
    target_speeds = []
    target_times = []
    current_time = 0.0

    plt.figure(figsize=(4.5, 2.25))
    plt.axvline(x = current_time, color='gray')

    for i in range(args.n):
        target_speeds.append(concat_inputs[i][target_speed_index].item() / 60)
        target_speeds.append(concat_inputs[i][target_speed_index].item() / 60)
        target_speeds.append(nan)

        label = concat_labels[i]
        start_speed = label[1].item()
        cruise_speed = label[2].item()
        end_speed = label[3].item()
        accel_time = label[4].item()
        cruise_time = label[5].item()
        decel_time = label[6].item()

        target_times.append(current_time)

        if model is not None:
            prediction = concat_predictions[i]
            model_accel_time = prediction[4].item()
            model_cruise_time = prediction[5].item()
            model_decel_time = prediction[6].item()
            model_start_speed = prediction[1].item()
            model_cruise_speed = prediction[2].item()
            model_end_speed = prediction[3].item()
            model_times.extend([current_time, current_time + model_accel_time, current_time + model_accel_time + model_cruise_time, current_time + model_accel_time + model_cruise_time + model_decel_time, nan])
            model_speeds.extend([model_start_speed, model_cruise_speed, model_cruise_speed, model_end_speed, nan])
            #plt.axvline(x = current_time + prediction[0].item(), color='orange', linestyle=':')

        if accel_time > 0:
            t_accel = torch.linspace(0, accel_time, steps=10)
            v_accel = torch.linspace(start_speed, cruise_speed, steps=10)
            label_times.extend((t_accel + current_time).tolist())
            label_speeds.extend(v_accel.tolist())
            current_time += accel_time
        if cruise_time > 0:
            t_cruise = torch.linspace(0, cruise_time, steps=2)
            v_cruise = torch.full((2,), cruise_speed)
            label_times.extend((t_cruise + current_time).tolist())
            label_speeds.extend(v_cruise.tolist())
            current_time += cruise_time
        if decel_time > 0:
            t_decel = torch.linspace(0, decel_time, steps=10)
            v_decel = torch.linspace(cruise_speed, end_speed, steps=10)
            label_times.extend((t_decel + current_time).tolist())
            label_speeds.extend(v_decel.tolist())
            current_time += decel_time
        target_times.append(current_time)
        target_times.append(nan)
        plt.axvline(x = current_time, color='gray')

    plt.plot(target_times, target_speeds, 'r--', label="Target Speed (F)")
    plt.plot(label_times, label_speeds, label="Actual Speed")
    if model is not None:
        plt.plot(model_times, model_speeds, linestyle='-.', label="Model predictions")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (mm/s)")
    plt.legend()
    plt.grid(axis='y')
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