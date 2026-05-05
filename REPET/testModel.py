import sys
try:
    import Logger
    import Utils
    import Constants
except:
    sys.path.insert(0,"..")
    import Logger
    import Utils
    import Constants
log = Logger.Logger(writesLog=False, verbose_level=0, prints_thread_id=False)
import VectorDataset as vd
import UpdateVectors as UV
from TransformerModel import *
from BiLSTMModel import *
from ConvolutionModel import *

import torch
import os
import time
import argparse
from glob import glob
import torch.utils.data as td
import numpy as np

"""
In order to print the boxplot values from the paper's Figure 9, uncomment the section labeled with "boxplot code".

WARNING: if you do execute the script for boxplot values on the full test dataset, note that this script will hence required around 250GB of memory!
"""

params = {
    "Filename": "Filename",
    "Slicer": "Slicer",
    "Printer": "Printer",
    "infillType": "Infill type",
    "infillDensity": "Infill density (%)"
}

# boxplot code
# isLogScale = False
# start = 0.0
# stop = 0.22
# ranges = list(Utils.frange(start, stop, 0.013333))
# if isLogScale:
#     n = len(ranges)

#     t = np.linspace(0.0, 1.0, n)

#     gamma = 3.0  # 1: linear, 2: moderate compression, 3-5: stronger compression
#     mapped = t ** gamma

#     ranges = (start + (stop - start) * mapped).tolist()
# sAPEs = [[] for _ in range(len(ranges) - 1)]

def arguments(parser: argparse.ArgumentParser) -> None:
    # Global arguments
    parser.add_argument('-v', '--verbose', required=False, action='count', default=0, \
                    help=f"Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this argument is specified")
    parser.add_argument('-q', '--quiet', required=False, action='count', default=0, \
                    help=f"Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time this argument is specified")
    
    # Optionnal arguments
    parser.add_argument('-o', required=False, default="./results/", metavar="result_output_dir", type=str, \
                        help=f"Result output directory, default is './results/' (will not save if 'None')")
    parser.add_argument('-l', required=False, default=0, metavar="label_vector_value", type=int, \
                        help=f"Label vector index value (aka which label vector to sum, and map for decile MSE), default is 0 max is \
{UV.TRUE_LABEL_VECTOR_SIZE - 1}")
    
    # Data arguments, mandatory and positional
    parser.add_argument("data_source", type=str, \
                    help="Path to the data (single file or folder, only files in the DAT format will be used).")
    parser.add_argument("model_path", type=str, \
                    help="Path to the saved Pytorch model.")
    

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if (not os.path.isfile(args.data_source)) and (not os.path.isdir(args.data_source)):
        log.error(f"Could not find the provided data source: {args.data_source}", 1)
    if not os.path.isfile(args.model_path):
        log.error(f"Could not find the provided model path: {args.model_path}", 1)
    if not (0 <= args.l < UV.TRUE_LABEL_VECTOR_SIZE):
        log.error(f"Label vector value must be within [0; {UV.TRUE_LABEL_VECTOR_SIZE}[", 1)
    return args

def predictFile(model: nn.Module, file_path: str, label_idx: int) -> tuple[Tensor, Tensor, float, float, float]:
    """
    Parameters:
    - model:        the nn.Module to use for testing
    - file_path:    the full path to the file to predict
    - label_idx:    an int indicating which value from the label vectors should be used for PE processing

    Returns:
    - The total sum prediction of the `label_idx` label vector value for the file located at `file_path`
    - The total sum label of the `label_idx` label vector value for the file located at `file_path`
    - The final PE of the prediction of the `label_idx` label vector value for the file located at `file_path`
    - The dataset build time for that file
    - The full file prediction time
    - A Tensor of shape [label_vector_size] containing the sMAPE values of each label vector values
    - The length of the dataloader for that file, used for sMAPE merging of all files
    - A Tensor of shape [label_vector_size] containing the sum of square or residuals of each label vector values, used for R2 score and MSE
    - A Tensor of shape [label_vector_size] containing the total sum of squares of each label vector values, used for R2 score
    - A Tensor of shape [label_vector_size] containing the total sum of absolute errors of each label vector values, used for MAE
    """
    # boxplot code
    # global ranges, sAPEs
    start_time = time.time()
    dataset = vd.VectorDataset(file_path, model.nbInstrPerSample, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, \
                                    model.left_offset, model.right_offset, model.device, floatType=model.floatType, log=model.log, random_seed=None, \
                                    masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, append_input_vectors=UV.UPDATING_FUNCTIONS)
    build_time = time.time() - start_time
    dataloader = td.DataLoader(dataset, batch_size=Constants.DEFAULT_BATCH_SIZE, shuffle=False)
    total_pred = torch.zeros((len(dataloader), 1), device=model.device, dtype=model.floatType)
    total_label = torch.zeros((len(dataloader), 1), device=model.device, dtype=model.floatType)
    total_instr_sape_sum = torch.zeros((len(dataloader), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device, dtype=model.floatType)
    ssres_sum: torch.Tensor = torch.zeros((len(dataloader), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device)
    sstot_sum: torch.Tensor = torch.zeros((len(dataloader), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device)
    all_se: torch.Tensor = torch.zeros((len(dataloader), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device, dtype=model.floatType)
    all_ae: torch.Tensor = torch.zeros((len(dataloader), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device, dtype=model.floatType)
    if UV.SELECTED_LABELS is not None:
        mean_values = torch.tensor([Constants.DEFAULT_LABEL_MEAN_VALUES[i] for i in UV.SELECTED_LABELS], dtype=torch.float32, device=model.device)
    else:
        mean_values = torch.tensor(Constants.DEFAULT_LABEL_MEAN_VALUES, dtype=torch.float32, device=model.device)
    total_instr_count = 0
    epsilon = torch.scalar_tensor(Constants.DEFAULT_DIVISION_BY_ZERO_EPSILON, device=model.device, dtype=model.floatType)
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            data: torch.Tensor = batch[0]
            target: torch.Tensor = batch[1]
            output: torch.Tensor = model(data)
            if model.trained_on_log_target and not (hasattr(model, "simple_rule_head") and model.simple_rule_head):
                output = torch.exp(output) - Constants.DEFAULT_LOG_NORM_EPSILON
            invalid = output[..., label_idx].isnan().logical_or(output[..., label_idx].isinf())
            if invalid.any():
                log.eraseLine(verbose_level=1)
                log.alert(f"File {file_path}, batch [{i + 1}/{len(dataloader)}]: detected either a NaN or an infinite value!")
                log.emptyLines(verbose_level=1)
                b_idx, t_idx = invalid.nonzero(as_tuple=True)
                output[b_idx, t_idx, label_idx] = 0
            if UV.SELECTED_LABELS is None or 0 in UV.SELECTED_LABELS or model.simple_rule_head:
                too_high = output[..., 0] > 10 * Constants.MEASURED_MAXIMUM_INSTR_TOTAL_TIME
                if too_high.any():
                    log.eraseLine(verbose_level=1)
                    log.alert(f"File {file_path}, batch [{i + 1}/{len(dataloader)}]: detected a predicted total instruction duration 10 times above the measured dataset maximum!")
                    log.emptyLines(verbose_level=1)
                    b_idx, t_idx = too_high.nonzero(as_tuple=True)
                    output[b_idx, t_idx, 0] = 0
            num = (output - target).abs()
            den = (output.abs() + target.abs())
            all_sape = torch.where(
                den < epsilon,
                torch.zeros_like(den),
                num / den
            )
            #all_sape = ((output - target).abs() / ((output.abs() + target.abs() + epsilon) / 2))
            ssres_sum[i] = ((target - output).square()).sum(dim=(0, 1))
            sstot_sum[i] = ((target - mean_values).square()).sum(dim=(0, 1))
            all_ae[i] = ((target - output).abs()).sum(dim=(0, 1))
            total_instr_sape_sum[i] = all_sape.sum(dim=(0, 1))
            # boxplot code
            # target_vals = target[:, :, label_idx].reshape(-1).cpu().numpy()
            # sape_vals = (all_sape[:, :, label_idx] * 100).reshape(-1).cpu().numpy()
            # for val, err in zip(target_vals, sape_vals):
            #     for idx in range(len(ranges) - 1):
            #         if ranges[idx] <= val < ranges[idx + 1]:
            #             sAPEs[idx].append(err)
            #             break
            total_instr_count += output.shape[0] * output.shape[1]
            total_pred[i, :] = torch.sum(output[:, :, label_idx])
            total_label[i, :] = torch.sum(target[:, :, label_idx])
        process_time = time.time() - start_time
    ssres_sum = ssres_sum.sum(dim=0).squeeze()
    sstot_sum = sstot_sum.sum(dim=0).squeeze()
    all_ae = all_ae.sum(dim=0).squeeze()
    total_instr_sape_sum = total_instr_sape_sum.sum(dim=0).squeeze()
    prediction = torch.sum(total_pred)
    label = torch.sum(total_label)
    percent_error = (prediction.item() * 100 / label.item() - 100)
    return prediction, label, percent_error, build_time, process_time, total_instr_sape_sum, total_instr_count, ssres_sum, sstot_sum, all_ae

def main() -> None:
    # Main variables initialization
    global log
    # boxplot code
    # global ranges, sAPEs
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    if args.verbose >= 0 and args.quiet == 0:
        verbose_level = args.verbose
    elif args.verbose == 0 and args.quiet > 0:
        verbose_level = - args.quiet
    else:
        log.error(f"Cannot execute the program in both quiet and verbose mode.", 1)
        
    log = Logger(verbose_level=verbose_level, writesLog=True, prints_thread_id=False)
    
    log.emptyLines()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_printoptions(precision=6)
    
    log.info(f"Device used: {device}")

    model, _, _, _ = Utils.parseModelArchitecture(args.model_path, device, torch.float32, 0.0, 1, log, True)
    model.summary()
    log.info(f"Selected input vector indexes: {UV.SELECTED_INPUTS} (if None, all are retained)")
    log.info(f"Selected label vector indexes: {UV.SELECTED_LABELS} (if None, all are retained)")
    funcs = ", ".join([f.__name__ for f in UV.UPDATING_FUNCTIONS])
    log.info(f"Used input vector updating functions: {funcs}")
    log.emptyLines()

    if not os.path.isdir(args.o):
        os.makedirs(args.o)

    if os.path.isfile(args.data_source):
        file_res = predictFile(model, args.data_source, args.l)
        log.info(f"Predicted {Utils.format_time(file_res[0].item())} print time (actual: {Utils.format_time(file_res[1].item())}), \
which equates to {file_res[2]}% error.")
        log.info(f"Took {Utils.format_time(file_res[4])} to process")
    else:
        csv_file = Utils.get_next_versioned_filename(os.path.join(args.o, f"testingResults_{model.getFilename()}_labelIdx-{args.l}.csv"))
        path = f"{args.data_source}/**/*.dat"
        filenames = glob(path, recursive=True)
        log.info(f"Found {len(filenames)} files to predict.")
        predict_times = torch.zeros(len(filenames), device=device)
        label_times = torch.zeros(len(filenames), device=device)
        total_instr_sape_sum: torch.Tensor = torch.zeros((len(filenames), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device)
        ssres_sums: torch.Tensor = torch.zeros((len(filenames), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device)
        sstot_sums: torch.Tensor = torch.zeros((len(filenames), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device)
        ae_all: torch.Tensor = torch.zeros((len(filenames), UV.TRUE_LABEL_VECTOR_SIZE), device=model.device)
        total_instr_count = 0
        j = 0
        with open(csv_file, 'w+') as f:
            start_time = time.time()
            f.write(f"Binary file name,Predicted time (s),Label time (s),Percentage error (%),Dataset build time (s),Prediction time (s)\n")
            log.emptyLines(verbose_level=1)
            for i, file in enumerate(filenames):
                log.eraseLine(verbose_level=1)
                log.debug(f"Predicting file [{i + 1}/{len(filenames)}] {file} ...")
                try:
                    file_res = predictFile(model, file, args.l)
                    f.write(f"{os.path.split(file)[1]},")
                    f.write(f"{','.join([f'{v:.6f}' for v in file_res[0:5]])}\n")
                    predict_times[j] = file_res[0]
                    label_times[j] = file_res[1]
                    total_instr_sape_sum[i] += file_res[5]
                    ssres_sums[i] += file_res[7]
                    sstot_sums[i] += file_res[8]
                    ae_all[i] += file_res[9]
                    total_instr_count += file_res[6]
                    j += 1
                except BaseException as e:
                    log.eraseLine(verbose_level=1)
                    log.warning(f"Could not predict {file}", e)
                    log.emptyLines(verbose_level=1)
                f.flush()
            log.eraseLine(verbose_level=1)
            processing_time = time.time() - start_time
            true_smape = (2 * total_instr_sape_sum.sum(dim=0) / total_instr_count) * 100
            square_error_sum = ssres_sums.sum(dim=0)
            r2_score = 1 - (square_error_sum/ sstot_sums.sum(dim=0))
            mse = square_error_sum / total_instr_count
            mae = ae_all.sum(dim=0) / total_instr_count
            log.info(f"Done. Took {Utils.format_time(processing_time)} to predict all files.")
            # boxplot code
            # log.info(f"Started processing boxplot values...")
            # start_time = time.time()
            # sub_sum = 0
            # sub_mean_sum = 0
            # for i, bucket in enumerate(sAPEs):
            #     if not bucket:
            #         continue
            #     arr = np.array(bucket)
            #     q1 = np.percentile(arr, 25)
            #     q3 = np.percentile(arr, 75)
            #     iqr = q3 - q1
            #     low = max(0, q1 - 1.5 * iqr)
            #     high = min(200, q3 + 1.5 * iqr)
            #     median = np.median(arr)
            #     mean = np.mean(arr)
            #     sub_sum += len(bucket)
            #     sub_mean_sum += len(bucket) * mean
            #     log.info(f"Range [{ranges[i]:.5f}, {ranges[i+1]:.5f}]: Q1-1.5IQR={low:.2f}, Q1={q1:.2f}, median={median:.2f}, Q3={q3:.2f}, Q3+1.5IQR={high:.2f}, mean={mean:.2f}, count={len(bucket)}")
            # processing_time = time.time() - start_time
            # log.info(f"Done. Took {Utils.format_time(processing_time)}")
            # log.notice(f"Theoretical sMAPE based on the processed boxplots: {sub_mean_sum / sub_sum:.2f}%")
            log.info(f"Predicted a total print time of {Utils.format_time(predict_times.sum().item())}, \
when the label time is {Utils.format_time(label_times.sum().item())}.")
            log.info(f"Each label vector value sMAPE is, for the full dataset: {', '.join([f'{v:.4f}%' for v in true_smape.tolist()])}")
            log.info(f"Each label vector value R2 score is, for the full dataset: {', '.join([f'{v:.4f}' for v in r2_score.tolist()])}")
            log.info(f"Each label vector value MSE is, for the full dataset: {', '.join([f'{v:.6f}' for v in mse.tolist()])}")
            log.info(f"Each label vector value MAE is, for the full dataset: {', '.join([f'{v:.6f}' for v in mae.tolist()])}")
            if args.o == "None":
                os.remove(csv_file)

if __name__ == "__main__":
    main()

"""
End of file.
"""