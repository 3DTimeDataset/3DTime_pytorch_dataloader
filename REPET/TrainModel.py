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
from FullyConnectedModel import *
from smallerModels.LinearRegression import *
from smallerModels.constantMeanPrediction import *
from smallerModels.SimpleRulePrediction import *
from smallerModels.FixedPredFromTrapez import *

import torch
import os
import time
import argparse
import psutil
from math import isnan
from torch import nn, Tensor
import torch.utils.data as td

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    ## Argument definitions.

    Usage: TrainModel.py [-h] [-v] [-q] [-b batch_size] [-e nb_epoch] [-s stop_condition] [-l learning_rate] [-d dropout_rate]
                         [-p torch_float_type] [-o model_folder] [-i instr_per_sample] [-m model_type] [-a nb_head] [-n tlayer_count]
                          [-z embed_dim] [-f feed_forward_dim] [-k kernel_lengths [kernel_lengths ...]] [-c convs_channels [convs_channels ...]]
                         [-w pretrain_weights]
                          train_source test_source

    ### Positional arguments:

    - train_source        Folder name where the files (DAT format) for training are stored
    - test_source         Folder name where the files (DAT format) for testing are stored

    ### Options:

    -h, --help                                    Show this help message and exit

    -v, --verbose                                 Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this\
                                                  argument is specified

    -q, --quiet                                   Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time\
                                                  this argument is specified

    -b batch_size                                 Batch size, default value is 256

    -e nb_epoch                                   Number of epoch for training, default value is 3

    -s stop_condition                             Stop condition for training, default value is 3

    -l learning_rate                              Learning rate for training, default value is 0.0001

    -d dropout_rate                               Dropout rate for training, default value is 0.0

    -p torch_float_type                           Pytorch float type (or precision), must be either 32 or 64, default value is 32

    -o model_folder                               Folder to store the trained model to, default is ./models, "None" if you do not want to save the model

    -i instr_per_sample                           Number of instructions per sample, default value is 100

    -m model_type                                 Model's architecture type, either 'Transformer' (default), 'BiLSTM', or 'Convolution'

    -a nb_head                                    Transformer attention head amount, default value is 1

    -n tlayer_count                               Number of layer (Transformer or BiLSTM), default value is 1

    -z embed_dim                                  Embed dimension for Transformer, or hidden size for BiLSTM, default value is 16

    -f feed_forward_dim                           Transformer feed forward dimension, default value is 128

    -k kernel_lengths [kernel_lengths ...]        Convolution model kernels' lengths (list), in order, output size will depend on these values.\
                                                  Default is [5, 5, 23], list length must match with 'convs_channels'

    -c convs_channels [convs_channels ...]        Convolution model layers' channels (list), in order, last value of the list must be 1.\
                                                  Default is [32, 12, 1], list length must match with 'kernel_lengths'

    -w pretrain_weights                           Full path to a pre-trained model's weights (pytorch .pt file), if provided, all other architecture\
                                                  params will be ignored
    """
    # Global arguments
    parser.add_argument('-v', '--verbose', required=False, action='count', default=0, \
                     help=f"Verbose level argument, if none specified, verbose level is 0, adds one to the latter for each time this argument is specified")
    parser.add_argument('-q', '--quiet', required=False, action='count', default=0, \
                     help=f"Quiet level argument, if none specified, verbose level is 0, subtracts one to the latter for each time this argument is specified")
    
    parser.add_argument('--rs', required=False, type=int, default=None, help=f"Manual random seed, default to None (uses the default value in Constants.py)")
    parser.add_argument('--lne', required=False, type=float, default=None, help=f"Log-normalization epsilon, default to None (uses the default value in Constants.py)")
    
    # Training hyper-parameter arguments
    parser.add_argument('--bs', required=False, default=Constants.DEFAULT_BATCH_SIZE, metavar="batch_size", type=int, \
                    help=f"Batch size, default value is {Constants.DEFAULT_BATCH_SIZE}")
    parser.add_argument('--ne', required=False, default=Constants.DEFAULT_NB_EPOCH, metavar="nb_epoch", type=int, \
                    help=f"Number of epoch for training, default value is {Constants.DEFAULT_NB_EPOCH}")
    parser.add_argument('--sc', required=False, default=Constants.DEFAULT_STOP_CONDITION, metavar="stop_condition", type=int, \
                    help=f"Stop condition for training, default value is {Constants.DEFAULT_STOP_CONDITION}")
    parser.add_argument('--lr', required=False, default=Constants.DEFAULT_LEARNING_RATE, metavar="learning_rate", type=float, \
                    help=f"Learning rate for training, default value is {Constants.DEFAULT_LEARNING_RATE}")
    parser.add_argument('--dp', required=False, default=Constants.DEFAULT_DROPOUT_RATE, metavar="dropout_rate", type=float, \
                    help=f"Dropout rate for training, default value is {Constants.DEFAULT_DROPOUT_RATE}")
    parser.add_argument('--ft', required=False, default=Constants.DEFAULT_FLOAT_TYPE, metavar="torch_float_type", type=int, \
                    help=f"Pytorch float type (or precision), must be either 32 or 64, default value is {Constants.DEFAULT_FLOAT_TYPE}")
    parser.add_argument('--mf', required=False, default=Constants.DEFAULT_MODEL_FOLDER, metavar="model_folder", type=str, \
                    help=f"Folder to store the trained model to, default is {Constants.DEFAULT_MODEL_FOLDER}, \"None\" if you do not want to save the model")
    parser.add_argument('--lo', required=False, default=Constants.DEFAULT_LEFT_OFFSET, metavar="left_offset", type=int, \
                    help=f"Prediction window's left offset, default value is {Constants.DEFAULT_LEFT_OFFSET}")
    parser.add_argument('--ro', required=False, default=Constants.DEFAULT_RIGHT_OFFSET, metavar="right_offset", type=int, \
                    help=f"Prediction window's right offset, default value is {Constants.DEFAULT_RIGHT_OFFSET}")
    
    # Loss function arguments
    parser.add_argument('--lf', required=False, default=Constants.DEFAULT_LOSS_TYPE, metavar="loss_type", type=str, choices=['mse', 'logcosh', 'huber', 'smape'], help=f"Loss function type, default value is {Constants.DEFAULT_LOSS_TYPE}")
    parser.add_argument('--kl', required=False, default=Constants.DEFAULT_KENDALL_MULTI_TASK, action="store_true", \
                    help=f"Wether the loss function should use the Kendall's multi task loss, default value is {Constants.DEFAULT_KENDALL_MULTI_TASK}")
    parser.add_argument('--ln', required=False, default=Constants.DEFAULT_LOSS_LOG_NORM, action="store_true", \
                    help=f"Wether the loss function should log normalize the values before the loss function, default value is {Constants.DEFAULT_LOSS_LOG_NORM}")
    parser.add_argument('--ea', required=False, default=Constants.DEFAULT_LOSS_EPOCH_ALPHA, metavar="loss_e_task_a", type=float, \
                    help=f"Loss function epoch task weighting alpha, default value is {Constants.DEFAULT_LOSS_EPOCH_ALPHA}. Set to 0 to prevent epoch task weighting.")
    parser.add_argument('--eb', required=False, default=Constants.DEFAULT_LOSS_EPOCH_BETA, metavar="loss_e_task_b", type=float, \
                    help=f"Loss function epoch task weighting beta, default value is {Constants.DEFAULT_LOSS_EPOCH_BETA}")
    
    # Model architecture related arguments
    parser.add_argument('-i', required=False, default=Constants.DEFAULT_SAMPLE_SIZE, metavar="instr_per_sample", type=int, \
                    help=f"Number of instructions per sample, default value is {Constants.DEFAULT_SAMPLE_SIZE}")
    parser.add_argument('-m', required=False, type=str, choices=['Transformer', 'BiLSTM', 'Convolution', 'FCC', 'LinearRegression', 'ConstantMean', 'SimpleRule', 'FromTrapez'],default='Transformer', metavar="model_type", \
                    help=f"Model's architecture type, either 'Transformer' (default), 'BiLSTM', 'Convolution', 'FCC', 'LinearRegression', 'constantMean', or 'FromTrapez'")
    parser.add_argument('-a', required=False, default=Constants.DEFAULT_NB_HEAD, metavar="nb_head", type=int, 
                    help=f"Transformer attention head amount, default value is {Constants.DEFAULT_NB_HEAD}")
    parser.add_argument('-n', required=False, default=Constants.DEFAULT_LAYER_COUNT, metavar="tlayer_count", type=int, 
                    help=f"Number of layer (Transformer or BiLSTM), default value is {Constants.DEFAULT_LAYER_COUNT}")
    parser.add_argument('-z', required=False, default=Constants.DEFAULT_EMBED_DIM, metavar="embed_dim", type=int, \
                    help=f"Embed dimension for Transformer, or hidden size for BiLSTM, default value is {Constants.DEFAULT_EMBED_DIM}")
    parser.add_argument('-f', required=False, default=Constants.DEFAULT_FEEDF_DIM, metavar="feed_forward_dim", type=int, \
                    help=f"Transformer feed forward dimension, default value is {Constants.DEFAULT_FEEDF_DIM}")
    parser.add_argument('-k', required=False, default=Constants.DEFAULT_KERNELS, metavar="kernel_lengths", nargs='+', type=int,\
                    help=f"Convolution model kernels' lengths (list), in order, output size will depend on these values. Default is {Constants.DEFAULT_KERNELS}, \
list length must match with 'convs_channels'")
    parser.add_argument('-c', required=False, default=Constants.DEFAULT_CHANNELS, metavar="convs_channels", nargs='+', type=int,\
                    help=f"Convolution model layers' channels (list), in order, last value of the list must be {UV.TRUE_LABEL_VECTOR_SIZE} (or the \
updated value, based on the dataset masked inputs/labels, and updated vectors (see VectorDataset.py)). Default is {Constants.DEFAULT_CHANNELS}, list \
length must match with 'kernel_lengths'")
    parser.add_argument('-s', required=False, default=Constants.DEFAULT_FCC_SUBWINDOW, metavar="sub_window_size", type=int, \
                    help=f"FCC model prediction sub-window size, default value is {Constants.DEFAULT_FCC_SUBWINDOW}")
    parser.add_argument('-p', required=False, default=Constants.DEFAULT_FCC_PREDICTED_INSTR, metavar="predicted_instr", type=int, \
                    help=f"FCC model predicted instruction among the sub-window, default value is {Constants.DEFAULT_FCC_PREDICTED_INSTR}th instr. \
Must be within [0; sub_window_size[")
    parser.add_argument('-l', required=False, default=Constants.DEFAULT_FCC_LAYERS, metavar="layer_sizes", nargs='+', type=int,\
                    help=f"FCC model layers' sizes (list), or the linear regression model hidden layer sizes, in order. For FCC model, last value of the list must be {UV.TRUE_LABEL_VECTOR_SIZE}. Also for the FCC model, first value must be equal to the number of FCC subwindow size times instr. input vector size. Default is {Constants.DEFAULT_FCC_LAYERS}")
    parser.add_argument('-w', required=False, default=None, metavar="pretrain_weights", type=str, \
                    help=f"Full path to a pre-trained model's weights (pytorch .pt file), if provided, all other architecture params will be ignored")
    
    # Data arguments, mandatory and positional
    parser.add_argument("train_source", type=str, \
                    help="Folder name where the files (DAT format) for training are stored")
    parser.add_argument("test_source", type=str, \
                    help="Folder name where the files (DAT format) for testing are stored, if 'None', testing will be ignored. If set to 'FromTrain', The testing dataset will be taken from the train dataset (separation at 80 percent).")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Script arguments parser, checks for the integrity of all given parameters.
    """
    args = parser.parse_args()
    log.info(f"Parsing arguments: {args}")
    if not (os.path.isdir(args.train_source) or os.path.isfile(args.train_source)):
        log.error("The specified path for data training files does not exist.", 1)
    log.info(f"Found training data folder: {args.train_source}")
    if args.test_source != "None" and args.test_source != "FromTrain" and not (os.path.isdir(args.test_source) or os.path.isfile(args.test_source)):
        log.error("The specified path for data test files does not exist.", 1)
    if args.test_source != "None" and args.test_source != "FromTrain":
        log.info(f"Found testing data folder: {args.test_source}")
    if args.bs < 1:
        log.error("Batch size should be superior or equal to 1.", 1)
    if args.ne < 1:
        log.error("The number of epoch should be superior or equal to 1.", 1)
    if args.sc < 1:
        log.error("The stop condition should be superior or equal to 1.", 1)
    if not (0 <= args.dp <= 1):
        log.error("The dropout rate must be between 0 and 1.", 1)
    if args.ft != 16 and args.ft != 32 and args.ft != 64:
        log.error("The pytorch float type must be either 32 or 64.", 1)
    else:
        if args.ft == 16:
            log.error("While pytorch does have a float16 type, it is not supported here because of a bug in pytorch's layer_norm.", 1)
        elif args.ft == 32:
            args.ft = torch.float32
        elif args.ft == 64:
            args.ft = torch.float64
    if (not (0 <= args.lo < args.ro <= args.i and args.ro - args.lo >= args.i - args.ro + args.lo)) and (not (args.m == "FCC" or args.m == "Convolution")):
        log.error(f"Wrong values of left and/or right offsets: {args.lo} and {args.ro}.They must follow these two conditions: \
'0 <= left_offset < right_offset <= instr_per_sample' and 'right_offset - left_offset >= instr_per_sample - right_offset + left_offset'.", 1)
        
    if not (0 <= args.ea <= 1):
        log.error(f"The given loss epoch task weight alpha is not within 0 and 1: {args.ea}", 1)
    if not (0 <= args.eb <= 1):
        log.error(f"The given loss epoch task weight beta is not within 0 and 1: {args.eb}", 1)

    if args.w != None and (not (os.path.isfile(args.w) and str(args.w).endswith(".pt"))):
        log.error("The specified model's weights does not exist.", 1)
    elif args.w != None:
        return args
    if args.i < 1:
        log.error("The number of instructions per sample should be superior or equal to 1.", 1)
    if args.a < 1:
        log.error("The number of attention head should be superior or equal to 1.", 1)
    if args.z < 1:
        log.error("The embed dimension, or hidden size for BiLSTM, should be superior or equal to 1.", 1)
    if args.n < 1:
        log.error("The number of layer should be superior or equal to 1.", 1)
    if args.f < 1:
        log.error("The feed forward dimension should be superior or equal to 1.", 1)
    if args.m == "Convolution" and len(args.k) != len(args.c):
        log.error("The lenghts of the lists of kernels lengths and number of channels must match.", 1)
    if args.m == "Convolution" and args.c[-1] != UV.TRUE_LABEL_VECTOR_SIZE:
        log.error(f"The last convolution layer's number of channels must be UV.TRUE_LABEL_VECTOR_SIZE.", 1)
    if args.m == "FCC" and not (0 < args.s < args.i // 2):
        log.error(f"The FCC model's sub-window size must be within [0; instr_per_sample / 2[.", 1)
    if args.m == "FCC" and not (0 <= args.p < args.s):
        log.error(f"The FCC model's predicted instr must be within [0; sub_windiw_size[.", 1)
    if args.m == "FCC" and not (len(args.l) > 1 and args.l[0] == (args.s * UV.TRUE_INPUT_VECTOR_SIZE) and args.l[-1] == UV.TRUE_LABEL_VECTOR_SIZE):
        log.alert(f"Given layers: {args.l}, first value should be {args.s * UV.TRUE_INPUT_VECTOR_SIZE} and last value should be {UV.TRUE_LABEL_VECTOR_SIZE}")
        log.error(f"The FCC model's layer sizes must follow these two conditions: \
first value must be equal to the sub-window size times the number of values per instruction input vector, and the last value must be equal to \
the instruction label vector size.", 1)
    if args.lne is not None and args.lne <= 0:
        log.error(f"The log-normalization epsilon value must be greater than 0.", 1)
    return args

@Utils.timeit
def trainModel(
        model: nn.Module,
        trainLoader: td.DataLoader,
        evalLoader: td.DataLoader,
        nb_epoch: int,
        stop_condition: int,
        modelfile: str,
        optimizer: torch.optim.Optimizer,
        lossFunction: Utils.LossFunction,
        lossfile: str,
        prev_epoch: int,
        is_special_startVel_loss: bool
    ) -> int:
    """
    Main training function.

    Parameters:
    - model:                the nn.Module model to train
    - trainLoader:          data loader for the training dataset
    - evalLoader:           data loader for the validation dataset
    - nb_epoch:             maximum number of training epoch
    - stop_condition:       number of epoch required to stop the training without improvement over the validation dataset
    - modelfile:            model save file name
    - optimizer:            a pytorch model training optimizer, like Adam or SGD
    - lossFunction          a pytorch module implementing the loss, must return a tuple of the total loss, and the individual losses of each tasks
    - lossfile:             file path to write the epoch loss
    - prev_epoch:           an int representing the number of epochs realised for the potential previous training

    Returns a tuple of two lists: mean mini-batch training losses (of each epoch), and mean mini-batch validation losses (of each epoch).
    """
    log.title("Start of training")
    log.notice(f"Optimizer used: {type(optimizer)}")
    
    # Training local variables initialization
    best_loss = float('inf')
    patience = stop_condition
    early_stopped = 0
    broke_out = False
    nb_batch_train = len(trainLoader)
    nb_batch_eval = len(evalLoader)
    process = psutil.Process()

    total_train_loss: torch.Tensor = torch.zeros((len(trainLoader), 1), device=model.device, dtype=model.floatType)
    total_val_loss: torch.Tensor = torch.zeros((len(evalLoader), 1), device=model.device, dtype=model.floatType)
    separate_train_loss: torch.Tensor = torch.zeros((len(trainLoader), lossFunction.num_tasks), device=model.device, dtype=model.floatType)
    separate_valid_loss: torch.Tensor = torch.zeros((len(evalLoader), lossFunction.num_tasks), device=model.device, dtype=model.floatType)

    # Temporary model save paths
    best_model_params_path = os.path.join("./tmp/", f"best_{modelfile}.pt")
    last_epoch_model_params_path = os.path.join("./tmp/last_epoch/", f"last_epoch_{modelfile}.pt")
    last_batch_model_params_path = os.path.join("./tmp/last_batch/", f"last_batch_{modelfile}.pt")

    # Epoch loop
    for epoch in range(nb_epoch):
        epoch += prev_epoch
        total_train_loss.zero_()
        total_val_loss.zero_()
        separate_train_loss.zero_()
        separate_valid_loss.zero_()
        
        # Train
        model.train()
        lossFunction.train()
        log.emptyLines(verbose_level=1)
        i = 0
        # Mini-batch loop
        for batch in trainLoader:
            data: torch.Tensor = batch[0]
            target: torch.Tensor = batch[1]
            optimizer.zero_grad()
            output = model(data)
            if not is_special_startVel_loss:
                loss: tuple[torch.Tensor, torch.Tensor] = lossFunction(output, target, epoch + 1)
            else:
                loss: tuple[torch.Tensor, torch.Tensor] = lossFunction(output, target, data)
            if isnan(loss[0].item()):
                log.eraseLine(1)
                log.warning(f"At epoch {epoch + 1}/{nb_epoch + prev_epoch}, batch {i}/{nb_batch_train}: reached NaN loss value.")
                log.warning(f"Separate loss values: {loss[1]}")
                log.warning(f"Do input or label contain NaN value: {'yes' if torch.any(data.isnan()).logical_or(torch.any(target.isnan())) else 'no'}")
                log.warning(f"Does prediction contain NaN value: {'yes' if torch.any(output.isnan()) else 'no'}")
                log.emptyLines(verbose_level=1)
                try:
                    checkpoint = torch.load(last_batch_model_params_path, weights_only=False)
                    model.load_state_dict(checkpoint["model_state"])
                    lossFunction.log_sigma.data.copy_(checkpoint["loss_state"])
                except BaseException as e:
                    log.eraseLine(1)
                    log.warning(f"Could not load last batch model's weights.", e)
                    log.emptyLines(verbose_level=1)
                broke_out = True
                break 
            loss[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_train_loss[i] = loss[0].detach()
            separate_train_loss[i, :] = loss[1].detach()
            i += 1
            if i % 100 == 0:
                cpu_mem_usage = process.memory_info().rss
                cpu_mem_max = psutil.virtual_memory().total
                if model.device.type == "cuda":
                    gpu_mem_usage = torch.cuda.memory_allocated()
                    gpu_mem_max = torch.cuda.memory_reserved()
                log.eraseLine(1)
                log.debug(f"Epoch {epoch + 1}, train batch [{i}/{nb_batch_train}]: train loss = {loss[0].item()}. CPU memory used: \
{Utils.format_memory_usage(cpu_mem_usage)}/{Utils.format_memory_usage(cpu_mem_max)} ({(cpu_mem_usage * 100 / cpu_mem_max):.2f} %){f', \
GPU memory used: {Utils.format_memory_usage(gpu_mem_usage)}/{Utils.format_memory_usage(gpu_mem_max)} ({(gpu_mem_usage * 100 / gpu_mem_max):.2f} %)' \
if model.device.type == "cuda" else ''}.")
            if i % 50000 == 0:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "loss_state": lossFunction.log_sigma.detach().cpu(),
                        "last_epoch": epoch + 1,
                        "loss_type": lossFunction.lossName,
                        "loss_log_norm": lossFunction.log_normalization,
                        "loss_kendall_mtask": lossFunction.multi_task,
                        "loss_epoch_alpha": lossFunction.alpha,
                        "loss_epoch_beta": lossFunction.beta,
                        "selected_inputs": UV.SELECTED_INPUTS,
                        "selected_labels": UV.SELECTED_LABELS,
                        "updating_functions": [f.__name__ for f in UV.UPDATING_FUNCTIONS],
                        "log_norm_epsilon": Constants.DEFAULT_LOG_NORM_EPSILON,
                        "optimizer": optimizer.state_dict()
                    },
                    last_batch_model_params_path
                    )
            del data, target, output, loss
        # Validation
        model.eval()
        lossFunction.eval()
        with torch.no_grad():
            i = 0
            # Mini-batch loop
            for batch in evalLoader:
                data: torch.Tensor = batch[0]
                target: torch.Tensor = batch[1]
                output: torch.Tensor = model(data)
                if not is_special_startVel_loss:
                    loss: tuple[torch.Tensor, torch.Tensor] = lossFunction(output, target, epoch + 1)
                else:
                    loss: tuple[torch.Tensor, torch.Tensor] = lossFunction(output, target, data)
                total_val_loss[i] = loss[0].detach()
                separate_valid_loss[i, :] = loss[1].detach()
                i += 1
                if i % 100 == 0:
                    cpu_mem_usage = process.memory_info().rss
                    cpu_mem_max = psutil.virtual_memory().total
                    if model.device.type == "cuda":
                        gpu_mem_usage = torch.cuda.memory_allocated()
                        gpu_mem_max = torch.cuda.memory_reserved()
                    log.eraseLine(1)
                    log.debug(f"Epoch {epoch + 1}, validation batch [{i}/{nb_batch_eval}]: validation loss = {loss[0].item()}. CPU memory used: \
{Utils.format_memory_usage(cpu_mem_usage)}/{Utils.format_memory_usage(cpu_mem_max)} ({(cpu_mem_usage * 100 / cpu_mem_max):.2f} %){f', \
GPU memory used: {Utils.format_memory_usage(gpu_mem_usage)}/{Utils.format_memory_usage(gpu_mem_max)} ({(gpu_mem_usage * 100 / gpu_mem_max):.2f} %)' \
if model.device.type == "cuda" else ''}.")
                del data, target, output, loss
        
        # Process the losses of this epoch
        train_loss = total_train_loss.mean().item()
        eval_loss = total_val_loss.mean().item()
        mean_train = separate_train_loss.mean(0)
        mean_valid = separate_valid_loss.mean(0)

        with open(lossfile, 'a') as f:
            line = f"{epoch + 1},{train_loss},{f','.join([f'{mean_train[j].item()}' for j in range(mean_train.shape[0])])}"
            line += f",{eval_loss},{f','.join([f'{mean_valid[j].item()}' for j in range(mean_valid.shape[0])])}\n"
            f.write(line)
        
        # Prints and model save if required
        cpu_mem_usage = process.memory_info().rss
        cpu_mem_max = psutil.virtual_memory().total
        if model.device.type == "cuda":
            gpu_mem_usage = torch.cuda.memory_allocated()
            gpu_mem_max = torch.cuda.memory_reserved()
        log.eraseLine(1)
        log.info(f"Epoch [{epoch+1}/{nb_epoch + prev_epoch}], Train loss: {train_loss}, Eval loss: {eval_loss}. CPU memory used: \
{Utils.format_memory_usage(cpu_mem_usage)}/{Utils.format_memory_usage(cpu_mem_max)} ({(cpu_mem_usage * 100 / cpu_mem_max):.2f} %){f', \
GPU memory used: {Utils.format_memory_usage(gpu_mem_usage)}/{Utils.format_memory_usage(gpu_mem_max)} ({(gpu_mem_usage * 100 / gpu_mem_max):.2f} %)' \
if model.device.type == "cuda" else ''}.")
        torch.save(
            {
                "model_state": model.state_dict(),
                "loss_state": lossFunction.log_sigma.detach().cpu(),
                "last_epoch": epoch + 1,
                "loss_type": lossFunction.lossName,
                "loss_log_norm": lossFunction.log_normalization,
                "loss_kendall_mtask": lossFunction.multi_task,
                "loss_epoch_alpha": lossFunction.alpha,
                "loss_epoch_beta": lossFunction.beta,
                "selected_inputs": UV.SELECTED_INPUTS,
                "selected_labels": UV.SELECTED_LABELS,
                "updating_functions": [f.__name__ for f in UV.UPDATING_FUNCTIONS],
                "log_norm_epsilon": Constants.DEFAULT_LOG_NORM_EPSILON,
                "optimizer": optimizer.state_dict()
            },
            last_epoch_model_params_path
            )
        if (eval_loss < best_loss): # If this epoch has the best resulsts on the validation dataset, save the weights
            best_loss = eval_loss
            patience = stop_condition
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "loss_state": lossFunction.log_sigma.detach().cpu(),
                    "last_epoch": epoch + 1,
                    "loss_type": lossFunction.lossName,
                    "loss_log_norm": lossFunction.log_normalization,
                    "loss_kendall_mtask": lossFunction.multi_task,
                    "loss_epoch_alpha": lossFunction.alpha,
                    "loss_epoch_beta": lossFunction.beta,
                    "selected_inputs": UV.SELECTED_INPUTS,
                    "selected_labels": UV.SELECTED_LABELS,
                    "updating_functions": [f.__name__ for f in UV.UPDATING_FUNCTIONS],
                    "log_norm_epsilon": Constants.DEFAULT_LOG_NORM_EPSILON,
                    "optimizer": optimizer.state_dict()
                },
                best_model_params_path
                )
        else:
            patience -= 1
            if patience == 0: # If the stop condition is met
                log.info(f"Early stopping at epoch: [{epoch+1}/{nb_epoch}]")
                early_stopped = epoch + 1
                break
        if broke_out:
            try:
                log.notice(f"Reloading the model's state of epoch {epoch + 1} (best validation results).")
                checkpoint = torch.load(best_model_params_path, weights_only=False)
                model.load_state_dict(checkpoint["model_state"])
                lossFunction.log_sigma.data.copy_(checkpoint["loss_state"])
            except BaseException as e:
                if not os.path.isfile(best_model_params_path):
                    log.warning(f"Training seems to have failed because no best model was found.")
                else:
                    log.warning(f"Failed to load the best model.", e)
            break
    
    # Training complete
    # Reloads the best model's weights
    try:
        if early_stopped != 0:
            log.notice(f"Reloading the model's state of epoch {early_stopped} (best validation results).")
            checkpoint = torch.load(best_model_params_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            lossFunction.log_sigma.data.copy_(checkpoint["loss_state"])
        log.notice(f"Model's weights backup saved at {best_model_params_path}")
    except BaseException as e:
        if not os.path.isfile(best_model_params_path):
            log.warning(f"Training seems to have failed because no best model was found.")
        else:
            log.warning(f"Failed to load the best model.", e)

    log.emptyLines(verbose_level=1)
    log.title("End of training\n")
    return epoch + 1

@Utils.timeit
def testModel(model: nn.Module, dataset: td.DataLoader, csv_file: str, lossFunction: Utils.LossFunction, is_special_startVel_loss: bool) -> tuple[Tensor, float, float]:
    """
    Test of the model on the test dataset.

    Parameters:
    - model:            the nn.Module variable of the model
    - dataset:          torch dataloader for the test dataset
    - csv_file:         path to the result CSV file
    - lossFunction:     a pytorch module implementing the loss, must return a tuple of the total loss and a tensor of each individual task loss

    Returns a tuple containing: mean test mini-batch loss, the predicted print time of the full test dataset, the real print time of the test dataset.
    """
    model.eval()
    lossFunction.eval()
    total_loss: torch.Tensor = torch.zeros(len(dataset), device=model.device, dtype=model.floatType)
    predict: torch.Tensor = torch.zeros(len(dataset), device=model.device, dtype=model.floatType)
    labels: torch.Tensor = torch.zeros(len(dataset), device=model.device, dtype=model.floatType)
    nb_batch = len(dataset)
    # No gradient processing, and opening the result file for writing
    with torch.no_grad(), open(csv_file, 'w+') as f:
        f.write(f"Batch prediction,Batch label,Batch loss,{','.join([f'Loss {i}' for i in range(1, model.label_vector_size + 1)])},Batch process time (s)\n")
        log.emptyLines(verbose_level=1)
        i = 0
        # Mini batch loop
        for i, batch in enumerate(dataset):
            data: torch.Tensor = batch[0]
            target: torch.Tensor = batch[1]
            tmp = time.time()
            output: torch.Tensor = model(data)
            process_time = time.time() - tmp
            if not is_special_startVel_loss:
                loss: tuple[torch.Tensor, torch.Tensor] = lossFunction(output, target, 1)
            else:
                loss: tuple[torch.Tensor, torch.Tensor] = lossFunction(output, target, data)
            if model.trained_on_log_target:
                output = torch.exp(output) - Constants.DEFAULT_LOG_NORM_EPSILON

            total_loss[i] = loss[0]
            predict[i] = output[..., 0].sum()
            labels[i] = target[..., 0].sum()

            f.write(
                f"{output[..., 0].sum().item():.6f},{target[..., 0].sum().item():.6f},{loss[0].item():.6f},{','.join([f'{l:.6f}' for l in loss[1]])},\
{process_time:.6f}\n"
                )
            i += 1
            if i % 100 == 0:
                log.eraseLine(1)
                log.debug(f"Model testing: batch [{i}/{nb_batch}]")
        log.eraseLine(1)
    return total_loss.mean().item(), predict.sum().item(), labels.sum().item()

def main() -> None:
    # Main variables initialization
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

    if args.rs is not None:
        Constants.RANDOM_SEED = args.rs
        log.info(f"Random seed used: {Constants.RANDOM_SEED}")
    if args.lne is not None:
        Constants.DEFAULT_LOG_NORM_EPSILON = args.lne
        log.info(f"Log-normalization epsilon used: {Constants.DEFAULT_LOG_NORM_EPSILON}")

    writesLog = (args.mf != "None")

    log = Logger(verbose_level=verbose_level, writesLog=writesLog, prints_thread_id=False)

    os.makedirs("./tmp", exist_ok=True)
    
    log.emptyLines()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_printoptions(sci_mode=True)
    
    log.info(f"Device used: {device}")

    cpu_generator = torch.Generator(device='cpu')
    cpu_generator.manual_seed(Constants.RANDOM_SEED)
    gpu_generator = torch.Generator(device=device)
    gpu_generator.manual_seed(Constants.RANDOM_SEED)

    lossFunction = None
    prev_epoch = None

    # Model architecture parsing
    if args.w == None:
        if args.m == "Transformer":
            model = TransformerModel(args.i, UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, args.lo,\
                    args.ro, device, embed_dim=args.z, nb_head=args.a, feedforward_dim=args.f,\
                    nlayers=args.n, dropout=args.dp, floatType=args.ft, log=log, generator=gpu_generator, trained_on_log_target=args.ln)
        elif args.m == "BiLSTM":
            model = BiLSTMModel(args.i, UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, args.lo,\
                    args.ro, device, embed_dim=args.z, nlayers=args.n, dropout=args.dp, floatType=args.ft, log=log, generator=gpu_generator, trained_on_log_target=args.ln)
        elif args.m == "Convolution":
            raise NotImplementedError("TODO")
            model = ConvolutionModel(args.i, UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, device,\
                        channels=args.c, kernels_size=args.k, dropout=args.d, floatType=args.p, log=log)
        elif args.m == "FCC":
            model = FullyConnectedModel(args.i, UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, device,\
                                        dropout=args.dp, floatType=args.ft, log=log, generator=gpu_generator,\
                                        subWindowSize=args.s, predictedInstr=args.p, layer_sizes=args.l)
        elif args.m == "LinearRegression":
            hidden_sizes = args.l if len(args.l) > 0 else []
            model = LinearRegressionModel(UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, hidden_sizes, device,\
                                          dropout=args.dp, floatType=args.ft, log=log, generator=gpu_generator, trained_on_log_target=args.ln)
        elif args.m == "ConstantMean":
            model = ConstantMeanModel(UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, device, floatType=args.ft, log=log)
        elif args.m == "SimpleRule":
            model = SimpleRuleModel(UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, device, floatType=args.ft, log=log)
        elif args.m == "FromTrapez":
            model = PredFromTrapez(UV.TRUE_INPUT_VECTOR_SIZE, UV.TRUE_LABEL_VECTOR_SIZE, args.i - 1, device, floatType=args.ft, log=log)
        optimizer = None
    else: # A model's weights were given as argument, parses the architecture based on the file name
        model, lossFunction, prev_epoch, optimizer = Utils.parseModelArchitecture(args.w, device, args.ft, args.dp, args.ne, log)
    filename = f"{model.getFilename()}_seed-{Constants.RANDOM_SEED}"

    needsSpecialLoss = all([v in UV.SELECTED_INPUTS for v in [6, 7]]) and UV.addDeltaCoordsAndAngle == UV.UPDATING_FUNCTIONS[-1] and UV.SELECTED_LABELS == [1]

    if lossFunction is None:
        if needsSpecialLoss:
            lossFunction = Utils.StartVelOnlyLoss(
                device=device,
                dtype=model.floatType,
                lambda_1=Constants.STARTVELLOSS_LAMBDA1,
                lambda_2=Constants.STARTVELLOSS_LAMBDA2,
                lambda_3=Constants.STARTVELLOSS_LAMBDA3,
                lambda_4=Constants.STARTVELLOSS_LAMBDA4,
                left_offset=0,
                right_offset=model.right_offset
            )
            if lossFunction.log_normalization != model.trained_on_log_target:
                log.error(f"The loss function and the model have a different target for the log normalization!", 1)
        else:
            lossFunction = Utils.LossFunction(
                model.label_vector_size,
                model.device,
                model.floatType,
                loss_function=args.lf,
                log_normalization=args.ln,
                kendall_multi_task=args.kl,
                alpha=args.ea,
                beta=args.eb
                )
    if prev_epoch is None:
        prev_epoch = 0
    if optimizer is None and not (args.m =="ConstantMean" or args.m == "SimpleRule" or args.m == "FromTrapez"):
        optimizer = torch.optim.Adam(list(model.parameters()) + list(lossFunction.parameters()), lr=args.lr)
    model.summary()
    log.info(f"Loss setup: function used = {lossFunction.lossName} ; does log normalization = {lossFunction.log_normalization} (with log norm epsilon = {Constants.DEFAULT_LOG_NORM_EPSILON}); uses Kendall's multi task weighting = {lossFunction.multi_task} ; epoch task weighting alpha = {lossFunction.alpha} ; epoch task weighting beta = {lossFunction.beta}")
    log.info(f"Selected input vector indexes: {UV.SELECTED_INPUTS} (if None, all are retained)")
    log.info(f"Selected label vector indexes: {UV.SELECTED_LABELS} (if None, all are retained)")
    funcs = ", ".join([f.__name__ for f in UV.UPDATING_FUNCTIONS])
    log.info(f"Used input vector updating functions: {funcs}")
    log.emptyLines()

    # Datasets loading
    log.info("Started building the dataset(s).")
    start_time = time.time()
    train_dataset_full = vd.VectorDataset(args.train_source, model.nbInstrPerSample, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, \
                                                        model.left_offset, model.right_offset, device, floatType=args.ft, log=log, \
                                                        random_seed=Constants.RANDOM_SEED, \
                                                        masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, \
                                                        append_input_vectors=UV.UPDATING_FUNCTIONS)
    train_dl_time = time.time() - start_time
    if args.test_source == "FromTrain":
        train_dataset_full, test_dataset = train_dataset_full.splitDataset(0.8)
    elif args.test_source != "None":
        start_time = time.time()
        test_dataset = vd.VectorDataset(args.test_source, model.nbInstrPerSample, Constants.DEFAULT_INPUT_SIZE, Constants.DEFAULT_LABEL_SIZE, \
                                                    model.left_offset, model.right_offset, device, floatType=args.ft, log=log, \
                                                    masked_inputs=UV.SELECTED_INPUTS, masked_labels=UV.SELECTED_LABELS, \
                                                    append_input_vectors=UV.UPDATING_FUNCTIONS)
        test_dl_time = time.time() - start_time
    trainDataset, evalDataset = train_dataset_full.splitDataset(0.8)
    log.info(f"Done. Took {Utils.format_time(train_dl_time)} for the training dataset\
{f' and {Utils.format_time(test_dl_time)} for the test dataset.' if args.test_source != 'None' and args.test_source != 'FromTrain' else '.'}")
    log.emptyLines()
    log.info(f"Training dataset summary.")
    trainDataset.summary()
    log.info(f"Validation dataset summary.")
    evalDataset.summary()
    if args.test_source != "None":
        log.info(f"Test dataset summary.")
        test_dataset.summary()
    train_dl = td.DataLoader(trainDataset, num_workers=0, batch_sampler=Utils.BatchSampler(len(trainDataset), args.bs, Constants.RANDOM_SEED))
    eval_dl = td.DataLoader(evalDataset, batch_size=args.bs, shuffle=False, generator=cpu_generator)
    if args.test_source != "None":
        test_dl = td.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, generator=cpu_generator)
    log.notice(f"Train dataset length: {len(trainDataset)} ; eval dataset length: {len(evalDataset)}\
{f' ; test dataset length: {len(test_dataset)}' if args.test_source != 'None' else ''}")
    log.emptyLines()

    loss_folder = "./loss"
    try:
        os.makedirs(loss_folder)
    except FileExistsError:
        pass
    except (OSError, PermissionError):
        log.warning("Failed to create the folder for the loss.csv file, it will be saved in the current directory.")
        loss_folder = "."
    loss_name = Utils.get_next_versioned_filename(f"{loss_folder}/loss_{filename}.csv")

    os.makedirs("./tmp/last_epoch/", exist_ok=True)
    os.makedirs("./tmp/last_batch/", exist_ok=True)
    
    # Training
    if not (args.m =="ConstantMean" or args.m == "SimpleRule" or args.m == "FromTrapez"):
        with open(loss_name, 'w+') as f:
            if not needsSpecialLoss:
                f.write(f"Epoch,Train loss,{','.join([f'Train value {i + 1} loss' for i in range(model.label_vector_size)])}")
                f.write(f",Eval loss,{','.join([f'Eval value {i + 1} loss' for i in range(model.label_vector_size)])}\n")
            else:
                f.write(f"Epoch,Train loss,{','.join([f'Train value {i + 1} loss' for i in range(Constants.DEFAULT_LABEL_SIZE)])}")
                f.write(f",Eval loss,{','.join([f'Eval value {i + 1} loss' for i in range(Constants.DEFAULT_LABEL_SIZE)])}\n")
        last_epoch, train_time = trainModel(
            model=model,
            trainLoader=train_dl,
            evalLoader=eval_dl,
            nb_epoch=args.ne,
            stop_condition=args.sc,
            modelfile=filename,
            optimizer=optimizer,
            lossFunction=lossFunction,
            lossfile=loss_name,
            prev_epoch=prev_epoch,
            is_special_startVel_loss=needsSpecialLoss
        )
        log.info(f"Training completed, took {Utils.format_time(train_time)}.")
    else:
        last_epoch = 1

    # Testing
    if args.test_source != "None":
        log.info(f"Started predicting on the test dataset with the trained model ...")
        result_csv_file = f"./tmp/test_results_{filename}.csv"
        (test_loss, predicted, actual), test_time = testModel(model, test_dl, result_csv_file, lossFunction, needsSpecialLoss)
        log.info(f"Done, testing took {Utils.format_time(test_time)}, test loss : {test_loss}.\n\tPredicted a print time of {Utils.format_time(predicted)} \
on the full test dataset, while the actual print time is {Utils.format_time(actual)} \
(on {len(test_dataset) * (test_dataset.right_offset - test_dataset.left_offset)} G-code instructions).")
    else:
        log.info(f"No test dataset folder was provided, skipped model testing.")
    
    # Train and test done, save results if asked (loss per epoch, test results, model weights)
    if args.mf != "None":
        test_name = Utils.get_next_versioned_filename(f"./results/batch_tests/test_results_{filename}.csv")
        try:
            os.makedirs("./results/batch_tests", exist_ok=True)
        except FileExistsError:
            pass # Should never happen
        except (OSError, PermissionError) as e:
            log.warning("Failed to create the test results folder, CSV file will be saved in the current directory.", e)
            test_name = Utils.get_next_versioned_filename(f"./test_results_{filename}.csv")
        if args.test_source != "None":
            os.replace(result_csv_file, test_name)

        try:
            os.makedirs(args.mf)
        except FileExistsError:
            pass
        except (OSError, PermissionError) as e:
            log.warning("Failed to created the save folder, model will be saved in the current directory.", e)
            args.mf = "./"
        finally:
            model_path = Utils.get_next_versioned_filename(os.path.join(args.mf, filename + ".pt"))
        try:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "loss_state": lossFunction.log_sigma.detach().cpu(),
                    "last_epoch": last_epoch,
                    "loss_type": lossFunction.lossName,
                    "loss_log_norm": lossFunction.log_normalization,
                    "loss_kendall_mtask": lossFunction.multi_task,
                    "loss_epoch_alpha": lossFunction.alpha,
                    "loss_epoch_beta": lossFunction.beta,
                    "selected_inputs": UV.SELECTED_INPUTS,
                    "selected_labels": UV.SELECTED_LABELS,
                    "updating_functions": [f.__name__ for f in UV.UPDATING_FUNCTIONS],
                    "log_norm_epsilon": Constants.DEFAULT_LOG_NORM_EPSILON,
                    "optimizer": optimizer.state_dict() if optimizer is not None else None
                },
                model_path
                )
            log.info(f"Model was successfully saved in the {args.mf} directory.")
        except BaseException as e:
            log.warning("Failed to save the model.", e)
    else:
        try:
            os.remove(result_csv_file)
        except:
            pass
        try:
            os.remove(loss_name)
        except:
            pass

    log.emptyLines()
    # End of train, test, and save.

if __name__ == "__main__":
    main()
    
"""
End of file.
"""