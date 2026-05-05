"""
Some utility functions for all kind of scripts in this repository.
"""

import os
import random
import subprocess
import time
from typing import Any, Callable, Generator, List, Iterator
import zipfile
import traceback
import numpy as np
import struct
import torch
from torch import nn
from torch.utils.data import Sampler
from math import ceil, log2, exp
import Constants
import REPET.UpdateVectors as UV
from REPET.TransformerModel import *
from REPET.BiLSTMModel import *
from REPET.ConvolutionModel import *
from REPET.FullyConnectedModel import *
from REPET.smallerModels.LinearRegression import *
from REPET.smallerModels.constantMeanPrediction import *
from REPET.smallerModels.SimpleRulePrediction import *
from REPET.smallerModels.FixedPredFromTrapez import *

################################################################################################################
# Classes

class BatchSampler(Sampler[List[int]]):

    def __init__(self, dataset_len: int, batch_size: int, seed: int) -> None:
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.indexes = list(range(0, dataset_len, batch_size))
        random.Random(seed).shuffle(self.indexes)

    def __iter__(self) -> Iterator[List[int]]:
        for start in self.indexes:
            yield list(range(start, min(start + self.batch_size, self.dataset_len)))

    def __len__(self) -> int:
        return len(self.indexes)

class SMAPELoss(nn.Module):

	def __init__(self, reduction: str = "mean"):
		super(SMAPELoss, self).__init__()
		self.reduction = reduction
	
	def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		num = torch.abs(prediction - target)
		denom = (torch.abs(prediction) + torch.abs(target))
		smape = num / (denom + 1e-8)
		if self.reduction == "mean":
			return smape.mean()
		elif self.reduction == "sum":
			return smape.sum()
		else:
			return smape

class LogCosHLoss(nn.Module):

	def __init__(self, reduction: str = "mean"):
		super(LogCosHLoss, self).__init__()
		self.reduction = reduction

	def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		x = prediction - target
		logCosH = x + torch.nn.functional.softplus(-2 * x) - math.log(2)
		# cosh(x) = (exp(x) + exp(-x)) / 2 => log(cosh(x)) = log(exp(x) * (1 + exp(-2x)) / 2) = x + log(1 + exp(-2x)) - log(2)
		# And log(1 + exp(-2x)) is equivalent to torch.nn.funcitonal.softplus
		# Only difference is that past a threshold (in this case, -2x > 20), the softplus function switches to the linear function
		# for numerical stability

		if self.reduction == "mean":
			return logCosH.mean()
		elif self.reduction == "sum":
			return logCosH.sum()
		else:
			return logCosH
		
class StartVelOnlyLoss(nn.Module):
	"""
	Secondary loss function, specifically designed for models trained to predict only the instruction start velocity, to which a trapeze simple rule prediction head is attached.
	"""

	def __init__(
			self,
			device: torch.device,
			dtype: torch.dtype,
			lambda_1: float,
			lambda_2: float,
			lambda_3: float,
			lambda_4: float,
			left_offset: int,
			right_offset: int
		):
		super().__init__()
		self.mse = torch.nn.MSELoss(reduction="none")
		# The name log_sigma is just to match the main LossFunction name. For this class, it is actually the 4 fixed coefficients for the full loss
		self.log_sigma = torch.tensor([lambda_1, lambda_2, lambda_3, lambda_4], dtype=dtype, device=device)
		self.lossName = "justStartVelMSE"
		self.log_normalization = False
		self.multi_task = False
		self.num_tasks = 4
		self.alpha = 0.0
		self.beta = 0.0

		self.left_offset = left_offset
		self.right_offset = right_offset

		# Printer configuration values
		self.def_accel = Constants.DEFAULT_XY_ACCEL
		self.def_decel = Constants.DEFAULT_XY_DECEL
		self.z_accel = Constants.DEFAULT_Z_ACCEL
		self.z_decel = Constants.DEFAULT_Z_DECEL
		if Constants.DEFAULT_E_ONLY_ACCEL is None:
			# If no E only acceleration is provided, process it based on the nozzle and filament diameters
			# This formula was taken from the Klipper source code
			self.e_accel = self.def_accel * ( (4 * Constants.DEFAULT_NOZZLE_DIAMETER**2) / (pi * (0.5 * Constants.DEFAULT_FILAMENT_DIAMETER)**2) )
			self.e_decel = self.e_accel
		else:
			self.e_accel = Constants.DEFAULT_E_ONLY_ACCEL
			self.e_decel = Constants.DEFAULT_E_ONLY_ACCEL

		self.target_speed_index = UV.SELECTED_INPUTS.index(6) if UV.SELECTED_INPUTS is not None else 6
		self.extr_len_index = UV.SELECTED_INPUTS.index(7) if UV.SELECTED_INPUTS is not None else 7
		if UV.addDeltaCoordsAndAngle == UV.UPDATING_FUNCTIONS[-1]:
			self.delta_x_index = -5
			self.delta_y_index = -4
			self.delta_z_index = -3
			self.seg_len_index = -2
		else:
			raise WrongVectorUpdates(f"For a model to have a simple rule prediction head, the vector updating function `addDeltaCoordsAndAngle` must be last in the `UPDATING_FUNCTIONS` list in `UpdateVectors.py` script.")

	def forward(self, output: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		output = output.squeeze()
		target = target.squeeze()
		cond1 = self.log_sigma[0] * torch.relu(-output)
		cond2 = self.log_sigma[1] * torch.relu(output - inputs[: , self.left_offset:self.right_offset, self.target_speed_index] / 60)

		accel = torch.full((inputs.shape[0], inputs.shape[1]), self.def_accel, device=inputs.device, dtype=inputs.dtype)
		decel = torch.full((inputs.shape[0], inputs.shape[1]), self.def_decel, device=inputs.device, dtype=inputs.dtype)

		justE = (inputs[..., self.seg_len_index] < 1e-12).logical_and(inputs[..., self.delta_x_index] < 1e-12).logical_and(inputs[..., self.delta_y_index] < 1e-12).logical_and(inputs[..., self.delta_z_index] < 1e-12)
		b_idx, t_idx = justE.nonzero(as_tuple=True)
		inputs[b_idx, t_idx, self.extr_len_index] = inputs[b_idx, t_idx, self.extr_len_index]
		accel[b_idx, t_idx] = self.e_accel
		decel[b_idx, t_idx] = self.e_decel

		contains_z_movement = (inputs[..., self.delta_z_index] > 1e-7)
		b_idx, t_idx = contains_z_movement.nonzero(as_tuple=True)
		accel[b_idx, t_idx] = self.z_accel
		decel[b_idx, t_idx] = self.z_decel

		vout = torch.cat((torch.zeros((inputs.shape[0], 1), device=inputs.device, dtype=inputs.dtype), output[:, 1:]), dim=-1)

		cond3 = self.log_sigma[2] * torch.relu(
			vout - output - accel[:, self.left_offset:self.right_offset] * inputs[: , self.left_offset:self.right_offset, self.seg_len_index]
		)

		cond4 = self.log_sigma[3] * torch.relu(
			output - vout - decel[:, self.left_offset:self.right_offset] * inputs[: , self.left_offset:self.right_offset, self.seg_len_index]
		)

		additionnalResults = torch.tensor([cond1.sum().item(), cond2.sum().item(), cond3.sum().item(), cond4.sum().item()], device=output.device, dtype=output.dtype)

		return (torch.mean(self.mse(output, target) + cond1 + cond2 + cond3 + cond4), additionnalResults)
	
	def summary(self) -> str:
		return f"Custom made loss function specifically for models trained to predict only the start velocity of an instruction. Uses the following coefficients for the four loss conditions: {', '.join([str(v) for v in self.log_sigma.tolist()])}."

class LossFunction(nn.Module):
	"""
	Main loss function.

	Can use several loss function types, such as mse or logcosh

	Multi-task loss function with learnable uncertainty parameters.

	Includes an implementation of the paper:

	Kendall A., Gal Y. & Cipolla R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.
	In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491).
	"""

	def __init__(
			self,
			num_tasks: int,
			device: torch.device,
			dtype: torch.dtype,
			loss_function: str = Constants.DEFAULT_LOSS_TYPE,
			log_normalization: bool = Constants.DEFAULT_LOSS_LOG_NORM,
			kendall_multi_task: bool = Constants.DEFAULT_KENDALL_MULTI_TASK,
			alpha: float = 0.5,
			beta: float = 0.13
			) -> None:
		"""
		Loss constructor.

		Parameters:
		- num_tasks:			number of predicted values per instruction
		- device:				pytorch device descriptor
		- dtype:				pytorch float type
		- loss_function:		string describing which individual loss to use. Must be one of: "huber", "mse", "smape", or "logcosh"
		- log_normalization:	boolean, indicates wether the model should be trained to predict log(1 + target) instead of target, also does influence the inference processing, as exp(output) - 1 instead of output
		- kendall_multi_task:	boolean, indicates wether the loss should include Kendall's multi-task loss system, with learnable `sigma` weights
		- alpha:				float, controls max deviation from 1 for task importance factors
		- beta:					float, controls the transition speed for task importance factors
		"""
		super(LossFunction, self).__init__()
		self.loss_function = loss_function
		if loss_function == "huber":
			self.loss = nn.HuberLoss(delta=1.0e-03)
			self.lossName = "huber"
		elif loss_function == "mse":
			self.loss = torch.nn.MSELoss()
			self.lossName = "mse"
		elif loss_function == "smape":
			self.loss = SMAPELoss()
			self.lossName = "smape"
		elif loss_function == "logcosh":
			self.loss = LogCosHLoss()
			self.lossName = "logcosh"
		else:
			raise TypeError(f"Unknown loss function: {loss_function}. Options are: 'huber', 'mse', 'smape', or 'logcosh'.")
		self.multi_task = kendall_multi_task
		self.log_normalization = log_normalization
		self.num_tasks = num_tasks
		self.alpha = alpha
		self.beta = beta
		# This is actually log(sigma^2), but I made a typo when creating this function and some trained models are already savec with this name instead :)
		self.log_sigma = nn.Parameter(torch.zeros(num_tasks, device=device, dtype=dtype))
		self.epsilon = torch.scalar_tensor(Constants.DEFAULT_LOG_NORM_EPSILON, dtype=dtype, device=device)

	def forward(self, output: torch.Tensor, target: torch.Tensor, epoch_id: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Loss call function.

		Parameters:
		- output:		the predicted tensor, of shape `[..., labelVectorSize]`
		- target:		the label tensor, of shape `[..., labelVectorSize]`
		- epoch_id:		the current epoch's number

		Returns a tuple of two values:
		- the total loss, as a pytorch scalar
		- the individual losses for all tasks, as a pytorch tensor
		"""
		assert(output.shape[-1] == self.num_tasks)
		assert(output.shape == target.shape)

		result = torch.zeros(output.shape[-1], dtype=output.dtype, device=output.device)
		individual_raw = torch.zeros(output.shape[-1], dtype=output.dtype, device=output.device)
		total_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)

		if self.log_normalization:
			normalized_target = torch.log(self.epsilon + target)
		else:
			normalized_target = target

		for i in range(self.num_tasks):

			res: torch.Tensor = self.loss(output[..., i], normalized_target[..., i])
			individual_raw[i] = res
			if self.multi_task:
				# Kendall's multi-task loss
				precision = torch.exp(-self.log_sigma[i])
				weighted_loss = 0.5 * precision * res + 0.5 * self.log_sigma[i]
			else:
				weighted_loss = res
			result[i] = weighted_loss			
			
		task_weights = [1 - self.alpha * (1 - exp(-self.beta * epoch_id))] * self.num_tasks
		task_weights[0] = 1 + self.alpha * (1 - exp(-self.beta * epoch_id))
		total_loss = sum(w * l for w, l in zip(task_weights, result))

		return total_loss, individual_raw
	
	def summary(self) -> str:
		return f"Loss function type: {self.lossName} (on {self.num_tasks} tasks)\nUses log normalization: {self.log_normalization}\nUses Kendall's multi task weight normalization: {self.multi_task}\nEpoch task weighting values: alpha = {self.alpha} and beta = {self.beta}"

class CustomError(Exception):
	def __init__(self, *args) -> None:
		super().__init__(*args)
		self.trace = traceback.extract_stack()
	
	def __str__(self):
		trace = ""
		for l in self.trace.format()[:-1]:
			trace += l
		return f"{trace}\n{super().__str__()}"

class TensorShapeError(CustomError):
	"""
	The specified tensor's shape is wrong.
	"""
	pass

class UnknownPrinter(CustomError):
	"""
	Unknown printer name
	"""
	pass

class UnknownConfigDefinitionFormat(CustomError):
	"""
	Unknown configuration definition format for slicing .ini file
	"""
	pass

class FilenameParsingError(CustomError):
	"""
	Could not parse the given file name.
	"""
	pass

class WrongVectorUpdates(CustomError):
	"""
	The given model was trained using a different set of vector updates specified in UpdateVectors.py
	"""
	pass

class NanInTensorDetected(CustomError):
	"""
	Detected at least one NaN value in a tensor.
	"""
	pass

################################################################################################################
# Functions

def timeit(func: Callable) -> (float | tuple[Any, float]):
	"""
	Time measurement execution decorator, returns a tuple of both the initial result of the function, and the processing time in seconds, or just the time
	if the function returned nothing.

	NOTE: this decorator changes the returning format of the original function:

		- if a function returns a single value by default, it will now return a tuple: `tuple( defaul_return_value, time )`

		- if the function returns a tuple, the returned values becomes `tuple( tuple(default_tuple_val_1, ..., default_tuple_val_n), time )`

		- if the function returns nothing, the return value becomes `time`
	"""
	def wrapper(*args: tuple[Any], **kwargs: dict[str, Any]) -> (float | tuple[Any, float]):
		start_time = time.time()
		res = func(*args, **kwargs)
		elapsed_time = time.time() - start_time
		if res == None:
			return elapsed_time
		else:
			return res, elapsed_time
	return wrapper

def format_time(elapsed_time: int | float) -> str:
	"""
	Time measure formatting into string.

	- elapsed_time:		the value of the time (in seconds) to format, as int or float.

	Returns a string representation of that time value.
	"""
	if not isinstance(elapsed_time, (float, int)):
		raise TypeError(f"The parameter 'elapsed_time' should be an int or a float")
	if math.isnan(elapsed_time):
		return f"NaN"
	if math.isinf(elapsed_time):
		return f"{'+' if np.sign(elapsed_time) == 1 else '-'}infinity"
	if elapsed_time < 0:
		negative = "-"
	else:
		negative = ""
	if elapsed_time < 1.0:
		if elapsed_time >= 1e-3:
			return f"{negative}{elapsed_time * 1e3:.3f} ms"
		elif elapsed_time >= 1e-6:
			return f"{negative}{elapsed_time * 1e6:.3f} µs"
		else:
			return f"{negative}{elapsed_time * 1e9:.3f} ns"
	elif elapsed_time < 60.0:
		return f"{negative}{elapsed_time:.3f} s"
	elif elapsed_time < 86400.0:
		if elapsed_time < 3600:
			return f"{negative} {int(elapsed_time / 60)} minutes {int(elapsed_time % 60)} seconds"
		else:
			return f"{negative} {int(elapsed_time / 3600)} hours {int((elapsed_time % 3600) / 60)} minutes"
	else:
		return f"{negative} {int(elapsed_time / 86400)} days, {int((elapsed_time % 86400) / 3600)} hours and {int(((elapsed_time % 86400) % 3600) / 60)} minutes"
		
def format_memory_usage(value: int | float) -> str:
	"""
	Memory usage metric formatting into string.

	- value:		the value of the memory usage to format, as int or float.

	Returns a string representation of that memory usage.
	"""
	if value < 1_000:
		return f"{value} bytes"
	elif value < 1_000_000:
		return f"{value / 1_000:.2f} kB"
	elif value < 1_000_000_000:
		return f"{value / 1_000_000:.2f} MB"
	else:
		return f"{value / 1_000_000_000:.2f} GB"

@DeprecationWarning
def unzip_files(func):
	"""
	Decorator, iterates through the parameters of the provided function. For each file path parameter that corresponds to a ZIP file, it extracts the contents
	of the ZIP file. Then, it invokes the original function with the first file extracted from each ZIP file as parameters (other parameters are unchanged).

	Once the original function execution is complete, the extracted files are deleted from the filesystem.
	"""
	def extract_first_file(zip_file: zipfile.ZipFile, output_dir: str) -> str:
		file = zip_file.namelist()[0]
		zip_file.extract(file, output_dir)
		return os.path.join(output_dir, os.path.basename(file))
	def wrapper(*args, **kwargs):
		nargs = []
		nkwargs = {}
		deleted_files = []
		for arg in args:
			if isinstance(arg, str) and os.path.isfile(arg) and arg.endswith(".zip"):
				with zipfile.ZipFile(arg, 'r') as zip_file:
					base_dir = os.path.dirname(arg)
					new_file = extract_first_file(zip_file, base_dir)
					deleted_files.append(new_file)
					nargs.append(new_file)
			else:
				nargs.append(arg)
		for key, value in kwargs.items():
			if isinstance(value, str) and os.path.isfile(value) and value.endswith(".zip"):
				with zipfile.ZipFile(value, 'r') as zip_file:
					base_dir = os.path.dirname(value)
					new_file = extract_first_file(zip_file, base_dir)
					deleted_files.append(new_file)
					nkwargs[key] = new_file
			else:
				nkwargs[key] = value
		try:
			res = func (*nargs, **nkwargs)
		except:
			for file in deleted_files:
				os.remove(file)
			raise
		else:
			for file in deleted_files:
				os.remove(file)
			return res
	return wrapper

def file_size(filename: str) -> int:
	"""
	Returns the number of bytes of the file path provided as argument.
	 	
	Raises IOError if failed.

	Parameter:
	- filename:		path to the file to analyse
	"""
	result = subprocess.run(['wc', '-c', filename], check=True, capture_output=True, text=True)
	if result.returncode != 0:
		raise IOError(result.stderr.strip())
	return int(result.stdout.strip().split()[0])

def __getDictValue(dict: dict[str, Any], key: str, defaultDict: dict[str, Any]) -> Any | None:
	try:
		return dict[key]
	except KeyError:
		return defaultDict[key]

def parseModelArchitecture(
			model_file: str,
			device: torch.device,
			floatType: torch.dtype,
			dropout: float,
			nb_epoch: int,
			log: Logger,
			is_for_test: bool = False
		) -> tuple[nn.Module, LossFunction, int]:
	"""
	Parses a torch module from its saved dict file.

	Parameters:
	- model_file:		file path of the model to parse

	Returns a torch.nn.Module
	"""
	checkpoint: dict[str, dict[str, Any]] = torch.load(model_file, weights_only=False, map_location=device)
	try:
		has_simple_rule_pred_head =  checkpoint["selected_labels"] == [1] and all([v in checkpoint["selected_inputs"] for v in [6, 7]]) and UV.addDeltaCoordsAndAngle.__name__ in checkpoint["updating_functions"]
		if checkpoint["model_state"]["modelType"] == "Transformer":
			model = TransformerModel(
						__getDictValue(checkpoint["model_state"], "instrPerSample", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "inputVectorSize", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "labelVectorSize", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "leftOffset", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "rightOffset", Constants.TRANSFORMER_DICT),
						device,
						embed_dim=__getDictValue(checkpoint["model_state"], "embedDim", Constants.TRANSFORMER_DICT),
						nb_head=__getDictValue(checkpoint["model_state"], "headCount", Constants.TRANSFORMER_DICT),
						feedforward_dim=__getDictValue(checkpoint["model_state"], "fForwardDim", Constants.TRANSFORMER_DICT),
						nlayers=__getDictValue(checkpoint["model_state"], "nbLayer", Constants.TRANSFORMER_DICT),
						dropout=dropout,
						floatType=floatType,
						log=log,
						generator=None,
						init_weights=False,
						trained_on_log_target=__getDictValue(checkpoint["model_state"], "trainedOnLogTarget", Constants.DEFAULT_LOSS_LOG_NORM),
						simple_rule_head=(has_simple_rule_pred_head and is_for_test)
						)
		elif checkpoint["model_state"]["modelType"] == "BiLSTM":
			model = BiLSTMModel(
						__getDictValue(checkpoint["model_state"], "instrPerSample", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "inputVectorSize", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "labelVectorSize", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "leftOffset", Constants.TRANSFORMER_DICT),
						__getDictValue(checkpoint["model_state"], "rightOffset", Constants.TRANSFORMER_DICT),
						device,
						embed_dim=__getDictValue(checkpoint["model_state"], "embedDim", Constants.TRANSFORMER_DICT),
						nlayers=__getDictValue(checkpoint["model_state"], "nbLayer", Constants.TRANSFORMER_DICT),
						dropout=dropout,
						floatType=floatType,
						log=log,
						generator=None,
						init_weights=False,
						trained_on_log_target=__getDictValue(checkpoint["model_state"], "trainedOnLogTarget", Constants.DEFAULT_LOSS_LOG_NORM),
						simple_rule_head=(has_simple_rule_pred_head and is_for_test)
						)
		elif checkpoint["model_state"]["modelType"] == "Convolution":
			raise NotImplementedError("TODO")
		elif checkpoint["model_state"]["modelType"] == "FullyConnected":
			model = FullyConnectedModel(
						__getDictValue(checkpoint["model_state"], "instrPerSample", Constants.FCC_DICT),
						__getDictValue(checkpoint["model_state"], "inputVectorSize", Constants.FCC_DICT),
						__getDictValue(checkpoint["model_state"], "labelVectorSize", Constants.FCC_DICT),
						device,
						subWindowSize=__getDictValue(checkpoint["model_state"], "subWindowSize", Constants.FCC_DICT),
						predictedInstr=__getDictValue(checkpoint["model_state"], "predictedInstr", Constants.FCC_DICT),
						layer_sizes=__getDictValue(checkpoint["model_state"], "layerSizes", Constants.FCC_DICT),
						dropout=dropout,
						floatType=floatType,
						log=log,
						generator=None,
						init_weights=False,
						trained_on_log_target=__getDictValue(checkpoint["model_state"], "trainedOnLogTarget", Constants.DEFAULT_LOSS_LOG_NORM)
						)
		elif checkpoint["model_state"]["modelType"] == "LinearRegression":
			model = LinearRegressionModel(
						__getDictValue(checkpoint["model_state"], "inputVectorSize", Constants.LINEAR_DICT),
						__getDictValue(checkpoint["model_state"], "labelVectorSize", Constants.LINEAR_DICT),
						__getDictValue(checkpoint["model_state"], "hiddenSizes", Constants.LINEAR_DICT),
						device,
						dropout=dropout,
						floatType=floatType,
						log=log,
						generator=None,
						init_weights=False,
						trained_on_log_target=__getDictValue(checkpoint["model_state"], "trainedOnLogTarget", Constants.DEFAULT_LOSS_LOG_NORM),
						simple_rule_head=(has_simple_rule_pred_head and is_for_test)
						)
		elif checkpoint["model_state"]["modelType"] == "ConstantMean":
			model = ConstantMeanModel(
						__getDictValue(checkpoint["model_state"], "inputVectorSize", Constants.LINEAR_DICT),
						__getDictValue(checkpoint["model_state"], "labelVectorSize", Constants.LINEAR_DICT),
						device,
						floatType=floatType,
						log=log,
						simple_rule_head=(has_simple_rule_pred_head and is_for_test)
						)
		elif checkpoint["model_state"]["modelType"] == "SimpleRule":
			model = SimpleRuleModel(
						__getDictValue(checkpoint["model_state"], "inputVectorSize", Constants.LINEAR_DICT),
						__getDictValue(checkpoint["model_state"], "labelVectorSize", Constants.LINEAR_DICT),
						device,
						floatType=floatType,
						log=log
			)
		elif checkpoint["model_state"]["modelType"] == "FromTrapez":
			model = PredFromTrapez(
						__getDictValue(checkpoint["model_state"], "inputVectorSize", Constants.LINEAR_DICT),
						__getDictValue(checkpoint["model_state"], "labelVectorSize", Constants.LINEAR_DICT),
						__getDictValue(checkpoint["model_state"], "instrPerSample", Constants.LINEAR_DICT),
						device,
						floatType=floatType,
						log=log
			)
		else:
			raise ValueError(f"Unknown model type: {checkpoint["model_state"]["modelType"]}")
	except KeyError as e:
		e.add_note(f"Missing key in model state dict, is the saved file from a previous version of the code?")
		raise e
	if has_simple_rule_pred_head and (model.left_offset != 0 and model.right_offset != 99):
		raise ValueError(f"A model trained only to predict the start velocity must have a left offset of 0 and a right offset of 99")
	model.load_state_dict(checkpoint["model_state"])
	if has_simple_rule_pred_head:
		UV.SELECTED_LABELS = None
		UV.TRUE_LABEL_VECTOR_SIZE = Constants.DEFAULT_LABEL_SIZE
		if is_for_test:
			lossFunction = LossFunction(
				UV.TRUE_LABEL_VECTOR_SIZE,
				device=device,
				dtype=floatType,
				loss_function=Constants.DEFAULT_LOSS_TYPE,
				log_normalization=Constants.DEFAULT_LOSS_LOG_NORM,
				kendall_multi_task=False,
				alpha=0.0,
				beta=0.0
				)
		else:
			lossFunction = StartVelOnlyLoss(device=device, dtype=floatType, lambda_1=Constants.STARTVELLOSS_LAMBDA1, lambda_2=Constants.STARTVELLOSS_LAMBDA2, lambda_3=Constants.STARTVELLOSS_LAMBDA3, lambda_4=Constants.STARTVELLOSS_LAMBDA4, left_offset=0, right_offset=99)
	else:
		try:
			lossFunction = LossFunction(
					UV.TRUE_LABEL_VECTOR_SIZE,
					device=device,
					dtype=floatType,
					loss_function=checkpoint["loss_type"],
					log_normalization=checkpoint["loss_log_norm"],
					kendall_multi_task=checkpoint["loss_kendall_mtask"],
					alpha=checkpoint["loss_epoch_alpha"],
					beta=checkpoint["loss_epoch_beta"]
					)
			lossFunction.log_sigma.data.copy_(checkpoint["loss_state"])
		except KeyError as e:
			e.add_note(f"No loss state key found in the given model file.")
			raise e
	try:
		if checkpoint["selected_inputs"] != UV.SELECTED_INPUTS:
			raise WrongVectorUpdates(f"The given model was trained using a different state of selected input vector values. Expected: {checkpoint['selected_inputs']}, but currently: {UV.SELECTED_INPUTS}.")
		if not has_simple_rule_pred_head and (checkpoint["selected_labels"] != UV.SELECTED_LABELS):
			raise WrongVectorUpdates(f"The given model was trained using a different state of selected label vector values. Expected: {checkpoint['selected_labels']}, but currently: {UV.SELECTED_LABELS}.")
		if checkpoint["updating_functions"] != [f.__name__ for f in UV.UPDATING_FUNCTIONS]:
			raise WrongVectorUpdates(f"The given model was trained using a different state of selected label vector values. Expected: {checkpoint['updating_functions']}, but currently: {[f.__name__ for f in UV.UPDATING_FUNCTIONS]}.")
	except KeyError as e:
		e.add_note(f"Missing key in model state dict, is the saved file from a previous version of the code?")
		raise e
	if "log_norm_epsilon" in checkpoint.keys():
		Constants.DEFAULT_LOG_NORM_EPSILON = checkpoint["log_norm_epsilon"]
	else:
		Constants.DEFAULT_LOG_NORM_EPSILON = 1
	if "optimizer" in checkpoint.keys() and not is_for_test:
		optimizer = torch.optim.Adam(list(model.parameters()) + list(lossFunction.parameters()), lr=Constants.DEFAULT_LEARNING_RATE)
		optimizer.load_state_dict(checkpoint["optimizer"])
	else:
		optimizer = None
	return model, lossFunction, checkpoint["last_epoch"], optimizer

def get_next_versioned_filename(filename: str) -> str:
	"""
	Checks wether the parameter file name exists, returns a valid filename.

	Parameter:
	- filename:		path to the file to analyse

	Returns the file name with a "(n)" appended before the extension if file already exists, n being the lowest possible number.
	"""
	if not os.path.isfile(filename):
		return filename
	name, ext = os.path.splitext(filename)
	counter = 1
	new_name = f"{name}_{counter}{ext}"
	while os.path.isfile(new_name):
		counter += 1
		new_name = f"{name}_{counter}{ext}"
	return new_name

def generate_sorted_random_list(i: int, size: int = 100) -> list[int]:
	"""
	Generates a list of sorted ints, without duplicates.

	Parameters:
	- size:			an int determining the size of the generated list
	- i:			an int, determining the range of the ints to generate, which is the interval ` [0, i[ `

	Returns the generated list.
	"""
	if i < size:
		size = i
	random_list = sorted(random.sample(range(i), size))
	return random_list

def frange(start: float, stop: float, step: float, n: int = None) -> Generator[float, None, None]:
	"""
	Inspired from: https://stackoverflow.com/a/67053708

	Parameters:
	- start:		float value, start of the range to generate
	- stop:			float value, end of the range to generate (excluded)
	- step:			float value, the increment/decrement of the range, must not be 0
	- n:			number of decimal points, default is based on the print values of the given parameters

	Return a WYSIWYG series of float values that mimic range behavior
	by excluding the end point and not printing extraneous digits beyond
	the precision of the input numbers (controlled by n and automatically
	detected based on the string representation of the numbers passed).

	EXAMPLES
	========

	non-WYSIWYS simple list-comprehension

	>>> [.11 + i*.1 for i in range(3)]
	[0.11, 0.21000000000000002, 0.31]

	WYSIWYG result for increasing sequence

	>>> list(frange(0.11, .33, .1))
	[0.11, 0.21, 0.31]

	and decreasing sequences

	>>> list(frange(.345, .1, -.1))
	[0.345, 0.245, 0.145]

	To hit the end point for a sequence that is divisibe by
	the step size, make the end point a little bigger by
	adding half the step size:

	>>> dx = .2
	>>> list(frange(0, 1 + dx/2, dx))
	[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	"""
	if step == 0:
		raise ValueError("Step must not be 0")		
	# how many decimal places are showing?
	if n is None:
		n = max([0 if '.' not in str(i) else len(str(i).split('.')[1]) for i in (start, stop, step)])
	if step*(stop - start) > 0:  # a non-null incr/decr range
		if step < 0:
			for i in frange(-start, -stop, -step, n):
				yield -i
		else:
			steps = round((stop - start)/step)
			while round(step*steps + start, n) < stop:
				steps += 1
			for i in range(steps):
				yield round(start + i*step, n)
	else:
		raise ValueError("Step value is not compatible with the start and stop values.")

def insertSearchInList[T](tab: list[T], elem: T) -> int:
	"""
	Returns the index `i` so that `tab[i] <= elem < tab[i + 1]`, or `tab[i] <= elem` if `i` is the last index of `tab`.

	For example, if `tab=[0, 21, 63, 107]` and `elem=25`, this function will return `1` (because `tab[1] <= elem < tab[2]`).

	Returns -1 if the value of `elem` is less than `tab[0]`.

	NOTE: this function does not check if the list `tab` is sorted or not. If the function cannot find a solution for any reason,\
		the function will simply exit after a maximum of `math.ceil(math.log2(len(tab)))` iterations, and return -1.
	"""
	j = 0
	try:
		max_iter = ceil(log2(len(tab)))
	except ValueError: # List is empty
		return -1
	k = len(tab) - 1
	while j <= k and max_iter >= 0:
		m = j + (k - j) // 2
		if elem >= tab[m] and (m == len(tab) - 1 or elem < tab[m + 1]):
			return m
		elif elem > tab[m]:
			j = m + 1
		else:
			k = m - 1
		max_iter -= 1
	return -1

@DeprecationWarning
def extractFileSlicingParams(filename: str) -> list[tuple[str, str]]:
	"""
	Takes as input a data file name, and extracts its slicing parameters, slicer and printer used, according to the data generation format of this
	repository. NOTE: this function will not work properly if the file was moved and/or renamed after data generation.

	Returns a list of tuples, each containing two strings, the parameter name and its value
	"""
	try:
		res = []
		res.append(("Filename", filename.split('/')[-1]))
		folder_names = filename.split('/')[:-1]
		slicing_params = folder_names[-1]
		printer = folder_names[-2]
		slicer = folder_names[-3]
		res.append(("Slicer", slicer))
		res.append(("Printer", printer))
		for param in slicing_params.split('_'):
			splits = param.split('-')
			res.append((splits[0], splits[1]))
	except:
		raise FilenameParsingError(f"Could not parse the given file name: {filename}, was it changed after data generation ?")
	return res

def decodeMoveTypeFromBitField(
		bitField: float | list | np.ndarray | torch.Tensor
	) -> dict[str, bool | list | np.ndarray | torch.Tensor]:
	"""
	Decodes the float value of the move type given by the Gato3D vector data extraction pass. This value is actually a bit field written in the binary files as a float, hence this decoder function.

	Parameter:
	- bitfield:		the float value of the move type, as extracted by `VectorDataset.py`. Must be either a single value, or an iterable of those values, as a python list, Numpy ndarray, or PyTorch tensor.

	Returns: A dict, with the name of the move types as keys, and a boolean as value in the case of a single input value, or an iterable of booleans of the same sort as the input. The move types are given by `Constants.MOVE_TYPES`.
	"""
	def __float_to_uint32(f: float) -> int:
		return struct.unpack("<I", struct.pack("<f", f))[0]
	def __decode_single(f: float) -> list[bool]:
		bits = __float_to_uint32(f)
		return [((bits >> i) & 1) == 1 for i in range(len(Constants.MOVE_TYPES))]
	
	# Scalar value handling
	if isinstance(bitField, float):
		decoded = __decode_single(bitField)
		return dict(zip(Constants.MOVE_TYPES, decoded))

	# List of scalars handling
	elif isinstance(bitField, list):
		if len(bitField) == 0:
			decoded = []
		elif isinstance(bitField[0], float):
			decoded = [__decode_single(f) for f in bitField]
		else:
			raise TypeError(f"The argument `bitField` must be one of: `float`, `list[float]`, `numpy.ndarray[float]`, or `torch.Tensor[torch.float32]`. Got: {type(bitField)}.")
		transposed = list(zip(*decoded))
		return dict(zip(Constants.MOVE_TYPES, [list(t) for t in transposed]))

	# Numpy array handling
	elif isinstance(bitField, np.ndarray) and bitField.dtype == np.float32:
		ints = np.frombuffer(bitField.astype(np.float32).tobytes(), dtype=np.uint32)
		decoded = ((ints[:, None] >> np.arange(len(Constants.MOVE_TYPES))) & 1).astype(bool)
		transposed = decoded.T
		return dict(zip(Constants.MOVE_TYPES, transposed))
	
	# PyTorch tensor handling
	elif isinstance(bitField, torch.Tensor) and bitField.dtype == torch.float32:
		ints = bitField.view(torch.int32)
		decoded = ((ints.unsqueeze(1) >> torch.arange(len(Constants.MOVE_TYPES), device=bitField.device, dtype=torch.int32)) & 1).bool()
		transposed = decoded.T
		return dict(zip(Constants.MOVE_TYPES, transposed))

	# Given bitField is not a handled type
	else:
		raise TypeError(f"The argument `bitField` must be one of: `float`, `list[float]`, `numpy.ndarray[float]`, or `torch.Tensor[torch.float32]`. Got: {type(bitField)}.")

def tensorMaskForMoveType(t: torch.Tensor, moveType: str | list[str], applyOrToMoveTypes: bool = True) -> torch.Tensor:
	"""
	Returns a PyTorch mask tensor of the same shape as the input tensor `t`, where the mask is true for instruction with a move type of the given `moveType`.
	"""
	moveTypes = [moveType] if isinstance(moveType, str) else list(moveType)	
	for mt in moveTypes:
		if mt not in Constants.MOVE_TYPES:
			raise ValueError(f"Unknown move type: {mt}")
	try:
		move_type_idx = 8 if UV.SELECTED_INPUTS is None else UV.SELECTED_INPUTS.index(8)
	except ValueError as e:
		e.add_note(f"The selected input values, set in `UpdateVectors.py`, most likely do not include the move type (which should be at index 8 of the default vectors).")
		raise e
	
	ints = t[..., move_type_idx].view(torch.int32)
	if applyOrToMoveTypes:
		mask = torch.zeros_like(ints, device=t.device, dtype=torch.bool)
	else:
		mask = torch.ones_like(ints, device=t.device, dtype=torch.bool)
	for mt in moveTypes:
		if applyOrToMoveTypes:
			mask |= ((ints >> torch.scalar_tensor(Constants.MOVE_TYPES.index(mt), device=t.device, dtype=torch.int32)) & 1).bool()
		else:
			mask &= ((ints >> torch.scalar_tensor(Constants.MOVE_TYPES.index(mt), device=t.device, dtype=torch.int32)) & 1).bool()

	return mask

"""
End of file
"""