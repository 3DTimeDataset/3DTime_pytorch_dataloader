import sys, os
import torch
import array
from torch.utils.data import Dataset
from math import ceil
from typing import Self, Callable, Tuple

try:
    import Logger
    import Utils
    import Constants
except:
    sys.path.insert(0,"..")
    import Logger
    import Utils
    import Constants

class VectorDataset(Dataset):
    """
    # Vector Dataset

    Custom Dataset loader. Since the data is too large to fit on (most) system's memory, this Dataset initiates arrays to setup data reading.

    When an index is looked up, data is read, converted to pytorch tensor, and sent to the specified device (CPU or GPU) at runtime.

    This class also overrides the `__getitems__` function, to allow accelerated batch sample reading.

    NOTE: the base function `__getitem__(index)` is used as `dataset[index]`, while the function `__getitems__(list_indexes)` can be used as `dataset[i:j]`.

    ## Class constructor

    The constructor can be divided into two sets of parameters, the first arguments, used for normal behavior, and the last three ones (aka `copyFrom`,
    `startIndex`, and `endIndex`) are only used internally for dataset division, and are not recommended to change for the user. For more information
    about these last parameters, see the function `splitDataset()`.

    ### First group of parameters:
    - data_source:          a string of the directory containing all the DAT files
    - nbInstrPerSample:     an int indicating the number of G-Code instruction wanted for the NN input
    - inputVectorSize:      an int describing the size of each instruction input vector (without the labels)
    - labelVectorSize:      an int describing the size of each instruction label vector (without the inputs). The total size of an instruction vector is
                            hence `inputVectorSize + labelVectorSize`
    - leftOffset:           the prediction models might only predict a sub-window of the total `nbInstrPerSample` instructions per sample (e.g. the default
                            value of `nbInstrPerSample` is 100, and the default left and right offsets are 10 and 90, meaning that from the 100 input
                            instructions, only 80 are predicted). This parameter describes the value of the left offset
    - rightOffset:          right offset of the aforementioned sub-window prediction. The two values of the offsets must follow these two rules:
                            `0 <= leftOffset < rightOffset <= nbInstrPerSample` (ensuring the values are ordered and within bounds), and
                            `rightOffset - leftOffset >= nbInstrPerSample - rightOffset + leftOffset` (simply meaning that more instructions are predicted
                            than ommited, due to a coding limitation)
    - device:               a torch device, either CPU or GPU, used to load the data on the correct device
    - floatType:            the float precision used to load the dataset, default is float32 (around 6 float points precision)
    - log:                  a Logger object (see Logger.py in parent directory). If None (by default), will create a new Logger without log
                            file writing
    - random_seed:          an int used as random seed for reproducibility. At initlalization, the files of the source directory are read in a random
                            order. If None (by default), the files will not be read randomly. This is used instead of runtime random to accelerate batch
                            loading

    ### Second group of parameters (not recommended for user, see the function splitDataset() instead):
    - copyFrom:             a VectorDataset to copy the data from (None by default)
    - startIndex:           an int of the start of the sub-array of samples for that copy
    - endIndex:             an int of the end of the sub-array of samples for that copy

    ### Visual representation of the offsets

    If there are offsets, the loading logic for a single sample is as follows:

    ```
    Input tensor: [[input0], [input1], [input2], ..., [input6], [input7], [input8], [input9]]
                    \\                                                                  /
                     \\                                                                /
                      \\                                                              /
                       \\                                                            /
                        \\                                                          /
                         \\                                                        /
                          \\                                                      /
    Label tensor:           [[label1], [label2], ..., [label6], [label7], [label8]]
    ```

    This scenario represents a dataset with the following values: `nbInstrPerSample = 10`, `leftOffset = 1`, and `rightOffset = 9`
    """

    # Constructor
    def __init__(
                self,
                data_source: str,           # Full path to the folder where the data files are located, either CSV or DAT (can be zipped)
                nbInstrPerSample: int,      # Number of G-code instruction wanted per sample of the model
                inputVectorSize: int,       # Number of input values per input vector (without the labels)
                labelVectorSize: int,       # Number of label values per input vector
                leftOffset: int,            # Left index of the predicted window from the total `nbInstrPerSample` of the input
                rightOffset: int,           # Right index of the predicted window from the total `nbInstrPerSample` of the input
                device: torch.device,       # Pytorch device, either CPU or CUDA, used to send the tensors to the wanted device
                *,
                floatType: torch.dtype = torch.float32, # Float type, or precision for the dataset: 2, 4 or 8 bytes per float (2 is not supported)
                log: Logger.Logger = None,  # Logger instance
                random_seed: int = None,    # Random seed for reproducibility. If None, the dataset will have no randomness.

                masked_inputs: list[int] = None,        # If provided, the input vectors will only consits of those feature vector indexes
                masked_labels: list[int] = None,        # If provided, the label vectors will only consits of those label vector indexes
                append_input_vectors: list[Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor | None, torch.Tensor | None]]] = [],
                # List of functions to call to add values to the input vectors, those functions must return a Tensor of shape
                # [batch_size, nbInstrPerSample, number_of_values_to_add]
                metadata: str = None,       # Path to a CSV file containing the metadata of the given dataset path
                column: str = None,         # Name of a column in the metadata CSV file
                values: list[str] = None,   # List of the values to consider for the generation of the dataset (aka a file in the dataset path will be added to this dataset only if its value in the metadata file in the given column is in the provided value list)

                includeUnchangedVectors: bool = False,  # Flag controlling wether item getters also include the base tensors without any updates that could come from UpdateVectors.py
                
                copyFrom: Self = None,      # VectorDataset to copy the data from
                startIndex: int = 0,        # Start index in the full dataset for the copy
                endIndex: int = 0           # End index in the full dataset for the copy
            ) -> None:
        
        # This if: normal constructor, build the dataset by reading data files in the provided folder
        if copyFrom is None:

            if log == None:
                self.log: Logger.Logger = Logger.Logger(writesLog=False)
            else:
                self.log: Logger.Logger = log

            assert(0 <= leftOffset < rightOffset <= nbInstrPerSample)
            assert(rightOffset - leftOffset >= nbInstrPerSample - rightOffset + leftOffset) # Ensures there are more predicted instructions that ommited ones
            self.data_source = data_source
            self.nbInstrPerSample = nbInstrPerSample
            self.nbValuesPerVector = inputVectorSize + labelVectorSize # Total size of an instruction vector: sum of input length and label length
            self.inputVectorSize = inputVectorSize
            self.labelVectorSize = labelVectorSize
            self.left_offset = leftOffset
            self.right_offset = rightOffset
            self.device = device
            self.floatType = floatType
            self.random_seed = random_seed

            self.masked_inputs = masked_inputs
            self.masked_labels = masked_labels
            self.append_input_vectors = append_input_vectors
            # Test if the masked input indexes correspond to the original vector sizes
            try:
                if self.masked_inputs is not None:
                    assert(all(0 <= i < self.inputVectorSize for i in self.masked_inputs))
                    assert(all(x1 < x2 for x1, x2 in zip(self.masked_inputs[:-1], self.masked_inputs[1:])))
                    self.masked_inputs = torch.tensor(self.masked_inputs, device=self.device)
            except AssertionError as e:
                self.log.error(f"The given masking input indexes ({self.masked_inputs}) is not correct.", 1, e)
            # Test if the masked label indexes correspond to the original vector sizes
            try:
                if self.masked_labels is not None:
                    assert(all(0 <= i < self.labelVectorSize for i in self.masked_labels))
                    assert(all(x1 < x2 for x1, x2 in zip(self.masked_labels[:-1], self.masked_labels[1:])))
                    self.masked_labels = torch.tensor(self.masked_labels, device=self.device)
            except AssertionError as e:
                self.log.error(f"The given masking label indexes ({self.masked_labels}) is not correct.", 1, e)
            # Test if the given functions used to add values to the vectors generate tensors of the correct shape
            __test_input = torch.randn(2, self.nbInstrPerSample, inputVectorSize)
            __test_label = torch.randn(2, rightOffset - leftOffset, labelVectorSize)
            __nb_added_values_input = 0
            __nb_added_values_label = 0
            for func in self.append_input_vectors:
                try:
                    i, l = func(__test_input, __test_label, leftOffset, rightOffset)
                except BaseException as e:
                    self.log.error(
                        f"For the given input vector extension function '{func.__name__}', there was an exception at runtime.", 1, e
                        )
                try:
                    if i is None:
                        pass
                    elif len(i.shape) == 3:
                        assert(i.shape[:2] == (2, self.nbInstrPerSample))
                        __nb_added_values_input += i.shape[2]
                    elif len(i.shape) == 2:
                        __nb_added_values_input += 1
                    else:
                        raise AssertionError
                except AssertionError as e:
                    self.log.error(
                        f"For the given input vector extension function '{func.__name__}', the result input tensor is not if the correct shape: {i.shape}.", 1, e
                        )
                except BaseException as e:
                    self.log.error(
                        f"For the given input vector extension function '{func.__name__}', failed to test result input shape ({i.shape}). Are you sure the result tensor is correct?", 1, e
                        )
                try:
                    if l is None:
                        pass
                    elif len(l.shape) == 3:
                        assert(l.shape[:2] == (2, rightOffset - leftOffset))
                        __nb_added_values_label += l.shape[2]
                    elif len(l.shape) == 2:
                        __nb_added_values_label += 1
                    else:
                        raise AssertionError
                except AssertionError as e:
                    self.log.error(
                        f"For the given input vector extension function '{func.__name__}', the result label tensor is not if the correct shape: {l.shape}.", 1, e
                        )
                except BaseException as e:
                    self.log.error(
                        f"For the given input vector extension function '{func.__name__}', failed to test result label shape ({l.shape}). Are you sure the result tensor is correct?", 1, e
                        )
                del i, l
            del __test_input, __test_label
            self.true_input_vector_size = self.inputVectorSize if self.masked_inputs is None else len(self.masked_inputs)
            self.true_input_vector_size += __nb_added_values_input
            self.true_label_vector_size = self.labelVectorSize if self.masked_labels is None else len(self.masked_labels)
            self.true_label_vector_size += __nb_added_values_label

            # Metadata file selection variable initialization
            if metadata is not None:
                import pandas as pd
                import numpy as np
                self.__metadata = pd.read_csv(metadata)
                if column not in self.__metadata.columns:
                    self.log.error(f"The given column name {column} is not a column name in the provided CSV file: {metadata}", 1)
                if "Binary file name" not in self.__metadata.columns:
                    self.log.error(f"The given CSV metadata file path does not seem to be the metadata of a dataset: it does not contain a \"Binary file name\" column.", 1)
                self.__column = column
                self.__valueType = self.__metadata[self.__column].dtype
                self.__npValues = np.array(values, dtype=self.__valueType)
            else:
                self.__metadata = None

            # Counter for the number of files used in the dataset
            self.nbFiles = 0
            
            # Initialization of local variables
            
            """
            Local variables used for file reading. For the `__indexes` and the `__filenames` lists, the indexes are linked (can be considered as a dict, but
            kept as two lists for easy index access), aka `__indexes[i]` corresponds to `__finenames[i]`.

            The `__lastIndex` variable is only used at class initialization and the `__length` variable is only used for error detection.

            ## Dataset initialization logic:

            Lists all data files found in the given input folder, process the number of samples per file depending on the other constructor parameters (such
            as `nbInstrPerSample`, and the offsets), and stores them accordingly. For example, if `file1.dat` contains 12 samples, `file2.dat` contains 7
            samples, and `file3.dat` contains 8 samples, the two lists will be:

            ```
            __filenames = ["file1.dat", "file2.dat", "file3.dat"]
            __indexes = [0, 12, 19, 27]
            ```

            To get sample `i` of the dataset, find `j` the index in the 2 lists so that `__filenames[j]` is the file containing the `i`-th sample.

            This can be done by searching in `__indexes` for the largest value `__indexes[j]` that is inferior or equal to i:

            ```
            max `j` so that `__indexes[j] <= i`
            ```
            """
            self.__indexes = []
            self.__filenames = []
            self.__lastIndex = 0
            self.__length = 0
            
            # Input data reading: is it a folder or a single file?

            if os.path.isdir(self.data_source):
                # Explores the sub-directories of the input folder
                roots = []
                for root, _, files in os.walk(data_source):
                    for file in files:
                        if file.endswith(".dat"):
                            if self.__metadata is None:
                                roots.append((root, file))
                            else:
                                row = self.__metadata[self.__metadata["Binary file name"] == file]
                                if row.empty:
                                    continue
                                if np.issubdtype(self.__valueType, np.floating):
                                    eps = 1e-5
                                    if np.any(np.abs(self.__npValues - row[self.__column].iloc[0]) < eps):
                                        roots.append((root, file))
                                else:
                                    if row[self.__column].iloc[0] in self.__npValues:
                                        roots.append((root, file))
                if self.__metadata is not None:
                    del self.__metadata, self.__column, self.__npValues

                # Randomizes the data file order if asked
                if self.random_seed != None:
                    self.log.notice(f"Randomizing dataset file access order.")
                    from random import seed, shuffle
                    seed(self.random_seed)
                    shuffle(roots)

                # Data file loop
                for (root, file) in roots:
                    file_path = os.path.join(root, file)
                    self.log.trace(f"Importing file {file_path}")
                    try:
                        self.nbFiles = self.nbFiles + self.__addVectorFileToDataset(file_path) # Adds the file to the dataset
                        self.__filenames.append(file_path)
                    except Exception as e:
                        self.log.alert(f"Failed to import file {file}", e)
                        continue                        
            elif os.path.isfile(self.data_source):
                try:
                    self.nbFiles = self.nbFiles + self.__addVectorFileToDataset(self.data_source)
                    self.__filenames.append(self.data_source)
                except Exception as e:
                    self.log.alert(f"Failed to import file {file}", e)
            else:
                self.log.error(f"The data source \"{data_source}\" does not exists.", 1)
            
            if self.nbFiles == 0:
                self.log.error("No data files were found in the specified data directory, or the given file was corrupted.", 1)

            self.__startIndex = 0
            self.__endIndex = self.__length
            self.__copyFrom: Self = None

            self.includeUnchangedVectors = includeUnchangedVectors

        # Else: the constructor will not build the dataset from the provided folder, but from the dataset provided as "copyFrom"
        # See the function splitDataset()
        # Hence, simply copy the needed values from the original dataset to this one
        else:
            self.log: Logger.Logger = copyFrom.log
            if startIndex <= 0 or endIndex <= 0 or startIndex >= endIndex or endIndex > len(copyFrom):
                self.log.error(f"Failed to copy the dataset, wrong parameters: {startIndex}, {endIndex}", 1)
            self.data_source: str = copyFrom.data_source
            self.nbInstrPerSample: int = copyFrom.nbInstrPerSample
            self.inputVectorSize: int = copyFrom.inputVectorSize
            self.labelVectorSize: int = copyFrom.labelVectorSize
            self.left_offset: int = copyFrom.left_offset
            self.right_offset: int = copyFrom.right_offset
            self.device: torch.device = copyFrom.device
            self.__startIndex: int = startIndex
            self.__endIndex: int = endIndex
            self.__length: int = copyFrom.__length
            self.__copyFrom: Self = copyFrom
            self.nbValuesPerVector: int = copyFrom.nbValuesPerVector
            self.nbFiles: int = copyFrom.nbFiles
            self.floatType: torch.dtype = copyFrom.floatType
            self.random_seed: int = copyFrom.random_seed
            self.masked_inputs: torch.Tensor | None = copyFrom.masked_inputs
            self.masked_labels: torch.Tensor | None = copyFrom.masked_labels
            self.append_input_vectors: list[Callable[[torch.Tensor], torch.Tensor]] = copyFrom.append_input_vectors
            self.true_input_vector_size: int = copyFrom.true_input_vector_size
            self.true_label_vector_size: int = copyFrom.true_label_vector_size
            self.includeUnchangedVectors: bool = includeUnchangedVectors
    
    def __addVectorFileToDataset(self, file_path: str) -> int:
        """
        Adds a single DAT file to the dataset.
        
        - file_path:     full path to the file

        Returns            1 if the file was added
        """
        nb_lines = Utils.file_size(file_path) // (4 * (self.nbValuesPerVector)) # 4 is for the size of a floats
        self.__indexes.append(self.__lastIndex)
        nb_samples = ceil(nb_lines / (self.right_offset - self.left_offset))
        self.__lastIndex += nb_samples
        self.__length = self.__lastIndex
        del nb_lines, nb_samples
        return 1
    
    def splitDataset(self, proportion: float) -> tuple[Dataset, Dataset]:
        """
        Splits the global Dataset into 2 different sub datasets.
        
        - proportion:       a float in [0.0,1.0] of the proportion of samples that will be added to the first sub-dataset.
                            The second sub-dataset will contain (1 - proportion) of the samples in the original dataset.

        Returns a tuple of the 2 datasets
        """
        assert(0.0 < proportion < 1.0)
        sub_start = int(self.__len__() * proportion)
        # This is where the other constructor params are used
        sub = VectorDataset("", 0, 0, 0, 0, 0, self.device, copyFrom=self, startIndex=sub_start, endIndex=self.__len__())
        self.__endIndex = sub_start - 1
        del sub_start
        return self, sub
    
    def summary(self):
        """
        Prints a summary of the dataset.
        """
        self.log.emptyLines()
        self.log.info("Dataset summary:")
        nbVectors = self.__len__() * self.nbInstrPerSample
        string = f" {'=' * 90}\n\t"
        string += f"Number of source files used for the dataset:    {self.nbFiles}\n\t"
        string += f"Total amount of samples:                        {self.__len__()} ({self.nbInstrPerSample} instructions per sample)\n\t"
        string += f"Total amount of G-code instructions:            {nbVectors} ({self.nbValuesPerVector} values per instruction)\n\t"
        string += f"Estimated max memory usage:                     {Utils.format_memory_usage(self.nbFiles * (4 + 100))}\n\t"
        string += f"{'=' * 90}\n"
        self.log.info(string)
        self.log.emptyLines()
    
    def __len__(self) -> int:
        """
        Gives the length of the dataset (number of samples).
        """
        return self.__endIndex - self.__startIndex
    
    def __getFileIndex(self, idx: int) -> int:
        """
        Binary search for the array index corresponding to sample index idx.

        For example, if `self.__indexes=[0, 21, 63, 107]` and `idx=25`, this function will return `1`
        (because `self.__indexes[1] <= idx < self.__indexes[2]`).
        """
        j = 0
        k = len(self.__indexes) - 1
        while j <= k:
            m = (j + k) // 2
            if idx >= self.__indexes[m] and (m == len(self.__indexes) - 1 or idx < self.__indexes[m + 1]):
                return m
            elif idx > self.__indexes[m]:
                j = m + 1
            else:
                k = m - 1
        raise ValueError(f"This error message can only appear if the data initialization was corrupted.")
    
    def __readBinaryBlock(
                    self,
                    filename: str,
                    start_index: int,
                    block_size: int,
                    line_size: int,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ## Main file reading function

        Reads bytes from file `filename`, starting at byte `start_index` until `start_index + block_size` excluded.

        Returns a tuple of two Tensors, each representing the inputs and labels of a batch (might be a batch size of 1, but always considered as a batch anyway).

        The inputs and labels are each a list of instructions (respectively of length `nbInstrPerSample` and `rightOffset - leftOffset`), each consisting of
        a list of `inputVectorSize` floats and `labelVectorSize` floats respectively.

        If EOF is reached, no error is raised, and the returned Tensors are simply shorter than requested (last sample is still filled with zeros if needed, but
        not the batch).

        Parameters:
        - filename:     the path to the file to read
        - start_index:  the byte number to start reading from. NOTE: if this value is negative, it means that there is a left offset, and that the current
                        requested sample is at the start of the file, meaning that the produced input list will start with some zeros
        - block_size:   total number of bytes to read from the file, processed elsewhere
        - line_size:    total number of bytes used for an instruction total vector (input and label)
        """
        with open(filename, 'rb') as f: # Open the file

            # Start block handling, if the first sample asked is at the start of a file, `start_index` is negative, meaning that the actual byte block
            # will be shorter by the size of the left offset, and the latter will be filled with zeros
            if start_index >= 0: # Not start of file, normal behavior
                try:
                    f.seek(start_index) # Go to start of block, avoids useless file reads
                except BaseException as e:
                    e.add_note(f"Index: {start_index}, filename: {filename}")
                    raise e
                block = f.read(block_size) # Read the block
                content = array.array("f", []) # Initiates the floar array (more memory and time complexity efficient that a list)
                content.frombytes(block) # Reads the raw bytes as floats (conversion is not needed, and implicitly done by the array type above)
            else: # Sample is a the start of a file, reduced block reading and filling with zeros
                block = f.read(block_size - self.left_offset * line_size) # Smaller block
                content = array.array("f", [0.0 for _ in range(self.nbValuesPerVector * self.left_offset)]) # Same array initialization, but with zeros
                content.frombytes(block) # Reads the raw bytes

            # At this point, the entire batch (potentially of size 1) is read, up to potentially EOF, meaning that the last sample might be shorter than
            # needed, so if that is the case, fill the end of the last sample with zeros

            # The if else statement is to diferentiate the case where the EOF was reached before or after the right offset
            if len(content) < (block_size // 4) - self.nbValuesPerVector * (self.nbInstrPerSample - self.right_offset):
                # Fill up to right offset
                while (len(content) - self.left_offset * self.nbValuesPerVector) % ((self.right_offset - self.left_offset) * self.nbValuesPerVector) != 0:
                    content.append(0.0)
                # Fill the right offset
                content.extend([0.0 for _ in range((self.nbInstrPerSample - self.right_offset) * self.nbValuesPerVector)])
            elif len(content) < block_size // 4: # Handles the case where the sample stopped exactly at the right offset
                # EOF was reached after right offset but before sample end, simply complete the offset
                while (len(content) != (block_size // 4)):
                    content.append(0.0)

            del block

            # Converts the python array to the pytorch tensor, this uses the highly efficient `torch.frombuffer()` command, and hence does not require
            # float conversion
            linear_tensor: torch.Tensor = torch.frombuffer(content, dtype=torch.float32).to(self.device)
            if self.device.type == "cuda":
                del content

            # Following commands only use pytorch tensor views and reshape, making them especially time and memory efficient

            # Separate the inputs from the labels
            inputs = linear_tensor.view(linear_tensor.shape[0] // self.nbValuesPerVector, self.nbValuesPerVector)[: ,:self.inputVectorSize]
            labels = linear_tensor.view(linear_tensor.shape[0] // self.nbValuesPerVector, self.nbValuesPerVector)[: ,self.inputVectorSize:]

            # Process the actual number of samples in this batch (can be less than wanted if EOF was reached)
            if start_index >= 0:
                nb_sample_in_block =labels.shape[0] // (self.right_offset - self.left_offset)
            else:
                nb_sample_in_block = (labels.shape[0] - self.left_offset) // (self.right_offset - self.left_offset)

            # Unfolds the inputs from [L, inputSize] to [batchSize, nbInstrPerSample, inputSize] with the wanter offset overlap
            batch_inputs = inputs.unfold(0, self.nbInstrPerSample, self.right_offset - self.left_offset).permute(0, 2, 1)
            # Reshapes the labels (no overlap needed, so quite simple, just without the useless leftmost and rightmost offsets)
            batch_labels = labels[
                self.left_offset:self.left_offset + nb_sample_in_block * (self.right_offset - self.left_offset)
            ].reshape(nb_sample_in_block, self.right_offset - self.left_offset, self.labelVectorSize)

            del nb_sample_in_block
            if torch.isnan(batch_inputs).any() or torch.isnan(batch_labels).any():
                raise Utils.NanInTensorDetected(f"Detected at least one NaN value in a tensor in file: {filename}")
            return batch_inputs, batch_labels

    def __getSampleFromFileIndex(self, file: str, file_index: int, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample from a file (aka simply calls the `__readBinaryBlock` function with a batch size of 1)
        """
        line_size = (self.nbValuesPerVector) * 4 # size of a "line" -> sizeof(float) * (nb_values_per_instr)
        start = (idx - self.__indexes[file_index]) * (self.right_offset - self.left_offset) * line_size - self.left_offset * line_size
        block_size = line_size * self.nbInstrPerSample
        data, label = self.__readBinaryBlock(file, start, block_size, line_size)
        return data[0], label[0]
    
    def __updateVectors(self, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: write doc
        """
        # Appending values processing
        to_append_i: list[torch.Tensor] = []
        to_append_l: list[torch.Tensor] = []
        if self.append_input_vectors is not None:
            for func in self.append_input_vectors:
                i, l = func(inputs, labels, self.left_offset, self.right_offset)
                to_append_i.append(i)
                to_append_l.append(l)
        # Input vector updates
        if self.masked_inputs is not None:
            inputs = inputs.index_select(dim=-1, index=self.masked_inputs)
        for t in to_append_i:
            if t is None:
                continue
            try:
                if len(t.shape) == 2 and len(inputs.shape) == 3:
                    t = t.unsqueeze(-1)
                elif len(t.shape) == 1 and len(inputs.shape) == 2:
                    assert(len(inputs.shape) == 2)
                    t = t.unsqueeze(-1)
                inputs = torch.cat([inputs, t], dim=-1)
            except RuntimeError as e:
                e.add_note(f"inputs shape: {inputs.shape}")
                e.add_note(f"Additionnal vector shape: {t.shape}")
                raise e
        # Label vector updates
        if self.masked_labels is not None:
            labels = labels.index_select(dim=-1, index=self.masked_labels)
        for t in to_append_l:
            if t is None:
                continue
            try:
                if len(t.shape) == 2 and len(labels.shape) == 3:
                    t = t.unsqueeze(-1)
                elif len(t.shape) == 1 and len(labels.shape) == 2:
                    assert(len(labels.shape) == 2)
                    t = t.unsqueeze(-1)
                labels = torch.cat([labels, t], dim=-1)
            except RuntimeError as e:
                e.add_note(f"labels shape: {labels.shape}")
                e.add_note(f"Additionnal vector shape: {t.shape}")
                raise e
        return inputs, labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameter:

        - `idx`:        an int of the index of the desired sample, within the interval `[0, len(self)[`

        Return value:

        Returns a tuple of tensors, the first one being the inputs (of shape: `[nbInstrPerSample, self.inputVectorSize]`), and the second
        being the labels (of shape `[rightOffset - leftOffset, self.labelVectorSize]`)

        If the class variable `includeUnchangedVectors` is set to true, it also returns two additionnal tensors, consisting of the input and
        label tensors, but before vector updates.
        """
        if not (0 <= idx < self.__len__()):
            raise IndexError(f"Index {idx} is out of bounds (dataset length: {self.__len__()}).")
        if self.__copyFrom == None:
            file_index = self.__getFileIndex(self.__startIndex + idx)
            input, label = self.__getSampleFromFileIndex(self.__filenames[file_index], file_index, self.__startIndex + idx)
            if self.includeUnchangedVectors:
                baseI = torch.zeros_like(input, device=torch.device("cpu"))
                baseL = torch.zeros_like(label, device=torch.device("cpu"))
                baseI.copy_(input, non_blocking=True)
                baseL.copy_(label, non_blocking=True)
            input, label = self.__updateVectors(input.unsqueeze(0), label.unsqueeze(0))
            return input.squeeze(), label.squeeze()
        else:
            actual_index = self.__copyFrom.__getFileIndex(self.__startIndex + idx)
            input, label = self.__copyFrom.__getSampleFromFileIndex(self.__copyFrom.__filenames[actual_index], actual_index, self.__startIndex + idx)
            if self.includeUnchangedVectors:
                baseI = torch.zeros_like(input, device=torch.device("cpu"))
                baseL = torch.zeros_like(label, device=torch.device("cpu"))
                baseI.copy_(input, non_blocking=True)
                baseL.copy_(label, non_blocking=True)
            input, label = self.__updateVectors(input.unsqueeze(0), label.unsqueeze(0))
            if not self.includeUnchangedVectors:
                return input.squeeze(), label.squeeze()
            else:
                return input.squeeze(), label.squeeze(), baseI, baseL

    def __getitems__(self, l_idx: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]] | list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Batch access for accelerated data loading.

        If the fetched indexes are random, this is not any faster than calling several times the __getitem__() function, but fetching speed is greatly
        increased if the indexes are ordered (aka sorted and spaced by 1 index each). If the fetched indexes are not ordered, this function simply calls
        the regular `__getitem__()` function several times.

        In the case of ordered fetched indexes, the data extraction logic is the same as for `__getitem__()`, but with a `block_size` multiplied by the
        number of samples asked in `l_idx`, which is the batch size.

        Parameter:
        - l_idx:        a list of indexes of the desired samples

        Returns a list of `len(l_idx)` tuples, each containing the 2 tensors of the corresponding sample: input data, label data.
        """
        if len(l_idx) == 0:
            raise IndexError(f"The list of indexes to fetch is empty.")

        # Checks wether the list l_idx is sorted and spaced by 1 for all elements
        # If len(l_idx)== 1 the condition is True, which does not really matter
        if not all(x2 - x1 == 1 for x1, x2 in zip(l_idx[:-1], l_idx[1:])): # To put it simply: checks if the list of indexes is shuffled or ordered
            # Result array of samples
            if self.includeUnchangedVectors:
                batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            else:
                batch: list[tuple[torch.Tensor, torch.Tensor]] = []
            for idx in l_idx:
                if self.includeUnchangedVectors:
                    i, l, bi, bl = self.__getitem__(idx)
                    batch.append((i, l, bi, bl))
                else:
                    i, l = self.__getitem__(idx)
                    batch.append((i, l))
        
        # Fetched indexes are ordered, optimized data extraction logic
        else:

            # Error handling
            if l_idx[-1] >= self.__length:
                raise IndexError(f"The last index of the index list is out of range: {l_idx[-1]} when the dataset length is {self.__length}.")
            
            # VectorDataset instance to use initialization, in case `self` is a copy of a VectorDataset
            if self.__copyFrom == None:
                to_use = self
            else: 
                to_use = self.__copyFrom

            # `j`-th index extraction: which index should be used for batch fetching in `self.__indexes` and `self.__filenames`
            start_file_index = to_use.__getFileIndex(self.__startIndex + l_idx[0])

            # Line size, in number of bytes
            line_size = (self.nbValuesPerVector) * 4

            # Total size of the binary block to read (might be shorter if EOF)
            max_block_size = line_size * (self.right_offset - self.left_offset) * len(l_idx) + \
                line_size * (self.nbInstrPerSample - self.right_offset + self.left_offset)

            # Start byte to read from
            start = (l_idx[0] - to_use.__indexes[start_file_index]) * (self.right_offset - self.left_offset) * line_size - self.left_offset * line_size

            # Binary data getter
            try:
                data, label = self.__readBinaryBlock(to_use.__filenames[start_file_index], start, max_block_size, line_size)
            except RuntimeError as e:
                try:
                    self.log.error(f"Failed to read the binary block")
                    e.add_note(f"Input shape: {data.shape}")
                    e.add_note(f"Label shape: {label.shape}")
                except:
                    pass
                raise e
            
            if self.includeUnchangedVectors:
                baseI = torch.zeros_like(data, device=torch.device("cpu"))
                baseL = torch.zeros_like(label, device=torch.device("cpu"))
                baseI.copy_(data, non_blocking=True)
                baseL.copy_(label, non_blocking=True)

            # Update vectors if needed
            try:
                data, label = self.__updateVectors(data, label)
            except RuntimeError as e:
                try:
                    self.log.error(f"Failed to update the vectors")
                    e.add_note(f"Input shape: {data.shape}")
                    e.add_note(f"Label shape: {label.shape}")
                except:
                    pass
                raise e

            # Re-arrange the batch from tuple[Tensor, Tensor] to list[tuple[Tensor, Tensor]]
            if self.includeUnchangedVectors:
                batch = [(data[i], label[i], baseI[i], baseL[i]) for i in range(data.shape[0])]
            else:
                batch = [(data[i], label[i]) for i in range(data.shape[0])]

            # If the batch is not yet complete (aka EOF of the first file read was reached), complete the batch with data from the
            # next file(s). Technically, an `if` should be enough, but just in case, this is a `while` (which should only iterate once
            # in most cases, more iterations only happen for combinations of large number of instr per sample and batch size, and small files).
            j = 1
            while len(batch) < len(l_idx):
                # Same data extraction logic, but with `start_file_index + j` instead
                new_block_size = line_size * (self.right_offset - self.left_offset) * (len(l_idx) - len(batch)) + line_size * \
                    (self.nbInstrPerSample - self.right_offset + self.left_offset)
                new_start = - line_size * self.left_offset
                data, label = self.__readBinaryBlock(to_use.__filenames[start_file_index + j], new_start, new_block_size, line_size)
                if self.includeUnchangedVectors:
                    baseI = torch.zeros_like(data, device=torch.device("cpu"))
                    baseL = torch.zeros_like(label, device=torch.device("cpu"))
                    baseI.copy_(data, non_blocking=True)
                    baseL.copy_(label, non_blocking=True)
                # Update vectors if needed
                try:
                    data, label = self.__updateVectors(data, label)
                except RuntimeError as e:
                    try:
                        self.log.error(f"Failed to update the vectors")
                        e.add_note(f"Input shape: {data.shape}")
                        e.add_note(f"Label shape: {label.shape}")
                    except:
                        pass
                    raise e
                if self.includeUnchangedVectors:
                    batch.extend([(data[i], label[i], baseI[i], baseL[i]) for i in range(data.shape[0])])
                else:
                    batch.extend([(data[i], label[i]) for i in range(data.shape[0])])
                j += 1

        # Uncomment the following code to test the produced tensor shapes
        # WARNING: this shape verification only works if no update was made to the tensors (aka 'masked_inputs', 'masked_labels', and 'append_input_vectors')
        # are set respectivelly to None, None, and [] (which is the case by default)
        """
        assert(len(batch) == len(l_idx))
        try:
            assert all(
                b[0].shape == (self.nbInstrPerSample, self.inputVectorSize) and
                b[1].shape == ((self.right_offset - self.left_offset), self.labelVectorSize)
                for b in batch
            )
        except AssertionError as e:
            e.add_note(f"First batch sample shape: input={batch[0][0].shape} and label={batch[0][1].shape}")
            e.add_note(f"Last batch sample shape: input={batch[-1][0].shape} and label={batch[-1][1].shape}")
            e.add_note(f"Batch length: {len(batch)}, asked for {len(l_idx)}")
            e.add_note(f"Start file index = {start_file_index}, with j = {j} (1 mean no other file was used)")
            e.add_note(f"Dataset start index: {self.__startIndex} and end index: {self.__endIndex}")
            e.add_note(f"File used: {self.__filenames[start_file_index]}, with a starting index of {self.__indexes[start_file_index]}")
            try:
                e.add_note(f"Next file: {self.__filenames[start_file_index + 1]}, with a starting index of {self.__indexes[start_file_index + 1]}")
            except:
                pass
            for i, b in enumerate(batch):
                if b[0].shape != (self.nbInstrPerSample, self.inputVectorSize) or \
                    b[1].shape != ((self.right_offset - self.left_offset), self.labelVectorSize):
                    e.add_note(f"Wrong shape on index {i}: input={b[0].shape} and label={b[1].shape}")
            raise e
        """
        
        # Return the final batch array
        return batch
    
def __addRandomValue(input: torch.Tensor, label: torch.Tensor, l_offset: int, r_offset: int) -> torch.Tensor:
    if len(input.shape) == 3:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        return torch.randn(batch_size, seq_len, 2, device=input.device), None
    else:
        seq_len = input.shape[0]
        return torch.randn(seq_len, 2, device=input.device), None

def main() -> None:
    torch.set_printoptions(precision=6)
    log = Logger.Logger(verbose_level=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(sys.argv) != 2:
        log.error("Wrong number of parameters, usage: python VectorDataset.py <input folder>", 1)
    folderName = sys.argv[1]
    ds = VectorDataset(folderName, 100, 11, 7, 10, 90, device, log=log, masked_inputs=[3, 5, 6, 7], masked_labels=None, append_input_vectors=[__addRandomValue])
    ds.summary()
    ds.log.title(f"Item access test")
    ds.log.title("Test access of a sample")
    i, l = ds.__getitem__(0)
    ds.log.info(f"Input: {i} of shape {i.shape}\nLabel: {l} of shape {l.shape}")
    i, l = ds.__getitem__(len(ds) - 1)
    ds.log.info(f"Input: {i} of shape {i.shape}\nLabel: {l} of shape {l.shape}")
    one, two = ds.splitDataset(0.8)
    ds.log.title("Splited the dataset at 80%, showing the 2 summaries:")
    one.summary()
    two.summary()
    ds.log.title("Test access of a sample for each")
    i1, l1 = one.__getitem__(len(one) - 1)
    one.log.info(f"Last sample of the first one:\nInput: {i1} of shape {i1.shape}\nLabel: {l1} of shape {l1.shape}")
    i2, l2 = two.__getitem__(0)
    two.log.info(f"First sample of the second one:\nInput: {i2} of shape {i2.shape}\nLabel: {l2} of shape {l2.shape}")
    i2, l2 = two.__getitem__(len(two) - 1)
    two.log.info(f"Last sample of the second one:\nInput: {i2} of shape {i2.shape}\nLabel: {l2} of shape {l2.shape}")

    import torch.utils.data as td
    import torch.multiprocessing as mp
    from TransformerModel import TransformerModel
    mp.set_start_method("spawn")
    from time import time
    one.log.title(f"Dataloader test")
    dl = td.DataLoader(one, num_workers=0, batch_sampler=Utils.BatchSampler(len(ds), 32, Constants.RANDOM_SEED))
    one.log.info(f"Number of batch: {len(dl)}")
    model = TransformerModel(ds.nbInstrPerSample, ds.true_input_vector_size, ds.true_label_vector_size, ds.left_offset, ds.right_offset, device)
    pred_times = torch.zeros(len(dl), device=device)
    access_times = torch.zeros(len(dl), device=device)
    start_time = time()
    temp_time = time()
    one.log.emptyLines()
    for i, (d, l) in enumerate(dl):
        access_times[i] = torch.tensor(time() - temp_time, device=device)
        temp_time = time()
        o = model(d)
        pred_times[i] = torch.tensor(time() - temp_time, device=device)
        temp_time = time()
        if i % 100 == 0:
            one.log.eraseLine()
            one.log.debug(f"Batch {i}/{len(dl)}")
    elapsed = time() - start_time
    one.log.eraseLine()
    mean_access = access_times.mean().item()
    mean_pred = pred_times.mean().item()
    one.log.info(f"Finished iterating through the dataloader, took {Utils.format_time(elapsed)} for {len(dl)} batches.")
    one.log.info(f"Mean batch access time: {Utils.format_time(mean_access)}")
    one.log.info(f"Mean pred time: {Utils.format_time(mean_pred)}")
    
if __name__ == "__main__":
    main()

"""
End of file
"""