"""
Python Logger

Color prints messages on the terminal, and exits with error code if asked.
Saves the printed logs to a file in the ./logs/ folder. Log file name is
"log_{executable name}_{date}_{time}.txt"

For debug: debug() and trace()
For informations: notice(), info() and title()
For warnings: alert() and warning()
For errors: error(), emergency() and critical()
Custom message type and verbose level: custom_verbose_level_message()
"""

import sys
import os
from datetime import datetime
import traceback
import threading
from platform import system as platsystem

class Colors:
	"""
	Color characters. If on windows, these values are blank (not supported on windows).
	"""
	def __init__(self) -> None:
		if platsystem() == 'Windows':
			# Windows CMD and PowerShell do not support ANSI escape sequences by default.
			# However, modern versions of Windows 10 support them if enabled.
			# For simplicity, we will use no-op strings for Windows.
			self.reset = ''
			self.bold = ''
			self.red = ''
			self.green = ''
			self.orange = ''
			self.cyan = ''
			self.blue = ''
			self.yellow = ''
			self.gray = ''
			self.purple = ''
		else:
			self.reset = '\033[0m'
			self.bold = '\033[01m'
			self.red = '\033[31m'
			self.green = '\033[32m'
			self.orange = '\033[93m'
			self.cyan = '\033[36m'
			self.blue = '\033[34m'
			self.yellow = '\033[33m'
			self.gray = '\033[90m'
			self.purple = '\033[35m'

class Logger:
	"""
	Python Logger

	Color prints messages on the terminal, and exits with error code if asked.
	Saves the printed logs to a file in the "./logs/" folder if asked to.

	- For debug: debug() and trace()
	- For informations: notice(), info() and title()
	- For warnings: alert() and warning()
	- For errors: error(), emergency() and critical()
	- Custom message type and verbose level: custom_verbose_level_message()

	Parameters:
	- verbose_level:		int, specifies the verbose level wanted for the logs, default is 0 (no verbose), a negative value will reduce the number of logs
	- prints_thread_id:		boolean, specifies if the logger must prints the thread id in the header of each message/ Default is False
	- writesLog: 			boolean, specifies if the logger writes the printed logs in a file in addition to the standart output.
							Default is False
	- logFileWritesDebug	boolean, specifies if the logger writes the debug and trace messages into the log file. Default is False
	- logFile:				string, file name for the log file, if not an absolute path, will be written to the current directory.
							Default is None, which creates the file: "./logs/log_{date}_{time}_{main script name}.log"
	"""

	# ==============================================================================================================
	# Constructor
	# ==============================================================================================================

	def __init__(
			self, *, verbose_level: int = 0, prints_thread_id: bool = False, writesLog: bool = False, logFileWritesDebug: bool = False, logFile: str = None
				) -> None:
		
		# Local variables
		self.__writesLog = writesLog
		self.__logFileWritesDebug = logFileWritesDebug
		self.__verbose_level = verbose_level
		self.__prints_thread_id = prints_thread_id
		self.__colors = Colors()

		# Log file handling: variables setting and checking if the log file is writable
		if self.__writesLog and logFile == None:
			now = datetime.now()
			filename = f"logs/log_{now.strftime('%Y%m%d_%H%M%S')}_{os.path.basename(sys.argv[0]).split('.')[0]}.log"
			if not os.path.isdir("logs/"):
				os.mkdir("logs/")
			try:
				with open(filename, "w+") as f:
					f.close()
				self.__log_file = filename
			except BaseException as e:
				self.__writesLog = False
				self.__log_file = None
				self.warning(f"Failed to open log file, log will not be written to file {filename}", e)
		elif self.__writesLog:
			try:
				with open(logFile, "a+") as f:
					f.close()
				self.__log_file = logFile
			except BaseException as e:
				self.__writesLog = False
				self.__log_file = None
				self.warning(f"Failed to open log file, log will not be written to file {logFile}", e)
		else:
			self.__log_file = None

	# ==============================================================================================================
	# Local functions
	# ==============================================================================================================

	def __printMessage(self, header: str, text: str, color: str, err: BaseException, exit_code: int, verbose_level: int, isDebug: bool = False) -> None:
		"""
		Main printing function, prints the message in the standart output, and writes it to the log file if necessary.

		Parameters:
		- header:			the message type, such as "info", or "warning"
		- text:				the core of the message to print
		- color:			the color to print the message with (no impact on the log file), must be one of red, green, orange, cyan, blue, yellow, gray, purple
		- err:				a python BaseException, if not None, the exception stack will be printed before the message
		- exit_code:		an int, if not None, the program will exit after handling the message with the specified exit code
		- verbose_level:	the verbose level of the message. If inferior to the class verbose level, the message will neither be printer nor written to the log file
		- isDebug:			a boolean used to control the writing to the log file depending on that value and the class value `logFileWritesDebug`
		"""
		# If the verbose level allows for the printing of that message
		if self.__verbose_level >= verbose_level:

			# Get the thread id
			thread_id = threading.get_ident()

			# Color handling, if the asked color does not exist, it is set to green
			try:
				color = getattr(self.__colors, color)
			except:
				color = self.__colors.green

			# Exception formatting
			if err is not None:
				error_trace = ""
				for l in traceback.format_exception(err):
					error_trace += l
			
			# Base header and core text formatting
			base_header = f"[{f'thread-{thread_id}:' if self.__prints_thread_id else ''}{header.upper()}]" if header is not None else ""
			base_text = f"{error_trace if err is not None else ''}{text}"

			# Writing to the standart output
			sys.stdout.write(f"{self.__colors.bold}{color}{base_header}{self.__colors.reset} {color}{base_text}{self.__colors.reset}\n")

			# Checks if the message should be written to the log file
			if self.__writesLog and self.__log_file is not None and header is not None and (not (isDebug and not self.__logFileWritesDebug)):
				now = datetime.now()
				with open(self.__log_file, 'a') as f:
					f.write(f"[{now.strftime('%d/%m/%Y_%H:%M:%S')}]{base_header} {base_text}\n")

			# Exits program if asked
			if exit_code is not None:
				sys.exit(exit_code)

	# ==============================================================================================================
	# Actual logger functions
	# ==============================================================================================================

	def eraseLine(self, verbose_level: int = 0) -> None:
		"""
		Erases the last printed line on the terminal, has no impact on the log file.

		NOTE: has no impact on Windows
		"""
		if platsystem() == "Windows":
			return
		elif self.__verbose_level >= verbose_level:
			sys.stdout.write("\033[F")
			sys.stdout.write("\033[K")


	def emptyLines(self, nbLines: int = 1, verbose_level: int = 0) -> None:
		"""
		Prints the asked number of empty lines, only if the specified verbose level is greater or equal to the Logger verbose level.

		Parameters:
		- nbLines:			an int of the wanted number of empty lines. If negative or zero, will print a single empty line.
		- verbose_level:	an int specifying the verbose level of the empty line to print
		"""
		if nbLines < 1:
			nbLines = 1
		self.__printMessage(None, f"{'\n' * (nbLines - 1)}", None, None, None, verbose_level)
	
	def critical(self, message: str, exit_code: int = 1, err: BaseException = None) -> None:
		"""
		Critical error: critical system fail, program stops here with given exit code.

		Will always print, no matter the verbose level.

		Parameters:
		- message:		printed message
		- exit_code:	exit code, default value is 1
		- err: 			the Exception to parse, None by default (not printed)
		"""
		self.__printMessage("critical", message, "red", err, exit_code, -999999)

	def emergency(self, message: str, exit_code: int = None, err: BaseException = None) -> None:
		"""
		Emergency message: immediate attention needed. Program stops if asked.

		Will always print, no matter the verbose level.

		Parameters:
		- message:		printed message
		- exit_code:	exit code, default value is None (no exit)
		- err: 			the Exception to parse, None by default (not printed)
		"""
		self.__printMessage("emergency", message, "red", err, exit_code, -999999)
	
	def error(self, message: str, exit_code: int = None, err: BaseException = None) -> None:
		"""
		Error message: an error occured. Program stops if asked.

		Will always print, no matter the verbose level.

		Parameters:
		- message:		printed message
		- exit_code:	exit code, default value is None (no exit)
		- err: 			the Exception to parse, None by default (not printed)
		"""
		self.__printMessage("error", message, "red", err, exit_code, -999999)
		
	def warning(self, message: str, err: BaseException = None) -> None:
		"""
		Warning message: potential problem in the program.

		Minimum verbose level for print: -2

		Parameters:
		- message:		printed message
		- err: 			the Exception to parse, None by default (not printed)
		"""
		self.__printMessage("warning", message, "orange", err, None, -2)

	def alert(self, message: str, err: BaseException = None) -> None:
		"""
		Alert message: attention needed.

		Minimum verbose level for print: -1

		Parameters:
		- message:		printed message
		- err: 			the Exception to parse, None by default (not printed)
		"""
		self.__printMessage("alert", message, "yellow", err, None, -1)

	def title(self, message: str) -> None:
		"""
		Title message: informative message, used to separate logs in sections.

		Minimum verbose level for print: -1

		Parameters:
		- message:		printed message
		"""
		self.__printMessage("info", message, "purple", None, None, 0)
		
	def info(self, message: str) -> None:
		"""
		Information message.

		Minimum verbose level for print: 0

		Parameters:
		- message:		printed message
		"""
		self.__printMessage("info", message, "green", None, None, 0)

	def notice(self, message: str) -> None:
		"""
		Notice message: not that important information.

		Minimum verbose level for print: 1

		Parameters:
		- message:		printed message
		"""
		self.__printMessage("notice", message, "gray", None, None, 1)

	def trace(self, message: str) -> None:
		"""
		Trace message: detailed debug message.

		Minimum verbose level for print: 2

		Parameters:
		- message:		printed message
		"""
		self.__printMessage("trace", message, "blue", None, None, 2, isDebug=True)
		
	def debug(self, message: str) -> None:
		"""
		Debug message.

		Minimum verbose level for print: 1

		Parameters:
		- message:		printed message
		"""
		self.__printMessage("debug", message, "cyan", None, None, 1, isDebug=True)

	def custom_message(self, message: str, header: str, color: str = "green", verbose_level: int = 0, err: BaseException = None, exit_code: int = -1):
		"""
		Custom verbose level message.

		Parameters:
		- message:			printed message
		- header:			header of the message, will be displayed as `f"[{HEADER}:{thread_number}] {message}"`
		- color:			a string describing the color of the message. Must be contained in the list:\
			 				red, green, orange, cyan, blue, yellow, gray, purple, if not, will be replaced by green (default).
		- verbose_level:	verbose level of the message, default is 0
		- err:				an exception, if given, its trace will be printed before the message
		- exit_code:		the exit code of the programm, -1 by default (no programm exit)
		"""
		self.__printMessage(header, message, color, err, exit_code, verbose_level)

# ==============================================================================================================
# End of Logger class
# 
# Test function, only executed by directly running this python script
# ==============================================================================================================

def test():
	log = Logger(writesLog=True, verbose_level=2, prints_thread_id=True)
	print("If you see this line, the terminal does not support line erasure.")
	log.eraseLine()
	log.debug("A debug message.")
	print("You also shouldn't see this line of text.")
	log.eraseLine()
	log.trace("A really detailed debug message followed by two empty line.")
	log.emptyLines(2)
	log.notice("Really not that important information.")
	log.info("Informations here.")
	log.title("New section !")
	log.alert("Attention, but not that much.")
	t = threading.Thread(target=log.info, args=["This should print from another thread."])
	t.start()
	log.warning("Potentially problematic, but not critical.")
	t2 = threading.Thread(target=log.custom_message, args=["This should also print from another thread", "custom_type"], kwargs={"color": "orange"})
	t2.start()
	log.error("An error occured, but the code can still execute. Program stops if 'exit_code' is >= 0 (default is -1).")
	log.emergency("DANGER ! Immediate attention needed. Program stops if 'exit_code' is >= 0 (default is -1).")
	try:
		_ = 10 / 0
	except BaseException as e:
		log.critical("System fails, program stops here.", exit_code=2, err=e)
	print("This should not print.")

if __name__ == "__main__":
	test()

"""
End of file
"""