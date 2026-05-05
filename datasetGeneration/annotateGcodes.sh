#!/bin/bash

usage() {
	echo -e "Usage: $1 [-m max number of background processes, default is 10] <input folder>" >&2
    cat >&2 <<-EOF

This script uses TimeKlip to process the print time of a folder containing potentially many G-code files, it is considered that TimeKlip is installed at: "~/timeklip/bin/timeklip"

The input folder's structure must be the following: "<input folder>/slicerName/printerName/slicingConfiguration/files.gcode",
potentially with several slicers, printers, and slicing configurations.

The names of the "slicingConfiguration" folders must satisfy the following syntax: "paramName-Value_paramName-Value" 

For each printer, the corresponding Klipper configuration file and klipper instruction dictionnary must be located inside the folder
"<input folder>/slicerName/printerName/<printerName.cfg and klipper.dict>".

NOTE: TimeKlip is considered to be installed locally instead of using docker installation for this script.

EOF
}

max_process=10

# Arguments definition
while getopts m:h opt; do
	case $opt in
    m) max_process="$OPTARG" ;;
    h) usage $0 ;
       exit 0 ;;
	*) usage $0 ;
       exit 1 ;;
 	esac
done

shift $(($OPTIND-1))

# Arguments parsing
if [ ! $# -eq 1 ]; then
    echo -e "[ERROR] Wrong  number of parameters." >&2
	usage $0
    exit 1
fi
if [[ ! -d $1 ]]; then
    echo -e "[ERROR] The provided input folder does not exist." >&2
    exit 1
fi
if [[ ! -f $HOME/timeklip/bin/timeklip ]]; then
    echo -e "[ERROR] Could not find the TimeKlip executable in '$HOME/timeklip/bin/timeklip'." >&2
    exit 1
fi
TK_PATH="$(dirname -- $HOME/timeklip/bin/timeklip)"
TK_PATH="$(cd -- "$TK_PATH" && pwd)"
if [[ -z "$TK_PATH" ]]; then
    echo -e "[ERROR] Could not get acces to TimeKlip executable path ('$TK_PATH')." >&2
    exit 1
fi

lock_csv="./tmp/.lock.csv"
lock_log="./tmp/.lock.log"
output_csv="./TimeKlipTimes.csv"
fail_log="./failedToAnnotate.txt"
progress_pipe="./tmp/.progress.pipe"

echo -e "G-code file name,Number of G0 and G1,Processing time (s),Print time (s)" > $output_csv
> "$fail_log"
rm -f "$progress_pipe"
mkfifo "$progress_pipe"

# Background progress monitor
(
  success=0
  fail=0
  while read line; do
    [[ $line == success* ]] && ((success++))
    [[ $line == fail* ]] && ((fail++))
    echo -en "\r\033[K[INFO] Successfully annotated $success files, failed $fail"
  done < "$progress_pipe"
) &

progress_pid=$!

# Actual annotation function, calls TimeKlip and creates a CSV file for stats, for a singular G-code
annotateOne() {
    # Parameters definition
    printer_config=$1
    klipper_dict=$2
    gcode_file=$3

    # Temp file name definition
    tmpfilename="./tmp/$(basename "$gcode_file" .gcode)_logs"

    # TimeKlip output file file name
    outputPath="${gcode_file%.gcode}_timeklip.gcode"

    # Actual TimeKlip call, and processing time measurment
    start_single=$(date +%s.%N)
    # Actual TimeKlip execution
    $HOME/timeklip/bin/timeklip -k 13 -d $klipper_dict -c $printer_config -i $gcode_file -o $outputPath > $tmpfilename.txt 2>&1
    success=$?
    end_single=$(date +%s.%N)
    processing_single=$(echo "$end_single - $start_single" | bc)
    processing_single=$(printf "%.2f" "$processing_single")

    # If TimeKlip execution was a success
    if [[ $success -eq 0 ]] && [[ -f $outputPath ]]; then
        # Replace old G-code with the new annotated one
        mv $outputPath $gcode_file > /dev/null 2>&1
        
        result=$(<"$tmpfilename.txt")
        printtime=$(echo "$result" | grep 'Print duration' | awk -F ': ' '{print $NF}')
        echo -n "$(basename "$gcode_file"),$(grep -E -c '^G[01]' "$gcode_file"),${processing_single},${printtime}"

        # Zip and delete old files
        tar -czf "${gcode_file%.gcode}.tgz" -C "$(dirname "$gcode_file")" "$(basename "$gcode_file")" > /dev/null 2>&1
        (( $? == 0 )) && rm -f "$gcode_file"
        rm -f "$tmpfilename.txt"
        return 0
    else
        # If TimeKlip failed, delete output file just in case, and keep the log file
        yes | rm $outputPath > /dev/null 2>&1
        return 1
    fi
}

export -f annotateOne

annotate_wrapper() {
    file="$1"
    output_csv="$2"
    fail_log="$3"
    lock_csv="$4"
    lock_log="$5"
    progress_pipe="$6"

    printerFolder=$(dirname "$(dirname "$file")")
    printer=$(basename "$printerFolder")
    slicer=$(basename "$(dirname "$printerFolder")")
    printer_config_file=$(find "$(dirname "$file")" -type f -name "*.cfg" -print -quit)
    klipper_dict_file=$(find "$(dirname "$file")" -type f -name "*.dict" -print -quit)

    if [[ ! -f $printer_config_file ]] || [[ ! -f $klipper_dict_file ]]; then
        echo "fail" > "$progress_pipe"
        exit 0
    fi

    csv_line=$(annotateOne "$printer_config_file" "$klipper_dict_file" "$file")
    if [[ $? -eq 0 && -n "$csv_line" ]]; then
        (
          flock -x 201
          echo "$csv_line" >> "$output_csv"
        ) 201>"$lock_csv"
        echo "success" > "$progress_pipe"
    else
        (
          flock -x 200
          echo "$file" >> "$fail_log"
        ) 200>"$lock_log"
        echo "fail" > "$progress_pipe"
    fi
}

export -f annotate_wrapper

exec 3>"$progress_pipe"

# Main loop
start_time=$(date +%s.%N)
find "$1" -type f -name "*.gcode" | \
  xargs -P "$max_process" -I {} bash -c 'annotate_wrapper "$@"' _ \
  {} "$output_csv" "$fail_log" "$lock_csv" "$lock_log" "$progress_pipe"

exec 3>&-
wait "$progress_pid"
rm "$progress_pipe"
end_time=$(date +%s.%N)
processing_time=$(echo "$end_time - $start_time" | bc)
echo -en "\r\033[K"
echo -e "[INFO] Finished annotating the files."
echo -e "[INFO] Wrote successful annotations into ./TimeKlipTimes.csv and failed ones into ./failedToAnnotate.txt"
echo -e "[INFO] The total annotation time was $processing_time s.\n"