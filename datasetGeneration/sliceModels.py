import sys
import os
import argparse
import configparser
import subprocess
import random
import shutil
from inspect import isfunction
import itertools
import matplotlib.pyplot as plt
import time
import numpy as np
import trimesh
from typing import Any

sys.path.insert(0,"..")
import Logger
import Utils
log = Logger.Logger(verbose_level=2)

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    parser.add_argument('--templates', required=True, type=str, help=f"Template directory, containing the printer config and klipper dict files for the Voron 2.4, and the default slicing config files for Prusa slicer for the Voron 2.4.")
    parser.add_argument('--slice-params', required=True, type=str, help=f"Slicing parameter .ini file specific to Prusa slicer and the Voron 2.4. Must follow the same structure given in the example file: `dataParamFiles/infillRandom_Voron24.ini`")
    parser.add_argument('--slice-section', required=False, default=None, type=str, help=f"Name of the slicing parameter .ini file section to use in addition to the \"general\" section. If not provided, no other section will be used.")
    parser.add_argument('--metadata-file', required=True, type=str, help=f"Path and filename of the metadata file that will be generated with this slicing process.")
    parser.add_argument('--failed-folder', required=True, type=str, help=f"Folder path where the failed slicing logs will be stored along with the corresponding 3D mesh file, sorted by type of error.")
    parser.add_argument('--input-folder', required=True, type=str, help=f"Path to the folder that will contain the resulting G-code files.")
    parser.add_argument('--output-folder', required=True, type=str, help=f"Path to the folder that will contain the resulting G-code files.")

    parser.add_argument('--previousslices', required=False, type=int, default=0, help=f"Because the random number generations are set with a manual seed, slicing batches of the same number of meshes can result in the exact same random choices. Therefore, add this flag with the number of previous slices to ensure new random numbers, while keeping the set seed controlled.")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if not os.path.isfile(args.slice_params):
        log.error(f"The given slicing parameter file does not exist: {args.slice_params}", 1)
    if not os.path.isdir(args.templates):
        log.error(f"The given template folder does not exist: {args.templates}", 1)
    if not os.path.isdir(args.input_folder):
        log.error(f"The given input folder does not exist: {args.input_folder}", 1)
    return args

SPARSE_COEFS: np.ndarray[np.float32] = np.asarray([0.1210847 , 0.09852355, 0.08306319, 0.07180294,
       0.06323422, 0.05649424, 0.05105362, 0.04656945, 0.04280976,
       0.03961204, 0.036859  , 0.03446388, 0.03236114, 0.03050029,
       0.02884186])
SPARSE_TYPES = ["rectilinear", "alignedrectilinear", "grid", "triangles", "stars", "cubic", "line", "adaptivecubic", "supportcubic", "lightning", "zigzag"]

GREEDY_COEFS: np.ndarray[np.float32] = np.asarray([0.360173  , 0.14592595, 0.09297623, 0.06837429, 0.054103  ,
       0.04477206, 0.03819095, 0.03329882, 0.02951885, 0.02651021,
       0.02405853, 0.02202216, 0.02030376, 0.01883424, 0.01756315])
GREEDY_TYPES = ["concentric", "honeycomb", "3dhoneycomb", "gyroid", "hilbertcurve", "archimedeanchords", "octagramspiral"]

MIN_INFILL_DENSITY = 5
MAX_INFILL_DENSITY = 80

NB_PERIMETERS = 2 * (1 + 2) * 10
NOZZLE_DIAMETER = 0.4

def fillrange(min: float, max: float) -> int:
    """
    Returns a random value between min and max, following a linear regressive probability distribution.
    """
    return int(random.triangular(min, max, min))

def fillDensityForPaper(min: int, max: int, infill_type: str) -> float:
    """
    TODO: doc
    """
    if infill_type in SPARSE_TYPES:
        __to_use = SPARSE_COEFS
    elif infill_type in GREEDY_TYPES:
        __to_use = GREEDY_COEFS
    else:
        log.error(f"Could not recognize infill pattern \"{infill_type}\".", 1)
    x_data = np.linspace(min, max, len(__to_use))
    cdf = np.cumsum((__to_use[:-1] + __to_use[1:]) / 2 * np.diff(x_data))
    cdf = np.insert(cdf, 0, 0)
    cdf /= cdf[-1]
    r = random.random()
    return np.interp(r, cdf, x_data)

def exprange(mn: float, mx: float, lbd: float = 0.04) -> int:
    """
    Returns a random value drawn from an exponential distribution between the provided min and max values.
    """
    assert(mn < mx)
    if lbd <= 0:
        return np.random.uniform(mn, mx)
    ea = np.exp(-lbd * mn)
    eb = np.exp(-lbd * mx)
    u = np.random.random()
    val = -np.log(ea - u * (ea - eb)) / lbd
    return int(min(max(val, mn), mx)) # Clamp to min and max just for safety

local_functions = {name: obj for name, obj in locals().items() if isfunction(obj)}

ERROR_PATTERNS: dict[str: str] = {
    "no_extrusions_first_layer": "There is an object with no extrusions in the first layer",
    "empty_print": "The print is empty. The model is not printable with current print settings",
    "no_layers_detected": "No layers were detected",
    "empty_file": "The input is an empty file",
    "model_loading_failed": "Loading of a model file failed",
    "unsupported_feature": "Unsupported G-code feature",
    "zero_size_object": "Object has zero size",
    "out_of_bed_bounds": "outside the print bed",
    "bad_resolution": "bad resolution",
    "too_thin": "is too thin and will not be printed",
    "missing_model": "No objects to slice",
    "outside_print_volume": "All objects are outside of the print volume"
}

def dummyRandoms(numberOfSlices: int) -> None:
    for _ in range(numberOfSlices):
        _ = random.random()
        _ = random.choice(SPARSE_TYPES)
        _ = random.random()

def classify_error(content: str) -> str:
    for key, pattern in ERROR_PATTERNS.items():
        if pattern.lower() in content.lower():
            return key
    return "unknown_error"

def boundingBoxProcessing(file_path: str, max_bounding_box: list[int, int, int]) -> tuple[float, str, tuple[float, float, float, float]]:
    """
    TODO: doc
    """
    if not os.path.isfile(file_path):
        log.error(f"The provided file name does not exist.", 1)
    mesh = trimesh.load_mesh(file_path)
    bounding_box: np.ndarray = mesh.bounding_box.extents
    ratio = np.max(bounding_box) / np.min(bounding_box)
    if bounding_box[0] > max_bounding_box[0] or bounding_box[1] > max_bounding_box[1] or bounding_box[2] > max_bounding_box[2]:
        return (-1, f"Scaled down the model to fit a bounding box of {max_bounding_box[0]}mm {max_bounding_box[1]}mm {max_bounding_box[2]}mm", (max_bounding_box[0], max_bounding_box[1], max_bounding_box[2], ratio))
    elif np.min(bounding_box) < NOZZLE_DIAMETER * NB_PERIMETERS:
        scale_factor = (NOZZLE_DIAMETER * NB_PERIMETERS) / np.min(bounding_box)
        if any([bb * scale_factor > max_bounding_box[i] for i, bb in enumerate(bounding_box)]):
            # raise ValueError(f"The mesh size is too big after a scale up to allow for print")
            pass
        new_bounding_box = bounding_box * scale_factor
        # Removed the scale up of models considered too small
        return (0, "", (bounding_box[0], bounding_box[1], bounding_box[2], ratio))
    else:
        return (0, "", (bounding_box[0], bounding_box[1], bounding_box[2], ratio))
    
def fmt_angle(a: float) -> str:
    """
    TODO: doc
    """
    return f'{a:.1f}'

def matrix_to_prusa_flags(
            R: tuple[np.ndarray[Any, np.dtype[np.floating]], Any] | np.ndarray[Any, np.dtype[np.floating]]
        ) -> tuple[tuple[str, str, str], str]:
    """
    TODO: doc
    """
    R = np.array(R[:3,:3], dtype=float)

    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    x_deg = np.degrees(x)
    y_deg = np.degrees(y)
    z_deg = np.degrees(z)

    return ((
        f"--rotate-x", f"{fmt_angle(x_deg)}",
        f"--rotate-y", f"{fmt_angle(y_deg)}",
        f"--rotate", f"{fmt_angle(z_deg)}"
    ), f"Rotated the mesh by {x_deg}° on the X axis {y_deg}° on the Y axis and {z_deg}° on the Z axis")
    
def findRotation(file_path: str, face_index: int = 0, degree_tolerance: int = 5) -> tuple[tuple[str], str]:
    """
    TODO: doc
    """
    if not os.path.isfile(file_path):
        log.error(f"The provided file name does not exist.", 1)
    mesh = trimesh.load_mesh(file_path)
    hull = mesh.convex_hull
    areas = hull.area_faces
    normals = hull.face_normals
    cos_tol = np.cos(np.radians(degree_tolerance))
    GROUND = np.array([0, 0, 1])
    valid = []
    for i, normal in enumerate(normals):
        if abs(np.dot(normal, GROUND)) < cos_tol:
            valid.append(i)
    valid = sorted(valid, key=lambda i: areas[i], reverse=True)

    # May throw IndexError, needs to be catched
    selected = valid[face_index]

    normal = hull.face_normals[selected]
    R = trimesh.geometry.align_vectors(normal, GROUND)
    return matrix_to_prusa_flags(R)
    
GCODE_TOO_SMALL_THRESHOLD = 2000
    
def getNbInstrGcode(file_path: str) -> int:
    """
    TODO: doc
    """
    result = subprocess.run(["grep", "-Ec", "^(G0|G1)", file_path], check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.strip() or 0)
    else:
        log.error(f"Failed to read the number of instruction for file {file_path}.", 1)
        return -1

def main() -> None:
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    if not os.path.isfile(os.path.expanduser("~/bin/PrusaSlicer/build/src/prusa-slicer")):
        log.error(f"Could not find the Prusa slicer executable. It should be located at '~/bin/PrusaSlicer/build/src/prusa-slicer'.", 1)

    random.seed(12345)

    dummyRandoms(args.previousslices)

    material_ini_files = {}
    for root, _, files in os.walk(os.path.join(args.templates, "materials/")):
        for file in files:
            if file.endswith(".ini"):
                material_ini_files[os.path.splitext(file)[0]] = os.path.join(root, file)
    if len(material_ini_files.keys()) == 0:
        log.warning(f"Could not find any material INI file in the template folder.")
    else:
        log.info(f"Found the following material config files in the template directory:\n{'\n'.join(material_ini_files.values())}")

    data_param = configparser.ConfigParser(interpolation=None)
    data_param.optionxform=str
    data_param.read(f"{args.slice_params}")

    sections = data_param.sections()
    if "general" not in sections:
        log.error(f"Could not find a 'general' section in the given slicing parameter file: {args.slice_params}", 1)

    try:
        printer_name = data_param["printer"]["name"]
    except:
        log.error(f"Could not find a printer name in file {args.slice_params}, it should be in section 'printer', in field 'name'.", 1)
    log.debug(f"Printer name: {printer_name}")

    try:
        max_bounding_box = [int(v) for v in data_param["printer"]["bounding_box"].split(",")]
        assert(len(max_bounding_box) == 3)
    except BaseException as e:
        log.error(f"Could not find a bounding box specification in file {args.slice_params}, it should be in section 'printer', in field 'bounding_box', using the following syntax: \"bounding_box = maxX,maxY,maxZ\". Note that this bounding box should be slightly smaller than the printer size, to allow the slicer to generate a skirt.", 1, e)

    if args.slice_section is not None and args.slice_section not in sections:
        log.error(f"Could not find the given optionnal slicing parameter section \"{args.slice_section}\" in the {args.slice_params} file.", 1)

    header_footer_file = os.path.join(args.templates, "prusaPrintHeadersFooters/", f"{printer_name}.ini")
    default_slicing_param_file = os.path.join(args.templates, f"{printer_name}/", "PrusaConfig.ini")
    printer_cfg_file = os.path.join(args.templates, f"{printer_name}/", f"{printer_name}.cfg")
    printer_klipper_dict_file = os.path.join(args.templates, f"{printer_name}/", f"{printer_name}_klipper.dict")
    if not os.path.isfile(header_footer_file):
        log.error(f"Could not find a G-code header footer file for printer {printer_name}, in the template folder: {args.templates}/prusaPrintHeadersFooters/", 1)
    if not os.path.isfile(default_slicing_param_file):
        log.error(f"Could not find a Prusa slicer slicing configuration file for printer {printer_name}, in the template folder: {args.templates}", 1)
    if not os.path.isfile(printer_cfg_file):
        log.error(f"Could not find a Klipper configuration file for printer {printer_name}, in the template folder: {args.templates}", 1)
    if not os.path.isfile(printer_klipper_dict_file):
        log.error(f"Could not find a Klipper dict file for printer {printer_name}, in the template folder: {args.templates}", 1)

    if os.path.isdir(args.metadata_file):
        log.error(f"The provided metadata file path ({args.metadata_file}) should be a file path and not a directory.", 1)
    
    if args.metadata_file != "/dev/null" and not args.metadata_file.endswith(".csv"):
        log.error(f"The metadata file path should end in '.csv', currently: {args.metadata_file}", 1)

    path, _ = os.path.split(args.metadata_file)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    metadata_file_path = Utils.get_next_versioned_filename(args.metadata_file)
    if not os.path.isdir(args.failed_folder):
        os.makedirs(args.failed_folder)
    os.makedirs(os.path.join(args.failed_folder, "meshes/"), exist_ok=True)
    failed_folder_path = args.failed_folder
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    output_folder_path = args.output_folder

    mesh_files = [os.path.join(root, file) for root, _, files in os.walk(args.input_folder) for file in files if file.endswith((".stl", ".obj", "3mf"))]

    constant_slicing_params = {f"--{k.replace("_", "-")}": data_param["general"].get(k, fallback="") for k in data_param["general"].keys()}

    if args.slice_section is not None:
        optional_params_constant = {f"--{k.replace("_", "-")}": data_param[args.slice_section].get(k, fallback="") for k in data_param[args.slice_section].keys() if not data_param[args.slice_section].get(k, fallback="").startswith(("funcrange", "range", "random", "expand", "DrawTypeFrom"))}
        optional_params_random = {f"--{k.replace("_", "-")}": data_param[args.slice_section].get(k, fallback="").split(":")[1:] for k in data_param[args.slice_section].keys() if data_param[args.slice_section].get(k, fallback="").startswith("random")}
        optional_params_baserange = {f"--{k.replace("_", "-")}": data_param[args.slice_section].get(k, fallback="").split(":")[1:] for k in data_param[args.slice_section].keys() if data_param[args.slice_section].get(k, fallback="").startswith("range")}
        optional_params_customrange = {f"--{k.replace("_", "-")}": (data_param[args.slice_section].get(k, fallback="").split(":")[0].split("_")[1], data_param[args.slice_section].get(k, fallback="").split(":")[1:]) for k in data_param[args.slice_section].keys() if data_param[args.slice_section].get(k, fallback="").startswith("funcrange")}
        optional_params_expands = {f"--{k.replace("_", "-")}": data_param[args.slice_section].get(k, fallback="").split(":")[1:] for k in data_param[args.slice_section].keys() if data_param[args.slice_section].get(k, fallback="").startswith("expand")}
        __draw_density_from_type = (data_param[args.slice_section].get("fill_density", fallback="") == "DrawDensityFromType")
        __draw_types_from = data_param[args.slice_section].get("fill_pattern", fallback="").startswith("DrawTypeFrom")
    else:
        optional_params_constant = {}
        optional_params_random = {}
        optional_params_baserange = {}
        optional_params_customrange = {}
        optional_params_expands = {}
        __draw_density_from_type = False
        __draw_types_from = False

    expanding_params = list(itertools.product(*[[(k, v) for v in values] for k, values in optional_params_expands.items()]))
    
    if len(expanding_params) == 1 and not expanding_params[0]:
        expanding_params = [(("", ""))]

    log.info(f"Found a total of {len(mesh_files)} 3D mesh files inside the folder {args.input_folder}. Each will be sliced a total of {len(expanding_params)}.")

    # Metadata file initialization
    with open(metadata_file_path, 'w+') as mdf:
        mdf.write(f"3D mesh name,3D mesh size (bytes),Bounding box X (mm),Bounding box Y (mm),Bounding box Z (mm),Bounding box ratio,Slice time (s),Infill type,Infill rotation (°),Infill density (%),Material,Slicing comment,G-code file name,G-code file size (bytes)\n")

    always_there_flags = ["--use-relative-e-distances", "--support-material=0", "--gcode-flavor", "klipper", "--machine-limits-usage", "ignore"]

    infill_type = ""
    infill_density = ""
    infill_rotation = ""
    material = ""

    nb_success = 0
    nb_fails = 0

    retained_densities = []
    retained_patterns: dict[str, int] = {}

    __dont_add = False
    for k, v in constant_slicing_params.items():
        if k == "--fill-pattern":
            if retained_patterns.get(v) is None:
                retained_patterns[v] = 1
            else:
                retained_patterns[v] += 1
            infill_type = v
        elif k == "--fill-density":
            if not __draw_density_from_type:
                retained_densities.append(int(v.replace("%", "")))
                infill_density = v
                v = f"{v}%"
            else:
                __dont_add = True
        elif k == "--fill-angle":
            infill_rotation = v
        elif k == "--filament-type":
            material = v
        elif k == "--material":
            material = v
            k = "--load"
            try:
                v = material_ini_files[v]
            except KeyError as e:
                log.error(f"Could not find the material INI file named \"{v}.ini\" inside the given template directory.", 1)
        if not __dont_add:
            always_there_flags.extend([k, v])
        __dont_add = False
    for k, v in optional_params_constant.items():
        if k == "--fill-pattern":
            if retained_patterns.get(v) is None:
                retained_patterns[v] = 1
            else:
                retained_patterns[v] += 1
            infill_type = v
        elif k == "--fill-density":
            if not __draw_density_from_type:
                retained_densities.append(int(v.replace("%", "")))
                infill_density = v
                v = f"{v}%"
            else:
                __dont_add = True
        elif k == "--fill-angle":
            infill_rotation = v
        elif k == "--filament-type":
            material = v
        elif k == "--material":
            material = v
            k = "--load"
            try:
                v = material_ini_files[v]
            except KeyError as e:
                log.error(f"Could not find the material INI file named \"{v}.ini\" inside the given template directory.", 1)
        if not __dont_add:
            always_there_flags.extend([k, v])
        __dont_add = False

    nb_slice = 1
    numberTooSmall = 0
    # Main loop
    main_time = time.time()
    log.emptyLines(verbose_level=1)
    for i, mesh in enumerate(mesh_files):
        for j, comb in enumerate(expanding_params):
            if comb[0] != "":
                output_file = os.path.join(output_folder_path, f"{os.path.splitext(os.path.split(mesh)[1])[0]}{'_' if comb[0][0] != '' else ''}{'_'.join([f"{key.replace('-', '')}-{v.replace('%', '')}" for key, v in comb])}.gcode")
            else:
                output_file = os.path.join(output_folder_path, f"{os.path.splitext(os.path.split(mesh)[1])[0]}.gcode")

            flags = [os.path.expanduser("~/bin/PrusaSlicer/build/src/prusa-slicer")]
            flags.extend(always_there_flags)
            flags.extend(["--export-gcode", "-o", f"{output_file}", "--load", default_slicing_param_file, "--load", header_footer_file])

            # Slicing parameter flag parsing from config file
            __dont_add = False
            for k, v in optional_params_random.items():
                value = random.choice(v)
                if k == "--fill-pattern":
                    if retained_patterns.get(value) is None:
                        retained_patterns[value] = 1
                    else:
                        retained_patterns[value] += 1
                    infill_type = value
                elif k == "--fill-density":
                    if not __draw_density_from_type:
                        retained_densities.append(int(value.replace("%", "")))
                        infill_density = value
                        value = f"{value}%"
                    else:
                        __dont_add = True
                elif k == "--fill-angle":
                    infill_rotation = value
                elif k == "--filament-type":
                    material = value
                elif k == "--material":
                    material = value
                    k = "--load"
                    try:
                        value = material_ini_files[value]
                    except KeyError as e:
                        log.error(f"Could not find the material INI file named \"{value}.ini\" inside the given template directory.", 1)
                if not __dont_add:
                    flags.extend([k, value])
                __dont_add = False
            for k, v in optional_params_baserange.items():
                value = str(int(random.uniform(float(v[0]), float(v[1]))))
                if k == "--fill-pattern":
                    if retained_patterns.get(value) is None:
                        retained_patterns[value] = 1
                    else:
                        retained_patterns[value] += 1
                    infill_type = value
                elif k == "--fill-density":
                    if not __draw_density_from_type:
                        retained_densities.append(int(value.replace("%", "")))
                        infill_density = value
                        value = f"{value}%"
                    else:
                        __dont_add = True
                elif k == "--fill-angle":
                    infill_rotation = value
                elif k == "--filament-type":
                    material = value
                elif k == "--material":
                    material = value
                    k = "--load"
                    try:
                        value = material_ini_files[value]
                    except KeyError as e:
                        log.error(f"Could not find the material INI file named \"{value}.ini\" inside the given template directory.", 1)
                if not __dont_add:
                    flags.extend([k, value])
                __dont_add = False
            for k, v in optional_params_customrange.items():
                try:
                    func = local_functions[v[0]]
                except KeyError:
                    log.error(f"Problem detected in the slicing configuration file {args.slice_params}, in section {args.slice_section}, for field {k}: could not find a local function called {v[0]}.", 1)
                value = str(func(float(v[1][0]), float(v[1][1])))
                if k == "--fill-pattern":
                    if retained_patterns.get(value) is None:
                        retained_patterns[value] = 1
                    else:
                        retained_patterns[value] += 1
                    infill_type = value
                elif k == "--fill-density":
                    if not __draw_density_from_type:
                        retained_densities.append(int(value.replace("%", "")))
                        infill_density = value
                        value = f"{value}%"
                    else:
                        __dont_add = True
                elif k == "--fill-angle":
                    infill_rotation = value
                elif k == "--filament-type":
                    material = value
                elif k == "--material":
                    material = value
                    k = "--load"
                    try:
                        value = material_ini_files[value]
                    except KeyError as e:
                        log.error(f"Could not find the material INI file named \"{value}.ini\" inside the given template directory.", 1)
                if not __dont_add:
                    flags.extend([k, value])
                __dont_add = False
            if comb[0] != "":
                for k, v in comb:
                    if k == "--fill-pattern":
                        if retained_patterns.get(v) is None:
                            retained_patterns[v] = 1
                        else:
                            retained_patterns[v] += 1
                        infill_type = v
                    elif k == "--fill-density":
                        if not __draw_density_from_type:
                            retained_densities.append(int(v.replace("%", "")))
                            infill_density = v
                            v = f"{v}%"
                        else:
                            __dont_add = True
                    elif k == "--fill-angle":
                        infill_rotation = v
                    elif k == "--filament-type":
                        material = v
                    elif k == "--material":
                        material = v
                        k = "--load"
                        try:
                            v = material_ini_files[v]
                        except KeyError as e:
                            log.error(f"Could not find the material INI file named \"{v}.ini\" inside the given template directory.", 1)
                    if not __dont_add:
                        flags.extend([k, v])
                    __dont_add = False

            if __draw_types_from:
                minAndMax = list(map(int, data_param[args.slice_section].get("fill_pattern", fallback="").split(":")[1:]))
                assert(len(minAndMax) == 2 and minAndMax[0] + minAndMax[1] == 100)
                r = int(random.random() * 100)
                if r < minAndMax[0]:
                    choice = random.choice(SPARSE_TYPES)
                else:
                    choice = random.choice(GREEDY_TYPES)
                if retained_patterns.get(choice) is None:
                    retained_patterns[choice] = 1
                else:
                    retained_patterns[choice] += 1
                infill_type = choice
                flags.extend(["--fill-pattern", choice])
            if __draw_density_from_type:
                density = int(fillDensityForPaper(MIN_INFILL_DENSITY, MAX_INFILL_DENSITY, infill_type))
                retained_densities.append(density)
                infill_density = str(density)
                flags.extend(["--fill-density", f"{density}%"])

            try:
                scale_factor, message, bounding_box = boundingBoxProcessing(mesh, max_bounding_box)
            except ValueError:
                nb_fails += 1
                log.eraseLine(verbose_level=1)
                log.warning(f"For file {mesh}, the shape of the mesh does not permit to scale up the model to allow for print (would become too large for printer)")
                log.emptyLines(verbose_level=1)
                shutil.copy(mesh, os.path.join(failed_folder_path, "meshes/"))
                retained_densities.pop()
                retained_patterns[infill_type] -= 1
                continue
            except BaseException as e:
                nb_fails += 1
                log.eraseLine(verbose_level=1)
                log.warning(f"Failed to slice file {mesh}", e)
                log.emptyLines(verbose_level=1)
                shutil.copy(mesh, os.path.join(failed_folder_path, "meshes/"))
                retained_densities.pop()
                retained_patterns[infill_type] -= 1
                continue

            if scale_factor == -1:
                flags.extend(["--scale-to-fit", f"{','.join([str(bd) for bd in max_bounding_box])}"])
            elif scale_factor != 0:
                flags.extend(["--scale", f"{scale_factor:.1f}"])

            flags.append(mesh)

            log.eraseLine(verbose_level=1)
            log.debug(f"Slicing file {nb_slice}/{len(mesh_files) * len(expanding_params)}: {os.path.split(output_file)[1]}")
            nb_slice += 1
            start_time = time.time()

            # Actual slicing
            result = subprocess.run(flags, check=False, capture_output=True, text=True)
            slice_time = time.time() - start_time
            error_log = result.stderr.strip()

            # Slicing failed
            if result.returncode != 0 or not os.path.isfile(output_file):

                rotation_count = 0
                while rotation_count < 4 and (result.returncode != 0 or not os.path.isfile(output_file)):

                    # Try to find a mesh rotation that allows the slice
                    try:
                        rotation_flags, messageBis = findRotation(mesh, face_index=rotation_count)
                    except BaseException:
                        break
                    else:
                        if os.path.isfile(output_file):
                            os.remove(output_file)
                        flags.pop()
                        flags.extend([rf for rf in rotation_flags])
                        flags.append(mesh)

                        start_time = time.time()
                        result = subprocess.run(flags, check=False, capture_output=True, text=True)
                        slice_time = time.time() - start_time
                        error_log = result.stderr.strip()
                    rotation_count += 1

                # If the slicing still fails, mark the file as a failed slice
                if result.returncode != 0 or not os.path.isfile(output_file):
                    log.eraseLine(verbose_level=1)
                    log.warning(f"Failed to slice the file {mesh} using the default parameters and after trying 5 rotations.")
                    log.emptyLines(verbose_level=1)
                    with open(f"{os.path.join(failed_folder_path, f"{os.path.splitext(os.path.split(output_file)[1])[0]}.stderr")}", "w+") as stef:
                        stef.write(error_log)
                    shutil.copy(mesh, os.path.join(failed_folder_path, "meshes/"))
                    nb_fails += 1
                    retained_densities.pop()
                    retained_patterns[infill_type] -= 1
                else:
                    # Successfully re-sliced the mesh, add the log to the result csv file
                    nb_success += 1
                    with open(metadata_file_path, 'a') as mdf:
                        mdf.write(f"{os.path.split(mesh)[1]},{os.path.getsize(mesh)},{','.join([f'{val:.2f}' for val in bounding_box])},{slice_time},{infill_type},{infill_rotation},{infill_density.replace("%", "")},{material},{messageBis if message == "" else f'{message} and {messageBis}'},{os.path.split(output_file)[1]},{os.path.getsize(output_file)}\n")
                    if getNbInstrGcode(output_file) < GCODE_TOO_SMALL_THRESHOLD:
                        log.eraseLine(verbose_level=1)
                        log.warning(f"The sliced G-code of file {mesh} is small (less than {GCODE_TOO_SMALL_THRESHOLD} G0 and G1 instructions).")
                        log.emptyLines(verbose_level=1)
                        numberTooSmall += 1
            else:
                # Successfully sliced the mesh, but check if it is too small, if yes delete the resulting G-code
                if getNbInstrGcode(output_file) < GCODE_TOO_SMALL_THRESHOLD:
                    log.eraseLine(verbose_level=1)
                    log.warning(f"The sliced G-code of file {mesh} is small (less than {GCODE_TOO_SMALL_THRESHOLD} G0 and G1 instructions).")
                    log.emptyLines(verbose_level=1)
                    numberTooSmall += 1
                    # shutil.copy(mesh, os.path.join(failed_folder_path, "too_small_meshes"))
                    # os.remove(output_file)
                    # retained_densities.pop()
                    # retained_patterns[infill_type] -= 1
                # Actual success of the slice, no errors detected. Write the csv file
                nb_success += 1
                with open(metadata_file_path, 'a') as mdf:
                    mdf.write(f"{os.path.split(mesh)[1]},{os.path.getsize(mesh)},{','.join([f'{val:.2f}' for val in bounding_box])},{slice_time},{infill_type},{infill_rotation},{infill_density.replace("%", "")},{material},{'Normal slice' if message == "" else message},{os.path.split(output_file)[1]},{os.path.getsize(output_file)}\n")

    total_time = time.time() - main_time

    log.eraseLine(verbose_level=1)
    log.info(f"Done slicing all {len(mesh_files)} files, {nb_success} with success and {nb_fails} failed.")
    log.info(f"Took a total time of {Utils.format_time(total_time)}, and {numberTooSmall} meshes were flagged as too small.")

    shutil.copy(printer_cfg_file, output_folder_path)
    shutil.copy(printer_klipper_dict_file, output_folder_path)

    # Plotting the infill densities and infill types that were used

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bin_width = 5.0
    bins = np.arange(0, 100 + bin_width, bin_width)
    counts, edges = np.histogram(retained_densities, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    retained_densities.sort()
    ax1.bar(bin_centers, counts, width=bin_width * 0.9, edgecolor="black", align="center")
    ax1.set_xlabel("Infill density (%)")
    ax1.set_ylabel("Number of sliced meshes")
    ax1.set_xlim([0, 100])
    ax1.grid(True)
    fig1.tight_layout(pad=0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    patterns = retained_patterns.keys()
    values = [retained_patterns[k] for k in patterns]
    ax2.bar(patterns, values, [0.5 for i in range(len(values))])
    ax2.set_xlabel("Infill patterns")
    ax2.set_xticks([i for i in range(len(values))])
    ax2.set_xticklabels(patterns, rotation=45)
    ax2.set_ylabel("Number of sliced meshes")
    ax2.grid(True)
    fig2.tight_layout(pad=0)
    plt.show()

if __name__ == "__main__":
    main()

"""
End of file.
"""