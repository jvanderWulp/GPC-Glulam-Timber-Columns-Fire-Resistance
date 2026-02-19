import re
from pathlib import Path
import numpy as np
import pandas as pd

def postprocess_failure(workdir: str | Path, filename: str | Path, t_end: float):
    """
    Read the last TIME entry in a SAFIR .OUT file and return the value in seconds.

    Parameters
    ----------
    workdir : str | Path
        Directory containing the .OUT file.
    filename : str | Path
        Name of the .OUT file.

    Returns
    -------
    float
        The last recorded time in minutes.

    """
    path = Path(workdir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    _TIME_SECONDS_RE = re.compile(r"TIME=\s*([0-9]+(?:\.[0-9]+)?)\s*SECONDS\b", re.IGNORECASE)

    last_seconds: float | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _TIME_SECONDS_RE.search(line)
            if m:
                last_seconds = float(m.group(1))
    if round(last_seconds/60, 0) == round(t_end, 0):
        failure = False
    else:
        failure = True
    
    return round(last_seconds/60, 2), failure

def get_simulation_time(workdir: str | Path, filename: str | Path):
    """
    Read the total calculation duration from a SAFIR .OUT file.

    Parameters
    ----------
    workdir : str | Path
        Directory containing the .OUT file.
    filename : str | Path
        Name of the .OUT file.

    Returns
    -------
    list[float]
        Duration of the simulation [hours, minutes, seconds]
    """
    path = Path(workdir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
        
    pattern = re.compile(
        r"DURATION OF THE CALCULATION\s*:\s*([0-9]+)\s*:\s*([0-9]+)\s*:\s*([0-9]+)",
    re.IGNORECASE
)

    hours = minutes = seconds = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                hours, minutes, seconds = map(int, match.groups())
    return [hours, minutes, seconds]    


def get_stiffness(workdir: str | Path, filename: str | Path, n_elements: int, *,  ng: int = 2):
    """
    Reads a SAFIR structural analysis .OUT file and collect [time_s, EA, ES, EI] for the middle element
    at a given Gauss point.


    Parameters
    ----------
    workdir : str | Path
        Directory containing the .OUT file.
    filename : str | Path
        .OUT file name.
    n_elements : int
        Total number of beam elements along the member.
    gauss_point : int, optional
        Gauss point to extract (default 2).

    Returns
    -------
    list[list[float]]
        Each row [time_seconds, EA, ES, EI], sorted by time.
    """
    if n_elements <= 0:
        raise ValueError("n_elements must be a positive integer.")

    if n_elements % 2 == 1:
        middle_elem = n_elements // 2 + 1
    else:
        middle_elem = n_elements // 2

    path = Path(workdir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    time_re = re.compile(r"TIME\s*=\s*([+-]?\d+(?:\.\d+)?)\s+SECONDS", re.IGNORECASE)
    stiff_hdr_re = re.compile(r"STIFFNESS\s+IN\s+THE\s+BEAM\s+ELEMENTS", re.IGNORECASE)
    num = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
    row_re = re.compile(rf"^\s*(\d+)\s+(\d+)\s+({num})\s+({num})\s+({num})\s*$")

    section_idxs = [i for i, l in enumerate(lines) if stiff_hdr_re.search(l)]
    if not section_idxs:
        raise ValueError("No 'STIFFNESS IN THE BEAM ELEMENTS.' sections found.")

    results = []

    for idx in section_idxs:
        time_seconds = None
        for j in range(idx, -1, -1):
            m = time_re.search(lines[j])
            if m:
                time_seconds = float(m.group(1))
                break
        if time_seconds is None:
            raise ValueError(f"No 'TIME=' line found before stiffness section at line {idx+1}.")

        k = idx + 4
        while k < len(lines):
            if row_re.match(lines[k]):
                break
            if lines[k].strip() == "" and k > idx + 1:
                break
            k += 1

        EA_ES_EI = None

        while k < len(lines):
            line = lines[k]
            if not line.strip():
                break
            mrow = row_re.match(line)
            if not mrow:
                break

            elem = int(mrow.group(1))
            gp   = int(mrow.group(2))
            if elem == middle_elem and gp == ng:
                EA = float(mrow.group(3))
                ES = float(mrow.group(4))
                EI = float(mrow.group(5))
                EA_ES_EI = (EA, ES, EI)
            k += 1

        if EA_ES_EI is None:
            raise ValueError(
                f"Middle element {middle_elem} at Gauss point {ng} not found "
                f"in stiffness block starting at line {idx+1}."
            )

        results.append((time_seconds, *EA_ES_EI))

    results.sort(key=lambda r: r[0])
    return np.array(results)
    
def make_description(workdir, filename, time_thermo, time_mech, time_tot, e0, rho, E, mu, f_c, f_t, w, h_ch, h_cc, eps, B, l, F, T0, t_h, T_max, r_cool, t_end, dt, fine_size, n_elements, failure, failure_time):
    """
    Makes a description file for the SAFIR simulations and stores the file in the same working directory. 

    Parameters:
    ------------
    See simulation notebooks. 

    """
    out_path = Path(workdir) / filename
    lines = []

    lines += ["SAFIR numerical analysis performed", 
        " ", 
        f"{time_thermo[0]}:{time_thermo[1]}:{time_thermo[2]} Time of thermal analysis [h:min:sec]", 
        f"{time_mech[0]}:{time_mech[1]}:{time_mech[2]} Time of structural analysis [h:min:sec]", 
        f"{time_tot[0]}:{time_tot[1]}:{time_tot[2]} Total time of simulation [h:min:sec]", 
        " ", 
        "INPUTS", 
        "----Material Variables----", 
        f"{e0} Eccentricity [m]", 
        f"{rho} Density [kg/m^3]", 
        f"{E} E-modulus [pa]", 
        f"{mu} Poisson ratio [-]", 
        f"{f_c} Compressive strength [pa]", 
        f"{f_t} Tensile strength [Pa]", 
        f"{w} Moisture content [%]", 
        " ", 
        f"{h_ch} Convection coefficient heating [W/m^2 K]", 
        f"{h_cc} Convection coefficient cooling [W/m^2 K]", 
        f"{eps} Relative emissivity [-]", 
        "--------------------"
        " ", 
        "----Structural Variables----", 
        f"{B} Column width [m]", 
        f"{l} Column length [m]", 
        f"{F} Applied force [N]",
        "--------------------",
        " ", 
        "----Fire Variables----", 
        f"{T0} Ambient temperature [°C]", 
        f"{t_h} Heating duration [min]", 
        f"{T_max} Maximum temperature [°C]", 
        f"{r_cool} Cooling rate [°C/min]", 
        f"{t_end} End time of thermal analysis [min]", 
        f"{dt} Time step of heating duration [min]", 
        "--------------------",
        " ", 
        "----Numerical Variables----", 
        f"{fine_size} Size of small element in 2D thermal mesh [m]",
        f"{n_elements} Amount of beam elements of structural analysis [-]",
        "--------------------", 
        " ", 
        " ", 
        " ", 
        " ", 
        "OUTPUTS", 
        f"{failure} Failure [True, False]", 
        f"{failure_time} Failure time [min]", 
    ]

    out_path.write_text("\n".join(lines))

def sum_times(times):
    """
    Sum a list of [h, m, s] time lists.
    """
    total_seconds = 0
    for h, m, s in times:
        total_seconds += h * 3600 + m * 60 + s

    # Convert back to hours, minutes, seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return [int(hours), int(minutes), int(seconds)]

def get_temperatures(workdir, file_name, node_ids, *, name_df: str = None, column_names : list[str] = None):
    """
    Parse a SAFIR thermal .OUT file and return a DataFrame with temperature time series
    for one or more nodes. Stores the created DataFrame in the according working directory. 

    Parameters
    ----------
    workdir : str | Path
        Path in which the thermal .OUT file is located
    file_name : str 
        Name of the thermal .OUT file, including the .OUT extension
    node_ids : int | list[int]
        One node id or a list of node ids to extract.
    name_df : str (Optional)
        Name of the data frame that will be stored in the work directory. Include extension (e.g. .csv), default is 'Node_Temperatures.csv'

    Returns
    -------
    pd.DataFrame
        Columns: ['time_s', <node1>, <node2>, ...], sorted by time.
    """
    if isinstance(node_ids, int):
        node_list = [node_ids]
    else:
        node_list = list(node_ids)

    file_path = Path(workdir) / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", errors="ignore") as f:
        lines = f.readlines()

    time_re = re.compile(r"TIME\s*=\s*([+-]?\d+(?:\.\d+)?)\s+SECONDS", re.IGNORECASE)
    pair_re = re.compile(r"\s*(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")

    section_indices = [i for i, l in enumerate(lines) if "TOTAL TEMPERATURES." in l.upper()]
    if not section_indices:
        raise ValueError("No 'TOTAL TEMPERATURES.' sections found in the file.")

    results = []
    any_node_seen = False

    for idx in section_indices:
        time_seconds = None
        for j in range(idx, -1, -1):
            m = time_re.search(lines[j])
            if m:
                try:
                    time_seconds = float(m.group(1))
                except ValueError:
                    time_seconds = None
                break

        if time_seconds is None:
            raise ValueError(f"No 'TIME=' line found before 'TOTAL TEMPERATURES.' at file line {idx+1}.")

        k = idx + 1
        while k < len(lines):
            if pair_re.findall(lines[k]):
                break
            if lines[k].strip() == "" and k > idx + 1:
                break
            k += 1

        temps = [np.nan] * len(node_list)
        found_any_here = False

        while k < len(lines):
            line = lines[k]
            if not line.strip():
                break
            pairs = pair_re.findall(line)
            if not pairs:
                break
            for n_str, t_str in pairs:
                n = int(n_str)
                if n in node_list:
                    idx_in_list = node_list.index(n)
                    try:
                        temp_val = float(t_str)
                    except ValueError:
                        raise ValueError(f"Could not parse temperature value '{t_str}' at line {k+1}.")
                    temps[idx_in_list] = temp_val
                    found_any_here = True
                    any_node_seen = True
            k += 1

        if found_any_here:
            results.append([time_seconds] + temps)

    if not any_node_seen:
        raise ValueError(f"None of the requested nodes {node_list} found in any 'TOTAL TEMPERATURES.' section.")

    results.sort(key=lambda r: r[0])

    if column_names:
        col_names = ["time_s"] + column_names
    else:
        col_names = ["time_s"] + [f'Node_{n}' for n in node_list]
    df = pd.DataFrame(results, columns=col_names)

    if name_df:
        data_name = name_df
    else:
        data_name = f'Node_Temperatures.csv'
    csv = Path(workdir) / data_name
    df.to_csv(csv, index=False)
    
    return df

def isotherm_times(data, target_temp):
    """
    Return an array of corresponding isotherm depths and time instances, giving the first time each depth reaches. 
    
    Parameters
    ----------
    data : str | Path | DataFrame
        CSV file or DataFrame. First column = time.
        Remaining columns: temperatures with headers like '10 mm', '25 mm', ...
    target_temp : float
        The temperature to find.

    Returns
    -------
    numpy.ndarray of shape (N, 2)
        Columns: time, depth.
    """
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("`data` must be a CSV path or a pandas DataFrame.")

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()

    _depth_re = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*mm\s*$", re.IGNORECASE)
    
    results = [[0, 0]]
    for col in df.columns[1:]:
        m = _depth_re.match(str(col))
        if not m:
            continue
        depth_mm = float(m.group(1))
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()

        idx_equal = np.where(y == target_temp)[0]
        if idx_equal.size:
            results.append([float(t[idx_equal[0]]), depth_mm])
            continue

        hit_time = np.nan
        for i in range(1, len(y)):
            y0, y1 = y[i-1], y[i]
            if not (np.isfinite(y0) and np.isfinite(y1)):
                continue
            if (y0 < target_temp <= y1) or ((y0 - target_temp) * (y1 - target_temp) < 0):
                denom = (y1 - y0)
                if denom == 0:
                    continue
                a = (target_temp - y0) / denom
                hit_time = float(t[i-1] + a * (t[i] - t[i-1]))
                break

        results.append([hit_time, depth_mm])

    arr = np.array(results, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2))
    return arr[np.argsort(arr[:, 1])]
