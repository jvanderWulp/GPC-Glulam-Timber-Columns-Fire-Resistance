import re
from pathlib import Path
import numpy as np

import running as rn

def thermal_restart(
    workdir,
    firstfile,
    B,
    fine_size,
    *,
    max_restarts=24,
    extend_step_s=3600,
    T0=20.0,
    show_output: bool = True,
):
    """
    Perform SAFIR thermal restarts until a negative temperature derivative is detected
    for the specified node_id, or until max_restarts is reached.

    Parameters
    ----------
    workdir : str | Path
        Directory containing the initial <firstfile>.IN/.OUT and where restarts will be created.
    firstfile : str
        Basename of the initial run.
    node_id : int
        Node to inspect for temperature derivatives.
    max_restarts : int, default 24
        Maximum number of restart attempts
    extend_step_s : float, default 3600
        Extra seconds to extend in each restart relative to the previous run's last printed time.

    Returns
    -------
    dict
        {
          "found_negative": bool,
          "runs": int,                         # total number of thermal runs executed (incl. initial)
          "last_out": str,                     # path to the .OUT of the last run executed
          "last_time": float,                  # last printed time [s] in that .OUT
          "filename_history": [str, ...],      # list of input filenames executed (order)
        }
    """
    workdir = Path(workdir)
    stem = Path(firstfile).stem

    first_out = workdir / f"{stem}.OUT"
    if not first_out.exists():
        raise FileNotFoundError(f"Initial OUT file not found: {first_out}")

    # k = B / (4 * fine_size)
    # node_id = ((k + (k / 2) + 1)**2) - (k + (k / 2)) # Calculation of the middle node according to the symmetrical mesh generator.

    k = int(round(B / (4.0 * fine_size)))  
    n_coarse = k // 2                      
    
    NX = 1 + n_coarse + k                   
    NY = NX                               
    
    node_id = NX * NY - (NX - 1)            

    
    check, der, last_time = Temp_derivatives(str(first_out), node_id)
    filename_history = [f"{stem}.IN"] 
    if check:
        return {
            "found_negative_derivative": True,
            "runs": 1,
            "last_out": str(first_out),
            "last_time": float(last_time),
            "filename_history": filename_history,
        }

    current_last_time = float(last_time)
    last_out_path = str(first_out)
    prev_restart_in_name = stem

    for i in range(1, max_restarts + 1):
        restart_in_name = f"{stem}_r{i}.IN"
        restart_in_path = workdir / restart_in_name

        make_restart_cross_section(
            workdir=workdir,
            filename=restart_in_name,
            firstfile=prev_restart_in_name,                  
            firsttime=current_last_time,     
            t_extend=extend_step_s,          
        )

        extend_firecurve(workdir, current_last_time + extend_step_s, T0)
        
        inp = workdir / f"{Path(restart_in_name).stem}"
        rn.run_safir(inp, env_var="SAFIREXE", workdir=workdir, show_output=show_output)

        restart_out_path = workdir / f"{Path(restart_in_name).stem}.OUT"
        if not restart_out_path.exists():
            raise FileNotFoundError(
                f"Expected OUT file not found after run: {restart_out_path}"
            )

        check, der, new_last_time = Temp_derivatives(str(restart_out_path), node_id)
        filename_history.append(restart_in_name)
        last_out_path = str(restart_out_path)
        current_last_time = float(new_last_time)
        prev_restart_in_name = restart_in_name

        if check:
            return {
                "check_negative_derivative": True,
                "runs": 1 + i,
                "last_out": last_out_path,
                "last_time": current_last_time,
                "filename_history": filename_history,
            }

    return {
        "found_negative": False,
        "runs": 1 + max_restarts,
        "last_out": last_out_path,
        "last_time": current_last_time,
        "filename_history": filename_history,
    }


def Temp_derivatives(file_path, node_id):
    """
    Parse a SAFIR thermal .OUT file and return a DataFrame with the temperature derivative time series
    for a specific node.

    Parameters
    ----------
    file_path : str | Path
        Path in which the SAFIR thermal .OUT is located. 
    node_id : int
        Node number at which the temperature derivates are computed.  
    
    Returns:
    ----------
        pd.DataFrame with columns ['time_s', 'temperature_C'], sorted by time.
    """
    file_path = Path(file_path)
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

        while k < len(lines):
            line = lines[k]
            if not line.strip():
                break
            pairs = pair_re.findall(line)
            if not pairs:
                break
            for n_str, t_str in pairs:
                if int(n_str) == int(node_id):
                    try:
                        temp_val = float(t_str)
                    except ValueError:
                        raise ValueError(f"Could not parse temperature value '{t_str}' at line {k+1}.")
                    results.append((time_seconds, temp_val))
            k += 1

    if not results:
        raise ValueError(f"Node {node_id} not found in any 'TOTAL TEMPERATURES.' section.")

    der = []
    for i in range(1, len(results)):
        t_prev, T_prev = results[i-1]
        t_curr, T_curr = results[i]

        dt = t_curr - t_prev
        if dt <= 0:
            raise ValueError(
                f"Non-increasing or duplicate time detected between entries {i-1} and {i}: "
                f"t_prev={t_prev}, t_curr={t_curr}"
            )

        dTdt = (T_curr - T_prev) / dt
        der.append(dTdt)
    
    der = np.array(der)
    if np.any(der < 0):
        return True, der, results[-1][0]
    else:
        return False, der, results[-1][0]

def make_restart_cross_section(workdir, filename, firstfile, firsttime, t_extend):
    """
    Create a SAFIR thermal restart input (.IN) by minimal modification of the original .IN.

    Parameters
    ----------
    workdir : str | Path
        Directory where the original <stem>.IN is located and where the new file will be written.
    filename : str
        Name of the new restart input file.
    firstfile : str
        Basename for the previous run.
    firsttime : float
        Restart time (s) that exists in the previous .OUT and from which to resume.
    t_extend : float
        Absolute end time (s) for the restart run.
    """
    workdir = Path(workdir)

    stem = Path(firstfile).stem
    template_in = workdir / f"{stem}.IN"
    prev_out = f"{stem}.OUT"

    if not template_in.exists():
        raise FileNotFoundError(f"Template input not found: {template_in}")

    lines = template_in.read_text(errors="ignore").splitlines()

    def find_idx(pred):
        for i, ln in enumerate(lines):
            if pred(ln):
                return i
        return -1

    i_temperat = find_idx(lambda l: l.strip().upper().startswith("TEMPERAT"))
    i_restartt = find_idx(lambda l: l.strip().upper().startswith("RESTARTT"))

    if i_temperat == -1 and i_restartt == -1:
        raise ValueError("Neither 'TEMPERAT' nor 'RESTARTT' found in the template .IN file.")

    if i_temperat != -1:
        lines[i_temperat] = "RESTARTT"
        insert_after = i_temperat
    else:
        insert_after = i_restartt

    j = insert_after + 1
    while j < len(lines):
        u = lines[j].strip().upper()
        if u.startswith("FIRSTFILE") or u.startswith("FIRSTTIME") or u.startswith("MATCH"):
            lines.pop(j)
            continue
        if re.match(r"^(TETA|TINITIAL|COMEBACK|DIAG_|MAKE\.TEM|NMAT|ELEMENTS|NODES|FRONTIER|SYMMETRY|PRECISION|MATERIALS|TIME|OUTPUT)\b", u):
            break
        j += 1

    ft_str = f"{float(firsttime):.6f}".rstrip('0').rstrip('.')
    lines.insert(insert_after + 1, f"FIRSTFILE {prev_out}")
    lines.insert(insert_after + 2, f"FIRSTTIME {ft_str}")
    lines.insert(insert_after + 3, "MATCHNODES")

    i_time = find_idx(lambda l: l.strip().upper() == "TIME")
    if i_time == -1 or i_time + 1 >= len(lines):
        raise ValueError("Malformed or missing TIME block.")

    time_vals = lines[i_time + 1].split()
    time_step = time_vals[2]
    time_min = time_vals[0]
    time_end = firsttime + t_extend

    lines[i_time + 1] = f"{time_min} {time_end} {time_step}"

    i_tp = find_idx(lambda l: l.strip().upper() == "TIMEPRINT")
    if i_tp == -1 or i_tp + 1 >= len(lines):
        raise ValueError("TIMEPRINT block not found or malformed.")

    tp_vals = lines[i_tp + 1].split()
    
    print_dt = tp_vals[0]
    lines[i_tp + 1] = f"{print_dt} {time_end}"

    out_path = workdir / filename
    out_path.write_text("\n".join(lines) + "\n")
    return

def extend_firecurve(workdir, time_s, T0):
    """
    Extend a SAFIR fire curve (.fct) file by appending one (time, temperature) pair.

    Parameters
    ----------
    workdir : str | Path
        Directory containing the .fct file.
    time_s : float | int
        Time value (in seconds) to add to the fire curve.
    T0 : float | int
        Corresponding temperature at that time.
    """
    workdir = Path(workdir)
    fct_files = list(workdir.glob("*.fct"))
    if not fct_files:
        raise FileNotFoundError(f"No .fct file found in {workdir}")
    if len(fct_files) > 1:
        print(f"Warning: Multiple .fct files found, using {fct_files[0].name}")

    fct_path = fct_files[0]

    with fct_path.open("a") as f:
        f.write(f"{time_s} {T0}\n")
    return
