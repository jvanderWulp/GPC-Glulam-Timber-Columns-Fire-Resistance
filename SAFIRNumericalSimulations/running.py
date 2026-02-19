import os
import subprocess
from pathlib import Path


def run_safir(
    input_file: str | Path,
    env_var: str = "SAFIREXE",
    workdir: str | Path | None = None,
    timeout: int | None = None,
    extra_args: list[str] | None = None,
    show_output: bool = True,
):
    """
    Run SAFIR using an executable path stored in an environment variable.

    Parameters
    ----------
    input_file : str | Path
        Path to your SAFIR input file (e.g. *.IN).
    env_var : str
        Name of the environment variable that holds the SAFIR executable path/name.
        Example: "SAFIREXE".
    workdir : str | Path | None
        Working directory to run in. If None, uses parent of input file.
    timeout : int | None
        Timeout in seconds (optional).
    extra_args : list[str] | None
        Optional extra CLI arguments to pass to SAFIR (default: none).
    show_output : bool
        If True (default), print SAFIR's stdout/stderr in the notebook.
        If False, suppress printing but still capture the output.
    """
    try:
        exe = os.environ[env_var]
    except KeyError:
        raise KeyError(
            f"Environment variable '{env_var}' is not set. "
            f"Set it to the SAFIR executable path/name (e.g. SAFIR.exe)."
        )

    input_path = Path(input_file).resolve()
    if workdir is None:
        workdir = input_path.parent
    workdir = Path(workdir)

    cmd = [exe]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(input_path.name)

    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        text=True,
        capture_output=True,
        shell=False,
        timeout=timeout,
        check=False,
    )

    if show_output:
        if proc.stdout:
            print("---- SAFIR STDOUT ----")
            print(proc.stdout)
        if proc.stderr:
            print("---- SAFIR STDERR ----")
            print(proc.stderr)
    return