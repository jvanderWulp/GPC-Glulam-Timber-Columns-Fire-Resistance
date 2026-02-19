from pathlib import Path
import numpy as np

def make_cross_section(
    workdir: str | Path,
    filename: str,
    B: float,
    fine_size: float,
    *,
    frontier_name: str = "FISO",
    teta: float = 0.9,
    T_initial: float = 20.0,
    comeback: float = 0.1,
    ng: int = 2,
    precision: float = 1e-3,
    dt0: float = 7.5,
    t_final: float = 3600.0,
    dtmax: float = 30.0,
    timeprint: float = 60.0,
    material_name: str = "WOODEC5",
    material_params: tuple = (410, 12, 35, 4, 0.8, 4, 0, 0, 1),
):
    """
    Generate a SAFIR thermal input (.IN) for a square section (B × B) with a refined mesh in the corner regions:
    - Corner regions (each width B/4) with mesh size = `fine_size`
    - Middle regions  (width B/2) with mesh size = 2*`fine_size` (coarse_size)
    All four sides are exposed with the same frontier name.

    IMPORTANT: `fine_size` must fit exactly:  B/(4*fine_size) must be an integer.
    
    Parameters
    ----------
    workdir : str | Path
        Directory where the output file will be written. Created if it does not exist.
    filename : str
        Name of the output file (e.g. "thermo.IN").
    B : float
        Section width (square) [m]
    fine_size: float
        Size of fine mesh element [m]
    frontier_name : str, optional
        Name of the thermal action to apply on all four sides (default: "FISO").
    teta : float, optional
        Time integration parameter (default 0.9).
    T_initial : float, optional
        Initial temperature [Celsius] (default 20.0).
    comeback : float, optional
        Minimum accepted time step during step control [s] (default 0.1).
    diag_capa : bool, optional
        If True, use diagonal heat capacity matrix (default True).
    make_tem : bool, optional
        If True, write a .TEM file for subsequent mechanical analysis (default True).
    ng : int, optional
        Gauss integration order for SOLID elements (default 2).
    precision : float, optional
        Numerical precision (default 1e-3).
    dt0 : float, optional
        Initial time step (s), default 7.5.
    t_final : float, optional
        End time (s), default 3600.0.
    dtmax : float, optional
        Maximum time step (s), default 30.0.
    timeprint : float, optional
        Output interval (s), default 60.0.
    material_name : str, optional
        SAFIR material keyword (default "WOODEC5").
    material_params : tuple, optional
        Parameters written on the line after `material_name`
        (Density including moisture content, Moisture content [%], convection coefficient (heating), convection coefficient (cooling), emmisivity constant, Ratio of anisotropy in conduction, direction of the grain x-coord, direction of the grain y-coord, direction of the grain z-coord)
        (default `(410, 12, 35, 4, 0.8, 4, 0, 0, 1)`).
    
    Returns
    -------
    int
         Number of fibres/elements in the cross-section grid: EX * EY.
    
    """
    out_path = Path(workdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k = B / (4.0 * fine_size)          
    if abs(round(k) - k) > 1e-9 or k < 1:
        raise ValueError(
            "fine_size must satisfy B/(4*fine_size) ∈ ℕ and ≥ 1. "
            f"Got B={B:g}, fine_size={fine_size:g}, B/(4*fine_size)={k:.12g}."
        )
    n_fine = int(round(k))
    n_coarse = n_fine

    coarse_size = 2.0 * fine_size

    xs = [0.0]
    for i in range(1, n_fine + 1):
        xs.append(i * fine_size)
    for i in range(1, n_coarse + 1):
        xs.append(B/4 + i * coarse_size)
    for i in range(1, n_fine + 1):
        xs.append(3*B/4 + i * fine_size)

    ys = xs  

    NX = len(xs)            
    NY = len(ys)
    EX = NX - 1             
    EY = NY - 1
    nfiber = EX * EY

    def nid(i: int, j: int) -> int:
        return j * NX + i + 1

    lines: list[str] = []
    lines += ["Safir Thermal Analysis", "Input file created with Python script"]
    lines.append("")

    lines += [
        f"NNODE {NX*NY}",
        "NDIM 2",
        "NDOFMAX 1",
        "TEMPERAT",
        f"TETA {teta}",
        f"TINITIAL {T_initial:g}",
        f"COMEBACK {comeback:g}",
    ]
    
    lines.append("DIAG_CAPA")
    lines.append("MAKE.TEM")
    
    lines += [
        "NMAT 1",
        "ELEMENTS",
        f"SOLID {EX*EY}",
        f"NG {ng}",
        "NVOID 0",
        "END_ELEM",
        "NODES",
    ]

    node_id = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            lines.append(f"NODE {node_id} {y:.6f} {x:.6f}")
            node_id += 1

    lines += [
        f"NODELINE {B/2:.6g} {B/2:.6g}",
        f"YC_ZC {B/2:.6g} {B/2:.6g}",
        "FIXATIONS",
        "END_FIX",
        "NODOFSOLID",
    ]

    elem_id = 0
    for j in range(EY):
        for i in range(EX):
            LD = nid(i,   j)
            RD = nid(i+1, j)
            RU = nid(i+1, j+1)
            LU = nid(i,   j+1)
            elem_id += 1
            lines.append(f"ELEM {elem_id} {LD} {RD} {RU} {LU} 1 0.")

    lines.append("FRONTIER")
    # left column
    for e in range(1, EX*EY + 1, EX):
        lines.append(f"F {e} NO NO NO {frontier_name}")
    # right column
    for e in range(EX, EX*EY + 1, EX):
        lines.append(f"F {e} NO {frontier_name} NO NO")
    # bottom row
    for e in range(1, EX + 1):
        lines.append(f"F {e} {frontier_name} NO NO NO")
    # top row
    start_top = (EY - 1) * EX + 1
    for e in range(start_top, EX*EY + 1):
        lines.append(f"F {e} NO NO {frontier_name} NO")
    lines.append("END_FRONT")

    lines += [
        "SYMMETRY",
        "END_SYM",
        f"PRECISION {precision:.1E}",
        "MATERIALS",
        material_name,
        " ".join(str(x) for x in material_params),
        "TIME",
        f"{dt0:g} {t_final:g} {dtmax:g}",
        "END_TIME",
        "OUTPUT",
        "TIMEPRINT",
        f"{timeprint:g} {t_final:g}",
        "END_TIMEPR",
    ]

    out_path.write_text("\n".join(lines))
    print(f"File saved at: {str(out_path)}")
    return nfiber

def make_cross_section_SYM(
    workdir: str | Path,
    filename: str,
    B: float,
    fine_size: float,
    *,
    frontier_name: str = "FISO",
    teta: float = 0.9,
    T_initial: float = 20.0,
    comeback: float = 0.1,
    ng: int = 2,
    precision: float = 1e-3,
    dt0: float = 7.5,
    t_final: float = 3600.0,
    dtmax: float = 30.0,
    timeprint: float = 60.0,
    material_name: str = "WOODEC5",
    material_params: tuple = (410, 12, 35, 4, 0.8, 4, 0, 0, 1),
    add_flux: bool = False,
    mesh_positions: np.ndarray | None = None,
    flux_name: str = 'myflux.txt',
):
    """
    Generate a SAFIR thermal input (.IN) for a quarter of a square section (B × B) with a refined mesh in the corner regions:
    - Corner region (B/4) with mseh size = 'fine_size'
    - Middle region  (B/4) with mesh size = 2*'fine_size' (coarse_size)
    All fire exposed sides are exposed with the same frontier name.

    IMPORTANT: 'fine_size' must fit exactly:  B/(4*fine_size) must be an integer.
    
    Parameters
    ----------
    workdir : str | Path
        Directory where the output file will be written. Created if it does not exist.
    filename : str
        Name of the output file (e.g. "thermo.IN").
    B : float
        Section width (square) [m]
    fine_size: float
        Size of fine mesh element [m]
    frontier_name : str, optional
        Name of the thermal action to apply on all four sides (default: "FISO").
    teta : float, optional
        Time integration parameter (default 0.9).
    T_initial : float, optional
        Initial temperature [Celsius] (default 20.0).
    comeback : float, optional
        Minimum accepted time step during step control [s] (default 0.1).
    diag_capa : bool, optional
        If True, use diagonal heat capacity matrix (default True).
    make_tem : bool, optional
        If True, write a .TEM file for subsequent mechanical analysis (default True).
    ng : int, optional
        Gauss integration order for SOLID elements (default 2).
    precision : float, optional
        Numerical precision (default 1e-3).
    dt0 : float, optional
        Initial time step (s), default 7.5.
    t_final : float, optional
        End time (s), default 3600.0.
    dtmax : float, optional
        Maximum time step (s), default 30.0.
    timeprint : float, optional
        Output interval (s), default 60.0.
    material_name : str, optional
        SAFIR material keyword (default "WOODEC5").
    material_params : tuple, optional
        Parameters written on the line after `material_name`
        (Density, Moisture content, convection coefficient (heating), convection coefficient (cooling), emmisivity constant, Ratio of anisotropy in conduction, direction of the grain x-coord, direction of the grain y-coord, direction of the grain z-coord)
        (default `(410, 12, 35, 4, 0.8, 4, 0, 0, 1)`).
    
    Returns
    -------
    int
         Number of fibres/elements in the cross-section grid: EX * EY.
    
    """

    allowed = {"WOODEC5", "USER1", "USER2"}
    if material_name not in allowed:
        raise ValueError(f"material_name must be one of {allowed}, got '{material_name}'.")

    if material_name in {"USER1", "USER2"} and len(material_params) != 9:
        raise ValueError(
            f"{material_name} expects 9 parameters: "
            "(rho, w, T_start, T_end, h_ch, h_cc, eps, r, tau). "
            f"Got {len(material_params)}."
        )
    if material_name == "WOODEC5" and len(material_params) != 9:
        raise ValueError(
            f"{material_name} expects 9 parameters: "
            "(rho, w, h_ch, h_cc, eps, r, l, m, n). "
            f"Got {len(material_params)}."
        )
    
    out_path = Path(workdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k = B / (4.0 * fine_size)          
    if abs(round(k) - k) > 1e-9 or k < 2:
        raise ValueError(
            "fine_size must satisfy B/(4*fine_size)"
            f"Got B={B:g}, fine_size={fine_size:g}, B/(4*fine_size)={k:.12g}."
        )
    n_fine = int(round(k))
    n_coarse = int(n_fine/2)

    coarse_size = 2.0 * fine_size
    k2 = B / (4.0 * coarse_size)
    if abs(round(k2) - k2) > 1e-9 or k2 < 1:
        raise ValueError(
            "fine_size must satisfy B/(4*coarse_size)"
            "  With coarse_size = 2*fine_size."
            f"  Got B={B:g}, fine_size={fine_size:g}, coarse_size={coarse_size:g}, B/(4*coarse_size)={k2:.12g}."
        )
        
    xs = [0.0]
    for i in range(1, n_coarse + 1):
        xs.append(i * coarse_size)
    for i in range(1, n_fine + 1):
        xs.append(B/4 + i * fine_size)

    ys = [0.0]
    for i in range(1, n_fine + 1):
        ys.append(i * fine_size)
    for i in range(1, n_coarse + 1):
        ys.append(B/4 + i * coarse_size)

    NX = len(xs)            
    NY = len(ys)
    EX = NX - 1             
    EY = NY - 1
    nfiber = 4 * (EX * EY)

    def nid(i: int, j: int):
        return j * NX + i + 1

    lines: list[str] = []
    lines += ["Safir Thermal Analysis with symmetry", "Input file created with Python script"]
    lines.append("")

    lines += [
        f"NNODE {NX*NY}",
        "NDIM 2",
        "NDOFMAX 1",
        "TEMPERAT",
        f"TETA {teta}",
        f"TINITIAL {T_initial:g}",
        f"COMEBACK {comeback:g}",
    ]
    
    lines.append("DIAG_CAPA")
    lines.append("MAKE.TEM")
    
    lines += [
        "NMAT 1",
        "ELEMENTS",
        f"SOLID {EX*EY}",
        f"NG {ng}",
        "NVOID 0",
        "END_ELEM",
        "NODES",
    ]

    node_id = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            lines.append(f"NODE {node_id} {y:.6f} {x:.6f}")
            node_id += 1

    lines += [
        f"NODELINE {B/2:.6g} {0.}",
        f"YC_ZC {B/2:.6g} {0.}",
        "FIXATIONS",
        "END_FIX",
        "NODOFSOLID",
    ]

    elem_id = 0
    for j in range(EY):
        for i in range(EX):
            LD = nid(i,   j)
            RD = nid(i+1, j)
            RU = nid(i+1, j+1)
            LU = nid(i,   j+1)
            elem_id += 1
            lines.append(f"ELEM {elem_id} {LD} {RD} {RU} {LU} 1 0.")

    lines.append("FRONTIER")
    # right column
    for e in range(EX, EX*EY + 1, EX):
        lines.append(f"F {e} NO {frontier_name} NO NO")
    # bottom row
    for e in range(1, EX + 1):
        lines.append(f"F {e} {frontier_name} NO NO NO")

    if add_flux:
        if mesh_positions is None or len(mesh_positions) == 0:
            raise ValueError(
            "No mesh_positions assigned. Need the mesh positions of the additional flux" 
            )
        for i in range(len(mesh_positions)):
                # right columns
                for e in range((EX*(i+1))-i, EX*EY - i + 1, EX):
                    lines.append(f"FLUX {e} NO flux{i+1}.txt NO NO")
                # bottom row
                for e in range(1 + EX*i, EX*(i+1)-i+1):
                    lines.append(f"FLUX {e} flux{i+1}.txt NO NO NO")
    
    # if add_flux:
    #     # right column
    #     for e in range(EX, EX*EY + 1, EX):
    #         lines.append(f"FLUX {e} NO {flux_name} NO NO")
    #     # bottom row
    #     for e in range(1, EX + 1):
    #         lines.append(f"FLUX {e} {flux_name} NO NO NO")
    lines.append("END_FRONT")

    lines += [
        "SYMMETRY",
        "YSYM",
        f"REALSYM {NX*NY - (NX - 1)} {NX*NY}",
        "END_SYM",
        f"PRECISION {precision:.1E}",
        "MATERIALS"
    ]

    if material_name == "WOODEC5":
        lines.append(material_name)
        lines.append(" ".join(str(x) for x in material_params))

    if material_name == "USER1":
        rho_wet, w_pct, Tstart, Tend, h_ch, h_cc, eps, rfac, tau = material_params
        k_tbl  = hopkins_conductivity(tau)
        c_tbl  = cachim_and_franssen(w_pct)
        rho_tbl = EC_density(rho_wet, w_pct)

        temps = k_tbl[0].tolist()
        ks = k_tbl[1].tolist()
        cs = c_tbl[1].tolist()
        rhos = rho_tbl[1].tolist()

        rho_dry = rho_wet / (1.0 + w_pct/100.0)
        w_kgm3  = (w_pct/100.0) * rho_dry

        lines.append(material_name + f" {len(temps)}")
        lines.append(f"{temps[0]:.0f}. {ks[0]:.6g} {cs[0]:.6g} {rhos[0]:.6g} "
                     f"{w_kgm3:.6g} {Tstart:.6g} {Tend:.6g} {h_ch:.6g} {h_cc:.6g} {eps:.6g} {rfac:.6g}")
        for i in range(1, len(temps)):
            lines.append(f"{temps[i]:.0f}. {ks[i]:.6g} {cs[i]:.6g} {rhos[i]:.6g}")

    if material_name == "USER2":
        rho_wet, w_pct, Tstart, Tend, h_ch, h_cc, eps, rfac, tau = material_params
        k_tbl  = hopkins_conductivity(tau)
        c_tbl  = EC_specific_heat()
        rho_tbl = EC_density(rho_wet, w_pct)

        temps = k_tbl[0].tolist(); ks = k_tbl[1].tolist()
        cs    = c_tbl[1].tolist(); rhos = rho_tbl[1].tolist()

        rho_dry = rho_wet / (1.0 + w_pct/100.0)
        w_kgm3  = (w_pct/100.0) * rho_dry

        lines.append(material_name + f" {len(temps)}")
        lines.append(f"{temps[0]:.0f}. {ks[0]:.6g} {cs[0]:.6g} {rhos[0]:.6g} "
                     f"{w_kgm3:.6g} {Tstart:.6g} {Tend:.6g} {h_ch:.6g} {h_cc:.6g} {eps:.6g} {rfac:.6g}")
        for i in range(1, len(temps)):
            lines.append(f"{temps[i]:.0f}. {ks[i]:.6g} {cs[i]:.6g} {rhos[i]:.6g}")

        
    lines += [
        "TIME",
        f"{dt0:g} {t_final:g} {dtmax:g}",
        "END_TIME",
        "OUTPUT",
        "TIMEPRINT",
        f"{timeprint:g} {t_final:g}",
        "END_TIMEPR",
    ]

    out_path.write_text("\n".join(lines))
    print(f"File saved at: {str(out_path)}")
    return nfiber



def make_column(
    workdir: str | Path,
    filename: str,
    length: float, 
    tem_file: str,     
    nfiber: int,
    *,
    elements: int = 20,
    ng: int = 2,
    comeback: float = 1e-5,
    precision: float = 1e-3,
    max_displ: float | None = None,
    F_function: str | None = None,
    nodeload1 = (0.0, 0.0, 0.0),
    nodeload2 = (0.0, 0.0, 0.0),
    q_dist: float | None = None,
    dt0: float = 1.0,
    t_final: float = 3600.0,
    dtmax: float = 30.0,
    timeprint: float = 30.0,
    material_name: str = "WOODEC5",
    material_params: tuple = (11.5e9, 0.3, 24e6, 19.2e6, 0.0), 
    stresses: bool = False
):
    """
    Create a SAFIR structural input for a vertical column using 3-noded BEAM elements. 
    The columns boundary conditions considered are pinned-roller. 
    
    Parameters
    ----------
    workdir : str | Path
        Directory where the output file will be written. Created if it does not exist.
    filename : str
        Output file name (e.g. "column.IN")
    length : float
        Column length l [m].
    tem_file : str
        Name of the thermal .TEM file.
    nfiber : int
        Number of fibres in the section.
    elements : int, default 20
        Number of BEAM elements.
    ng : int, default 2
        Gauss integration points.
    comeback : float, default 1e-5
        Minimum solver time step.
    precision : float, default 1e-3
        Numerical precision.
    max_displ : float | None, default None
        If provided, writes `MAX_DISPL <value>`; if None, the line is omitted.
    F_function : str | None, default None
        Loading function
    nodeload1 : tuple, default (0.0, 0.0, 0.0)
        Load at the bottom node (y=0) as (Fx, Fy, Mz).
    nodeload2 : tuple, default (0.0, 0.0, 0.0)
        Load at the top node (y=L) as (Fx, Fy, Mz).
    q_dist : float | None, default None
        Uniform distributed load per element in Fy.
    dt0 : float, default 1.0
        Initial time step [s]
    t_final : float, default 3600.0
        End of the run [s]
    dtmax : float, default 30.0
        Maximum time step [s]
    timeprint : float, default 30.0
        Output creation time of any multiple [s]
    material_name : str, {"WOODEC5", "WOODPRBWE", "WOODECPF"}, default "WOODEC5"
        SAFIR material module. 
    material_params : tuple, default GL24h (11.5e9, 0.3, 24e6, 19.2e6, 0.0)
        Required length depends on 'material_name':
        - WOODEC5: 5 parameters  (E_modulus [Pa], Poisson ratio [-], Compression strength [Pa], Tensile strength [Pa], Max allowed compression strain [-])
        - WOODPRBWE: 6 parameters (E_modulus [Pa], Poisson ratio [-], Compression strength [Pa], Tensile strength [Pa], Weibull quantile [-], Max allowed compression strain [-])
        - WOODECPF: 6 parameters (E_modulus [Pa], Poisson ratio [-], Compression strength [Pa], Tensile strength [Pa], air cooling [1] or water cooling [-1], Max allowed compression strain [-])
        Values are written in the order provided.
    Stresses : bool, default False
        When True, prints the stresses at the middle element (elements/2 + 1) at the second integration point. 
    """
    
    allowed_materials = {"WOODEC5": 5, "WOODPRBWE": 6, "WOODECPF": 6}
    if material_name not in allowed_materials:
        raise ValueError(
            f"Unsupported material_name '{material_name}'. "
            f"Allowed: {', '.join(allowed_materials.keys())}."
        )
    expected_n = allowed_materials[material_name]
    if len(material_params) != expected_n:
        raise ValueError(
            f"{material_name} requires {expected_n} material parameters, "
            f"but got {len(material_params)}: {material_params!r}"
        )
    
    
    E = elements
    L = float(length)
    N = 2*E + 1
    h = L / (2*E)

    out_path = Path(workdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    lines += ["Safir Structural Analysis", "Input file created with Python script"]
    lines.append("")

    lines += [
        f"NNODE {N}",
        "NDIM 2",
        "NDOFMAX 3",
        "STATIC PURE_NR",
        "NLOAD 1",
        "HYDROST 0",
        "OBLIQUE 0",
        f"COMEBACK {comeback:g}",
        "NMAT 1",
        "ELEMENTS",
        f"BEAM {E} 1",
        f"NG {ng}",
        f"NFIBER {nfiber}",
        "END_ELEM",
        "NODES",
    ]

    for i in range(N):
        y = i * h
        lines.append(f"NODE {i+1} 0.000000 {y:.6f}")

    lines += [
        "FIXATIONS",
        "BLOCK 1 F0 F0 NO",
        f"BLOCK {N} F0 NO NO",
        "END_FIX",
        "NODOFBEAM",
        tem_file,
        "TRANSLATE 1 1",
        "END_TRANS",
    ]

    for e in range(1, E+1):
        left_id = 2*e - 1
        mid_id  = 2*e
        right_id= 2*e + 1
        lines.append(f"ELEM {e} {left_id} {mid_id} {right_id} 1")

    lines.append(f"PRECISION {precision:.3f}")
    if max_displ is not None:
        lines.append(f"MAX_DISPL {max_displ:.6g}")


    lines.append("LOADS")

    if F_function is not None:
        lines.append(f"FUNCTION {F_function}")
    if any(nodeload1) is True: 
        lines.append(f"NODELOAD 1 {nodeload1[0]} {nodeload1[1]} {nodeload1[2]}")
    if any(nodeload2) is True: 
        lines.append(f"NODELOAD {N} {nodeload2[0]} {nodeload2[1]} {nodeload2[2]}")
    if q_dist is not None:
        for e in range(1, E+1):
            lines.append(f"DISTRBEAM {e} 0.0 {q_dist}")
    lines.append("END_LOAD")

    lines += [
        "MASS", 
        "END_MASS",
        "MATERIALS",
        material_name,
        " ".join(str(x) for x in material_params),
    ]

    lines += [
        "TIME",
        f"{dt0:g} {t_final:g} {dtmax:g}",
        "END_TIME",
        "EPSTH",
        "IMPRESSION",
        "TIMEPRINT",
        f"{timeprint:g} {t_final:g}",
        "END_TIMEPR",
        "PRINTREACT",
        "PRINTMN",
    ]
    lines.append("PRNEIBEAM")
    if stresses:
        lines.append(f"PRNSIGMABM {elements//2 + 1} {2}")

    out_path.write_text("\n".join(lines))
    print(f"File saved at: {str(out_path)}")
    return



def hopkins_conductivity(tau):
    k_mod = 1.45 * (tau**(-0.48))
    temps = [20, 99, 100, 120, 121, 200, 250, 300, 350, 400, 500, 600, 800, 1200, 1345, 20000]
    values_1 = np.array([99, 100, 120, 121])
    interp_1 = ((0.15 - 0.12) / (200 - 20)) * (values_1 - 20) + 0.12
    values_2 = np.array([250, 300])
    interp_2 = ((0.07 - 0.15) / (350 - 200)) * (values_2 - 200) + 0.15
    values_3 = np.array([400])
    interp_3 = (((0.09 * k_mod) - 0.07) / (500 - 350)) * (values_3 - 350) + 0.07
    values_4 = np.array([600])
    interp_4 = (((0.35 * k_mod) - (0.09 * k_mod)) / (800 - 500)) * (values_4 - 500) + (0.09 * k_mod)
    values_5 = np.array([1345, 20000])
    interp_5 = (((1.5 * k_mod) - (0.35 * k_mod)) / (1200 - 800)) * (values_5 - 800) + (0.35 * k_mod)
    k = [0.12, interp_1[0], interp_1[1], interp_1[2], interp_1[3], 0.15,
         interp_2[0], interp_2[1], 0.07, interp_3[0], 0.09 * k_mod,
         interp_4[0], 0.35 * k_mod, 1.50 * k_mod, interp_5[0], interp_5[1]]
    return np.array((temps, k))

def cachim_and_franssen(w):
    temps = [20, 99, 100, 120, 121, 200, 250, 300, 350, 400, 500, 600, 800, 1200, 1345, 20000]
    w = w / 100.0
    G1 = 1 + w
    G2 = 1.0
    c1 = (1210 +  4190*w)     / G1
    c2 = (1480 +  4190*w)     / G1
    c3 = (1480 + 114600*w)    / G1
    c4 = (2120 +  95500*w)    / G2
    c5 = 2120 / G2
    c6 = 2000 / G2
    c  = [c1, c2, c3, c4, c5, c6, 1620, 710, 850, 1000, 1200, 1400, 1650, 1650, 1650, 1650]
    return np.array((temps, c))

def EC_density(rho_wet, w_pct):
    rho_d = rho_wet / (1.0 + w_pct/100.0)  # dry ref density
    temps = [20, 99, 100, 120, 121, 200, 250, 300, 350, 400, 500, 600, 800, 1200, 1345, 20000]
    density = [rho_wet, rho_wet, rho_wet, rho_d, rho_d, rho_d,
               0.93*rho_d, 0.76*rho_d, 0.52*rho_d, 0.38*rho_d,
               0.33*rho_d, 0.28*rho_d, 0.26*rho_d, 0.0, 0.0, 0.0]
    return np.array((temps, density))

def EC_specific_heat():
    temps = [20, 99, 99.01, 120, 120.01, 200, 250, 300, 350, 400, 500, 600, 800, 1200, 1345, 20000]
    c = [1530, 1770, 13600, 13500, 2120, 2000, 1620, 710, 850, 1000, 1200, 1400, 1650, 1650, 1650, 1650]
    return np.array([temps, c])

def compute_fine_size_closest(width: float, threshold: float):
    """
    Compute a mesh size by repeatedly dividing the width by 2, and
    return the value (either just above or just below the threshold)
    that is closest to the threshold.

    Parameters
    ----------
    width : float
        Initial cross-sectional width.
    threshold : float
        Target threshold value.

    Returns
    -------
    float
        The mesh size closest to the threshold.
    """
    fine_size = width
    history = [fine_size]

    # Keep dividing until we fall below the threshold
    while fine_size > threshold:
        fine_size /= 2.0
        history.append(fine_size)

    if len(history) == 1:
        return fine_size

    above = history[-2]   
    below = history[-1]  

    if abs(above - threshold) <= abs(below - threshold):
        return above
    else:
        return below

def compute_mesh_and_nelements(length: float, threshold: float):
    """
    Determine an odd number of elements and a mesh size that is closest to the given threshold, while ensuring the column length is divided exactly into an odd integer number of mesh intervals.

    Parameters
    ----------
    length : float
        Total column length.
    threshold : float
        Desired mesh size threshold.

    Returns
    -------
    (int, float)
        A tuple containing:
        - n_elements : odd integer
        - mesh_size : float
    """
    n_ideal = length / threshold

    n_low = int(n_ideal) // 2 * 2 + 1             
    n_high = n_low + 2                             

    if n_low < 1:
        n_low = 1
        n_high = 3

    mesh_low = length / n_low
    mesh_high = length / n_high

    if abs(mesh_low - threshold) <= abs(mesh_high - threshold):
        return n_low, mesh_low
    else:
        return n_high, mesh_high
