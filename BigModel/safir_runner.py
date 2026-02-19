from pathlib import Path
import numpy as np
import pandas as pd

import firecurve as fc
import running as rn
import makefiles as mf
import thermalrestart as tr
from Analytical_solution_Dirichlet import analytical_dirichlet
import Post_Processor as pp

class SafirRunner:
    def __init__(self, base_dir: str | Path, column_names: list[str]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.column_names = column_names

    def run(self, phys: dict, df: pd.DataFrame):
        """
        Run a SAFIR simulation given the physcial units in a dictionary. Like in the Simulation_Notebook. 

        Stores a dataframe of all inputs and results. 

        Returns:
        ---------
        failre_label : int
            Label to include for the GPC training. 0 for non-failure, 1 for failure. 
            
        df : Pandas.DataFrame
            Dataframe of all the variables and outcomes. 
        """

        i = len(df)
        analysis_id = i + 1

        workdir = self.base_dir / f"Simulation_{analysis_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        name_firecurve = f"fr_{analysis_id}.fct"
        name_thermalfile = f"thermo_{analysis_id}.IN"
        name_mechfile = f"COLUMN_{analysis_id}.IN"

        # Structural Design variables
        B = float(phys.get("B", 0.280))            # [m]
        l = float(phys.get("l", 3.65))         # [m]
        F = float(phys.get("F", 322000.0))     # [N]

        # Fire Variablles
        T0 = 20.0
        t_h = float(phys.get("t_h", 15))           # [min]

        # Choice of independent T_max or Dependent T_max on the heating duration (t_h)
        if "T_max" in phys:
            T_max = float(phys["T_max"])
        else:
            T_max = round(T0 + 1325 * fc.phi(t_h / 60.0), 2)     # Heating following the ISO 834 firecurve

        r_cool = float(phys.get("r_cool", 10.4))
        dt = 0.5

        # Material variables
        e0  = float(phys.get("e0", 0.02))
        rho = float(phys.get("rho", 420))
        E   = float(phys.get("E", 12.8e9))
        f_c = float(phys.get("f_c", 40.4e6))
        f_t = float(phys.get("f_t", 26.4e6))
        w   = float(phys.get("w", 12))

        mu  = float(phys.get("mu", 0.3))
        h_ch = float(phys.get("h_ch", 35))
        h_cc = float(phys.get("h_cc", 4))
        eps  = float(phys.get("eps", 0.8))

        struc_material_name   = "WOODEC5"
        struc_material_params = (E, mu, f_c, f_t, 1)
        therm_material_name   = "WOODEC5"
        therm_material_params = (rho, w, h_ch, h_cc, eps, 4, 0, 0, 1)

       # Mesh generation
        fine_size = mf.compute_fine_size_closest(B, 0.005)
        n_elements, mesh_size = mf.compute_mesh_and_nelements(l, 0.25)

        # Calculations
        q_dist = B * B * rho * 10
        M0 = e0 * F

        # Analytical first estimate of the end time of the simulations
        t_end  = round(
            analytical_dirichlet(
                B, rho, T_max, t_h, r_cool,
                c=1530, k=0.12, plot=False
            )[0] * 1.3,
            2
        )
        t_end_guess = t_end

        # Firecurve
        t_all, T_all, tau = fc.build_fire_curve(
            T0=T0, T_max=T_max, t_h=t_h,
            r_cool=r_cool, t_end=t_end, dt=dt,
            plot_curve=False
        )
        H = fc.area_firecurve(T0, T_max, t_h, r_cool, t_end, tau)
        outfile = fc.save_fct(t_all, T_all, filename=name_firecurve, out_dir=workdir)

        # Thermal analysis
        nfiber = mf.make_cross_section_SYM(
            workdir=workdir, filename=name_thermalfile,
            B=B, fine_size=fine_size,
            frontier_name=name_firecurve,
            T_initial=20.0, comeback=0.001,
            precision=1e-3, dt0=7.5, t_final=t_end*60,
            dtmax=30.0, timeprint=60.0,
            material_name=therm_material_name,
            material_params=therm_material_params
        )
        inp = workdir / Path(name_thermalfile).stem
        rn.run_safir(input_file=inp, env_var="SAFIREXE", workdir=workdir, show_output=False)
        restart = tr.thermal_restart(workdir, name_thermalfile, B, fine_size, show_output=False)

        for r in range(1, 15):
            if restart["runs"] > 1:
                t_end = restart["last_time"] / 60
                workdir = workdir / f"Batch0{r}"
                t_all, T_all, tau = fc.build_fire_curve(
                    T0=T0, T_max=T_max, t_h=t_h,
                    r_cool=r_cool, t_end=t_end, dt=dt,
                    plot_curve=False
                )
                H = fc.area_firecurve(T0, T_max, t_h, r_cool, t_end, tau)
                outfile = fc.save_fct(t_all, T_all, filename=name_firecurve, out_dir=workdir)
                nfiber = mf.make_cross_section_SYM(
                    workdir=workdir, filename=name_thermalfile,
                    B=B, fine_size=fine_size,
                    frontier_name=name_firecurve,
                    T_initial=20.0, comeback=0.01,
                    precision=1e-3, dt0=7.5, t_final=t_end*60,
                    dtmax=30.0, timeprint=60.0,
                    material_name=therm_material_name,
                    material_params=therm_material_params
                )
                inp = workdir / Path(name_thermalfile).stem
                rn.run_safir(input_file=inp, env_var="SAFIREXE", workdir=workdir, show_output=False)
                restart = tr.thermal_restart(workdir, name_thermalfile, B, fine_size, show_output=False)
            else:
                break

        # Structural analysis
        tem_file = f"{Path(name_thermalfile).stem}.TEM"
        mf.make_column(
            workdir=workdir, filename=name_mechfile,
            length=l, tem_file=tem_file, nfiber=nfiber,
            elements=n_elements, ng=3, comeback=1e-09,
            F_function="FLOAD",
            nodeload1=(0.0, 0.0, -M0),
            nodeload2=(0.0, -F, M0),
            q_dist=-q_dist, t_final=t_end*60,
            timeprint=60.0,
            material_name=struc_material_name,
            material_params=struc_material_params,
            stresses=True
        )
        inp_2 = workdir / Path(name_mechfile).stem
        rn.run_safir(input_file=inp_2, env_var="SAFIREXE", workdir=workdir, show_output=False)

        # Post processing
        failure_time, failure = pp.postprocess_failure(
            workdir, f"{Path(name_mechfile).stem}.OUT", t_end
        )
        time_thermo = pp.get_simulation_time(
            workdir, f"{Path(name_thermalfile).stem}.OUT"
        )
        time_mech = pp.get_simulation_time(
            workdir, f"{Path(name_mechfile).stem}.OUT"
        )
        time_tot = pp.sum_times([time_thermo, time_mech])
        pp.make_description(
            workdir, "AA_Description.txt",
            time_thermo, time_mech, time_tot,
            e0, rho, E, mu, f_c, f_t, w, h_ch, h_cc, eps,
            B, l, F, T0, t_h, T_max, r_cool, t_end, dt,
            fine_size, n_elements, failure, failure_time
        )
        stiffness = pp.get_stiffness(
            workdir, f"{Path(name_mechfile).stem}.OUT",
            n_elements=n_elements
        )

        df.loc[i] = [
            analysis_id, e0, rho, E, mu, f_c, f_t, w, h_ch, h_cc, eps, B, l, F,
            t_h, T_max, r_cool, H, t_end, t_end_guess, fine_size, n_elements,
            failure, failure_time, time_thermo, time_mech, time_tot, stiffness
        ]

        csv = self.base_dir / "Data_AL.csv"
        df.to_csv(csv, index=False)

        failure_label = int(failure)
        return failure_label, df