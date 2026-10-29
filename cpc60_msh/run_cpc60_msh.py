#!/usr/bin/env python3
"""从原始 CSV 构建 CPC60 MSH，采用三激发态、600 坐标 Ehrenfest。

python run_cpc60_msh.py --conf 3 --prepare-only
python run_cpc60_msh.py --conf 3 --ntraj 8 --t-end-fs 20 --no-show

200 个频率各重复三次，坐标顺序 mu=3*j+a（j 从 0、a 从 0）。
保持原始位移坐标；基态只用于初态中心，不参与电子传播。
内部单位：Hartree、原子时间、质量加权原子单位；输出图时间为 fs。
短时间少轨迹结果只用于检查实现，不是收敛的文献复现。
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

from ehrenfest_multistate_multimode import (
    Model, simulate, reorganization_matrix, make_figure,
)

HARTREE_EV = 27.211386245988
AU_TIME_FS = 0.024188843265857
KB_HARTREE_K = 3.166811563e-6
ROOT = Path(__file__).resolve().parent

# JCP 160, 034108 (2024), Table I (paper p. 8 / supplied PDF page 9).
# epsilon is the minimum energy of each displaced PES, NOT vertical energy.
TABLE = {
    2: dict(
        epsilon_ev=[0.0, -0.268, -0.779],
        gamma_ev=[[0, 0.088, 0.00011], [0.088, 0, 0.00012],
                  [0.00011, 0.00012, 0]],
        lambda_ev=[[0,.311,1.507,.127], [.311,0,1.316,.559],
                   [1.507,1.316,0,1.699], [.127,.559,1.699,0]],
    ),
    3: dict(
        epsilon_ev=[0.0, -0.828, -0.640],
        gamma_ev=[[0,-0.015,0.0072], [-0.015,0,-0.029],
                  [0.0072,-0.029,0]],
        lambda_ev=[[0,.340,.492,.150], [.340,0,.041,.910],
                   [.492,.041,0,1.134], [.150,.910,1.134,0]],
    ),
}


def load_csv(path):
    expected = ["j", "omega_j/au"] + [
        f"Sj_{a}{n}" for n in range(1, 5) for a in range(1, 4)]
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected:
            raise ValueError(
                f"Unexpected CSV header: {reader.fieldnames}; expected {expected}")
        rows = list(reader)
    if len(rows) != 200:
        raise ValueError(f"Expected 200 frequency rows, found {len(rows)}")
    indices = np.array([int(r["j"]) for r in rows])
    if not np.array_equal(indices, np.arange(1, 201)):
        raise ValueError("Mode indices must be 1..200 in order.")
    w = np.array([float(r["omega_j/au"]) for r in rows])
    # s[n,j,a]：第二个下标是频率行，第三个是几何方向。
    s = np.array([
        [[float(r[f"Sj_{a}{n}"]) for a in range(1, 4)] for r in rows]
        for n in range(1, 5)])
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        raise ValueError("CSV contains NaN or Inf.")
    if np.any(w <= 0) or np.any(np.diff(w) <= 0):
        raise ValueError("Frequencies must be positive and strictly increasing.")
    return w, s


def build_model(path, conf=3, temperature_K=300.0):
    if conf not in TABLE:
        raise ValueError("conf must be 2 or 3")
    if not np.isfinite(temperature_K) or temperature_K < 0:
        raise ValueError("Temperature must be nonnegative and finite.")
    w, s = load_csv(path)
    table = TABLE[conf]
    m = Model(
        omega=np.repeat(w, 3),
        epsilon=np.array(table["epsilon_ev"])/HARTREE_EV,
        gamma=np.array(table["gamma_ev"])/HARTREE_EV,
        shifts=s[:3].reshape(3, -1),
        ground_shift=s[3].reshape(-1),
        kBT=KB_HARTREE_K*temperature_K,
        hbar=1.0, labels=("pi-pi*", "CT1", "CT2"),
        units="Hartree / atomic time / mass-weighted atomic coordinates",
    )
    # 核验全部态对，不用取整后的重组能反推原始位移。
    lam = reorganization_matrix(m)*HARTREE_EV
    tab_lam = np.array(table["lambda_ev"])
    mismatch = np.max(np.abs(lam-tab_lam))
    if mismatch > 0.0005001:
        raise ValueError(f"CSV does not match Table I rounding: {mismatch} eV")

    # 展开到 G 中心后的系统参数，为后续 EH-TEMPO 留出公共输入。
    d = m.shifts-m.ground_shift
    c = -(m.omega**2)*d
    vertical = m.epsilon + 0.5*np.sum((m.omega*d)**2, axis=1)
    # 在真实尺度的初始位置附近，检查平方形式与展开形式逐点相等。
    rng = np.random.default_rng(18)
    sd = np.sqrt(
        (1/np.tanh(m.omega/(2*m.kBT)) if m.kBT > 0 else 1)/(2*m.omega))
    qs = rng.normal(size=(5, m.nb))*sd
    R = qs+m.ground_shift
    shifted = m.epsilon[None, :] + 0.5*np.sum(
        (m.omega*(R[:, None, :]-m.shifts))**2, axis=2)
    expanded = (0.5*np.sum((m.omega*qs)**2, axis=1)[:, None]
                + vertical[None, :] + qs @ c.T)
    equality_error = np.max(np.abs(shifted-expanded))
    if equality_error > 1e-11:
        raise ValueError(f"Hamiltonian expansion check failed: {equality_error} Ha")

    audit = dict(
        doi="10.1063/5.0185076", table="Table I, paper p. 8 / PDF p. 9",
        csv_name=Path(path).name,
        csv_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        conformation=conf, temperature_K=temperature_K,
        state_order=["pi-pi*", "CT1", "CT2", "G"],
        frequency_rows=200, directions_per_frequency=3, nuclear_coordinates=m.nb,
        csv_column_interpretation="Sj_an = S_j^(a,n); a=1..3, n=1..4",
        flattened_coordinate_order="mu=3*j+a (j,a zero-based)",
        epsilon_ev=table["epsilon_ev"], gamma_ev=table["gamma_ev"],
        reorganization_ev_from_csv=lam.tolist(),
        reorganization_ev_table=tab_lam.tolist(),
        max_reorganization_difference_ev=float(mismatch),
        all_six_pairs_match_table_rounding=True,
        min_omega_au=float(w.min()), max_omega_au=float(w.max()),
        shortest_period_fs=float(2*np.pi/w.max()*AU_TIME_FS),
        longest_period_fs=float(2*np.pi/w.min()*AU_TIME_FS),
        expanded_system_diagonal_ev= (vertical*HARTREE_EV).tolist(),
        max_expansion_identity_error_hartree=float(equality_error),
        nuclear_initial_state="thermal Wigner on G; a0=(1,0,0)",
        electronic_propagation="three excited states only; Gamma_nG=0",
    )
    return m, audit


def model_json(m):
    return dict(
        omega=m.omega.tolist(), epsilon=m.epsilon.tolist(),
        gamma=m.gamma.real.tolist(), gamma_imag=m.gamma.imag.tolist(),
        shifts=m.shifts.tolist(), ground_shift=m.ground_shift.tolist(),
        kBT=m.kBT, hbar=m.hbar, labels=list(m.labels), units=m.units,
        a0_real=m.a0.real.tolist(), a0_imag=m.a0.imag.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conf", type=int, choices=[2, 3], default=3)
    p.add_argument("--csv", type=Path)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--ntraj", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--t-end-fs", type=float, default=20.0)
    p.add_argument("--output-dt-fs", type=float, default=0.1)
    p.add_argument("--max-step-fs", type=float, default=0.1)
    p.add_argument("--rtol", type=float, default=1e-11)
    p.add_argument("--atol", type=float, default=1e-13)
    p.add_argument("--output-dir", type=Path, default=Path("cpc60_results"))
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    csv_path = args.csv or ROOT/"data"/f"MSH-4State-conf{args.conf}-param.csv"
    m, audit = build_model(csv_path, args.conf, args.temperature_K)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"conf{args.conf}_T{args.temperature_K:g}K"
    # Prepare-only outputs can be recreated deterministically.
    audit_path = args.output_dir/f"{prefix}_input_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (args.output_dir/f"{prefix}_model_au.json").write_text(
        json.dumps(model_json(m), indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if args.prepare_only:
        return
    if args.t_end_fs <= 0 or args.output_dt_fs <= 0:
        raise ValueError("Time and output spacing must be positive.")
    nstep = int(round(args.t_end_fs/args.output_dt_fs))
    if nstep < 1 or not np.isclose(nstep*args.output_dt_fs, args.t_end_fs,
                                  rtol=0, atol=1e-9):
        raise ValueError("t-end-fs must be a positive multiple of output-dt-fs.")
    import time
    stem = (f"{prefix}_N{args.ntraj}_t{args.t_end_fs:g}fs_"
            f"seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}")
    data_path = args.output_dir/f"{stem}.npz"
    fig_path = args.output_dir/f"{stem}.png"
    summary_path = args.output_dir/f"{stem}_summary.json"
    if any(path.exists() for path in [data_path, fig_path, summary_path]):
        raise FileExistsError("Output already exists; choose another output directory.")
    result = simulate(
        m, ntraj=args.ntraj, seed=args.seed,
        t_end=args.t_end_fs/AU_TIME_FS, ntime=nstep+1,
        rtol=args.rtol, atol=args.atol,
        max_step=args.max_step_fs/AU_TIME_FS)
    arrays = {k:v for k,v in result.items() if isinstance(v, np.ndarray)}
    metadata = dict(
        audit=audit, summary=result["summary"],
        method="Ehrenfest with DOP853 (not NE-FGR)",
        integration=dict(
            rtol=args.rtol, atol=args.atol, max_step_fs=args.max_step_fs,
            output_dt_fs=args.output_dt_fs, t_end_fs=args.t_end_fs),
        note="Small ensemble/short-time implementation check; not converged.",
        constants=dict(
            hartree_ev=HARTREE_EV, au_time_fs=AU_TIME_FS,
            kb_hartree_K=KB_HARTREE_K))
    np.savez_compressed(
        data_path, **arrays, times_fs=result["times"]*AU_TIME_FS,
        omega_au=m.omega, shifts=m.shifts, ground_shift=m.ground_shift,
        epsilon_au=m.epsilon, gamma_au=m.gamma,
        metadata_json=np.array(json.dumps(metadata)))
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Dynamics summary:",json.dumps(result["summary"], indent=2), flush=True)
    # Only convert the plotting time axis; saved result["times"] remains atomic time.
    for_plot = dict(result)
    for_plot["times"] = result["times"]*AU_TIME_FS
    fig = make_figure(for_plot, m)
    for ax in fig.axes:
        ax.set_xlabel("Time (fs)")
    fig.suptitle(
        f"CPC60 MSH Conf. {args.conf}, {args.temperature_K:g} K | "
        f"Ehrenfest, N={args.ntraj}, 600 nuclear coordinates")
    fig.savefig(fig_path, dpi=180)
    print(f"Saved {data_path.resolve()}\nSaved {fig_path.resolve()}")
    import matplotlib.pyplot as plt
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
