#!/usr/bin/env python3
"""多态、多模式、共享谐振环境的 Ehrenfest 轨迹平均。

模型采用质量加权坐标（每个模式质量为 1）：
    V_n(R) = epsilon[n] + 1/2 sum_mu omega[mu]^2
                                      * (R[mu]-shifts[n,mu])^2
    i*hbar*da/dt = (diag(V) + gamma) @ a
    dR/dt = P
    dP/dt = -sum_n |a[n]|^2 * omega^2 * (R-shifts[n])

运行：
    python ehrenfest_multistate_multimode.py
    python ehrenfest_multistate_multimode.py --demo three-state --ntraj 200
    python ehrenfest_multistate_multimode.py --model-json model.json --no-show

默认 two-state 是之前 M=1 的两态单模式模型。
three-state 是人为设置的三态双模式教学例子，不是文献 CPC60 参数。

JSON 输入字段（所有量必须事先统一到一套单位）：
    omega:   [Nmode]，正频率，按角频率使用，不能直接填波数或 Hz
    epsilon: [Nstate]，势阱最低点能量，不是展开后含位移常数的能量
    shifts:  [Nstate,Nmode]，激发态势阱中心
    gamma:   [Nstate,Nstate]，实部，主对角必须为零
    gamma_imag: 可选，同形状虚部，组合后的 gamma 必须厄米
    ground_shift: [Nmode]，只定义初始热态的参考中心
    kBT: 热能，必须与 epsilon 的能量单位一致，非开尔文温度
    hbar: 默认 1，必须与其余单位一致
    a0_real、a0_imag: 可选 [Nstate]，默认电子位于第一个态
    labels: 可选电子态名称；units: 可选单位说明文本

温度只通过初态采样进入，无额外摩擦、随机力或恒温器。
G 不作为传播电子态，除非显式把它加入 epsilon/gamma/shifts。
不手动归一化振幅，不跳过失败轨迹。
输出四面板图及包含轨迹、密度矩阵、重组能等的 NPZ。
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Model:
    omega: np.ndarray
    epsilon: np.ndarray
    gamma: np.ndarray
    shifts: np.ndarray
    ground_shift: np.ndarray
    kBT: float
    hbar: float = 1.0
    a0: np.ndarray | None = None
    labels: tuple | None = None
    units: str = "dimensionless"

    def __post_init__(self):

        self.omega = np.array(self.omega, dtype=float, copy=True)
        self.epsilon = np.array(self.epsilon, dtype=float, copy=True)
        self.gamma = np.array(self.gamma, dtype=complex, copy=True)
        self.shifts = np.array(self.shifts, dtype=float, copy=True)
        self.ground_shift = np.array(self.ground_shift, dtype=float, copy=True)

        if self.omega.ndim != 1 or len(self.omega) < 1:
            raise ValueError("omega 必须是非空一维数组。")
        
        if self.epsilon.ndim != 1 or len(self.epsilon) < 1:
            raise ValueError("epsilon 必须是非空一维数组。")
        
        ns, nb = len(self.epsilon), len(self.omega)
        if self.shifts.shape != (ns, nb):
            raise ValueError(f"shifts 应为 {(ns, nb)}，实际 {self.shifts.shape}")
        
        if self.gamma.shape != (ns, ns):
            raise ValueError("gamma 必须是 Nstate x Nstate 矩阵。")
        
        if self.ground_shift.shape != (nb,):
            raise ValueError("ground_shift 必须是长度 Nmode 的数组。")
        
        for arr in (self.omega, self.epsilon, self.gamma,
                    self.shifts, self.ground_shift):
            if not np.all(np.isfinite(arr)):
                raise ValueError("模型参数不能含 NaN 或 Inf。")
            
        if not np.isfinite(self.hbar) or self.hbar <= 0:
            raise ValueError("hbar 必须为正有限值。")
        
        if not np.isfinite(self.kBT) or self.kBT < 0:
            raise ValueError("kBT 必须是非负有限值。")
        
        if np.any(self.omega <= 0):
            raise ValueError("所有 omega 必须为正。")
        
        if not np.allclose(self.gamma, self.gamma.conj().T,
                           rtol=1e-12, atol=1e-14):
            raise ValueError("gamma 必须厄米；不能静默对称化。")
        
        if np.any(np.diag(self.gamma) != 0):
            raise ValueError("gamma 主对角必须为零，对角能量放入 epsilon。")
        
        if self.a0 is None:
            self.a0 = np.zeros(ns, dtype=complex)
            self.a0[0] = 1.0
        else:
            self.a0 = np.array(self.a0, dtype=complex, copy=True)

        if self.a0.shape != (ns,) or not np.all(np.isfinite(self.a0)):
            raise ValueError("a0 形状或数值错误。")
        
        if not np.isclose(np.vdot(self.a0, self.a0).real, 1.0,
                          rtol=0.0, atol=1e-12):
            raise ValueError("请提供归一化的电子初态；程序不会自动归一化。")
        
        if self.labels is None:
            self.labels = tuple(f"state {n+1}" for n in range(ns))
        else:
            self.labels = tuple(str(x) for x in self.labels)
            
        if len(self.labels) != ns:
            raise ValueError("labels 数量必须等于 Nstate。")

    @property
    def ns(self):
        return len(self.epsilon)

    @property
    def nb(self):
        return len(self.omega)


def demo_model(name):
    if name == "two-state":
        return Model(
            omega=[1.0], epsilon=[0.0, 0.0],
            gamma=[[0.0, 0.3], [0.3, 0.0]],
            shifts=[[-1.0], [1.0]], ground_shift=[-1.0], kBT=1.0,
            labels=("state 1", "state 2"),
        )
    return Model(
        omega=[1.0, 0.7], epsilon=[0.0, -0.2, 0.1],
        gamma=[[0, 0.25, 0.10], [0.25, 0, -0.15], [0.10, -0.15, 0]],
        shifts=[[-1.0, 0.0], [1.0, 0.5], [0.2, -0.8]],
        ground_shift=[-1.0, 0.0], kBT=1.0,
        labels=("demo 1", "demo 2", "demo 3"),
    )


def load_model_json(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    gamma = np.asarray(d["gamma"], float).astype(complex)
    if "gamma_imag" in d:
        imag = np.asarray(d["gamma_imag"], float)
        if imag.shape != gamma.shape:
            raise ValueError("gamma_imag 必须与 gamma 形状相同。")
        gamma += 1j*imag
    a0 = None
    if "a0_real" in d or "a0_imag" in d:
        ns = len(d["epsilon"])
        re = np.asarray(d.get("a0_real", [0.0]*ns), float)
        im = np.asarray(d.get("a0_imag", [0.0]*ns), float)
        if re.shape != (ns,) or im.shape != (ns,):
            raise ValueError("a0_real、a0_imag 长度必须等于 Nstate。")
        a0 = re + 1j*im
    return Model(
        omega=d["omega"], epsilon=d["epsilon"], gamma=gamma,
        shifts=d["shifts"], ground_shift=d["ground_shift"],
        kBT=d["kBT"], hbar=d.get("hbar", 1.0), a0=a0,
        labels=d.get("labels"), units=d.get("units", "unspecified"),
    )


def pack_state(R, P, a):
    """通用顺序：[全部 R，全部 P，全部 Re(a)，全部 Im(a)]。"""
    return np.concatenate((R, P, a.real, a.imag))


def unpack_state(y, m):
    """支持单个状态 [D] 和求解结果 [D,Ntime]。"""
    b, s = m.nb, m.ns
    R = y[:b]
    P = y[b:2*b]
    a = y[2*b:2*b+s] + 1j*y[2*b+s:2*b+2*s]
    return R, P, a


def potential_and_gradient(R, m):
    displacement = R[None, :] - m.shifts       # [Nstate,Nmode]
    gradient = displacement * m.omega[None, :]**2
    V = m.epsilon + 0.5*np.sum(displacement*gradient, axis=1)
    return V, gradient


def rhs(t, y, m):
    R, P, a = unpack_state(y, m)
    V, gradient = potential_and_gradient(R, m)
    da = -1j*(V*a + m.gamma @ a) / m.hbar
    dP = -(np.abs(a)**2 @ gradient)
    return pack_state(P, dP, da)


def sample_initial_conditions(m, ntraj, seed):
    factor = (np.ones(m.nb) if m.kBT == 0 else
              1.0/np.tanh(m.hbar*m.omega/(2*m.kBT)))
    sigma_R = np.sqrt(m.hbar/(2*m.omega)*factor)
    sigma_P = np.sqrt(m.hbar*m.omega/2*factor)
    # 独立随机流；固定模式数与 seed 时，增加轨迹数保留原有样本。
    sr, sp = np.random.SeedSequence(seed).spawn(2)
    R0 = np.random.default_rng(sr).normal(
        m.ground_shift, sigma_R, size=(ntraj, m.nb))
    P0 = np.random.default_rng(sp).normal(
        0.0, sigma_P, size=(ntraj, m.nb))
    return R0, P0, sigma_R, sigma_P


def reorganization_matrix(m):
    """最后一行/列为初始参考态 G；能量单位与 epsilon 相同。"""
    shifts = np.vstack((m.shifts, m.ground_shift))
    diff = shifts[:, None, :] - shifts[None, :, :]
    return 0.5*np.sum((diff*m.omega)**2, axis=-1)


def energy_components(R, P, a, m):
    """R/P: [Ntime,Nmode]；a: [Ntime,Nstate]。"""
    kin = 0.5*np.sum(P**2, axis=1)
    diag = np.zeros(len(R))
    for n in range(m.ns):
        Vn = m.epsilon[n] + 0.5*np.sum(
            (m.omega*(R-m.shifts[n]))**2, axis=1)
        diag += np.abs(a[:, n])**2*Vn
    coupling = np.einsum("ti,ij,tj->t", a.conj(), m.gamma, a).real
    return np.column_stack((kin, diag, coupling))


def simulate(m, ntraj=200, seed=2026, t_end=30.0, ntime=601,
             rtol=1e-9, atol=1e-11, max_step=0.05, progress=True):
    if ntraj < 2 or ntime < 2 or not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("ntraj、ntime 至少为 2，t_end 必须为正有限值。")
    if any(not np.isfinite(x) or x <= 0 for x in (rtol, atol, max_step)):
        raise ValueError("rtol、atol、max_step 必须为正有限值。")
    times = np.linspace(0.0, t_end, ntime)
    R0, P0, sigma_R, sigma_P = sample_initial_conditions(m, ntraj, seed)
    R_samples = np.empty((ntraj, ntime, m.nb))
    P_samples = np.empty_like(R_samples)
    a_samples = np.empty((ntraj, ntime, m.ns), complex)
    parts = np.empty((ntraj, ntime, 3))
    for ell in range(ntraj):
        y0 = pack_state(R0[ell], P0[ell], m.a0)
        sol = solve_ivp(
            rhs, (0.0, t_end), y0, args=(m,), t_eval=times,
            method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
        )
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise RuntimeError(f"Trajectory {ell} failed: {sol.message}")
        R, P, a = unpack_state(sol.y, m)
        R_samples[ell], P_samples[ell], a_samples[ell] = R.T, P.T, a.T
        parts[ell] = energy_components(R.T, P.T, a.T, m)
        if progress and ((ell+1) % max(1, ntraj//10) == 0 or ell+1 == ntraj):
            print(f"Completed {ell+1}/{ntraj}", flush=True)

    rho_avg = np.einsum(
        "lti,ltj->tij", a_samples, a_samples.conj(), optimize=True)/ntraj
    pops = np.abs(a_samples)**2
    pop_avg = rho_avg.diagonal(axis1=1, axis2=2).real
    pop_sem = pops.std(axis=0, ddof=1)/np.sqrt(ntraj)
    norm_error = np.abs(pops.sum(axis=2)-1)
    energy = parts.sum(axis=2)
    drift = energy-energy[:, :1]
    trace_error = np.abs(np.trace(rho_avg, axis1=1, axis2=2)-1)
    herm_error = np.max(np.abs(
        rho_avg-rho_avg.conj().transpose(0, 2, 1)))
    pop0 = np.abs(m.a0)**2
    # 所有量均为质量加权形式，初始电子态可为叠加态。
    initial_V_mean = m.epsilon + 0.5*np.sum(
        m.omega**2*(sigma_R**2+(m.ground_shift-m.shifts)**2), axis=1)
    theory_energy = (
        0.5*np.sum(sigma_P**2) + pop0 @ initial_V_mean
        + np.vdot(m.a0, m.gamma @ m.a0).real)
    summary = {
        "nstate": m.ns, "nmode": m.nb, "ntraj": ntraj, "seed": seed,
        "max_norm_error": float(norm_error.max()),
        "max_energy_drift": float(np.abs(drift).max()),
        "max_trace_error": float(trace_error.max()),
        "max_hermiticity_error": float(herm_error),
        "minimum_density_eigenvalue": float(np.linalg.eigvalsh(rho_avg).min()),
        "sample_mean_initial_energy": float(energy[:, 0].mean()),
        "theory_mean_initial_energy": float(theory_energy),
        "initial_energy_mean_sem": float(energy[:, 0].std(ddof=1)/np.sqrt(ntraj)),
        "final_populations": pop_avg[-1].tolist(),
        "final_population_sem": pop_sem[-1].tolist(),
    }
    return dict(
        times=times, R0=R0, P0=P0, sigma_R=sigma_R, sigma_P=sigma_P,
        R_samples=R_samples, P_samples=P_samples, a_samples=a_samples,
        rho_avg=rho_avg, pop_avg=pop_avg, pop_sem=pop_sem,
        norm_error=norm_error, energy_drift=drift, energy_parts=parts,
        trace_error=trace_error, reorganization=reorganization_matrix(m),
        summary=summary,
    )


def make_figure(r, m):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    t = r["times"]
    for n, label in enumerate(m.labels):
        line, = axs[0, 0].plot(t, r["pop_avg"][:, n], label=label)
        axs[0, 0].fill_between(
            t, r["pop_avg"][:, n]-2*r["pop_sem"][:, n],
            r["pop_avg"][:, n]+2*r["pop_sem"][:, n],
            color=line.get_color(), alpha=0.16)
    axs[0, 0].set(title="Populations: mean +/- 2 SEM",
                  ylabel="Population", ylim=(-0.03, 1.03))
    for ax, error, title in (
        (axs[0, 1], r["norm_error"], "Norm error"),
        (axs[1, 0], np.abs(r["energy_drift"]), "Energy error"),
    ):
        # 零值只在绘图时使用下限，原始数据不作修改。
        ax.semilogy(t, np.maximum(error.max(axis=0), 1e-18), label="Max")
        ax.semilogy(t, np.maximum(error.mean(axis=0), 1e-18), "--",
                    label="Mean absolute")
        ax.set(title=title, ylabel="Absolute error")
    tracked = min(1, m.ns-1)
    count = len(r["R0"])
    for ell in np.unique(np.linspace(0, count-1, min(count, 8), dtype=int)):
        axs[1, 1].plot(t, np.abs(r["a_samples"][ell, :, tracked])**2,
                       color="0.5", alpha=0.3, lw=0.8)
    axs[1, 1].plot(t, r["pop_avg"][:, tracked], lw=2, label="Mean")
    axs[1, 1].set(title=f"Selected trajectories: {m.labels[tracked]}",
                  ylabel="Population", ylim=(-0.03, 1.03))
    for ax in axs.flat:
        ax.set_xlabel("Time")
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"{m.ns} states, {m.nb} modes, "
                 f"{count} trajectories | units: {m.units}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--demo", choices=["two-state", "three-state"],
                        default="two-state")
    source.add_argument("--model-json", type=Path)
    p.add_argument("--ntraj", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--t-end", type=float, default=30.0)
    p.add_argument("--ntime", type=int, default=601)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-11)
    p.add_argument("--max-step", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=Path("multimode_results"))
    p.add_argument("--tag", default=None, help="可选输出文件前缀")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = load_model_json(args.model_json) if args.model_json else demo_model(args.demo)
    if args.model_json is None and args.demo == "three-state":
        print("Teaching example only: these are NOT CPC60 literature parameters.")
    result = simulate(
        m, args.ntraj, args.seed, args.t_end, args.ntime,
        args.rtol, args.atol, args.max_step)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print("Reorganization matrix order:", (*m.labels, "G (initial reference)"))
    print(result["reorganization"])
    print("SEM measures sampling error only. Conservation is checked on t_eval.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import time
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    stem = f"{tag}_Ns{m.ns}_Nb{m.nb}_N{args.ntraj}_seed{args.seed}"
    fig_path = args.output_dir/f"{stem}.png"
    data_path = args.output_dir/f"{stem}.npz"
    # 两个输出的共同前缀不得意外覆盖已有结果。
    if fig_path.exists() or data_path.exists():
        raise FileExistsError("输出已存在，请换 --tag 或 --output-dir。")
    metadata = dict(
        summary=result["summary"], labels=list(m.labels), units=m.units,
        hbar=m.hbar, kBT=m.kBT, method="DOP853", rtol=args.rtol,
        atol=args.atol, max_step=args.max_step,
        model_source=str(args.model_json) if args.model_json else args.demo,
        state_order="R(all), P(all), Re(a)(all), Im(a)(all)",
    )
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(
        data_path, **arrays, omega=m.omega, epsilon=m.epsilon,
        gamma=m.gamma, shifts=m.shifts, ground_shift=m.ground_shift,
        a0=m.a0, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    fig = make_figure(result, m)
    fig.savefig(fig_path, dpi=180)
    print(f"Saved data: {data_path.resolve()}")
    print(f"Saved figure: {fig_path.resolve()}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
