# -*- coding: utf-8 -*-
"""Лабораторная работа №1.2. Регрессионный анализ."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
RES = BASE / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Tahoma"]
plt.rcParams["axes.unicode_minus"] = False

ALPHA = 0.05
PLOT_N = 8000
RNG = np.random.default_rng(42)
TERMS = ["bedrooms", "square_feet", "pets_yes"]

VAR_RU = {
    "price": "Арендная плата, USD/мес.",
    "square_feet": "Площадь, кв. футы",
    "bedrooms": "Число спален",
    "pets_yes": "Можно с животными (фиктивная)",
}


def sturges_bins(series: pd.Series) -> tuple[int, float]:
    """Число и ширина интервалов по формуле Стерджесса."""
    n = len(series.dropna())
    k = max(1, int(round(1 + 3.322 * math.log10(n))))
    width = (series.max() - series.min()) / k
    return k, width


def load_data() -> pd.DataFrame:
    """Очищенный массив аренды."""
    df = pd.read_csv(BASE / "data_clean.csv")
    df["pets_yes"] = df["pets_yes"].astype(int)
    return df


def standardized_residuals(resid: np.ndarray) -> np.ndarray:
    """Остаток, делённый на своё стандартное отклонение."""
    s = float(resid.std(ddof=1))
    return resid / s if s > 0 else resid * np.nan


def equation_line(params: pd.Series, terms: list[str], left: str) -> str:
    """Уравнение по оценённым коэффициентам."""
    parts = [f"{float(params.get('const', 0.0)):.6f}"]
    for term in terms:
        if term in params.index:
            coef = float(params[term])
            parts.append(f" {'+' if coef >= 0 else '-'} {abs(coef):.6f}*{term}")
    return left + "".join(parts)


def adj_r2(r2: float, n: int, k: int) -> float:
    """Скорректированный коэффициент детерминации."""
    if n <= k + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def plot_residuals(
    y_hat: np.ndarray,
    resid: np.ndarray,
    std_resid: np.ndarray,
    title_prefix: str,
    file_stem: str,
) -> dict:
    """График остатков и гистограмма стандартизированных остатков."""
    n = len(resid)
    idx = RNG.choice(n, size=min(PLOT_N, n), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    fig.suptitle(title_prefix, fontsize=12, fontweight="bold")

    axes[0].scatter(y_hat[idx], resid[idx], s=6, alpha=0.35, c="steelblue")
    axes[0].axhline(0, color="crimson", lw=1.2)
    axes[0].set_xlabel("Предсказанные значения ŷ")
    axes[0].set_ylabel("Остатки e = y − ŷ")
    axes[0].set_title("График остатков")

    axes[1].hist(std_resid, bins=40, color="steelblue", edgecolor="white", density=True)
    xs = np.linspace(-4, 4, 200)
    axes[1].plot(xs, stats.norm.pdf(xs), "r-", lw=1.5, label="Нормальное N(0,1)")
    axes[1].set_xlabel("Стандартизированные остатки")
    axes[1].set_ylabel("Плотность")
    axes[1].set_title("Гистограмма стандартизированных остатков")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.savefig(FIG / f"{file_stem}_residuals.png")
    plt.close(fig)

    q = pd.qcut(y_hat, q=4, duplicates="drop")
    grouped = pd.Series(resid).groupby(q, observed=True)
    mean_by_q = grouped.mean()
    var_by_q = grouped.var(ddof=1)
    var_ratio = float(var_by_q.max() / var_by_q.min()) if var_by_q.min() > 0 else np.nan
    resid_std = float(resid.std(ddof=1))
    max_abs_qmean = float(np.max(np.abs(mean_by_q.to_numpy(dtype=float))))
    return {
        "resid_mean": float(resid.mean()),
        "resid_std": resid_std,
        "max_abs_quartile_mean": max_abs_qmean,
        "var_ratio_q": var_ratio,
        "std_resid_skew": float(pd.Series(std_resid).skew()),
        "std_resid_kurtosis": float(pd.Series(std_resid).kurtosis()),
    }


def breusch_pagan(resid: np.ndarray, exog: np.ndarray) -> dict:
    """Тест Бройша–Пагана на постоянство дисперсии остатков."""
    try:
        lm, p, _, _ = het_breuschpagan(resid, exog)
        return {"lm": float(lm), "p": float(p), "homo_at_alpha": p >= ALPHA}
    except Exception as exc:
        return {"error": str(exc)}


def r2_bars(names, values, colors, ylabel, title, path, text_dy, figsize) -> None:
    """Столбчатое сравнение R^2."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    vals = list(values)
    bars = ax.bar(names, vals, color=colors, width=0.55)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    top = max(vals) if vals else 0
    ax.set_ylim(0, top * 1.35 if top > 0 else 1)
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + text_dy,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    fig.savefig(path)
    plt.close(fig)


def save_coef(file_stem: str, columns: dict) -> pd.DataFrame:
    """Таблица коэффициентов."""
    coef = pd.DataFrame(columns)
    coef.to_csv(RES / f"{file_stem}_coefficients.csv", encoding="utf-8-sig")
    return coef


def evaluate_ols(model, name: str, file_stem: str, formula_terms: list[str]) -> dict:
    """для МНК: остатки, уравнение, R^2, F-тест, коэффициенты."""
    y_hat = np.asarray(model.fittedvalues)
    resid = np.asarray(model.resid)
    std_resid = standardized_residuals(resid)
    n, k = int(model.nobs), int(model.df_model)
    r2, r2_adj = float(model.rsquared), float(model.rsquared_adj)
    f_stat, f_p = float(model.fvalue), float(model.f_pvalue)
    coef = save_coef(
        file_stem,
        {
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p": model.pvalues,
            "significant": model.pvalues < ALPHA,
        },
    )
    summary = {
        "name": name,
        "n": n,
        "k": k,
        "r2": r2,
        "r2_adj": r2_adj,
        "f_stat": f_stat,
        "f_pvalue": f_p,
        "model_significant": f_p < ALPHA,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "equation": equation_line(model.params, formula_terms, "ŷ = "),
        "resid_info": plot_residuals(y_hat, resid, std_resid, name, file_stem),
        "breusch_pagan": breusch_pagan(resid, model.model.exog),
        "coefficients": coef.to_dict(orient="index"),
    }
    print(f"  [{name}] R2={r2:.4f}, F_p={f_p:.3g}")
    return summary


def evaluate_glm_poisson(model, y: np.ndarray, name: str, file_stem: str, formula_terms: list[str]) -> dict:
    """для Пуассона. При φ > 1 значимость считается по квази-LR и квази-z."""
    y_hat = np.asarray(model.fittedvalues)
    raw_resid = y - y_hat
    n, k = int(model.nobs), int(model.df_model)
    llf, llnull = float(model.llf), float(model.llnull)
    pseudo_r2 = 1 - llf / llnull if llnull != 0 else np.nan
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_var = 1 - float(np.sum(raw_resid**2)) / ss_tot if ss_tot > 0 else np.nan
    dispersion = float(np.sum(np.asarray(model.resid_pearson) ** 2) / float(model.df_resid))
    se_scale = math.sqrt(dispersion)
    se_quasi = model.bse * se_scale
    z_quasi = model.params / se_quasi
    p_quasi = pd.Series(
        2 * stats.norm.sf(np.abs(np.asarray(z_quasi, dtype=float))),
        index=model.params.index,
    )
    lr_stat, lr_p = float(model.llr), float(model.llr_pvalue)
    lr_quasi = lr_stat / dispersion
    lr_quasi_p = float(stats.chi2.sf(lr_quasi, k))
    coef = save_coef(
        file_stem,
        {
            "coef": model.params,
            "std_err_poisson": model.bse,
            "std_err_quasi": se_quasi,
            "z_quasi": z_quasi,
            "p": p_quasi,
            "significant": p_quasi < ALPHA,
        },
    )
    summary = {
        "name": name,
        "n": n,
        "k": k,
        "r2": r2_var,
        "r2_adj": adj_r2(r2_var, n, k),
        "pseudo_r2_mcfadden": pseudo_r2,
        "f_stat": lr_quasi,
        "f_pvalue": lr_quasi_p,
        "model_significant": lr_quasi_p < ALPHA,
        "rmse": float(np.sqrt(np.mean(raw_resid**2))),
        "equation": equation_line(model.params, formula_terms, "log(μ) = ") + " ;  μ = E[Y] = exp(…)",
        "resid_info": plot_residuals(y_hat, raw_resid, standardized_residuals(raw_resid), name, file_stem),
        "breusch_pagan": breusch_pagan(raw_resid, np.asarray(model.model.exog, dtype=float)),
        "coefficients": coef.to_dict(orient="index"),
        "dispersion": dispersion,
        "lr_poisson": lr_stat,
        "lr_poisson_p": lr_p,
    }
    print(f"  [{name}] R2(var)={r2_var:.4f}, quasi-LR p={lr_quasi_p:.3g}, phi={dispersion:.2f}")
    return summary


def points_1_to_3(df: pd.DataFrame) -> None:
    """П.1–3. Описание переменных, дескриптивный анализ, гистограммы Стерджесса."""
    print("\n=== П.1–3 ===")
    n = len(df)
    cols = ["price", "bedrooms", "square_feet"]
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurtosis()
    desc.to_csv(RES / "01_03_descriptive.csv", encoding="utf-8-sig")

    norm_rows = []
    for col in cols:
        x = df[col].astype(float).to_numpy()
        jb, jp = stats.jarque_bera(x)
        norm_rows.append(
            {
                "variable": col,
                "skew": float(stats.skew(x)),
                "kurtosis_excess": float(stats.kurtosis(x)),
                "jarque_bera": float(jb),
                "p_value": float(jp),
                "normal_at_alpha": bool(jp >= ALPHA),
            }
        )
    pd.DataFrame(norm_rows).to_csv(RES / "01_03_normality.csv", index=False, encoding="utf-8-sig")
    n_no = int((df["pets_yes"] == 0).sum())
    n_yes = int((df["pets_yes"] == 1).sum())
    pd.DataFrame(
        {"pets_yes": [0, 1], "count": [n_no, n_yes], "share": [n_no / n, n_yes / n]}
    ).to_csv(RES / "01_03_pets.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    titles = {"price": "Арендная плата", "square_feet": "Площадь", "bedrooms": "Спальни"}
    for ax, col in zip(axes, ["price", "square_feet", "bedrooms"]):
        if col == "bedrooms":
            bins = sorted(df[col].dropna().unique())
            ax.hist(
                df[col],
                bins=np.arange(min(bins) - 0.5, max(bins) + 1.5, 1),
                color="steelblue",
                edgecolor="white",
            )
        else:
            k, _ = sturges_bins(df[col])
            ax.hist(df[col], bins=k, color="steelblue", edgecolor="white")
        ax.set_title(titles[col])
        ax.set_xlabel(VAR_RU[col])
        ax.set_ylabel("Частота")
    fig.savefig(FIG / "03_histograms.png")
    plt.close(fig)


def point_4(df: pd.DataFrame) -> None:
    """Для арендной платы берётся линейная модель, не пуассоновская."""
    print("\n=== П.4 ===")
    y = df["price"]
    print(f"  Y=price, диапазон {y.min():.0f}…{y.max():.0f}, среднее {y.mean():.2f}")
    print("  модель пунктов 5–10: линейная")


def points_5_to_10_linear(df: pd.DataFrame) -> dict:
    """Линейная регрессия с фиктивной pets_yes."""
    print("\n=== П.5–10 ===")
    y = df["price"].astype(float)
    model = sm.OLS(y, sm.add_constant(df[TERMS].astype(float))).fit()
    return evaluate_ols(model, "Линейная регрессия (МНК) + фиктивная", "05_linear", TERMS)


def point_11_poisson(df: pd.DataFrame, linear: dict) -> dict:
    """Пуассоновская регрессия и сравнение с линейной."""
    print("\n=== П.11 ===")
    y = np.clip(np.rint(df["price"].astype(float)).astype(int), 0, None)
    X = sm.add_constant(df[TERMS].astype(float))
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    null = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson()).fit()
    model.llnull = null.llf
    model.llr = 2 * (model.llf - null.llf)
    model.llr_pvalue = stats.chi2.sf(model.llr, model.df_model)
    summary = evaluate_glm_poisson(
        model, y.astype(float), "Пуассоновская регрессия + фиктивная", "11_poisson", TERMS
    )
    r2_bars(
        ["Линейная (п.5)", "Пуассоновская (п.11)"],
        [linear["r2"], summary["r2"]],
        ["steelblue", "darkorange"],
        "R² (для пуассона — по дисперсии Y)",
        "П.11. Сравнение: линейная и пуассоновская модели",
        FIG / "11_linear_vs_poisson.png",
        0.008,
        (7, 4.8),
    )
    return summary


def point_12_nonlinear(df: pd.DataFrame, linear: dict) -> dict:
    """Нелинейная регрессия по площади: x, x^2, x^2*log(x), спальни и pets_yes."""
    print("\n=== П.12 ===")
    x = df["square_feet"].astype(float).to_numpy()
    design = pd.DataFrame(
        {
            "square_feet": x,
            "square_feet_sq": x**2,
            "square_feet_sq_log": (x**2) * np.log(x),
            "bedrooms": df["bedrooms"].astype(float).to_numpy(),
            "pets_yes": df["pets_yes"].astype(float).to_numpy(),
        }
    )
    y = df["price"].astype(float)
    X = sm.add_constant(design)
    model = sm.OLS(y, X).fit()
    exog = np.asarray(X, dtype=float)
    vif = {
        name: float(variance_inflation_factor(exog, i))
        for i, name in enumerate(X.columns)
        if name != "const"
    }
    pd.Series(vif, name="VIF").to_csv(RES / "12_nonlinear_vif.csv", encoding="utf-8-sig")
    summary = evaluate_ols(
        model,
        "Нелинейная регрессия (МНК) + фиктивная",
        "12_nonlinear",
        list(design.columns),
    )
    summary["vif"] = vif
    summary["condition_number"] = float(model.condition_number)
    r2_bars(
        ["Линейная (п.5)", "Нелинейная (п.12)"],
        [linear["r2"], summary["r2"]],
        ["steelblue", "seagreen"],
        "R²",
        "П.12. Сравнение: линейная и нелинейная модели",
        FIG / "12_linear_vs_nonlinear.png",
        0.008,
        (7, 4.8),
    )
    return summary


def point_13(linear: dict, poisson: dict, nonlinear: dict) -> None:
    """Сводная таблица трёх моделей."""
    print("\n=== П.13 ===")
    rows = []
    for summary, label in (
        (linear, "Линейная (п.5)"),
        (poisson, "Пуассоновская (п.11)"),
        (nonlinear, "Нелинейная (п.12)"),
    ):
        rows.append(
            {
                "модель": label,
                "R2": summary["r2"],
                "R2_adj": summary["r2_adj"],
                "RMSE": summary["rmse"],
                "pvalue_модели": summary["f_pvalue"],
                "значима": summary["model_significant"],
                "среднее_остатков": summary["resid_info"]["resid_mean"],
                "отношение_дисперсий_квартили": summary["resid_info"]["var_ratio_q"],
            }
        )
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RES / "13_model_comparison.csv", index=False, encoding="utf-8-sig")
    r2_bars(
        ["Линейная\n(п.5)", "Пуассоновская\n(п.11)", "Нелинейная\n(п.12)"],
        cmp["R2"],
        ["steelblue", "darkorange", "seagreen"],
        "R² (для пуассона — по дисперсии Y)",
        "Итоговое сравнение качества моделей (после п.12)",
        FIG / "13_model_comparison.png",
        0.006,
        (9, 5),
    )


def main() -> None:
    df = load_data()
    print(f"n={len(df)}")
    points_1_to_3(df)
    point_4(df)
    linear = points_5_to_10_linear(df)
    poisson = point_11_poisson(df, linear)
    nonlinear = point_12_nonlinear(df, linear)
    point_13(linear, poisson, nonlinear)


if __name__ == "__main__":
    main()
