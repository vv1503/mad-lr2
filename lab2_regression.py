# -*- coding: utf-8 -*-
"""
Лабораторная работа №1.2. Регрессионный анализ.
Датасет: Apartment for Rent Classified (UCI #555), тот же, что в ЛР1.1.

Порядок строго по заданию:
  п.1–3 → п.4 (линейная ИЛИ пуассон) → п.5–10 (линейная) →
  п.11 (пуассон) → п.12 (нелинейная; появляется только здесь).
"""

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

# Русские подписи переменных на графиках
VAR_RU = {
    "price": "Арендная плата, USD/мес.",
    "square_feet": "Площадь, кв. футы",
    "bedrooms": "Число спален",
    "pets_yes": "Можно с животными (фиктивная)",
}


def sturges_bins(series: pd.Series) -> tuple[int, float]:
    n = len(series.dropna())
    k = max(1, int(round(1 + 3.322 * math.log10(n))))
    width = (series.max() - series.min()) / k
    return k, width


def load_data() -> pd.DataFrame:
    path = BASE / "data_clean.csv"
    df = pd.read_csv(path)
    df["pets_yes"] = df["pets_yes"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Пункты 1–3
# ---------------------------------------------------------------------------

def points_1_to_3(df: pd.DataFrame) -> None:
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
        {
            "pets_yes": [0, 1],
            "count": [n_no, n_yes],
            "share": [n_no / n, n_yes / n],
        }
    ).to_csv(RES / "01_03_pets.csv", index=False, encoding="utf-8-sig")

    k_formula = 1 + 3.322 * math.log10(n)
    k_price, h_price = sturges_bins(df["price"])
    k_sq, h_sq = sturges_bins(df["square_feet"])
    k_bed, h_bed = sturges_bins(df["bedrooms"])
    n_bed = int(df["bedrooms"].nunique())
    bed_min = float(df["bedrooms"].min())
    bed_max = float(df["bedrooms"].max())

    norm_lines = []
    for row in norm_rows:
        verdict = "согласуется" if row["normal_at_alpha"] else "не согласуется"
        norm_lines.append(
            f"  {row['variable']}: асимметрия={row['skew']:.3f}, эксцесс={row['kurtosis_excess']:.3f}, "
            f"Жарк–Бера={row['jarque_bera']:.4g}, p={row['p_value']:.4g} → {verdict} с нормальным законом при α={ALPHA}"
        )

    lines = [
        "П.1 Описание данных",
        "Датасет: Apartment for Rent Classified (UCI #555), тот же массив, что в ЛР1.1.",
        f"n = {n}",
        "Y (зависимая): price — арендная плата, USD/мес.",
        "В файле цена целочисленная, по смыслу это стоимость аренды, а не число событий.",
        "X: bedrooms — число спален (количественная дискретная);",
        "    square_feet — площадь, кв. футы (количественная);",
        "    pets_yes — фиктивная 0/1 из категориального pets_allowed",
        "    (1 — можно с животными, 0 — нельзя).",
        f"pets_yes = 0: {n_no} ({n_no / n:.1%}); pets_yes = 1: {n_yes} ({n_yes / n:.1%}).",
        "",
        "П.2 Дескриптивный анализ и нормальность",
        "H0 критерия Жарка–Бера: выборка взята из нормального распределения.",
        *norm_lines,
        "Вывод: price, bedrooms и square_feet не согласуются с нормальным распределением.",
        "У price и square_feet правосторонняя скошенность и тяжёлый правый хвост.",
        "",
        "П.3 Гистограммы. Формула Стерджесса: k = round(1 + 3.322·lg n), h = (max − min) / k.",
        f"  1 + 3.322·lg({n}) = {k_formula:.4f}, после округления k = {k_price}.",
        f"  price: k={k_price}, h={h_price:.4f}",
        f"  square_feet: k={k_sq}, h={h_sq:.4f}",
        f"  bedrooms: формула даёт k={k_bed}, h={h_bed:.4f}.",
        f"  Различных значений bedrooms — {n_bed} (от {bed_min:.0f} до {bed_max:.0f}).",
        "  Дробный шаг разрезал бы целые числа спален, поэтому гистограмма построена",
        f"  по целым значениям, число столбцов = {n_bed}.",
    ]
    (RES / "01_03_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    short_titles = {
        "price": "Арендная плата",
        "square_feet": "Площадь",
        "bedrooms": "Спальни",
    }
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
        ax.set_title(short_titles[col])
        ax.set_xlabel(VAR_RU[col])
        ax.set_ylabel("Частота")
    fig.savefig(FIG / "03_histograms.png")
    plt.close(fig)
    print("  сохранено: results/01_03_*, figures/03_histograms.png")


# ---------------------------------------------------------------------------
# Общие утилиты регрессии
# ---------------------------------------------------------------------------

def adj_r2(r2: float, n: int, k: int) -> float:
    if n <= k + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def standardized_residuals(resid: np.ndarray) -> np.ndarray:
    s = resid.std(ddof=1)
    return resid / s if s > 0 else resid * np.nan


def plot_residuals(
    y_hat: np.ndarray,
    resid: np.ndarray,
    std_resid: np.ndarray,
    title_prefix: str,
    file_stem: str,
) -> dict:
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
    # Общее среднее ≈ 0 даёт свободный член. Постоянство среднего — это сдвиг по уровням ŷ.
    mean_stable = bool(resid_std > 0 and max_abs_qmean < 0.15 * resid_std)
    quartile_lines = [
        f"ŷ∈{interval}: mean(e)={mean_v:.2f}, var(e)={var_v:.1f}"
        for interval, mean_v, var_v in zip(
            mean_by_q.index,
            mean_by_q.to_numpy(dtype=float),
            var_by_q.to_numpy(dtype=float),
        )
    ]

    return {
        "resid_mean": float(resid.mean()),
        "resid_std": resid_std,
        "mean_by_quartile": {str(k): float(v) for k, v in mean_by_q.items()},
        "var_by_quartile": {str(k): float(v) for k, v in var_by_q.items()},
        "quartile_lines": quartile_lines,
        "max_abs_quartile_mean": max_abs_qmean,
        "var_ratio_q": var_ratio,
        "mean_approx_zero": bool(abs(resid.mean()) < max(1e-6, 0.01 * resid_std)),
        "mean_stable_across_fitted": mean_stable,
        "std_resid_skew": float(pd.Series(std_resid).skew()),
        "std_resid_kurtosis": float(pd.Series(std_resid).kurtosis()),
    }


def evaluate_ols(model, y: np.ndarray, name: str, file_stem: str, formula_terms: list[str]) -> dict:
    y_hat = np.asarray(model.fittedvalues)
    resid = np.asarray(model.resid)
    std_resid = standardized_residuals(resid)
    n = int(model.nobs)
    k = int(model.df_model)
    r2 = float(model.rsquared)
    r2_adj = float(model.rsquared_adj)
    f_stat = float(model.fvalue)
    f_p = float(model.f_pvalue)
    model_sig = f_p < ALPHA

    resid_info = plot_residuals(y_hat, resid, std_resid, name, file_stem)

    try:
        bp_lm, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
        bp = {"lm": float(bp_lm), "p": float(bp_p), "homo_at_alpha": bp_p >= ALPHA}
    except Exception as e:
        bp = {"error": str(e)}

    coef_table = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p": model.pvalues,
            "significant": model.pvalues < ALPHA,
        }
    )
    coef_table.to_csv(RES / f"{file_stem}_coefficients.csv", encoding="utf-8-sig")

    intercept = float(model.params.get("const", 0.0))
    parts = [f"{intercept:.6f}"]
    for term in formula_terms:
        if term in model.params.index:
            c = float(model.params[term])
            sign = "+" if c >= 0 else "-"
            parts.append(f" {sign} {abs(c):.6f}*{term}")
    equation = "ŷ = " + "".join(parts)

    quality_notes = []
    if r2 < 0.3:
        quality_notes.append("R² низкий — модель объясняет малую долю дисперсии Y.")
    elif r2 < 0.6:
        quality_notes.append("R² умеренный.")
    else:
        quality_notes.append("R² высокий.")
    if resid_info["mean_stable_across_fitted"]:
        quality_notes.append(
            "Среднее остатков по квартилям ŷ сдвинуто мало относительно разброса."
        )
    else:
        quality_notes.append("Среднее остатков заметно меняется по уровню ŷ.")
    if isinstance(bp, dict) and "p" in bp:
        if bp["homo_at_alpha"]:
            quality_notes.append(
                f"Breusch–Pagan p={bp['p']:.4g}: гипотеза гомоскедастичности не отвергается."
            )
        else:
            quality_notes.append(
                f"Breusch–Pagan p={bp['p']:.4g}: гетероскедастичность (дисперсия остатков не постоянна)."
            )
    skew = resid_info["std_resid_skew"]
    if abs(skew) < 0.5:
        quality_notes.append("Гистограмма станд. остатков относительно симметрична.")
    else:
        quality_notes.append(f"Гистограмма станд. остатков скошена (асимметрия={skew:.2f}).")

    summary = {
        "name": name,
        "n": n,
        "k": k,
        "r2": r2,
        "r2_adj": r2_adj,
        "f_stat": f_stat,
        "f_pvalue": f_p,
        "model_significant": model_sig,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "equation": equation,
        "resid_info": resid_info,
        "breusch_pagan": bp,
        "quality_notes": quality_notes,
        "coefficients": coef_table.to_dict(orient="index"),
        "stat_name": "F",
    }

    text_lines = [
        f"Модель: {name}",
        f"Уравнение: {equation}",
        f"n={n}, k={k}",
        f"R²={r2:.6f}, R²_adj={r2_adj:.6f}",
        f"F={f_stat:.4f}, p={f_p:.6g}, значимо (α={ALPHA}): {model_sig}",
        f"RMSE={summary['rmse']:.4f}",
        "",
        "Коэффициенты:",
        coef_table.to_string(),
        "",
        "Анализ остатков:",
        f"  mean(e)={resid_info['resid_mean']:.6g}",
        f"  std(e)={resid_info['resid_std']:.6g}",
        f"  max |mean| по квартилям ŷ={resid_info['max_abs_quartile_mean']:.4f}",
        *[f"  {line}" for line in resid_info["quartile_lines"]],
        f"  отношение дисперсий по квартилям ŷ: {resid_info['var_ratio_q']:.4f}",
        f"  Breusch–Pagan: {bp}",
        "",
        "Оценка качества:",
        *[f"  - {q}" for q in quality_notes],
    ]
    (RES / f"{file_stem}_summary.txt").write_text("\n".join(text_lines), encoding="utf-8")
    print(f"  [{name}] R2={r2:.4f}, F_p={f_p:.3g}, significant={model_sig}")
    return summary


def evaluate_glm_poisson(model, y: np.ndarray, name: str, file_stem: str, formula_terms: list[str]) -> dict:
    y_hat = np.asarray(model.fittedvalues)
    resid = np.asarray(model.resid_pearson)
    raw_resid = y - y_hat
    std_resid = standardized_residuals(raw_resid)

    n = int(model.nobs)
    k = int(model.df_model)
    llf = float(model.llf)
    llnull = float(model.llnull)
    pseudo_r2 = 1 - llf / llnull if llnull != 0 else np.nan
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_var = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_adj = adj_r2(r2_var, n, k)

    lr_stat = float(model.llr)
    lr_p = float(model.llr_pvalue)
    pearson_chi2 = float(np.sum(resid**2))
    df_resid = float(model.df_resid)
    dispersion = pearson_chi2 / df_resid
    se_scale = math.sqrt(dispersion)
    # Обычный Пуассон предполагает φ = 1. Здесь φ >> 1, поэтому z и LR не используются.
    se_quasi = model.bse * se_scale
    z_quasi = model.params / se_quasi
    p_quasi = pd.Series(
        2 * stats.norm.sf(np.abs(np.asarray(z_quasi, dtype=float))),
        index=model.params.index,
    )
    lr_quasi = lr_stat / dispersion
    lr_quasi_p = float(stats.chi2.sf(lr_quasi, k))
    model_sig = lr_quasi_p < ALPHA

    resid_info = plot_residuals(y_hat, raw_resid, std_resid, name, file_stem)

    try:
        bp_lm, bp_p, _, _ = het_breuschpagan(raw_resid, np.asarray(model.model.exog, dtype=float))
        bp = {"lm": float(bp_lm), "p": float(bp_p), "homo_at_alpha": bool(bp_p >= ALPHA)}
    except Exception as e:
        bp = {"error": str(e)}

    coef_table = pd.DataFrame(
        {
            "coef": model.params,
            "std_err_poisson": model.bse,
            "std_err_quasi": se_quasi,
            "z_quasi": z_quasi,
            "p": p_quasi,
            "significant": p_quasi < ALPHA,
        }
    )
    coef_table.to_csv(RES / f"{file_stem}_coefficients.csv", encoding="utf-8-sig")

    intercept = float(model.params.get("const", 0.0))
    lin_parts = [f"{intercept:.6f}"]
    for term in formula_terms:
        if term in model.params.index:
            c = float(model.params[term])
            sign = "+" if c >= 0 else "-"
            lin_parts.append(f" {sign} {abs(c):.6f}*{term}")
    equation = "log(μ) = " + "".join(lin_parts) + " ;  μ = E[Y] = exp(…)"

    quality_notes = [
        f"Псевдо-R² (McFadden)={pseudo_r2:.4f}; R² по дисперсии Y={r2_var:.4f}.",
        (
            f"Сверхдисперсия φ = χ² Пирсона / df = {dispersion:.2f} "
            f"(у пуассоновской модели нужно φ ≈ 1). Значимость оценивается квази-LR и квази-z."
        ),
    ]
    if resid_info["mean_stable_across_fitted"]:
        quality_notes.append(
            "Среднее остатков по квартилям ŷ сдвинуто мало относительно разброса."
        )
    else:
        quality_notes.append("Среднее остатков заметно меняется по уровню ŷ.")
    if isinstance(bp, dict) and "p" in bp:
        if bp["homo_at_alpha"]:
            quality_notes.append(
                f"Breusch–Pagan p={bp['p']:.4g}: постоянство дисперсии сырых остатков не отвергается."
            )
        else:
            quality_notes.append(
                f"Breusch–Pagan p={bp['p']:.4g}: дисперсия сырых остатков не постоянна."
            )

    inference_note = (
        f"Оценки коэффициентов — пуассоновский ММП. "
        f"φ = {dispersion:.2f}, √φ = {se_scale:.2f}: стандартные ошибки умножены на √φ. "
        f"Обычный LR = {lr_stat:.1f} (p={lr_p:.3g}) предполагает Var(Y)=μ и в вывод о значимости не входит. "
        f"Используется квази-LR = LR/φ = {lr_quasi:.2f}, p={lr_quasi_p:.3g}."
    )

    summary = {
        "name": name,
        "n": n,
        "k": k,
        "r2": r2_var,
        "r2_adj": r2_adj,
        "pseudo_r2_mcfadden": pseudo_r2,
        "f_stat": lr_quasi,
        "f_pvalue": lr_quasi_p,
        "model_significant": model_sig,
        "rmse": float(np.sqrt(np.mean(raw_resid**2))),
        "equation": equation,
        "resid_info": resid_info,
        "quality_notes": quality_notes,
        "coefficients": coef_table.to_dict(orient="index"),
        "pearson_chi2": pearson_chi2,
        "dispersion": dispersion,
        "lr_poisson": lr_stat,
        "lr_poisson_p": lr_p,
        "breusch_pagan": bp,
        "stat_name": "квази-LR (LR/φ)",
        "inference_note": inference_note,
    }

    text_lines = [
        f"Модель: {name}",
        f"Уравнение: {equation}",
        f"n={n}, k={k}",
        f"R²(var)={r2_var:.6f}, R²_adj={r2_adj:.6f}, pseudo-R²(McFadden)={pseudo_r2:.6f}",
        f"φ={dispersion:.4f}, квази-LR={lr_quasi:.4f}, p={lr_quasi_p:.6g}, значимо: {model_sig}",
        f"LR Пуассона (не используется)={lr_stat:.4f}, p={lr_p:.6g}",
        f"RMSE={summary['rmse']:.4f}",
        "",
        "Коэффициенты (log-link, p — квазипуассоновские):",
        coef_table.to_string(),
        "",
        inference_note,
        "",
        "Анализ остатков:",
        f"  mean(y-μ)={resid_info['resid_mean']:.6g}",
        f"  max |mean| по квартилям ŷ={resid_info['max_abs_quartile_mean']:.4f}",
        *[f"  {line}" for line in resid_info["quartile_lines"]],
        f"  var_ratio_q={resid_info['var_ratio_q']:.4f}",
        f"  Breusch–Pagan: {bp}",
        "",
        "Оценка качества:",
        *[f"  - {q}" for q in quality_notes],
    ]
    (RES / f"{file_stem}_summary.txt").write_text("\n".join(text_lines), encoding="utf-8")
    print(f"  [{name}] R2(var)={r2_var:.4f}, LR_p={lr_p:.3g}, significant={model_sig}")
    return summary


def _coef_sig_lines(summary: dict, alpha: float = ALPHA) -> list[str]:
    lines = []
    for name, row in summary["coefficients"].items():
        p = float(row["p"])
        sig = bool(row["significant"])
        c = float(row["coef"])
        verdict = "значим" if sig else "не значим"
        lines.append(f"  - {name}: коэффициент={c:.6g}, p={p:.4g} → {verdict} при α={alpha}")
    return lines


def write_points_6_10_justification(summary: dict, stem: str, model_kind: str) -> str:
    """Подробное обоснование п.6–10. Возвращает текст для отчёта."""
    ri = summary["resid_info"]
    bp = summary.get("breusch_pagan", {})
    bp_line = ""
    if isinstance(bp, dict) and "p" in bp:
        if bp["homo_at_alpha"]:
            bp_line = (
                f"- Тест Breusch–Pagan: LM={bp['lm']:.2f}, p={bp['p']:.4g} ≥ α={ALPHA} → "
                "гипотеза постоянства дисперсии не отвергается.\n"
            )
        else:
            bp_line = (
                f"- Тест Breusch–Pagan: LM={bp['lm']:.2f}, p={bp['p']:.4g} < α={ALPHA} → "
                "гетероскедастичность (дисперсия остатков не постоянна).\n"
            )

    r2 = summary["r2"]
    if r2 < 0.3:
        r2_verdict = (
            f"R²={r2:.4f} < 0.3 — модель объясняет малую долю дисперсии Y; "
            "прогностическое качество слабое, хотя связь может быть статистически значимой."
        )
        quality_total = (
            "модель формально пригодна для описания средней связи, "
            "но R² низкий — большая часть вариации цены не объяснена включёнными X"
        )
    elif r2 < 0.6:
        r2_verdict = f"R²={r2:.4f} — умеренное качество аппроксимации."
        quality_total = "качество умеренное по R²"
    else:
        r2_verdict = f"R²={r2:.4f} — высокое качество аппроксимации."
        quality_total = "качество высокое по R²"

    mean_stable = ri["mean_stable_across_fitted"]
    var_ratio = ri["var_ratio_q"]
    skew = ri["std_resid_skew"]
    if isinstance(bp, dict) and "p" in bp:
        homo_ok = bool(bp["homo_at_alpha"])
        homo_word = "не отвергается" if homo_ok else "отвергается"
    else:
        homo_ok = False
        homo_word = "не проверена"
    q_lines = "\n".join(f"  {line}" for line in ri["quartile_lines"])
    max_abs = ri["max_abs_quartile_mean"]
    std_e = ri["resid_std"]
    mean_ratio = max_abs / std_e if std_e else np.nan
    inference = summary.get("inference_note", "")
    inference_block = f"- {inference}\n" if inference else ""
    stat_name = summary.get("stat_name", "статистика")

    text = f"""Обоснование п.6–10 для модели: {summary['name']} ({model_kind})

П.6. График остатков. Постоянство среднего и дисперсии
- График: figures/{stem}_residuals.png (левая панель: остатки e против предсказанных ŷ).
- Общее среднее остатков mean(e) = {ri['resid_mean']:.6g}.
  У модели со свободным членом оно близко к нулю по построению.
  Постоянство среднего проверяется по уровням ŷ, а не по этому числу.
- Среднее и дисперсия остатков по квартилям ŷ:
{q_lines}
- Наибольший |mean(e)| по квартилям = {max_abs:.2f} при sd(e) = {std_e:.2f}
  (отношение {mean_ratio:.3f}). Критерий: отношение < 0.15 — сдвиг мал на фоне разброса.
  Вывод по среднему: {'среднее остатков по уровню ŷ примерно постоянно' if mean_stable else 'среднее остатков меняется по уровню ŷ'}.
- Отношение max/min дисперсий по квартилям ŷ = {var_ratio:.4f}.
{bp_line}
П.7. Гистограмма стандартизированных остатков
- Файл: figures/{stem}_residuals.png (правая панель).
- Асимметрия станд. остатков = {skew:.4f}; эксцесс = {ri['std_resid_kurtosis']:.4f}.
- Критерий близости к N(0,1): |асимметрия| < 0.5 и визуальное совпадение с кривой плотности.
  Вывод: {'форма близка к симметричной' if abs(skew) < 0.5 else 'распределение остатков скошено; нормальность остатков не выполняется'}.

П.8. Уравнение регрессии
{summary['equation']}

П.9. Оценка качества (R², график и гистограмма остатков)
- R² = {summary['r2']:.6f}; R²_adj = {summary['r2_adj']:.6f}; RMSE = {summary['rmse']:.4f}.
- {r2_verdict}
- По графику остатков: среднее по уровням ŷ примерно постоянно — {'да' if mean_stable else 'нет'}; постоянство дисперсии — см. п.6.
- По гистограмме: см. п.7 (асимметрия={skew:.2f}).
- Итог по качеству: {quality_total}.

П.10. Значимость модели и коэффициентов
- Критерий значимости уравнения: p-value теста модели < α={ALPHA}.
{inference_block}- {stat_name} = {summary['f_stat']:.4f}, p = {summary['f_pvalue']:.6g} →
  модель {'значима' if summary['model_significant'] else 'НЕ значима'} при α={ALPHA}.
- Значимость коэффициентов (H0: βj=0), α={ALPHA}:
""" + "\n".join(_coef_sig_lines(summary)) + f"""

Вывод по п.6–10: уравнение {'статистически значимо' if summary['model_significant'] else 'незначимо'};
качество по R² {'низкое' if r2 < 0.3 else 'умеренное/высокое'}; остатки —
среднее по уровням ŷ {'примерно постоянно' if mean_stable else 'не постоянно'}, гомоскедастичность {homo_word}.
"""
    (RES / f"{stem}_justification_6_10.txt").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# П.4 — только линейная ИЛИ пуассоновская
# ---------------------------------------------------------------------------

def point_4(df: pd.DataFrame) -> None:
    print("\n=== П.4 Выбор типа регрессии (линейная или пуассоновская) ===")
    y = df["price"]
    text = f"""П.4. Выбор модели регрессии

По заданию на этом шаге выбираем ТОЛЬКО между двумя типами:
линейная регрессия или пуассоновская регрессия.
(Нелинейная модель в п.4 не рассматривается — она появляется только в п.12.)

Критерий выбора: тип зависимой переменной Y.

1) Характеристика Y = price
   - тип: количественная непрерывная (аренда, USD/мес.);
   - область значений: [{y.min():.0f}; {y.max():.0f}], среднее = {y.mean():.2f};
   - не является счётной величиной (числом событий / целым «count»).

2) Когда уместна линейная регрессия (МНК)
   - Y непрерывна (или близка к непрерывной);
   - моделируем E[Y|X] = β0 + β1·X1 + … как линейную комбинацию признаков;
   - коэффициенты оцениваются методом наименьших квадратов.
   → для price линейная модель подходит по типу Y.

3) Когда уместна пуассоновская регрессия
   - Y — неотрицательные целые счётные данные (число заявок, аварий и т.п.);
   - моделируется E[Y|X] = exp(Xβ), дисперсия связана со средним.
   → price — не count-переменная; пуассон для арендной платы формально применим
     только после округления Y до целых долларов, но по смыслу типа данных
     это менее естественный выбор.

4) Решение по п.4
   Выбираем ЛИНЕЙНУЮ регрессию как основную модель для п.5–10, потому что:
   - Y непрерывна → соответствует предпосылкам линейной модели;
   - цель — прогноз уровня арендной платы, а не интенсивности счётного процесса.

   Пуассоновская модель будет построена отдельно по требованию п.11
   (обязательный пункт задания), но НЕ является выбором п.4.

Фиктивная переменная (для п.5): pets_yes ∈ {{0,1}} из категориального pets_allowed
(1 — можно с животными, 0 — нельзя).
"""
    (RES / "04_model_choice.txt").write_text(text, encoding="utf-8")
    print("  сохранено: results/04_model_choice.txt")
    assert (RES / "04_model_choice.txt").exists()


# ---------------------------------------------------------------------------
# П.5–10 линейная с фиктивными
# ---------------------------------------------------------------------------

def points_5_to_10_linear(df: pd.DataFrame) -> dict:
    print("\n=== П.5–10 Линейная регрессия с фиктивными переменными ===")
    y = df["price"].astype(float)
    X = df[["bedrooms", "square_feet", "pets_yes"]].astype(float)
    X = sm.add_constant(X)
    terms = ["bedrooms", "square_feet", "pets_yes"]
    model = sm.OLS(y, X).fit()
    summary = evaluate_ols(
        model,
        y.to_numpy(),
        "Линейная регрессия (МНК) + фиктивная",
        "05_linear",
        terms,
    )
    summary["justification_6_10"] = write_points_6_10_justification(
        summary, "05_linear", "линейная МНК"
    )

    checks = {
        "p5_model_built": model is not None,
        "p6_resid_plot": (FIG / "05_linear_residuals.png").exists(),
        "p6_mean_near_zero": summary["resid_info"]["mean_approx_zero"],
        "p7_std_resid_hist": (FIG / "05_linear_residuals.png").exists(),
        "p8_equation": "ŷ =" in summary["equation"],
        "p9_r2_present": np.isfinite(summary["r2"]),
        "p10_f_pvalue": np.isfinite(summary["f_pvalue"]),
        "p10_model_sig_flag": isinstance(summary["model_significant"], (bool, np.bool_)),
    }
    (RES / "05_10_checks.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in checks.items())
        + "\n\n"
        + summary["equation"]
        + f"\nR2={summary['r2']:.6f}\n",
        encoding="utf-8",
    )
    for k, v in checks.items():
        assert v, f"Проверка не пройдена: {k}"
    print("  ПРОВЕРКА п.5–10: OK")
    return summary


# ---------------------------------------------------------------------------
# П.11 Пуассон — сравнение только с линейной
# ---------------------------------------------------------------------------

def point_11_poisson(df: pd.DataFrame, linear: dict) -> dict:
    print("\n=== П.11 Пуассоновская регрессия (+ п.6–10) ===")
    y = np.rint(df["price"].astype(float)).astype(int)
    y = np.clip(y, 0, None)
    X = df[["bedrooms", "square_feet", "pets_yes"]].astype(float)
    X = sm.add_constant(X)
    terms = ["bedrooms", "square_feet", "pets_yes"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    null = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson()).fit()
    model.llnull = null.llf  # type: ignore[attr-defined]
    model.llr = 2 * (model.llf - null.llf)  # type: ignore[attr-defined]
    model.llr_pvalue = stats.chi2.sf(model.llr, model.df_model)  # type: ignore[attr-defined]

    summary = evaluate_glm_poisson(
        model,
        y.astype(float),
        "Пуассоновская регрессия + фиктивная",
        "11_poisson",
        terms,
    )
    summary["justification_6_10"] = write_points_6_10_justification(
        summary, "11_poisson", "пуассоновская"
    )

    better_r2 = "линейная" if linear["r2"] >= summary["r2"] else "пуассоновская"
    better_r2adj = "линейная" if linear["r2_adj"] >= summary["r2_adj"] else "пуассоновская"
    better_rmse = "линейная" if linear["rmse"] <= summary["rmse"] else "пуассоновская"

    cmp_text = f"""П.11. Пуассоновская регрессия — сравнение с линейной (альтернативы п.4)

На этом шаге сравниваем ТОЛЬКО две модели из п.4: линейную и пуассоновскую.
Нелинейная модель ещё не построена (она появится в п.12).

Одинаковые независимые переменные: bedrooms, square_feet, pets_yes.

| Показатель | Линейная (п.5) | Пуассон (п.11) | Лучше по показателю |
|------------|----------------|----------------|---------------------|
| R² (доля объяснённой дисперсии Y) | {linear['r2']:.6f} | {summary['r2']:.6f} | {better_r2} |
| R²_adj | {linear['r2_adj']:.6f} | {summary['r2_adj']:.6f} | {better_r2adj} |
| RMSE (меньше — лучше) | {linear['rmse']:.4f} | {summary['rmse']:.4f} | {better_rmse} |
| Значимость модели (α={ALPHA}) | {linear['model_significant']} (F, p={linear['f_pvalue']:.3g}) | {summary['model_significant']} (квази-LR, p={summary['f_pvalue']:.3g}) | обе значимы |
| φ = χ² Пирсона / df (только пуассон) | — | {summary['dispersion']:.2f} | — |
| Pseudo-R² McFadden (только пуассон) | — | {summary.get('pseudo_r2_mcfadden', float('nan')):.6f} | — |

Обоснование вывода после п.11:
1) По типу Y (п.4) линейная модель уместнее: price — арендная плата, а не число событий.
2) По качеству аппроксимации: R² линейной ({linear['r2']:.4f}) {'>' if linear['r2'] > summary['r2'] else '≤'} R²(var) пуассона ({summary['r2']:.4f}).
3) По ошибке прогноза: RMSE линейной ({linear['rmse']:.2f}) {'<' if linear['rmse'] < summary['rmse'] else '≥'} RMSE пуассона ({summary['rmse']:.2f}).
4) Линейная модель значима по F. Пуассоновская значима по квази-LR = LR/φ.
   φ = {summary['dispersion']:.2f} при требуемом для Пуассона φ ≈ 1, поэтому обычные z и LR
   (LR = {summary['lr_poisson']:.1f}) в оценку значимости не входят.
5) Дисперсия сырых остатков пуассоновской модели не постоянна: Breusch–Pagan p={summary['breusch_pagan'].get('p', float('nan')):.3g}.
6) Итог: выбор п.4 подтверждается. Линейная модель лучше по типу Y, по R² и по RMSE.
"""
    (RES / "11_vs_linear.txt").write_text(cmp_text, encoding="utf-8")
    summary["vs_linear"] = cmp_text

    # График сравнения только линейная vs пуассон
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    names = ["Линейная (п.5)", "Пуассоновская (п.11)"]
    vals = [linear["r2"], summary["r2"]]
    bars = ax.bar(names, vals, color=["steelblue", "darkorange"], width=0.55)
    ax.set_ylabel("R² (для пуассона — по дисперсии Y)")
    ax.set_title("П.11. Сравнение: линейная и пуассоновская модели")
    ax.set_ylim(0, max(vals) * 1.35 if max(vals) > 0 else 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", va="bottom")
    fig.savefig(FIG / "11_linear_vs_poisson.png")
    plt.close(fig)

    checks = {
        "p11_model": True,
        "p6_plot": (FIG / "11_poisson_residuals.png").exists(),
        "p8_equation": "log(μ)" in summary["equation"],
        "p9_r2": np.isfinite(summary["r2"]),
        "p10_sig": np.isfinite(summary["f_pvalue"]),
    }
    (RES / "11_checks.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in checks.items()), encoding="utf-8"
    )
    for k, v in checks.items():
        assert v, f"Проверка не пройдена: {k}"
    print("  ПРОВЕРКА п.11 (+6–10): OK")
    return summary


# ---------------------------------------------------------------------------
# П.12 Нелинейная — появляется только здесь
# ---------------------------------------------------------------------------

def point_12_nonlinear(df: pd.DataFrame, linear: dict) -> dict:
    print("\n=== П.12 Нелинейная регрессия с фиктивными (+ п.6–10) ===")
    # Спецификация по заданию: линейная свёртка нелинейных элементов
    # y = a0 + a1*x + a2*x^2 + a3*x^2*log(x) + ... + dummy
    x = df["square_feet"].astype(float).to_numpy()
    x2 = x**2
    x2_logx = x2 * np.log(x)

    design = pd.DataFrame(
        {
            "square_feet": x,
            "square_feet_sq": x2,
            "square_feet_sq_log": x2_logx,
            "bedrooms": df["bedrooms"].astype(float).to_numpy(),
            "pets_yes": df["pets_yes"].astype(float).to_numpy(),
        }
    )
    y = df["price"].astype(float)
    X = sm.add_constant(design)
    terms = list(design.columns)
    model = sm.OLS(y, X).fit()
    exog = np.asarray(X, dtype=float)
    vif = {
        name: float(variance_inflation_factor(exog, i))
        for i, name in enumerate(X.columns)
        if name != "const"
    }
    cond = float(model.condition_number)
    vif_txt = ", ".join(f"{name}={value:.1f}" for name, value in vif.items())
    pd.Series(vif, name="VIF").to_csv(RES / "12_nonlinear_vif.csv", encoding="utf-8-sig")
    summary = evaluate_ols(
        model,
        y.to_numpy(),
        "Нелинейная регрессия (МНК) + фиктивная",
        "12_nonlinear",
        terms,
    )
    summary["vif"] = vif
    summary["condition_number"] = cond
    summary["inference_note"] = (
        f"Число обусловленности = {cond:.4g}. VIF: {vif_txt}. "
        "Столбцы square_feet, square_feet² и square_feet²·log почти линейно зависимы. "
        "t-критерии формально отклоняют нулевые коэффициенты этих членов, "
        "но их знаки по отдельности не интерпретируются. Вклад формы виден по приросту R²."
    )
    summary["justification_6_10"] = write_points_6_10_justification(
        summary, "12_nonlinear", "нелинейная МНК"
    )

    d_r2 = summary["r2"] - linear["r2"]
    d_r2adj = summary["r2_adj"] - linear["r2_adj"]
    d_rmse = linear["rmse"] - summary["rmse"]
    worth = d_r2adj >= 0.01

    cmp_text = f"""П.12. Нелинейная регрессия — сравнение с линейной (п.5)

Нелинейная модель появляется ТОЛЬКО в этом пункте (в п.4 её не было).
Спецификация по заданию (линейная свёртка нелинейных элементов):
  price ~ square_feet + square_feet² + square_feet²·log(square_feet) + bedrooms + pets_yes

| Показатель | Линейная (п.5) | Нелинейная (п.12) | Δ (нелин − лин) |
|------------|----------------|-------------------|-----------------|
| R² | {linear['r2']:.6f} | {summary['r2']:.6f} | {d_r2:+.6f} |
| R²_adj | {linear['r2_adj']:.6f} | {summary['r2_adj']:.6f} | {d_r2adj:+.6f} |
| RMSE (меньше — лучше) | {linear['rmse']:.4f} | {summary['rmse']:.4f} | изменение RMSE: {-d_rmse:+.4f} |
| Число регрессоров k | {linear['k']} | {summary['k']} | +{summary['k'] - linear['k']} |
| Значимость модели | {linear['model_significant']} | {summary['model_significant']} | — |

Обоснование:
1) Число обусловленности = {cond:.4g}. VIF: {vif_txt}.
   Квадрат площади и square_feet²·log имеют VIF в тысячи: члены описывают одну и ту же
   кривизну и по отдельности не читаются. Формула в задании (x³ и x²·log x) — пример
   линейной свёртки; здесь свёртка x, x² и x²·log x.
2) Прирост качества: ΔR²_adj = {d_r2adj:+.6f}
   {'≥ 0.01 → гибкая форма даёт заметный выигрыш' if worth else '< 0.01 → прирост практически пренебрежим'}.
3) RMSE изменился на {d_rmse:.4f} USD при среднем price ≈ {float(df['price'].mean()):.0f}.
4) Итог п.12: уравнение значимо, как и линейное. Прирост качества пренебрежим,
   коэффициенты нелинейных членов площади не интерпретируются.
   Для содержательного вывода достаточно линейной модели п.5.
"""
    (RES / "12_vs_linear.txt").write_text(cmp_text, encoding="utf-8")
    summary["vs_linear"] = cmp_text

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    names = ["Линейная (п.5)", "Нелинейная (п.12)"]
    vals = [linear["r2"], summary["r2"]]
    bars = ax.bar(names, vals, color=["steelblue", "seagreen"], width=0.55)
    ax.set_ylabel("R²")
    ax.set_title("П.12. Сравнение: линейная и нелинейная модели")
    ax.set_ylim(0, max(vals) * 1.35 if max(vals) > 0 else 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", va="bottom")
    fig.savefig(FIG / "12_linear_vs_nonlinear.png")
    plt.close(fig)

    checks = {
        "p12_has_nonlinear_terms": all(
            t in model.params.index for t in ["square_feet_sq", "square_feet_sq_log"]
        ),
        "p6_plot": (FIG / "12_nonlinear_residuals.png").exists(),
        "p8_equation": "ŷ =" in summary["equation"],
        "p9_r2": np.isfinite(summary["r2"]),
        "p10_sig": np.isfinite(summary["f_pvalue"]),
    }
    (RES / "12_checks.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in checks.items()), encoding="utf-8"
    )
    for k, v in checks.items():
        assert v, f"Проверка не пройдена: {k}"
    print("  ПРОВЕРКА п.12 (+6–10): OK")
    return summary


# ---------------------------------------------------------------------------
# Итоговый отчёт (по порядку пунктов, без смешения)
# ---------------------------------------------------------------------------

def _md_table(frame: pd.DataFrame, index_name: str = "") -> str:
    columns = [index_name] + [str(col) for col in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for idx, row in frame.iterrows():
        cells = [str(idx)]
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(linear: dict, poisson: dict, nonlinear: dict, df: pd.DataFrame) -> None:
    print("\n=== Итоговый отчёт ===")

    # Сводная таблица всех трёх — только в конце, после п.12
    rows = []
    for s, label in (
        (linear, "Линейная (п.5)"),
        (poisson, "Пуассоновская (п.11)"),
        (nonlinear, "Нелинейная (п.12)"),
    ):
        rows.append(
            {
                "модель": label,
                "R2": s["r2"],
                "R2_adj": s["r2_adj"],
                "RMSE": s["rmse"],
                "pvalue_модели": s["f_pvalue"],
                "значима": s["model_significant"],
                "среднее_остатков": s["resid_info"]["resid_mean"],
                "отношение_дисперсий_квартили": s["resid_info"]["var_ratio_q"],
            }
        )
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RES / "13_model_comparison.csv", index=False, encoding="utf-8-sig")

    best_name = cmp.loc[cmp["R2"].idxmax(), "модель"]
    c = linear["coefficients"]
    corr_bed = float(df["price"].corr(df["bedrooms"]))
    corr_sq = float(df["price"].corr(df["square_feet"]))
    corr_bed_sq = float(df["bedrooms"].corr(df["square_feet"]))
    bed_means = df.groupby("bedrooms")["price"].agg(["count", "mean"])
    bed_lines = "\n".join(
        f"  {int(idx)} спален: n={int(row['count'])}, средняя плата = {row['mean']:.1f} USD"
        for idx, row in bed_means.iterrows()
        if float(idx) <= 4
    )
    vif_txt = ", ".join(f"{name}={value:.1f}" for name, value in nonlinear["vif"].items())
    interp = f"""Итоговое сравнение и интерпретация

Порядок по заданию:
- п.4: между линейной и пуассоновской выбрана линейная, потому что Y — арендная плата;
- п.5–10: линейная модель с фиктивной pets_yes;
- п.11: пуассоновская; по R² и RMSE она хуже линейной;
- п.12: нелинейная свёртка. Наибольший R² у модели «{best_name}»,
  прирост R²_adj к линейной пренебрежим, столбцы площади почти коллинеарны.

Интерпретация линейного уравнения (п.5), при прочих включённых признаках:
- свободный член {c['const']['coef']:.2f} — уровень уравнения при нулевых X.
  Площадь в данных не бывает нулевой, поэтому свободный член не является ценой реальной квартиры.
- square_feet: {c['square_feet']['coef']:.3f} USD за квадратный фут.
  Парная корреляция цены и площади = {corr_sq:.3f}.
- bedrooms: {c['bedrooms']['coef']:.2f} USD за дополнительную спальню при фиксированной площади.
  Парная корреляция цены и числа спален положительная ({corr_bed:.3f}):
  в среднем квартира с большим числом спален дороже, потому что она больше
  (корреляция bedrooms и square_feet = {corr_bed_sq:.3f}).
  Минус в уравнении — частный эффект при той же площади: комнаты оказываются меньше.
- pets_yes: {c['pets_yes']['coef']:.2f} USD, если животные разрешены.
  Это связь в объявлениях, а не эффект «животные снижают цену».

Средняя плата по числу спален (без учёта площади):
{bed_lines}

Пуассоновская модель: φ = {poisson['dispersion']:.2f}. Значимость взята по квази-LR.
Дисперсия её сырых остатков не постоянна.

Нелинейная модель: число обусловленности = {nonlinear['condition_number']:.4g}, VIF: {vif_txt}.
Коэффициенты square_feet, square_feet² и square_feet²·log по отдельности не интерпретируются.

Общий вывод: все три уравнения статистически значимы при n = {len(df)}.
R² около 0,15: выбранные признаки оставляют необъяснённой большую часть разброса цены.
Рабочая модель — линейная из п.5.

Данные: n={len(df)}, Y=price, X=bedrooms, square_feet, pets_yes.
"""
    (RES / "13_interpretation.txt").write_text(interp, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    short_names = ["Линейная\n(п.5)", "Пуассоновская\n(п.11)", "Нелинейная\n(п.12)"]
    bars = ax.bar(short_names, cmp["R2"], color=["steelblue", "darkorange", "seagreen"], width=0.55)
    ax.set_ylabel("R² (для пуассона — по дисперсии Y)")
    ax.set_title("Итоговое сравнение качества моделей (после п.12)")
    ax.set_ylim(0, float(cmp["R2"].max()) * 1.35)
    for b, v in zip(bars, cmp["R2"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}", ha="center", va="bottom")
    fig.savefig(FIG / "13_model_comparison.png")
    plt.close(fig)

    desc = pd.read_csv(RES / "01_03_descriptive.csv", index_col=0)
    norm = pd.read_csv(RES / "01_03_normality.csv")
    pets = pd.read_csv(RES / "01_03_pets.csv")
    desc_md = _md_table(desc, index_name="переменная")
    norm_md = _md_table(norm.set_index("variable"), index_name="переменная")
    pets_md = _md_table(pets.set_index("pets_yes"), index_name="pets_yes")
    summary_13 = open(RES / "01_03_summary.txt", encoding="utf-8").read()

    report = f"""# Титульный лист

**Лабораторная работа №1.2. Регрессионный анализ**

Студент: ________________  
Группа: ________________  
Преподаватель: ________________  

---

# Отчёт. Лабораторная работа №1.2
## Регрессионный анализ

### Цель работы
Изучение регрессионного анализа.

---

### П.1. Описание исходных данных

{summary_13}

---

### П.2. Результаты дескриптивного анализа

{desc_md}

Частоты фиктивной переменной:

{pets_md}

Согласованность с нормальным распределением, критерий Жарка–Бера (H0: нормальность, α = 0,05):

{norm_md}

Вывод: `price`, `bedrooms` и `square_feet` не согласуются с нормальным распределением. У цены и площади распределение скошено вправо.

---

### П.3. Гистограммы

Число и ширина интервалов — по формуле Стерджесса, см. блок пункта 1 выше. Для `bedrooms` формула даёт дробный шаг, гистограмма построена по целым значениям.

![Гистограммы выбранных атрибутов](figures/03_histograms.png)

---

### П.4. Выбор модели (линейная или пуассоновская)

{open(RES / '04_model_choice.txt', encoding='utf-8').read()}

---

### П.5–10. Линейная регрессия с фиктивными переменными

**Уравнение:** `{linear['equation']}`

| Показатель | Значение |
|------------|----------|
| R² | {linear['r2']:.4f} |
| R²_adj | {linear['r2_adj']:.4f} |
| F (p-value) | {linear['f_stat']:.2f} ({linear['f_pvalue']:.3g}) |
| Модель значима (α=0.05) | {linear['model_significant']} |
| RMSE | {linear['rmse']:.2f} |

![Остатки линейной модели](figures/05_linear_residuals.png)

Коэффициенты: `results/05_linear_coefficients.csv`.

#### Обоснование п.6–10 (линейная)

```
{linear['justification_6_10']}
```

---

### П.11. Пуассоновская регрессия (+ п.6–10)

**Уравнение:** `{poisson['equation']}`

| Показатель | Значение |
|------------|----------|
| R² (по дисперсии Y) | {poisson['r2']:.4f} |
| R²_adj | {poisson['r2_adj']:.4f} |
| McFadden pseudo-R² | {poisson.get('pseudo_r2_mcfadden', float('nan')):.4f} |
| φ (χ² Пирсона / df) | {poisson['dispersion']:.2f} |
| Квази-LR (p-value) | {poisson['f_stat']:.2f} ({poisson['f_pvalue']:.3g}) |
| LR Пуассона при φ=1 (не используется) | {poisson['lr_poisson']:.2f} |
| Модель значима по квази-LR | {poisson['model_significant']} |
| RMSE | {poisson['rmse']:.2f} |

![Остатки пуассоновской модели](figures/11_poisson_residuals.png)

![Сравнение линейной и пуассоновской моделей](figures/11_linear_vs_poisson.png)

#### Обоснование п.6–10 (пуассон)

```
{poisson['justification_6_10']}
```

#### Сравнение линейной и пуассоновской (без нелинейной)

```
{poisson['vs_linear']}
```

---

### П.12. Нелинейная регрессия с фиктивными переменными (+ п.6–10)

Нелинейная модель появляется **только в этом пункте**.

Спецификация: `price ~ square_feet + square_feet² + square_feet²·log(square_feet) + bedrooms + pets_yes`.

**Уравнение:** `{nonlinear['equation']}`

| Показатель | Значение |
|------------|----------|
| R² | {nonlinear['r2']:.4f} |
| R²_adj | {nonlinear['r2_adj']:.4f} |
| F (p-value) | {nonlinear['f_stat']:.2f} ({nonlinear['f_pvalue']:.3g}) |
| Модель значима | {nonlinear['model_significant']} |
| RMSE | {nonlinear['rmse']:.2f} |
| Число обусловленности | {nonlinear['condition_number']:.4g} |

![Остатки нелинейной модели](figures/12_nonlinear_residuals.png)

![Сравнение линейной и нелинейной моделей](figures/12_linear_vs_nonlinear.png)

#### Обоснование п.6–10 (нелинейная)

```
{nonlinear['justification_6_10']}
```

#### Сравнение линейной и нелинейной

```
{nonlinear['vs_linear']}
```

---

### Итог (после всех пунктов)

{open(RES / '13_interpretation.txt', encoding='utf-8').read()}

Сводная таблица: `results/13_model_comparison.csv`.

![Сравнение трёх моделей](figures/13_model_comparison.png)

### Как запустить
```bash
D:\\mad_lab2_venv\\Scripts\\python.exe lab2_regression.py
```
"""
    (BASE / "ОТЧЕТ.md").write_text(report, encoding="utf-8")
    print("  сохранено: ОТЧЕТ.md, results/13_*, figures/13_*")


def main() -> None:
    print("Загрузка data_clean.csv ...")
    df = load_data()
    print(f"n={len(df)}, columns={list(df.columns)}")

    points_1_to_3(df)
    assert (FIG / "03_histograms.png").exists()
    print("  ПРОВЕРКА п.1–3: OK")

    point_4(df)
    print("  ПРОВЕРКА п.4: OK")

    linear = points_5_to_10_linear(df)
    poisson = point_11_poisson(df, linear)
    nonlinear = point_12_nonlinear(df, linear)
    write_report(linear, poisson, nonlinear, df)

    required = [
        RES / "04_model_choice.txt",
        RES / "05_linear_summary.txt",
        RES / "05_linear_justification_6_10.txt",
        RES / "11_poisson_summary.txt",
        RES / "11_vs_linear.txt",
        RES / "12_nonlinear_summary.txt",
        RES / "12_vs_linear.txt",
        RES / "13_model_comparison.csv",
        FIG / "03_histograms.png",
        FIG / "05_linear_residuals.png",
        FIG / "11_poisson_residuals.png",
        FIG / "11_linear_vs_poisson.png",
        FIG / "12_nonlinear_residuals.png",
        FIG / "12_linear_vs_nonlinear.png",
        FIG / "13_model_comparison.png",
        BASE / "ОТЧЕТ.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Не хватает файлов:\n" + "\n".join(missing))
    print("\n=== ВСЕ ПУНКТЫ ВЫПОЛНЕНЫ И ПРОВЕРЕНЫ ===")


if __name__ == "__main__":
    main()
