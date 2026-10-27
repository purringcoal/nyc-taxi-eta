"""Бэктест стратегий переобучения на истории.

Проживаем месяцы по очереди и для каждой стратегии смотрим, какая модель работала
бы в этом месяце и какую ошибку она дала бы на его поездках. Модель для месяца
всегда обучена только на данных до него.

Обозначение. M[s] это модель, обученная на 12 месяцах перед месяцем s
(11 месяцев обучения и последний для ранней остановки). Её можно использовать
начиная с месяца s.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.features import TARGET, load_sample, make_model_input
from src.monitor import GRID, MUTED, SERIES_COLORS, TEXT, log
from src.months import month_range, months_before
from src.retrain import LEARNING_RATE, MAX_TREES, WINDOW_MONTHS, compare
from src.training import train_model
from src.validation import validate_month

FIRST_MONTH = "2024-01"
LAST_MONTH = "2025-06"
SAMPLE_PER_MONTH = 100_000  # одна подвыборка на месяц и для обучения, и для оценки
THRESHOLDS = [0.0, 0.01, 0.02]  # пороги выкатки для правила champion/challenger
SEED = 42
EXPERIMENT = "taxi-eta-backtest"
REPORTS_DIR = Path("reports")


class Backtest:
    """Хранит подвыборки месяцев, обученные модели и посчитанные ошибки.

    Модели и ошибки считаются по требованию и запоминаются. Одна и та же модель
    нужна нескольким стратегиям, и обучать её второй раз незачем.
    """

    def __init__(self, samples: dict[str, pd.DataFrame]):
        self.samples = samples
        self.models = {}
        self.errors = {}

    def model(self, month: str):
        """Модель M[month], обученная на 12 месяцах перед month."""
        if month not in self.models:
            window = months_before(month, WINDOW_MONTHS)
            train = pd.concat([self.samples[m] for m in window[:-1]], ignore_index=True)
            valid = self.samples[window[-1]]
            model, info = train_model(train, valid, LEARNING_RATE, MAX_TREES)
            log(f"  обучена M[{month}] на {window[0]}..{window[-2]}, "
                f"{info['n_trees']} деревьев, {info['train_seconds']:.0f} с")
            self.models[month] = model
        return self.models[month]

    def abs_errors(self, model_month: str, month: str) -> np.ndarray:
        """Абсолютные ошибки модели M[model_month] на поездках месяца month."""
        key = (model_month, month)
        if key not in self.errors:
            df = self.samples[month]
            pred = self.model(model_month).predict(make_model_input(df))
            self.errors[key] = np.abs(df[TARGET].to_numpy() - pred)
        return self.errors[key]

    def mae(self, model_month: str, month: str) -> float:
        return float(self.abs_errors(model_month, month).mean())


def never(months: list[str]) -> dict[str, str]:
    """Модель первого месяца работает всё время."""
    return {s: months[0] for s in months}


def always_fresh(months: list[str]) -> dict[str, str]:
    """Каждый месяц новая модель на самых свежих данных, без проверки."""
    return {s: s for s in months}


def always_lagged(months: list[str]) -> dict[str, str]:
    """Каждый месяц новая модель без проверки, но с тем же отставанием на месяц, что у правила.

    Нужна, чтобы отделить цену отставания от цены отказов в выкатке.
    """
    return {s: months[max(i - 1, 0)] for i, s in enumerate(months)}


def rule(bt: Backtest, months: list[str], threshold: float) -> tuple[dict[str, str], list[dict]]:
    """Правило champion/challenger из src/retrain.py с заданным порогом."""
    deployed, decisions = {}, []
    champion = months[0]
    for i, t in enumerate(months):
        deployed[t] = champion
        if i == len(months) - 1:
            break  # после последнего месяца решение уже ни на что не влияет

        # В конце месяца t претендент M[t] сравнивается с текущей моделью на поездках t.
        # В первом месяце претендент совпадает с текущей моделью, сравнивать нечего.
        if champion == t:
            continue
        d = bt.abs_errors(champion, t) - bt.abs_errors(t, t)
        days = bt.samples[t]["tpep_pickup_datetime"].dt.day.to_numpy()
        stats = compare(d, days)
        promote = stats["ci_low"] > threshold * bt.mae(champion, t)
        decisions.append({"threshold": threshold, "month": t, "champion": champion,
                          "challenger": t, "promoted": promote, **stats})
        if promote:
            champion = t
    return deployed, decisions


def plot_backtest(monthly: pd.DataFrame, path: Path) -> Path:
    """MAE стратегий по месяцам и их разница со стратегией "никогда"."""
    strategies = [c for c in monthly.columns if c != "month"]
    x = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for color, name in zip(SERIES_COLORS, strategies):
        axes[0].plot(x, monthly[name], color=color, linewidth=2, marker="o", markersize=4, label=name)
        axes[1].plot(x, monthly[name] - monthly[strategies[0]], color=color, linewidth=2,
                     marker="o", markersize=4, label=name)

    axes[0].set_title("MAE работающей модели, мин", loc="left", fontsize=11, color=TEXT)
    axes[1].set_title(f"Разница со стратегией \"{strategies[0]}\", мин (ниже лучше)",
                      loc="left", fontsize=11, color=TEXT)
    axes[1].axhline(0, color=MUTED, linewidth=1)
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    axes[-1].set_xticks(x, monthly["month"], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    months = month_range(FIRST_MONTH, LAST_MONTH)
    data_months = months_before(FIRST_MONTH, WINDOW_MONTHS) + months
    log(f"Бэктест {FIRST_MONTH}..{LAST_MONTH}, нужны данные {data_months[0]}..{data_months[-1]}")

    # Проверяем и загружаем все месяцы заранее. Каждый месяц читается с диска один раз.
    samples = {}
    for m in data_months:
        report = validate_month(m)
        if not report.ok:
            log(f"Месяц {m} не прошёл проверку: {'; '.join(report.errors)}. Бэктест остановлен")
            return
        samples[m] = load_sample([m], SAMPLE_PER_MONTH, seed=SEED)
        log(f"  загружен {m}")
    bt = Backtest(samples)

    log("Считаю стратегии без проверки")
    plans = {
        "никогда": never(months),
        "каждый месяц, свежая": always_fresh(months),
        "каждый месяц, с отставанием": always_lagged(months),
    }
    all_decisions = []
    for threshold in THRESHOLDS:
        log(f"Считаю правило с порогом {threshold:.0%}")
        deployed, decisions = rule(bt, months, threshold)
        plans[f"правило, порог {threshold:.0%}"] = deployed
        all_decisions += decisions

    # Для каждой стратегии и каждого месяца MAE модели, которая в нём работала.
    monthly = pd.DataFrame({"month": months})
    for name, deployed in plans.items():
        monthly[name] = [bt.mae(deployed[s], s) for s in months]

    summary = pd.DataFrame([
        {
            "strategy": name,
            "mean_mae": monthly[name].mean(),
            "vs_never": monthly[name].mean() - monthly["никогда"].mean(),
            # Смена модели это месяц, в котором работает не та модель, что в предыдущем.
            "switches": sum(deployed[a] != deployed[b] for a, b in zip(months, months[1:])),
        }
        for name, deployed in plans.items()
    ])
    decisions = pd.DataFrame(all_decisions)

    REPORTS_DIR.mkdir(exist_ok=True)
    paths = {
        "monthly": REPORTS_DIR / "backtest_monthly.csv",
        "summary": REPORTS_DIR / "backtest_summary.csv",
        "decisions": REPORTS_DIR / "backtest_decisions.csv",
    }
    monthly.to_csv(paths["monthly"], index=False)
    summary.to_csv(paths["summary"], index=False)
    decisions.to_csv(paths["decisions"], index=False)
    png_path = plot_backtest(monthly, REPORTS_DIR / "backtest.png")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"backtest-{FIRST_MONTH}-{LAST_MONTH}"):
        mlflow.log_params({"first_month": FIRST_MONTH, "last_month": LAST_MONTH,
                           "sample_per_month": SAMPLE_PER_MONTH, "learning_rate": LEARNING_RATE,
                           "max_trees": MAX_TREES, "thresholds": ",".join(map(str, THRESHOLDS))})
        for path in [*paths.values(), png_path]:
            mlflow.log_artifact(str(path))

    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 1 else f"{x:.3f}"))
    print()
    shown = decisions.assign(threshold=decisions["threshold"].map(lambda x: f"{x:.0%}"))
    print(shown[["threshold", "month", "champion", "improvement", "ci_low", "ci_high", "promoted"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    log(f"Отчёты сохранены в {REPORTS_DIR}")


if __name__ == "__main__":
    main()