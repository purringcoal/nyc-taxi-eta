"""Мониторинг модели по месяцам.

Для каждого месяца после окончания обучающего окна считаем ошибку модели
на поездках этого месяца и дрейф признаков относительно обучающих данных.
Результаты пишутся в MLflow и в папку reports.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.cbd_zones import load_cbd_zones
from src.drift import psi_categorical, psi_continuous
from src.features import FEATURES, TARGET, load_month, make_model_input
from src.months import parse_month_range_args
from src.validation import validate_month

MODEL_NAME = "taxi-eta"
MODEL_ALIAS = "champion"
EXPERIMENT = "taxi-eta-monitoring"
REPORTS_DIR = Path("reports")
FEE_START_MONTH = "2025-01"  # месяц введения платы за въезд, отмечается на графике

# Метрики считаются по случайной подвыборке. Предсказания модели для всех поездок
# заняли бы часы, а погрешность метрик на подвыборке такого размера пренебрежимо мала.
SAMPLE_SIZE = 100_000
REFERENCE_SAMPLE_SIZE = 200_000
SEED = 42


def log(message: str) -> None:
    """Печатает сообщение с текущим временем.

    flush=True заставляет Python вывести строку сразу, а не копить её в буфере.
    Иначе при долгой работе сообщения могут появляться пачками с опозданием.
    """
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Случайная подвыборка из n строк без повторений. Если строк меньше, возвращает все."""
    return df.sample(n=min(n, len(df)), random_state=SEED)


def resolve_model(name: str, alias: str) -> tuple[str, str, list[str]]:
    """Находит версию модели по алиасу и месяцы, на которых она обучалась.

    Возвращаем URI с номером версии, а не с алиасом. Если во время мониторинга
    алиас перевесят на другую версию, мы всё равно будем знать, какую модель проверяли.
    """
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(version.run_id)
    train_months = run.data.params["train_months"].split(",")
    return f"models:/{name}/{version.version}", version.version, train_months


def compute_metrics(df: pd.DataFrame, pred: np.ndarray, reference: pd.DataFrame,
                    reference_pred: np.ndarray, cbd_zones: set[int]) -> dict[str, float]:
    """Ошибка модели и дрейф данных по готовым предсказаниям.

    Функция ничего не загружает и не печатает. Её можно вызывать откуда угодно,
    в том числе из пайплайна и из тестов.
    """
    residual = df[TARGET].to_numpy() - pred  # положительный остаток значит, что модель недооценила

    # Поездка относится к зоне платы, если начинается или заканчивается в ней.
    in_cbd = (df["PULocationID"].isin(cbd_zones) | df["DOLocationID"].isin(cbd_zones)).to_numpy()

    metrics = {
        "share_cbd": float(in_cbd.mean()),
        "mae": float(np.abs(residual).mean()),
        "mae_cbd": float(np.abs(residual[in_cbd]).mean()),
        "mae_outside": float(np.abs(residual[~in_cbd]).mean()),
        # Модель обучена на потере L1, поэтому без сдвига данных медиана остатков близка к нулю.
        "bias_cbd": float(np.median(residual[in_cbd])),
        "bias_outside": float(np.median(residual[~in_cbd])),
    }
    metrics["bias_gap"] = metrics["bias_cbd"] - metrics["bias_outside"]

    for col in FEATURES:
        metrics[f"psi_{col}"] = psi_categorical(reference[col], df[col])
    metrics["psi_duration"] = psi_continuous(reference[TARGET], df[TARGET])
    metrics["psi_prediction"] = psi_continuous(reference_pred, pred)
    return metrics


# Цвета линий взяты из проверенной палитры, которая различима при нарушениях цветового зрения.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MUTED = "#8a8985"
TEXT = "#52514e"
GRID = "#e4e3df"


def plot_monitoring(table: pd.DataFrame, path: Path) -> Path:
    """Четыре графика друг под другом с общей осью месяцев."""
    panels = [
        ("Ошибка модели (MAE), мин",
         [("mae", "все поездки"), ("mae_cbd", "через зону платы"), ("mae_outside", "вне зоны платы")]),
        ("Медиана остатков в зоне платы минус вне её, мин",
         [("bias_gap", None)]),
        ("PSI признаков относительно обучающих данных",
         [(f"psi_{col}", col) for col in FEATURES]),
        ("PSI длительности и предсказаний",
         [("psi_duration", "фактическая длительность"), ("psi_prediction", "предсказание")]),
    ]
    x = np.arange(len(table))
    months = table["month"].tolist()

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 13), sharex=True)
    for ax, (title, series) in zip(axes, panels):
        for color, (col, label) in zip(SERIES_COLORS, series):
            ax.plot(x, table[col], color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if len(series) > 1:
            # Легенда справа от графика, чтобы не закрывать линии.
            ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))

    axes[1].axhline(0, color=MUTED, linewidth=1)
    # Пороги PSI, принятые на практике. Выше 0.1 сдвиг заметный, выше 0.25 серьёзный.
    for ax in axes[2:]:
        for level in (0.1, 0.25):
            ax.axhline(level, color=MUTED, linewidth=1, linestyle="--")

    if FEE_START_MONTH in months:
        position = months.index(FEE_START_MONTH) - 0.5
        for ax in axes:
            ax.axvline(position, color=MUTED, linewidth=1, linestyle=":")
        axes[0].text(position + 0.1, 1.0, "плата за въезд", transform=axes[0].get_xaxis_transform(),
                     va="top", fontsize=9, color=TEXT)

    axes[-1].set_xticks(x, months, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    months = parse_month_range_args("Мониторинг модели по месяцам")
    started_all = time.perf_counter()

    log(f"Ищу модель {MODEL_NAME}@{MODEL_ALIAS} в реестре MLflow")
    model_uri, version, train_months = resolve_model(MODEL_NAME, MODEL_ALIAS)
    log(f"Нашёл {model_uri}, обучена на {', '.join(train_months)}")

    log("Загружаю модель")
    model = mlflow.pyfunc.load_model(model_uri)
    cbd_zones = load_cbd_zones()
    log(f"Модель загружена, зон внутри зоны платы {len(cbd_zones)}")

    # Эталон для дрейфа это данные, на которых модель обучалась.
    log(f"Читаю эталонные месяцы {', '.join(train_months)}")
    reference = pd.concat([load_month(m) for m in train_months], ignore_index=True)
    reference = sample(reference, REFERENCE_SAMPLE_SIZE)
    log(f"Делаю предсказания для эталона ({len(reference):,} поездок)")
    started = time.perf_counter()
    reference_pred = model.predict(make_model_input(reference))
    log(f"Эталон готов за {time.perf_counter() - started:.0f} с")

    # Месяцы, на которых модель обучалась, проверять не нужно. Модель видела там ответы,
    # и ошибка на них не показательна.
    to_check = [m for m in months if m not in train_months]
    skipped = [m for m in months if m in train_months]
    if skipped:
        log(f"Пропускаю месяцы обучения {', '.join(skipped)}")
    log(f"Месяцев для проверки {len(to_check)}")

    rows = []
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"monitor-v{version}"):
        mlflow.log_params({
            "model_uri": model_uri,
            "train_months": ",".join(train_months),
            "first_month": months[0],
            "last_month": months[-1],
            "n_cbd_zones": len(cbd_zones),
            "sample_size": SAMPLE_SIZE,
            "reference_sample_size": REFERENCE_SAMPLE_SIZE,
        })

        month_times = []
        for step, month in enumerate(to_check, start=1):
            started = time.perf_counter()
            prefix = f"[{step}/{len(to_check)}] {month}"

            report = validate_month(month)
            if not report.ok:
                log(f"{prefix}  пропущен, данные не прошли проверку: {'; '.join(report.errors)}")
                continue

            full = load_month(month)
            df = sample(full, SAMPLE_SIZE)
            log(f"{prefix}  прочитано {len(full):,} поездок, предсказываю для {len(df):,}")

            pred = model.predict(make_model_input(df))
            metrics = {"n_trips": len(full), "n_sample": len(df),
                       **compute_metrics(df, pred, reference, reference_pred, cbd_zones)}
            mlflow.log_metrics(metrics, step=step)
            rows.append({"month": month, **metrics})

            # Оценка оставшегося времени по среднему времени уже обработанных месяцев.
            month_times.append(time.perf_counter() - started)
            left = len(to_check) - step
            eta = sum(month_times) / len(month_times) * left
            log(f"{prefix}  MAE {metrics['mae']:.3f}, разница смещений {metrics['bias_gap']:+.3f}, "
                f"за {month_times[-1]:.0f} с, осталось примерно {eta / 60:.1f} мин")

        if not rows:
            log("Нет ни одного месяца для мониторинга")
            return

        log("Сохраняю таблицу и график")
        table = pd.DataFrame(rows)
        REPORTS_DIR.mkdir(exist_ok=True)
        # Диапазон месяцев в имени файла, чтобы отчёты разных запусков не перезаписывали друг друга.
        stem = f"monitoring_{months[0]}_{months[-1]}"
        csv_path = REPORTS_DIR / f"{stem}.csv"
        table.to_csv(csv_path, index=False)
        png_path = plot_monitoring(table, REPORTS_DIR / f"{stem}.png")
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(png_path))

    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    log(f"Готово за {(time.perf_counter() - started_all) / 60:.1f} мин. Отчёт в {csv_path} и {png_path}")


if __name__ == "__main__":
    main()