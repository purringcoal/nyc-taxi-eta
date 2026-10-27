"""График излома тренда: разность смещений по годам для одних и тех же месяцев.

Собирает все отчёты мониторинга из папки reports и рисует, как менялась
разность смещений от года к году. По одной линии на каждый календарный месяц.
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.monitor import GRID, MUTED, REPORTS_DIR, SERIES_COLORS, TEXT

MONTHS = {3: "март", 4: "апрель", 5: "май", 6: "июнь"}
FEE_START_YEAR = 2025


def main():
    # Склеиваем все отчёты. Если какой-то месяц попал в несколько файлов,
    # оставляем значение из последнего по имени файла.
    files = sorted(REPORTS_DIR.glob("monitoring_*.csv"))
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    table = table.drop_duplicates("month", keep="last")

    # Разбиваем 'ГГГГ-ММ' на год и номер месяца и оставляем только нужные месяцы.
    table["year"] = table["month"].str[:4].astype(int)
    table["month_num"] = table["month"].str[5:7].astype(int)
    table = table[table["month_num"].isin(MONTHS)]

    # Таблица год на месяц, в ячейках разность смещений.
    pivot = table.pivot(index="year", columns="month_num", values="bias_gap").sort_index()
    print(pivot.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    for color, (month_num, name) in zip(SERIES_COLORS, MONTHS.items()):
        ax.plot(pivot.index, pivot[month_num], color=color, linewidth=2, marker="o", markersize=5, label=name)

    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(FEE_START_YEAR - 0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.text(FEE_START_YEAR - 0.45, 1.0, "плата за въезд", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color=TEXT)

    ax.set_title("Медиана остатков в зоне платы минус вне её, мин", loc="left", fontsize=11, color=TEXT)
    ax.set_xticks(pivot.index)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))

    path = REPORTS_DIR / "trend_break.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nГрафик сохранён в {path}")


if __name__ == "__main__":
    main()