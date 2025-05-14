# %%

import matplotlib.pyplot as plt
import pandas as pd

from gwsw import analytical_np as analytical
from gwsw.results_reader import Results

# %%

def setup_comparison(results: Results, df: pd.DataFrame, xmax) -> pd.DataFrame:
    comparison = pd.DataFrame(
        data={
            "c_drain": results.cell_drainage_c(xmax=xmax),
            "c_modflow": analytical.c_entry(L=df["L"], B=df["B"], c0=df["c0"]),
            "c_ernst": analytical.c_ernst(
                L=df["L"],
                B=df["B"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                kv=results.df["k33"],
                c0=df["c0"],
            ),
            "c_de_lange_2022": analytical.c_de_lange_2022(
                L=df["L"],
                B=df["B"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                kv=results.df["k33"],
                c0=df["c0"],
                c1=df["c1"],
            ),
        }
    )
    return comparison


def scatter_plot(comparison: pd.DataFrame, name: str) -> None:
    fig, axes = plt.subplots(ncols=3, figsize=(20, 6))
    fig.suptitle(name.replace("-", " "))
    lower = 0.0
    upper = comparison.max().max()
    titles = ("Waterbodem (MODFLOW)", "Ernst", "De Lange 2020")
    for column, ax, title in zip(comparison.columns[1:], axes, titles):
        comparison.plot.scatter(ax=ax, x="c_drain", y=column, alpha=0.2)
        ax.set_xlabel("c iGrOw (d)")
        ax.set_ylabel("")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.axline((0.0, 0.0), (upper, upper), color="r", alpha=0.5)
        ax.set_aspect(1.0)
        ax.set_title(title)

        if column == "c_ernst":
            comparison.plot.scatter(ax=ax, x="c_drain", y="c_ernst_cv0", alpha=0.2, color="orange")
    axes[0].set_ylabel("c celdrain (d)")
    fig.savefig(f"../figures/{name}.png", dpi=200)
    return


def summary_csv(results: Results, df: pd.DataFrame, comparison: pd.DataFrame, name: str) -> None:
    out = pd.concat(
        (
            results.df.drop("dewatering_depth", axis=1),
            df.drop("c1", axis=1),
            comparison,
        ),
        axis=1,
    )
    out.to_csv(f"../figures/summary-{name}.csv")
    return


def process_results(name: str, xmax=None) -> None:
    print(f"processing {name}")
    results = Results(f"../modflow6/{name}", f"../sectiondata/{name}.csv", dz=0.05)

    # Check mass balance error
    budgets = results.budgets()
    by = budgets.sum(("z", "x"))
    mass_balance_gap = by["drn_drn"] + by["riv_riv"] + by["rch_rch"]
    if "ghb_ghb" in by:
        mass_balance_gap += by["ghb_ghb"]
    mass_balance_error = abs(mass_balance_gap / by["rch_rch"] * 100)
    print(
        f"maximum mass balance error at any profile: {mass_balance_error.max().item()} %"
    )

    df = pd.DataFrame(
        data={
            "L": 2.0 * (results.total_width(xmax=xmax) - results.df["ditch_width"]),
            "B": 2.0 * results.df["ditch_width"],
            "kh": results.df["k11"],
            "kv": results.df["k33"],
            "c0": results.df["c0"],
            "c1": results.df["c1"].fillna(1.0e7),
        }
    )
    
    comparison = setup_comparison(results=results, df=df, xmax=xmax)
    
    if xmax is not None:
        label = f"{name}-width{2 * xmax}"
    else:
        label = name

    #scatter_plot(comparison, label)
    #summary_csv(results, df, comparison, label)
    return comparison


# %%

CASES = (
#   "single-layer-peat",
#    "single-layer-clay",
#    "single-layer-sand",
#    "single-layer-brook",
    "two-layer-brook",
    "two-layer-clay",
#    "two-layer-peat",
    "two-layer-sand",
)

dataframes = {}
for case in CASES:
    dataframes[case] = process_results(case, xmax=50.0)

scatter_plot(dataframes["two-layer-brook"], name="Beek cel 100 m c_v 0")
scatter_plot(dataframes["two-layer-clay"], name="Klei cel 100 m c_v 0")
scatter_plot(dataframes["two-layer-sand"], name="Zand cel 100 m c_v 0")

dataframes = {}
for case in CASES:
    dataframes[case] = process_results(case, xmax=12.5)

scatter_plot(dataframes["two-layer-brook"], name="Beek cel 25 m c_v 0")
scatter_plot(dataframes["two-layer-clay"], name="Klei cel 25 m c_v 0")
scatter_plot(dataframes["two-layer-sand"], name="Zand cel 25 m c_v 0")
# %%
