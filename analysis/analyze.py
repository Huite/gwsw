# %%

import matplotlib.pyplot as plt
import pandas as pd

from gwsw import analytical
from gwsw.results_reader import Results

# %%


def process_results(name: str):
    print(f"processing {name}")
    results = Results(f"../modflow6/{name}", f"../sectiondata/{name}.csv", dz=0.05)

    # Check mass balance error
    budgets = results.budgets()
    by = budgets.sum(("z", "x"))
    mass_balance_gap = by["drn"] + by["riv"] + by["rch"]
    if "ghb" in by:
        mass_balance_gap += by["ghb"]
    mass_balance_error = abs(mass_balance_gap / by["rch"] * 100)
    print(
        f"maximum mass balance error at any profile: {mass_balance_error.max().item()} %"
    )

    df = pd.DataFrame(
        data={
            "L": 2.0 * (results.total_width() - results.df["ditch_width"]),
            "B": 2.0 * results.df["ditch_width"],
            "kh": results.df["k11"],
            "kv": results.df["k33"],
            "c0": results.df["c0"],
            "c1": results.df["c1"].fillna(1.0e7),
        }
    )

    comparison = pd.DataFrame(
        data={
            "c_fine": results.cell_drainage_c(),
            "c_modflow": analytical.c_modflow(L=df["L"], B=df["B"], c0=df["c0"]),
            "c_ernst": analytical.c_ernst(
                L=df["L"],
                B=df["B"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                kv=results.df["k33"],
                c0=df["c0"],
            ),
            "c_de_lange_1997": analytical.c_de_lange_1997(
                L=df["L"],
                B=df["B"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                kv=results.df["k33"],
                c0=df["c0"],
                c1=df["c1"],
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

    fig, axes = plt.subplots(ncols=4, figsize=(20, 6))
    fig.suptitle(name.replace("-", " "))
    lower = 0.0
    upper = comparison.max().max()
    for column, ax in zip(comparison.columns[1:], axes):
        comparison.plot.scatter(ax=ax, x="c_fine", y=column, alpha=0.2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.axline((0.0, 0.0), (upper, upper), color="r")
        ax.set_aspect(1.0)
    fig.savefig(f"../figures/{name}.png", dpi=200)

    out = pd.concat(
        (
            results.df.drop("dewatering_depth", axis=1),
            df.drop("c1", axis=1),
            comparison,
        ),
        axis=1,
    )
    out.to_csv(f"../figures/summary-{name}.csv")


# %%

CASES = (
    "single-layer-brook",
    "single-layer-clay",
    "single-layer-peat",
    "single-layer-sand",
    "two-layer-brook",
    "two-layer-clay",
    "two-layer-peat",
    "two-layer-sand",
)

for case in CASES:
    process_results(case)

# %%
