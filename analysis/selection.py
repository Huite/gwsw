# %%

import matplotlib.pyplot as plt
import pandas as pd

from gwsw import analytical
from gwsw.results_reader import Results

# %%

def setup_comparison(results: Results, df: pd.DataFrame, xmax) -> pd.DataFrame:
    comparison = pd.DataFrame(
        data={
            "c_fine": results.cell_drainage_c(xmax=xmax),
            "c_modflow": analytical.c_modflow(L=df["L"], B=df["B"], c0=df["c0"]),
            "c_ernst": analytical.c_ernst(
                L=df["L"],
                B=df["B"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                kv=results.df["k33"],
                c0=df["c0"],
            ),
            "c_de_lange": analytical.c_de_lange_1997(
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
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
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
    return


def process_results(name: str, xmax=None) -> None:
    print(f"processing {name}")
    results = Results(f"../modflow6/{name}", f"../sectiondata/{name}.csv", dz=0.05)
    print("done")

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

    scatter_plot(comparison, label)
# %%


process_results("two-layer-clay", xmax=None)
# %%

def setup_comparison(results: Results, df: pd.DataFrame, xmax) -> pd.DataFrame:
    comparison = pd.DataFrame(
        data={
            "c_fine": results.cell_drainage_c(xmax=xmax),
            "c_modflow": analytical.c_modflow(L=df["L"], B=df["B"], c0=df["c0"]),
            "c_vertical": analytical.c_vertical(
                D=results.df["domain_height"],
                kv=results.df["k33"],
            ),
            "c_horizontal": analytical.c_horizontal(
                L=df["L"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
            ),
            "c_radial": analytical.c_radial(
                L=df["L"],
                B=df["B"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                kv=results.df["k33"],
            ),
            "c_bypass": analytical.c_horizontal(
                L=df["L"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
            ) - analytical.c_horizontal_multilayer(
                L=df["L"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                c1=df["c1"],
            ),
            "c_ernst": analytical.c_ernst(
                L=df["L"],
                B=df["B"],
                D=results.df["domain_height"],
                kh=results.df["k11"],
                kv=results.df["k33"],
                c0=df["c0"],
            ),
        }
    )
    return comparison


def scatter_plot(comparison: pd.DataFrame, name: str) -> None:
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
    axes = axes.ravel()
    fig.suptitle(name.replace("-", " "))
    lower = 0.0
    upper = comparison.max().max()
    for column, ax in zip(comparison.columns[1:], axes):
        comparison.plot.scatter(ax=ax, x="c_fine", y=column, alpha=0.2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.axline((0.0, 0.0), (upper, upper), color="r")
        ax.set_aspect(1.0)
    return


def process_results(name: str, xmax=None) -> None:
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

    scatter_plot(comparison, label)
    
# %%

process_results("two-layer-clay")
# %%
