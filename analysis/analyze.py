# %%

from pathlib import Path

import imod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr

from gwsw import analytical

# %%

def z_coordinate(
    obj: xr.DataArray | xr.Dataset, z: np.ndarray
) -> xr.DataArray | xr.Dataset:
    return obj.assign_coords(z=("layer", z)).swap_dims({"layer": "z"})


class Results:
    def __init__(self, mf6_directory: str, section_csv: str, dz: float):
        mf6_directory = Path(mf6_directory)
        self._head = (
            imod.mf6.open_hds(
                mf6_directory / "GWF_1/GWF_1.hds",
                mf6_directory / "GWF_1/dis.dis.grb",
            )
            .isel(time=0, drop=True)
            .compute()
        )
        self._budgets = (
            xr.Dataset(
                imod.mf6.open_cbc(
                    mf6_directory / "GWF_1/GWF_1.cbc",
                    mf6_directory / "GWF_1/dis.dis.grb",
                )
            )
            .isel(time=0, drop=True)
            .compute()
        )
        # Compute midpoint z for visualization
        nlayer = self._head["layer"].size
        self.z = np.arange(nlayer * dz, 0.0, -dz) - 0.5 * dz
        self.df = pd.read_csv(section_csv, index_col=0)

    def head(self) -> xr.DataArray:
        head = self._head.where(self._budgets["npf-sat"] > 0)
        return z_coordinate(head, self.z)

    def budgets(self):
        return z_coordinate(self._budgets, self.z)

    def streamfunction(self) -> xr.DataArray:
        flip = slice(None, None, -1)
        dx = self._budgets["dx"]
        frf = (
            self._budgets["flow-right-face"]
            .isel(layer=flip)
            .cumsum("layer")
            .isel(layer=flip)
            .where(self._budgets["npf-sat"] > 0)
            .drop_vars("dx")
        )
        frf["x"] = frf["x"] + 0.5 * dx
        x0 = frf["x"].isel(x=0) - 0.5 * dx.isel(x=0)
        frf0 = frf.isel(x=0)
        first_column = xr.zeros_like(frf0).where(frf0.notnull()).assign_coords(x=x0)
        out = xr.concat([first_column, frf], dim="x")
        return z_coordinate(out, self.z)

    def groundwatertable(self) -> xr.DataArray:
        npf_sat = self._budgets["npf-sat"]
        is_top = npf_sat["layer"] == npf_sat["layer"].where(npf_sat > 0).max("layer")
        return self._head.where(is_top).max("layer")

    def total_width(self):
        gwt = self.groundwatertable()
        total_width = gwt["dx"].where(gwt.notnull()).sum("x")
        return total_width

    def cell_drainage_c(self, xmax=None):
        budgets = self.budgets().sum(("z", "x"))
        gwt = self.groundwatertable()
        Q = -budgets["riv"] + -budgets["drn"]
        total_width = self.total_width()
        h_mean = (gwt * gwt["dx"]).sum("x") / total_width
        dH = h_mean - self.df["stage"]
        c_cell_drain = (dH / Q) * total_width
        return c_cell_drain


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

    out = pd.concat((results.df.drop("dewatering_depth", axis=1), df.drop("c1", axis=1), comparison), axis=1)
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

results = Results(
    "../modflow6/two-layer-clay", "../sectiondata/two-layer-clay.csv", dz=0.05
)
# %%

head = results.head()
budgets = results.budgets()
streamfunction = results.streamfunction()
watertable = results.groundwatertable()

# %%

# %%
# Check mass balance error

by = budgets.sum(("z", "x"))
mass_balance_gap = by["drn"] + by["riv"] + by["rch"]
if "ghb" in by:
    mass_balance_gap += by["ghb"]
mass_balance_error = abs(mass_balance_gap / by["rch"] * 100)
print(f"maximum mass balance error at any profile: {mass_balance_error.max().item()} %")

# %%

y = -2
fig, ax = plt.subplots(figsize=(50, 10))
head.isel(y=y).plot.contour(ax=ax, levels=20)
streamfunction.isel(y=y).plot.contour(ax=ax, levels=20, cmap="turbo")
watertable.isel(y=y).plot(ax=ax, color="black")
# ax.set_aspect(1.0)
fig.savefig("../figures/check.png", dpi=300)
