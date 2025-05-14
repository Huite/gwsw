from pathlib import Path

import imod
import pandas as pd
import numpy as np
import xarray as xr


def z_coordinate(
    obj: xr.DataArray | xr.Dataset, z: np.ndarray
) -> xr.DataArray | xr.Dataset:
    return obj.assign_coords(z=("layer", z)).swap_dims({"layer": "z"})

# %%

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

    def total_width(self, xmax=None):
        gwt = self.groundwatertable()
        if xmax is None:
            total_width = gwt["dx"].where(gwt.notnull()).sum("x")
        else:
            gwt = gwt.sel(x=slice(None, xmax))
            total_width = gwt["dx"].sel(x=slice(None, xmax)).where(gwt.notnull()).sum("x")
        return total_width

    def cell_drainage_c(self, xmax=None):
        if xmax is None:
            budgets = self.budgets().sum(("z", "x"))
            gwt = self.groundwatertable()
        else:
            budgets = self.budgets().sel(x=slice(None, xmax)).sum(("z", "x"))
            gwt = self.groundwatertable().sel(x=slice(None, xmax))

        Q = -budgets["riv_riv"] + -budgets["drn_drn"]
        total_width = self.total_width(xmax=xmax)
        h_mean = (gwt * gwt["dx"]).sum("x") / total_width
        dH = h_mean - self.df["stage"]
        c_cell_drain = (dH / Q) * total_width
        return c_cell_drain
