from typing import Any

import numpy as np
import imod
import xarray as xr


def rand_within(n: int, low: float, high: float) -> np.ndarray:
    delta = high - low
    return np.random.rand(n) * delta + low


def data_along_y(idomain, data):
    return xr.DataArray(
        data=data,
        coords={"y": idomain["y"]},
        dims=["y"],
    )


def xy_spacing(
    xleft: float, xright: float, ny: int, width: float, dx0: float, power: float
) -> dict[str, Any]:
    if power != 1.0:
        remainder = width - xright
        n_extra = int(np.ceil(np.log(remainder * np.log(power) / dx0) / np.log(power)))
        dx = np.concatenate(
            (
                np.full(int(xright - xleft / dx0), dx0),
                np.full(n_extra, dx0) * power ** np.arange(1, n_extra + 1),
            )
        )
    else:
        n = int(width / dx0)
        dx = np.full(n, dx0)
    x = xleft + dx.cumsum() - 0.5 * dx
    return {"x": x, "dx": ("x", dx), "y": np.arange(ny, 0.0, -1.0) - 0.5}


def create_dis(
    xycoords: dict[str, np.ndarray],
    dz: float,
    ditch_elevation: np.ndarray,
    ditch_width: np.ndarray,
    domain_height: np.ndarray,
) -> imod.mf6.StructuredDiscretization:
    # x=0 at is the location of the water-land divide.
    bottom_values: np.ndarray = np.arange(domain_height.max() - dz, -dz, -dz)
    nx = xycoords["x"].size
    ny = xycoords["y"].size
    nz = bottom_values.size

    coords = xycoords.copy()
    layer = np.arange(1, 1 + nz)
    coords["layer"] = layer
    idomain = xr.DataArray(
        data=np.ones((nz, ny, nx), dtype=int),
        coords=coords,
        dims=("layer", "y", "x"),
    )
    top = xr.full_like(
        idomain.isel(layer=0, drop=True), domain_height.max(), dtype=float
    )
    bottom = xr.ones_like(idomain, dtype=float) * xr.DataArray(
        bottom_values, coords={"layer": layer}, dims=["layer"]
    )
    elevation = data_along_y(idomain, ditch_elevation)
    top_boundary = data_along_y(idomain, domain_height)
    left_boundary = data_along_y(idomain, -ditch_width)
    exclude = (
        (bottom >= top_boundary)
        | (idomain["x"] < left_boundary)
        | (idomain["x"] < 0) & (bottom > elevation)
    )
    idomain = idomain.where(~exclude, other=0)
    return imod.mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)


def create_npf(idomain, k11, k33, unconfined: bool) -> imod.mf6.NodePropertyFlow:
    ones = xr.ones_like(idomain, dtype=float)
    k11 = (ones * data_along_y(idomain, k11)).where(idomain > 0)
    k33 = (ones * data_along_y(idomain, k33)).where(idomain > 0)
    if unconfined:
        icelltype = 1
    else:
        icelltype = 0
    return imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k11,
        k22=1.0e-300,
        k33=k33,
        save_flows=True,
        save_saturation=True,
    )


def create_rch(idomain, rate) -> imod.mf6.Recharge:
    rate = idomain * xr.DataArray(rate, dims=("y",))
    # only keep recharge on land part in top cell
    active_layer = idomain["layer"].where(idomain > 0).min("layer")
    rate = rate.where((rate["layer"] == active_layer) & (rate["x"] > 0))
    rate = rate.dropna("layer", how="all")
    return imod.mf6.Recharge(
        rate=rate,
        save_flows=True,
    )


def create_riv(
    idomain, bottom, c0, k11, k33, stage, elevation, dx0, dz
) -> imod.mf6.River:
    # horizontal part
    ones = xr.ones_like(idomain, dtype=float)
    c0 = ones * data_along_y(idomain, c0)
    stage = ones * data_along_y(idomain, stage)
    elevation = ones * data_along_y(idomain, elevation)

    # Place river where elevation is above midpoint of cell.
    # Then find the lowest layer where this occurs.
    height_diff = elevation - bottom
    height_diff = height_diff.where((height_diff > 0.5 * dz))
    height_diff = height_diff.where(height_diff == height_diff.min("layer"))

    # Compute the total resistance from the midpoint of the cell to the cell edge.
    # Conductance is cross-sectional area (dx * dy)
    c_cell = height_diff / k33
    c_total = c_cell + c0
    dy = 1.0
    # Do 1.0 / c_total to preserve (layer, y, x) dimension order.
    conductance = 1.0 / c_total * idomain["dx"] * dy
    # River is by definition located left of x=0.
    is_river = height_diff.notnull() & (idomain["x"] < 0) & (idomain > 0)

    # Now find the vertical face of the river.
    # It is found anywhere the midpoint is submerged, and above the elevation.
    # Horizontally, it is found in the first cell to the right of x = 0.
    midpoint = bottom + 0.5 * dz
    is_vertical_river = (
        (midpoint <= stage)
        & (midpoint > elevation)
        & (idomain["x"] > 0.1 * dx0)
        & (idomain["x"] < 0.9 * dx0)
        & (idomain == 1)
    )

    # vertical part
    vertical_stage = stage.where(is_vertical_river)
    vertical_elevation = bottom.where(is_vertical_river)
    c_cell_vertical = 0.5 * dz / k11
    vertical_conductance = (0.1 / (c_cell_vertical + c0)).where(is_vertical_river)

    # only keep in river part in top cell
    stage = stage.where(is_river)
    conductance = conductance.where(is_river)
    elevation = elevation.where(is_river)

    # Combine with vertical face and remove all layers without input.
    stage = stage.combine_first(vertical_stage).dropna("layer", how="all")
    conductance = conductance.combine_first(vertical_conductance).dropna(
        "layer", how="all"
    )
    elevation = elevation.combine_first(vertical_elevation).dropna("layer", how="all")

    return imod.mf6.River(
        stage=stage,
        conductance=conductance,
        bottom_elevation=elevation,
        save_flows=True,
    )


def create_seepage(idomain, bottom, k11, stage, dx0, dz) -> imod.mf6.Drainage:
    # horizontal part
    ones = xr.ones_like(idomain, dtype=float)
    stage = ones * data_along_y(idomain, stage)

    midpoint = bottom + 0.5 * dz
    is_seepage = (
        (midpoint > stage)
        & (idomain["x"] > 0.1 * dx0)
        & (idomain["x"] < 0.9 * dx0)
        & (idomain == 1)
    )
    elevation = bottom.where(is_seepage).dropna("layer", how="all")
    c_cell = 0.5 * dx0 / k11
    conductance = (dz / c_cell).where(is_seepage).dropna("layer", how="all")
    scaling_depth = (
        xr.full_like(conductance, dz).where(is_seepage).dropna("layer", how="all")
    )

    return imod.mf6.Drainage(
        elevation=elevation,
        conductance=conductance,
        scaling_depth=scaling_depth,
        save_flows=True,
    )


def create_aquifer(idomain, c1, head) -> imod.mf6.GeneralHeadBoundary:
    ones = xr.ones_like(idomain.isel(layer=[-1]), dtype=float)
    area = ones * idomain["dx"]  # dy = 1.0
    c1 = ones * data_along_y(idomain, c1)
    head = ones * data_along_y(idomain, head)
    conductance = area / c1
    return imod.mf6.GeneralHeadBoundary(
        head=head, conductance=conductance, save_flows=True
    )


def create_ic(idomain, stage) -> imod.mf6.InitialConditions:
    ones = xr.ones_like(idomain, dtype=float)
    head = ones * data_along_y(idomain, stage).where(idomain > 0)
    return imod.mf6.InitialConditions(start=head)


def z_coordinate(
    obj: xr.DataArray | xr.Dataset, z: np.ndarray
) -> xr.DataArray | xr.Dataset:
    return obj.assign_coords(z=("layer", z)).swap_dims({"layer": "z"})


class ManyCrossSections:
    """
    Class to contain many cross sections.

    Values between _lower and _upper are randomly generated from a uniform
    distribution. The vertical discretization is fixed. The horizontal
    discretization is constant from -B to B (where B is the width of the
    ditch). From B to W (domain width), the cells grow exponentially in the
    horizontal direction.

    Parameters
    ----------
    n: int
        Number of cross sections
    domain_width: float
        Maximum horizontal size. Must be positive.
    ditch_width_lower: float
        Lower bound of the width of the ditch.
    ditch_width_upper: float
        Upper bound of the width of the ditch.
    dewatering_depth_lower: float
        Lower bound of the ditch stage. Must be positive.
    dewatering_depth_upper: float
        Upper bound of the ditch stage. Must be positive.
    ditch_depth_lower: float
        Lower bound of the water column depth in the ditch. Must be positive.
    ditch_depth_upper: float
        Upper bound of the water column depth in the ditch. Must be positive.
    dx0: float, default value is 0.1 m.
        Initial horizontal cell size. Forms the base for the growing cell size in the x-direction.
    growth_rate: float, default value is 1.1.
        Exponential growth rate. Each subsequent cell is multiplied by this factor.
    dz: float, default value is 0.1 m.
        Vertical cell size. Constant in the domain.
    """

    def __init__(
        self,
        n: int,
        domain_width: float,
        domain_height_lower: float,
        domain_height_upper: float,
        ditch_width_lower: float,
        ditch_width_upper: float,
        dewatering_depth_lower: float,
        dewatering_depth_upper: float,
        ditch_depth_lower: float,
        ditch_depth_upper: float,
        dx0: float = 0.1,
        dx_growth_rate: float = 1.1,
        dz: float = 0.1,
    ):
        self.n = n
        self.dx0 = dx0
        self.dz = dz
        self.domain_height = (
            rand_within(n=n, low=domain_height_lower, high=domain_height_upper) / dz
        ).astype(int) * dz
        self.ditch_width = (
            rand_within(n=n, low=ditch_width_lower, high=ditch_width_upper) / dx0
        ).astype(int) * dx0
        self.xycoords = xy_spacing(
            xleft=-ditch_width_upper,
            xright=ditch_width_upper,
            ny=n,
            width=domain_width,
            dx0=dx0,
            power=dx_growth_rate,
        )
        self.dewatering_depth = rand_within(n=n, low=dewatering_depth_lower, high=dewatering_depth_upper)
        self.stage = self.domain_height - self.dewatering_depth
        if (self.stage <= 0).any():
            raise ValueError("Dewatering depth should not exceed domain height")
        self.depth = rand_within(n=n, low=ditch_depth_lower, high=ditch_depth_upper)
        self.elevation = self.stage - self.depth
        self.dis = create_dis(
            xycoords=self.xycoords,
            dz=dz,
            ditch_elevation=self.elevation,
            ditch_width=self.ditch_width,
            domain_height=self.domain_height,
        )
        self.z = self.dis["bottom"].isel(x=0, y=0).to_numpy() + 0.5 * dz
        self.ic = create_ic(self.dis["idomain"], self.stage)
        self.npf = None
        self.rch = None
        self.riv = None
        self.drn = None
        self.ghb = None

    def set_conductivity(
        self,
        kh_lower: float,
        kh_upper: float,
        anisotropy_lower: float,
        anisotropy_upper: float,
        unconfined: bool,
    ) -> None:
        self.anisotropy = rand_within(
            n=self.n, low=anisotropy_lower, high=anisotropy_upper
        )
        self.k11 = rand_within(n=self.n, low=kh_lower, high=kh_upper)
        self.k33 = self.k11 * self.anisotropy
        self.npf = create_npf(
            self.dis["idomain"], k11=self.k11, k33=self.k33, unconfined=unconfined
        )

    def set_recharge(
        self,
        rate_lower: float,
        rate_upper: float,
    ) -> None:
        self.recharge_rate = rand_within(n=self.n, low=rate_lower, high=rate_upper)
        self.rch = create_rch(self.dis["idomain"], rate=self.recharge_rate)

    def set_ditch(self, c0_lower: float, c0_upper: float) -> None:
        self.c0 = rand_within(n=self.n, low=c0_lower, high=c0_upper)
        dis = self.dis
        npf = self.npf
        if npf is None:
            raise ValueError("set_conductivity has not been called yet")

        self.riv = create_riv(
            dis["idomain"],
            bottom=dis["bottom"],
            c0=self.c0,
            k11=npf["k"],
            k33=npf["k33"],
            stage=self.stage,
            elevation=self.elevation,
            dx0=self.dx0,
            dz=self.dz,
        )

    def set_seepage_phase(self):
        dis = self.dis
        npf = self.npf
        self.drn = create_seepage(
            dis["idomain"],
            bottom=dis["bottom"],
            k11=npf["k"],
            stage=self.stage,
            dx0=self.dx0,
            dz=self.dz,
        )

    def set_aquifer(
        self, c1_lower: float, c1_upper: float, dhead_lower: float, dhead_upper: float
    ) -> None:
        self.c1 = rand_within(n=self.n, low=c1_lower, high=c1_upper)
        self.dhead = rand_within(n=self.n, low=dhead_lower, high=dhead_upper)
        self.aquifer_head = self.stage + self.dhead
        self.ghb = create_aquifer(
            self.dis["idomain"], c1=self.c1, head=self.aquifer_head
        )

    def setup_simulation(self, directory: str, binary: bool = True) -> None:
        self.directory = directory
        gwf = imod.mf6.GroundwaterFlowModel(newton=True)
        gwf["dis"] = self.dis
        gwf["npf"] = self.npf
        gwf["ic"] = self.ic
        gwf["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
        gwf["sto"] = imod.mf6.StorageCoefficient(
            storage_coefficient=0.0, specific_yield=0.0, transient=False, convertible=0
        )
        if self.rch is not None:
            gwf["rch"] = self.rch
        if self.riv is not None:
            gwf["riv"] = self.riv
        if self.drn is not None:
            gwf["drn"] = self.drn
        if self.ghb is not None:
            gwf["ghb"] = self.ghb

        simulation = imod.mf6.Modflow6Simulation(name="clay")
        simulation["GWF_1"] = gwf
        # Essentially simple preset with no_ptc
        simulation["solver"] = imod.mf6.Solution(
            modelnames=["GWF_1"],
            print_option="all",
            outer_dvclose=1.0e-5,
            outer_maximum=100,
            inner_dvclose=1.0e-6,
            inner_maximum=200,
            inner_rclose=1.0e-6,
            under_relaxation=None,
            under_relaxation_theta=0.0,
            under_relaxation_kappa=0.0,
            under_relaxation_gamma=0.0,
            under_relaxation_momentum=0.0,
            backtracking_number=0,
            backtracking_tolerance=0.0,
            backtracking_reduction_factor=0.0,
            backtracking_residual_limit=0.0,
            rclose_option="strict",
            linear_acceleration="bicgstab",
            relaxation_factor=0.0,
            preconditioner_levels=0,
            preconditioner_drop_tolerance=0,
            number_orthogonalizations=0,
            no_ptc="all",  # crucial setting
        )
        simulation.create_time_discretization(
            additional_times=["2020-01-01", "2020-01-02"]
        )
        simulation.write(directory=directory, binary=binary)
        self.simulation = simulation

    def run(self):
        self.simulation.run()
        self._head = self.simulation.open_head().isel(time=0, drop=True).compute()
        self._budgets = (
            self.simulation.open_flow_budget().isel(time=0, drop=True).compute()
        )

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
