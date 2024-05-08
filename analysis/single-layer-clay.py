# %%
import numpy as np

np.random.seed(12345)

from gwsw.numerical_model import ManyCrossSections
import matplotlib.pyplot as plt

# %%

sections = ManyCrossSections(
    n=100,
    domain_width=200.0,
    domain_height=5.0,
    ditch_width_lower=1.0,
    ditch_width_upper=3.0,
    ditch_stage_lower=2.0,
    ditch_stage_upper=2.0,
    ditch_depth_lower=0.5,
    ditch_depth_upper=0.5,
    dx0=0.05,
    dx_growth_rate=1.1,
    dz=0.05,
)
# %%

sections.set_conductivity(
    kh_lower=0.5,
    kh_upper=2.5,
    anisotropy_lower=0.1,
    anisotropy_upper=0.3,
    unconfined=True,
)
sections.set_recharge(rate_lower=0.0005, rate_upper=0.0005)
sections.set_aquifer(
    c1_lower=100.0, c1_upper=1000.0, dhead_lower=-0.2, dhead_upper=-0.2
)
sections.set_ditch(c0_lower=1.0, c0_upper=2.5)
sections.set_seepage_phase()
sections.setup_simulation("../modflow6/single-layer-clay", binary=False)

# %%

sections.run()

# %%

head = sections.head()
budgets = sections.budgets()
streamfunction = sections.streamfunction()
watertable = sections.groundwatertable()

# %%

fig, ax = plt.subplots(figsize=(50, 10))
head.isel(y=0).plot.contour(ax=ax, levels=20)
streamfunction.isel(y=0).plot.contour(ax=ax, levels=20, cmap="turbo")
watertable.isel(y=0).plot(ax=ax, color="black")
ax.set_aspect(1.0)
fig.savefig("../figures/check.png", dpi=300)

# %%

y = 4.5
x = slice(None, 5.4)

fig, ax = plt.subplots(figsize=(10, 5))
head.sel(y=y, x=x).plot(ax=ax, levels=10)
# %%

b0 = budgets.isel(y=0)
print(b0["drn"].sum())
print(b0["rch"].sum())
print(b0["riv"].sum())
print(b0["ghb"].sum())

# %%
