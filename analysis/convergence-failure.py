# %%
import numpy as np
np.random.seed(12345)

from gwsw.numerical_model import ManyCrossSections
import matplotlib.pyplot as plt

# %%
# Model setup

sections = ManyCrossSections(
    n=2,
    domain_width=125.0,
    domain_height=3.0,
    ditch_width_lower=0.5,
    ditch_width_upper=0.5,
    ditch_stage_lower=2.0,
    ditch_stage_upper=2.0,
    ditch_depth_lower=0.3,
    ditch_depth_upper=0.3,
    dx0=0.05,
    dx_growth_rate=1.01,
    dz=0.05,
)
sections.set_conductivity(
    kh_lower=0.7,
    kh_upper=0.7,
    anisotropy_lower=0.3,
    anisotropy_upper=0.3,
    unconfined=True,
)
sections.set_recharge(rate_lower=0.0005, rate_upper=0.0005)
sections.set_ditch(c0_lower=1.0, c0_upper=2.5)
sections.set_seepage_phase()
sections.setup_simulation("../modflow6/sand-single-layer", binary=False)
# %%
sections.run()

# %%



# %%
# Gather results

head = sections.head()
budgets = sections.budgets()
streamfunction = sections.streamfunction()
watertable = sections.groundwatertable()

# %%
# Diagnostic plots

y = 1 

fig, ax = plt.subplots(figsize=(15, 5))
head.isel(y=y).plot.contour(ax=ax, levels=10)
streamfunction.isel(y=y).plot.contour(ax=ax, levels=10, cmap="turbo")
watertable.isel(y=y).plot(ax=ax, color="black")
#ax.set_aspect(1.0)
#fig.savefig("../figures/check.png", dpi=300)

# %%
# Analytical comparison

y = 4.5
x = slice(None, 5.4)

fig, ax = plt.subplots(figsize=(10, 5))
head.sel(y=y, x=x).plot(ax=ax, levels=10)
# %%


# %%
