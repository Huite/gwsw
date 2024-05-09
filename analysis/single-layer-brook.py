# %%
import numpy as np

np.random.seed(12345)

from gwsw.numerical_model import ManyCrossSections
import matplotlib.pyplot as plt

# %%
# Model setup

sections = ManyCrossSections(
    n=100,
    domain_width=500.0,
    domain_height_lower=5.0,
    domain_height_upper=15.0,
    ditch_width_lower=5.0,
    ditch_width_upper=10.0,
    dewatering_depth_lower=2.0,
    dewatering_depth_upper=2.0,
    ditch_depth_lower=1.5,
    ditch_depth_upper=1.5,
    dx0=0.05,
    dx_growth_rate=1.1,
    dz=0.05,
)
sections.set_conductivity(
    kh_lower=2.0,
    kh_upper=10.0,
    anisotropy_lower=0.1,
    anisotropy_upper=0.5,
    unconfined=True,
)
sections.set_recharge(rate_lower=0.0005, rate_upper=0.0005)
sections.set_ditch(c0_lower=1.0, c0_upper=2.5)
sections.set_seepage_phase()
sections.setup_simulation("../modflow6/single-layer-brook", binary=True)

# %%
sections.run()

# %%
# Gather results

head = sections.head()
budgets = sections.budgets()
streamfunction = sections.streamfunction()
watertable = sections.groundwatertable()

# %%
# Diagnostic plots

y = 0
x = slice(None, 500)

fig, ax = plt.subplots(figsize=(15, 5))
head.isel(y=y, x=x).plot.contour(ax=ax, levels=10)
streamfunction.isel(y=y, x=x).plot.contour(ax=ax, levels=10, cmap="turbo")
watertable.isel(y=y, x=x).plot(ax=ax, color="black")
#ax.set_aspect(1.0)
# fig.savefig("../figures/check.png", dpi=300)

# %%
# Analytical comparison
