# %%
import numpy as np

np.random.seed(12345)

from gwsw.numerical_model import ManyCrossSections
from gwsw.results_reader import Results
import matplotlib.pyplot as plt
# %%

case = {
    "section": dict(
        n=2,
        domain_width=125.0,
        domain_height_lower=3.0,
        domain_height_upper=10.0,
        ditch_width_lower=1.5,
        ditch_width_upper=1.5,
        dewatering_depth_lower=1.0,
        dewatering_depth_upper=2.0,
        ditch_depth_lower=0.3,
        ditch_depth_upper=0.3,
        dx0=0.05,
        dx_growth_rate=1.1,
        dz=0.05,
    ),
    "conductivity": dict(
        kh_lower=0.5,
        kh_upper=3.0,
        anisotropy_lower=0.1,
        anisotropy_upper=0.5,
        unconfined=True,
    ),
    "recharge": dict(rate_lower=0.008, rate_upper=0.008),
    "ditch": dict(c0_lower=1.0, c0_upper=2.5),
    "aquifer": dict(c1_lower=50.0, c1_upper=1000.0, dhead_lower=0.1, dhead_upper=0.1),
}

name = "basic"
sections = ManyCrossSections(**case["section"])
sections.set_conductivity(**case["conductivity"])
sections.set_recharge(**case["recharge"])
sections.set_ditch(**case["ditch"])
sections.set_seepage_phase()
sections.setup_simulation(f"../modflow6/single-layer-{name}", binary=True)
sections.run()
sections.to_dataframe().to_csv(f"../sectiondata/single-layer-{name}.csv")
# %%

name = f"single-layer-{name}"
results = Results(f"../modflow6/{name}", f"../sectiondata/{name}.csv", dz=0.05)
# %%

y = 0 
x = slice(None, None)

fig, ax = plt.subplots(figsize=(25, 5))
head.isel(y=y, x=x, drop=True).plot.contour(ax=ax, levels=20, colors="blue")
streamfunction.isel(y=y, x=x, drop=True).plot.contour(ax=ax, levels=10, colors="red")
watertable.isel(y=y, x=x, drop=True).plot(ax=ax, color="black")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

fig.savefig("../figures/check-sand.png", dpi=200)

# %%
