# %%
import numpy as np

np.random.seed(12345)

from gwsw.numerical_model import ManyCrossSections

# %%

N = 500
CASES = {
    "clay": {
        "section": dict(
            n=N,
            domain_width=50.0,
            domain_height_lower=3.0,
            domain_height_upper=3.0,
            ditch_width_lower=1.0,
            ditch_width_upper=3.0,
            dewatering_depth_lower=1.0,
            dewatering_depth_upper=1.0,
            ditch_depth_lower=0.5,
            ditch_depth_upper=0.5,
            dx0=0.05,
            dx_growth_rate=1.1,
            dz=0.05,
        ),
        "conductivity": dict(
            kh_lower=0.5,
            kh_upper=2.5,
            anisotropy_lower=0.1,
            anisotropy_upper=0.3,
            unconfined=True,
        ),
        "recharge": dict(rate_lower=0.0005, rate_upper=0.0005),
        "ditch": dict(c0_lower=1.0, c0_upper=2.5),
        "aquifer": dict(c1_lower=50.0, c1_upper=1000.0, dhead_lower=0.1, dhead_upper=0.1),
    },
#
    "peat": {
        "section": dict(
            n=N,
            domain_width=15.0,
            domain_height_lower=1.6,
            domain_height_upper=1.6,
            ditch_width_lower=0.5,
            ditch_width_upper=2.5,
            dewatering_depth_lower=0.6,
            dewatering_depth_upper=0.6,
            ditch_depth_lower=0.3,
            ditch_depth_upper=0.3,
            dx0=0.05,
            dx_growth_rate=1.1,
            dz=0.05,
        ),
        "conductivity": dict(
            kh_lower=0.5,
            kh_upper=1.0,
            anisotropy_lower=0.1,
            anisotropy_upper=0.5,
            unconfined=True,
        ),
        "recharge": dict(rate_lower=0.0005, rate_upper=0.0005),
        "ditch": dict(c0_lower=1.0, c0_upper=2.5),
        "aquifer": dict(c1_lower=50.0, c1_upper=1000.0, dhead_lower=0.1, dhead_upper=0.1),
    },
#
    "sand": {
        "section": dict(
            n=N,
            domain_width=125.0,
            domain_height_lower=3.0,
            domain_height_upper=10.0,
            ditch_width_lower=0.5,
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
        "recharge": dict(rate_lower=0.0005, rate_upper=0.0005),
        "ditch": dict(c0_lower=1.0, c0_upper=2.5),
        "aquifer": dict(c1_lower=50.0, c1_upper=1000.0, dhead_lower=0.1, dhead_upper=0.1),
    },
#
    "brook": {
        "section": dict(
            n=N,
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
        ),
        "conductivity": dict(
            kh_lower=2.0,
            kh_upper=10.0,
            anisotropy_lower=0.1,
            anisotropy_upper=0.5,
            unconfined=True,
        ),
        "recharge": dict(rate_lower=0.0005, rate_upper=0.0005),
        "ditch": dict(c0_lower=1.0, c0_upper=2.5),
        "aquifer": dict(c1_lower=50.0, c1_upper=1000.0, dhead_lower=0.1, dhead_upper=0.1),
    },
}

# %%

for name, case in CASES.items():
    print(f"Setting up {name} single-layer")
    sections = ManyCrossSections(**case["section"])
    sections.set_conductivity(**case["conductivity"])
    sections.set_recharge(**case["recharge"])
    sections.set_ditch(**case["ditch"])
    sections.set_seepage_phase()
    sections.setup_simulation(f"../modflow6/single-layer-{name}", binary=True)
    sections.to_dataframe().to_csv(f"../sectiondata/single-layer-{name}.csv")
    print(f"Running {name} single-layer")
    sections.run()
    
    print(f"Setting up {name} two-layer")
    sections.set_aquifer(**case["aquifer"])
    sections.setup_simulation(f"../modflow6/two-layer-{name}", binary=True)
    sections.to_dataframe().to_csv(f"../sectiondata/two-layer-{name}.csv")
    print(f"Running {name} two-layer")
    sections.run()

# %%