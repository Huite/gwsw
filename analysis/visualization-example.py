# %%

import matplotlib.pyplot as plt

from gwsw.results_reader import Results

# %%

name = "single-layer-clay"
results = Results(f"../modflow6/{name}", f"../sectiondata/{name}.csv", dz=0.05)

# %%

head = results.head()
budgets = results.budgets()
streamfunction = results.streamfunction()
watertable = results.groundwatertable()

# %%

y = 100
x = slice(None, None)

fig, ax = plt.subplots(figsize=(15, 5))
head.isel(y=y, x=x).plot.contour(ax=ax, levels=20, cmap="viridis")
streamfunction.isel(y=y, x=x).plot.contour(ax=ax, levels=20, cmap="turbo")
watertable.isel(y=y, x=x).plot(ax=ax, color="black")
ax.set_aspect(1.0)

# %%
