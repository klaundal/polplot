# Code for plotting on latitude/local time grid

A class for making plots in polar coordinates. By default the polar coordinates are called lat (latitude) and LT (local time), since that is what we use this code for most often. lat is 0 at equator and 90 at the pole. lt ranges from 0 to 24 (hours).

This is how you use it:

```python

import matplotlib.pyplot as plt
from polplot import pp

fig, ax = plt.subplots()
pax = pp(ax)

# get your data as functions of mlat and mlt, and plot it:
pax.plot(mlat, mlt)
pax.scatter(mlat, mlt)
pax.contour(mlat, mlt, Z)
pax.contourf(mlat, mlt, Z)
# and so on. The functions are wrappers for the corresponding matplotlib function
# so you can use the same set of keywords to manipulate lines, colors, etc. 
```
The `pax` object above has the original matplotlib.axes subplot object as a member variable called `ax`. 

## Note:
Not everything in the code is well documented, and there may be errors. If in doubt, please ask. It is published here as is because we use it in many other projects. 
