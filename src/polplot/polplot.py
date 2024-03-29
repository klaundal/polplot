"""
Polplot Module
==============

This module provides the Polarplot class for creating and managing plots in a polar latitude-local 
time coordinate system. It is designed to facilitate the visualization of geophysical or astronomical 
data where such a coordinate system is more meaningful than traditional Cartesian coordinates.

The Polarplot class in this module offers a range of methods that mirror those in Matplotlib but 
are adapted for plotting in polar coordinates. These methods include plotting data points, lines, 
text, and complex figures like contours and scatter plots. The class handles the transformation 
from geographic or magnetic coordinates to the polar latitude-local time system, making it a 
useful tool for researchers and scientists working in fields like space physics or meteorology.

Key Features:
- Conversion between latitude/local time and Cartesian coordinates.
- Plotting methods adapted for polar latitude-local time coordinates.
- Customizable plotting and visualization options.
- Integration with standard Matplotlib AxesSubplot for ease of use.

Example Usage:
    import matplotlib.pyplot as plt
    import polplot

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pax = polplot.Polarplot(ax, minlat=50)
    pax.plot(data_lat, data_lt, **kwargs)
    plt.show()

The module is designed to be intuitive for those familiar with Matplotlib, 
providing a straightforward transition to polar latitude-local time plotting.

"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.interpolate import griddata
from datetime import datetime
from matplotlib import rc
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PolyCollection, LineCollection

d2r = np.pi / 180

datapath = os.path.dirname(os.path.abspath(__file__)) + '/data/'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Verdana']})
rc('text', usetex=True)

class Polarplot(object):
    def __init__(self, ax, minlat = 50, plotgrid = True, sector = 'all', lt_label='lt', lat_label='lat', **kwargs):
        """
        Initialize a Polarplot object for plotting data in polar latitude-local time coordinates.

        This class facilitates plotting in a polar coordinate system, where the radial dimension
        represents latitude and the angular dimension represents local time. It offers various
        functions for plotting data, adding text, and more, in this coordinate system.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Matplotlib AxesSubplot object where the plot will be drawn.
        minlat : float, optional
            The minimum latitude boundary of the plot. This value sets the innermost circle of the
            polar plot. The default is 50 degrees.
        plotgrid : bool, optional
            If True, a grid is plotted on initialization. The grid consists of circles at constant
            latitudes and lines at constant local times. The default is True.
        sector : str, optional
            Specifies the sector to be plotted. Can be 'all' for a full polar plot or other
            values to represent specific sectors. The default is 'all'.
        lt_label : str, optional
            The label used for local time coordinates. Default is 'lt'.
        lat_label : str, optional
            The label used for latitude coordinates. Default is 'lat'.
        **kwargs : dict
            Additional keyword arguments that are passed to the plotting functions for the grid.
            These can include parameters like color, linestyle, etc.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> import polplot
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> pax = polplot.Polarplot(ax)
        >>> pax.plot(lat, lt, **kwargs)
        >>> plt.show()

        The class provides various member functions including plot(), text(), scatter(),
        contour(), and contourf(), which are analogous to their Matplotlib counterparts but
        are adapted for the polar latitude-local time coordinate system.
        """

        self.minlat = minlat # the lower latitude boundary of the plot
        self.ax = ax
        self.ax.axis('equal')
        self.sector = sector

        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = .5

        if 'color' not in kwargs.keys():
            kwargs['color'] = 'lightgrey'

        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'

        # define the lt limits
        if sector.lower() == 'all':
            self.ltlims = 'lt >= 0'
        elif sector.lower() == 'dusk':
            self.ltlims = '(lt >= 12) | (lt == 0)'
        elif sector.lower() == 'dawn':
            self.ltlims = '(lt <= 12) | (lt == 24)'
        elif sector.lower() == 'night':
            self.ltlims = '(lt >= 18) | (lt <= 6)'
        elif sector.lower() == 'day':
            self.ltlims = '(lt >= 6) & (lt <=18)'
        else:
            sector = [float(s) for s in sector.split('-')]
            if sector[0]> sector[-1]:
                self.ltlims= f'(lt>={sector[0]})|(lt<={sector[-1]})'
            else:
                self.ltlims= f'(lt>={sector[0]})&(lt<={sector[-1]})'

        self.ax.set_axis_off()

        self.ax.format_coord= self._create_coordinate_formatter(lt_label, lat_label)
        self.lat_labels=False
        self.lt_labels= False
        if plotgrid:
            self.plotgrid(**kwargs)

        # set suitable plot limits by drawing a circle at minlat:
        x, y = self._latlt2xy(np.full(100, self.minlat), np.linspace(0, 24, 100))
        x, y = np.hstack((x, 0)), np.hstack((y, 0)) # add origin
        self.ax.set_xlim(np.nanmin(x) - 0.1, np.nanmax(x) + 0.1)
        self.ax.set_ylim(np.nanmin(y) - 0.1, np.nanmax(y) + 0.1)


    def plot(self, lat, lt, **kwargs):
        """
        Wrapper for matplotlib's plot function. Accepts lat, lt instead of x and y

        Keywords passed to this function are passed on to matplotlib's plot. 

        The return parameter is the return parameter from matplotlib's plot.


        Parameters
        ----------
        lat: array-like
            array of latitudes [degrees]
        lt: array-like
            array of local times [hours]

        """

        x, y = self._latlt2xy(lat, lt)
        return self.ax.plot(x, y, **kwargs)


    def text(self, lat, lt, text, ignore_plot_limits = False, **kwargs):
        """
        Wrapper for matplotlib's text function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's text.

        The return parameter is the return parameter from matplotlib's text.

        Parameters
        ----------
        lat: float
            latitude of text [degrees]
        lt: float
            local time of text [hours]
        text: string
            the text to be shown
        ignore_plot_limits: bool, optional
            set to True to allow plotting outside plot limits. Default is False
        """

        x, y = self._latlt2xy(lat, lt, ignore_plot_limits=ignore_plot_limits)

        if np.isfinite(x + y):
            return self.ax.text(x, y, text, **kwargs)
        else:
            print('text outside plot limit - set "ignore_plot_limits = True" to override')


    def write(self, lat, lt, text, ignore_plot_limits = False, **kwargs):
        """
        Wrapper for matplotlib's text function. Accepts lat, lt instead of x and y

        keywords passed to this function is passed on to matplotlib's text.

        The return parameter is the return parameter from matplotlib's text.

        Parameters
        ----------
        lat: float
            latitude of text [degrees]
        lt: float
            local time of text [hours]
        text: string
            the text to be shown
        ignore_plot_limits: bool, optional
            set to True to allow plotting outside plot limits. Default is False
        """

        return self.text(lat, lt, text, ignore_plot_limits, **kwargs)


    def scatter(self, lat, lt, **kwargs):
        """
        Wrapper for matplotlib's scatter function. Accepts lat, lt instead of x and y

        Keywords passed to this function are passed on to matplotlib's scatter. 

        The return parameter is the return parameter from matplotlib's scatter.


        Parameters
        ----------
        lat: array-like
            array of latitudes [degrees]
        lt: array-like
            array of local times [hours]

        """

        x, y = self._latlt2xy(lat, lt)
        return self.ax.scatter(x, y, **kwargs)


    def plotgrid(self, labels=False, **kwargs):
        """ plot lt, lat-grid on self.ax

        parameters
        ----------
        labels: bool
            set to True to include lat/lt labels
        **kwarsgs: dictionary
            passed to matplotlib's plot function (for linestyle etc.)

        """
        returns= []
        for lt in [0, 6, 12, 18]:
            returns.append(self.plot([self.minlat, 90], [lt, lt], **kwargs))

        lts = np.linspace(0, 24, 100)
        for lat in np.r_[90: self.minlat -1e-12 :-10]:
            returns.append(self.plot(np.full(100, lat), lts, **kwargs))

        # add LAT and LT labels to axis
        if labels:
            if self.lt_labels:
                try:
                    for label in self.lt_labels:
                        label.remove()
                except:
                    pass
            if self.lat_labels:
                try:
                    for label in self.lat_labels:
                        label.remove()
                except:
                    pass
            self.lat_labels= self.writeLATlabels()
            self.lt_labels=self.writeLTlabels()
        return tuple(returns)


    def writeLTlabels(self, lat = None, degrees = False, **kwargs):
        """ 
        Write local time labels at given latitude (default minlat)
        
        If degrees is True, longitude will be used instead of hour
        """
        
        if lat is None:
            lat = self.minlat
        
        labels=[]
        
        if degrees:
            if self.sector in ['all', 'night', 'dawn', 'dusk']:
                labels.append(self.write(lat, 0,    '0$^\circ$', verticalalignment = 'top'   , horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            if self.sector in ['all', 'night', 'dawn', 'day']:
                labels.append(self.write(lat, 6,   '90$^\circ$', verticalalignment = 'center', horizontalalignment = 'left'  , ignore_plot_limits=True, **kwargs))
            if self.sector in ['all', 'dusk', 'dawn', 'day']:
                labels.append(self.write(lat, 12, '180$^\circ$', verticalalignment = 'bottom', horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            if self.sector in ['all', 'night', 'dusk', 'day']:
                labels.append(self.write(lat, 18, '-90$^\circ$', verticalalignment = 'center', horizontalalignment = 'right' , ignore_plot_limits=True, **kwargs))
        else:
            lt=np.array([0, 24])
            if any(eval(self.ltlims)):
                labels.append(self.write(lat, 0, '00', verticalalignment = 'top'    , horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            lt=6
            if eval(self.ltlims):
                labels.append(self.write(lat, 6, '06', verticalalignment = 'center' , horizontalalignment = 'left'  , ignore_plot_limits=True, **kwargs))
            lt=12
            if eval(self.ltlims):
                labels.append(self.write(lat, 12, '12', verticalalignment = 'bottom', horizontalalignment = 'center', ignore_plot_limits=True, **kwargs))
            lt=18
            if eval(self.ltlims):
                labels.append(self.write(lat, 18, '18', verticalalignment = 'center', horizontalalignment = 'right' , ignore_plot_limits=True, **kwargs))
        if self.lt_labels:
            try:
                for label in self.lt_labels:
                    label.remove()
            except:
                pass
        self.lt_labels= labels
        return labels


    def writeLATlabels(self, lt=3, lats=None, rotation=45, north = True, color='lightgrey', backgroundcolor='white', zorder=2, **kwargs):
        """
        Write latitude labels at a specified meridian.

        This function adds latitude labels to a plot, with customizable appearance. 
        The labels are placed at specified longitude and latitudes at intervals of 10 degrees 
        starting from `self.minlat` up to 80 degrees.

        kwargs are passed to self.text

        Parameters
        ----------
        lt : int, default 3
            Longitude for placing the labels.
        rotation : int, default 45
            Text rotation angle.
        north : bool, default True
            Set to False to add a minus sign for the labels
        color : str, default 'lightgrey'
            Text color. Set to (0, 0, 0, 0) to disable background
        backgroundcolor : str, default 'white'
            Background color of the text.
        zorder : int, default 2
            Z-order for layering on the plot.


        Notes
        -----
        The function internally calls `self.text` method to generate each label. 
        """

        label_params = {'rotation': rotation, 'color': color, 'backgroundcolor': backgroundcolor, 'zorder': zorder}
        label_params.update(kwargs)

        labels = []

        sign = '' if north else '-'

        if not lats:
            lats=np.r_[80:self.minlat-1e-12:-10][::-1].astype('int64')
        for lat in lats:
            labels.append(self.write(lat, lt, sign + f'{lat}°', ignore_plot_limits = False, **label_params))
        if self.lat_labels:
            try:
                for label in self.lat_labels:
                    label.remove()
            except:
                pass
        self.lat_labels= labels
        return labels


    def plotpins(self, lats, lts, poleward, eastward, reverse=False, rotation=0, SCALE=None, size=10, unit='', colors='black', markercolor='black', marker='o', markersize=20, **kwargs):
        """
        Plot a vector field with dots and lines representing vectors.

        Each vector is depicted as a dot at its origin and a line indicating its direction and magnitude.

        Parameters
        ----------
        lats : array_like
            Array of latitudes [deg] for each vector. Must be positive
        lts : array_like
            Array of local times [h] for each vector.
        poleward : array_like
            Poleward component of each vector.
        eastward : array_like
            Eastward component of each vector.
        reverse : bool, optional
            If True, the pins will be placed at the end of the vectors. Default is False
        rotation : float, optional
            Rotation angle for vectors, in degrees. Default is 0.
        SCALE : float or None, optional
            Scale factor for vector magnitude. If None, a default scale of 1 is used.
        size : int, optional
            Font size for the unit text. Default is 10.
        unit : str, optional
            Unit label for the scale bar. Default is an empty string.
        colors : str or array_like, optional
            Color for the vector lines. Default is 'black'.
        markercolor : str or array_like, optional
            Color for the markers. Default is 'black'.
        marker : str, optional
            Marker style. Default is 'o' (circle).
        markersize : int, optional
            Size of the markers. Default is 20.
        **kwargs : dict, optional
            Additional keyword arguments passed to `ax.plot` for line customization.

        Returns
        -------
        tuple
            A tuple of Matplotlib objects representing the plotted elements.


        """

        lts, lats, poleward, eastward = [np.asarray(a).flatten() for a in (lts, lats, poleward, eastward)]
        R = np.array([[np.cos(np.deg2rad(rotation)), -np.sin(np.deg2rad(rotation))], 
                      [np.sin(np.deg2rad(rotation)), np.cos(np.deg2rad(rotation))]])

        scale = 0.1 / SCALE if SCALE is not None else 1.

        x, y = self._latlt2xy(lats, lts)
        dx, dy = R.dot(self._northEastToCartesian(poleward, eastward, lts)) * scale
        segments = np.dstack((np.vstack((x, x + dx)).T, np.vstack((y, y + dy)).T))

        returns = [self.ax.add_collection(LineCollection(segments, colors=colors, **kwargs))]

        if unit and SCALE is not None:
            returns.extend([
                self.ax.plot([0.9, 1], [0.95, 0.95], color=colors, linestyle='-', linewidth=2),
                self.ax.text(0.9, 0.95, f'{SCALE:.1f} {unit}', ha='right', va='center', fontsize=size)
            ])

        if markersize != 0:
            if reverse:
                returns.append(self.ax.scatter(x + dx, y + dy, marker=marker, c=markercolor, s=markersize, edgecolors=markercolor))
            else:
                returns.append(self.scatter(lats, lts, marker=marker, c=markercolor, s=markersize, edgecolors=markercolor))

        return tuple(returns)


    def quiver(self, lat, lt, poleward, eastward, rotation=0, qkeyproperties=None, **kwargs):
        """
        Wrapper for Matplotlib's quiver function for latitude/local time coordinates.

        This function plots vectors defined by poleward and eastward components at specified
        latitudes and longitudes. An optional rotation can be applied to the vectors.

        Parameters
        ----------
        lat : array_like
            Array of latitudes [deg] for each vector. Must be positive
        lt : array_like
            Array of local times [h] for each vector.
        poleward : array_like
            Poleward component of each vector.
        eastward : array_like
            Eastward component of each vector.
        rotation : float, optional
            Rotation angle for the vectors, in radians. Default is 0.
        qkeyproperties : dict or None, optional
            Properties for the quiver key. If set to a dictionary, it must contain 'label' and 'U'.
            'label' is the label written next to the arrow, and 'U' is the length of the arrow key in
            the same unit as the vector components.
        **kwargs : dict, optional
            Additional keyword arguments passed to Matplotlib's quiver function.

        Returns
        -------
        matplotlib.quiver.Quiver
            The quiver plot object.

        Notes
        -----
        The function is a convenience wrapper around Matplotlib's quiver function, tailored for
        geographic coordinates and vector components. It allows easy plotting of vector fields
        such as wind or ocean current data.
        """

        lt, lat, poleward, eastward = [np.asarray(a).flatten() for a in (lt, lat, poleward, eastward)]
        R = np.array([[np.cos(rotation), -np.sin(rotation)], 
                      [np.sin(rotation), np.cos(rotation)]])

        x, y = self._latlt2xy(lat, lt)
        dx, dy = R.dot(self._northEastToCartesian(poleward, eastward, lt))

        q = self.ax.quiver(x, y, dx, dy, **kwargs)

        if qkeyproperties is not None:
            qkeykwargs = {'X': .8, 'Y': .9, 'labelpos': 'E'}
            qkeykwargs.update(qkeyproperties)
            self.ax.quiverkey(q, **qkeykwargs)

        return q


    def contour(self, lat, lt, f, **kwargs):
        """
        Wrapper for Matplotlib's contour function using latitude and local time.

        This function creates contour plots using latitude and local time coordinates.
        It translates these coordinates into a Cartesian grid suitable for contour plotting.

        Parameters
        ----------
        lat : array_like
            Array of latitudes [deg] for the contour plot. Must be positive
        lt : array_like
            Array of local times [h] for the contour plot.
        f : array_like
            Array of values to contour.
        **kwargs : dict, optional
            Additional keyword arguments passed to Matplotlib's contour function.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            The contour plot object.

        Notes
        -----
        The function flattens the input arrays and uses a grid interpolation to map the data
        onto a uniform Cartesian grid for contour plotting. 
        """

        xea, yea = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits = True)
        mask, _  = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits = False)
        f = f.flatten()
        f[~np.isfinite(mask)] = np.nan

        # Convert to Cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack((xea, yea)).T
        gridf = griddata(points, f, (xx, yy), method = 'linear')

        # Plot and return the contour
        return self.ax.contour(xx, yy, gridf, **kwargs)


    def contourf(self, lat, lt, f, **kwargs):
        """
        Wrapper for Matplotlib's contourf function using latitude and local time.

        This function creates filled contour plots (contourf) using positive latitude values and 
        local time coordinates. It maps these coordinates onto a Cartesian grid suitable for contour plotting.

        Parameters
        ----------
        lat : array_like
            Array of latitudes [deg] for the contour plot. Must be positive.
        lt : array_like
            Array of local times [h] for the contour plot.
        f : array_like
            Array of values to contour.
        **kwargs : dict, optional
            Additional keyword arguments passed to Matplotlib's contourf function.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            The filled contour plot object.

        Notes
        -----
        The function flattens the input arrays and uses a grid interpolation to map the data
        onto a uniform Cartesian grid for contour plotting. 
        """

        xea, yea = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits=True)
        mask, _ = self._latlt2xy(lat.flatten(), lt.flatten(), ignore_plot_limits=False)
        f = f.flatten()
        f[~np.isfinite(mask)] = np.nan

        # Convert to Cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack((xea, yea)).T
        gridf = griddata(points, f, (xx, yy), method = 'linear')

        # Plot and return the filled contour
        return self.ax.contourf(xx, yy, gridf, **kwargs)


    def fill(self, lat, lt, **kwargs):
        """
        Fill a polygon defined by latitude and local time coordinates.

        This function creates a filled polygon on the plot using latitude and local time coordinates.
        The filling properties can be customized using additional keyword arguments.

        Parameters
        ----------
        lat : array_like
            Array of latitudes [deg] defining the vertices of the polygon. Must be positive
        lt : array_like
            Array of local times (in hours) corresponding to each latitude, defining the vertices of the polygon.
        **kwargs : dict, optional
            Additional keyword arguments passed to Matplotlib's ax.fill function for customizing the polygon's appearance.

        Returns
        -------
        list of matplotlib.patches.Polygon
            A list containing the filled polygon object(s).


        """

        xx, yy = self._latlt2xy(lat.flatten(), lt.flatten())

        # Plot and return the filled polygon
        return self.ax.fill(xx, yy, **kwargs)


    def plot_terminator(self, time, sza=90, north=True, apex=None, shadecolor=None, shade_kwargs={}, **kwargs):
        """
        Plot the sunlight terminator, and (if shadecolor != None) shade area antisunward of the terminator.

        Parameters
        ----------
        time : float or datetime
            If datetime, calculates the subsolar point and sunlight terminator for the specified time.
            If float, draws a horizontal terminator line at the specified degree from the pole towards the night side.
        sza : float, optional
            The terminator is defined as contours of this solar zenith angle. Default is 90 degrees.
        north : bool, optional
            Set to True for terminator in the north (default), False for south.
        apex : apexpy.Apex object, optional
            If provided, converts the terminator to magnetic apex coordinates. Default is geographic.
        shadecolor : str, optional
            Color for shading the night side of the terminator. Default is None (no shading).
        shade_kwargs : dict
            Keywords passed to matplotlib.patches.Polygon function if shadecolor is not None.
        kwargs : dict
            Keywords passed to matplotlib's plot function for the terminator line.

        Returns
        -------
        tuple
            A tuple containing the plot objects created (terminator line, optional shading).
        """

        returns = []
        hemisphere = 1 if north else -1

        if isinstance(time, (float, int)):  # time is interpreted as dipole tilt angle
            y0 = -(hemisphere * time + sza - 90) / (90. - self.minlat)  # noon-midnight coordinate
            xr = np.sqrt(1 - y0**2)

            x = np.array([-xr, xr])
            y = np.array([y0, y0])

        elif isinstance(time, datetime):  # time is a datetime object
            sslat, sslon = subsol(time)  # Substitute with the actual method to get subsolar lat, lon

            # Cartesian vector for subsolar point
            ss = np.array([
                np.cos(np.deg2rad(sslat)) * np.cos(np.deg2rad(sslon)),
                np.cos(np.deg2rad(sslat)) * np.sin(np.deg2rad(sslon)),
                np.sin(np.deg2rad(sslat))
            ])

            # Vector pointing towards dawn, normalized
            dawn_vector = np.cross(ss, np.array([0, 0, 1]))
            dawn_vector /= np.linalg.norm(dawn_vector)

            # Vector pointing northward at the 90 degree SZA contour
            sza90_vector = np.cross(dawn_vector, ss)

            # Rotate to get specified SZA contour (Rodrigues' rotation formula)
            rotation_angle = np.deg2rad(sza - 90)
            sza_vector = sza90_vector * np.cos(rotation_angle) + \
                         np.cross(dawn_vector, sza90_vector) * np.sin(rotation_angle) + \
                         dawn_vector * np.dot(dawn_vector, sza90_vector) * (1 - np.cos(rotation_angle))

            # Trace out the trajectory
            angles = np.linspace(-np.pi/2, 3*np.pi/2, 360)
            r = sza_vector[:, np.newaxis] * np.cos(angles) + \
                np.cross(ss, sza_vector)[:, np.newaxis] * np.sin(angles) + \
                ss[:, np.newaxis] * np.dot(dawn_vector, sza90_vector) * (1 - np.cos(rotation_angle))

            # Convert to spherical coordinates
            t_lat = 90 - np.rad2deg(np.arccos(r[2]))
            t_lon = np.rad2deg(np.arctan2(r[1], r[0])) % 360

            # Longitude difference and conversion to local time
            londiff = (t_lon - sslon + 180) % 360 - 180
            t_lt = (180. + londiff) / 15.

            if apex is not None:
                mlat, mlon = apex.geo2apex(t_lat, t_lon, apex.refh)
                mlt = apex.mlon2mlt(mlon, time)
                t_lat, t_lt = mlat, mlt

            # Limit contour to correct hemisphere
            valid_indices = (t_lat >= self.minlat) if north else (t_lat <= -self.minlat)
            if not np.any(valid_indices):
                return None  # Terminator is outside plot
            t_lat = t_lat[valid_indices]
            t_lt = t_lt[valid_indices]

            x, y = self._latlt2xy(t_lat, t_lt)

        else:
            raise ValueError("time must be a float, int, or datetime object")

        terminator_line = self.ax.plot(x, y, **kwargs)
        returns.append(terminator_line)

        if not x.size:  # Terminator is not in the plot
            return None

        # Shade area if shadecolor is provided
        if shadecolor is not None:
            x0, x1 = np.min(x), np.max(x)
            y0, y1 = np.min(y), np.max(y)

            a0 = np.arctan2(y0, x0)
            a1 = np.arctan2(y1, x1)
            if a1 < a0:
                a1 += 2 * np.pi

            a = np.linspace(a0, a1, 100)[::-1]
            xx, yy = np.cos(a), np.sin(a)

            shade = Polygon(np.vstack((np.hstack((x[::hemisphere], xx)), np.hstack((y[::hemisphere], yy)))).T,
                            closed=True, color=shadecolor, linewidth=0, **shade_kwargs)
            shade_patch = self.ax.add_patch(shade)
            returns.append(shade_patch)

        return tuple(returns)


    def filled_cells(self, lat, lt, latres, ltres, data, resolution=10, crange=None, levels=None, bgcolor=None, verbose=False, **kwargs):
        """
        Create a color plot of specified cells in latitude and local time with associated data.

        Parameters
        ----------
        lat : array_like
            Array of latitudes.
        lt : array_like
            Array of local times.
        latres : array_like
            Array of latitude resolutions.
        ltres : array_like
            Array of local time resolutions.
        data : array_like
            Data array to color the cells.
        resolution : int, optional
            Resolution for cell vertices. Default is 10.
        crange : tuple or None, optional
            Color range for the plot. Default is None.
        levels : array_like or None, optional
            Specific levels for color mapping. Default is None.
        bgcolor : str or None, optional
            Background color. Default is None (no background).
        verbose : bool, optional
            If True, prints additional information. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to PolyCollection.

        Returns
        -------
        matplotlib.collections.PolyCollection
            The added PolyCollection object.

        Raises
        ------
        ValueError
            If input arrays are of unequal length.
        """

        # Validate input lengths
        try: 
            lat, lt, latres, ltres, data = np.broadcast_arrays(lat, lt, latres, ltres, data)
        except:
            raise ValueError("Input arrays to filled_cells are not broadcastable.")

        # Flatten input arrays
        lat, lt, latres, ltres, data = map(np.ravel, [lat, lt, latres, ltres, data])

        # Apply mask
        mask = np.ones_like(data)
        mask[(~eval(self.ltlims)) | (lat < self.minlat)] = np.nan
        data *= mask

        if verbose:
            print(f"Shapes - lt: {lt.shape}, lat: {lat.shape}, latres: {latres.shape}, ltres: {ltres.shape}")

        # Calculate vertices
        la = np.vstack([(lt - 6) / 12 * np.pi + i * ltres / (resolution - 1) / 12 * np.pi for i in range(resolution)]).T
        ua = la[:, ::-1]
        radial_factor = (90 - lat)[:, np.newaxis] / (90 - self.minlat)
        radial_res_factor = (90 - lat - latres)[:, np.newaxis] / (90 - self.minlat)

        vertslo = np.dstack((radial_factor * np.cos(la), radial_factor * np.sin(la)))
        vertshi = np.dstack((radial_res_factor * np.cos(ua), radial_res_factor * np.sin(ua)))
        verts = np.concatenate((vertslo, vertshi), axis=1)

        if verbose:
            print(f"Vertices shape: {verts.shape}")

        # Determine color map
        cmap = kwargs.pop('cmap', plt.cm.viridis)

        # Create PolyCollection
        if levels is not None:
            nlevels = len(levels)
            lmin, lmax = levels.min(), levels.max()
            norm_func = plt.Normalize(lmin, lmax)
            colornorm = lambda x: norm_func(np.floor((x - lmin) / (lmax - lmin) * nlevels) / nlevels * (lmax - lmin) + lmin)
            coll = PolyCollection(verts, facecolors=cmap(colornorm(data.flatten())), **kwargs)
        else:
            coll = PolyCollection(verts, array=data, cmap=cmap, **kwargs)
            if crange is not None:
                coll.set_clim(*crange)

        # Add background if specified
        if bgcolor:
            radius = 2 * ((90 - self.minlat) / (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder=0, facecolor=bgcolor)
            self.ax.add_artist(bg)

        # Add collection to axis
        return self.ax.add_collection(coll)


    def plotimg(self, lat, lt, image, corr=True, crange=None, bgcolor=None, **kwargs):
        """
        Displays an image in polar coordinates.

        Parameters
        ----------
        lat : array_like
            2D array of latitudes.
        lt : array_like
            2D array of local times.
        image : array_like
            2D array representing the image to be plotted.
        corr : bool, optional
            Controls how latitude and local time arrays are interpreted:
            - If True, lat and lt are assumed to represent cell center coordinates. The function calculates edge coordinates by 
              averaging adjacent center coordinates, leading to a plotted image of size (m-2) x (n-2).
            - If False, lat and lt are assumed to be edge coordinates. The function plots the image with a small offset, 
              resulting in a plotted image of size (m-1) x (n-1).
        crange : tuple or None, optional
            Lower and upper colormap limits.
        bgcolor : str or None, optional
            Background color inside the polar region.

        Returns
        -------
        matplotlib.collections.PolyCollection
            The added PolyCollection object.
        """

        # Validate input array shapes
        if lat.shape != lt.shape or lat.shape != image.shape:
            raise ValueError("lat, lt, and image must have the same shape.")

        x, y = self._latlt2xy(lat, lt)

        # Correct coordinates if required
        if corr:
            xe = (x[:-1, :-1] + x[1:, 1:]) / 2
            ye = (y[:-1, :-1] + y[1:, 1:]) / 2
            x, y = xe, ye
            data = image[1:-1, 1:-1]
        else:
            data = image[:-1, :-1]

        # Define vertices
        ll = (x[:-1, :-1].flatten(), y[:-1, :-1].flatten())
        lr = (x[:-1, 1:].flatten(), y[:-1, 1:].flatten())
        ul = (x[1:, :-1].flatten(), y[1:, :-1].flatten())
        ur = (x[1:, 1:].flatten(), y[1:, 1:].flatten())

        # Create vertex arrays
        vertsx = np.vstack((ll[0], lr[0], ur[0], ul[0])).T
        vertsy = np.vstack((ll[1], lr[1], ur[1], ul[1])).T

        # Filter vertices within unit circle
        warnings.filterwarnings("ignore", message="overflow encountered in square")
        valid_indices = np.any(vertsx**2 + vertsy**2 <= 1, axis=1)
        verts = np.dstack((vertsx[valid_indices], vertsy[valid_indices]))
        warnings.resetwarnings()

        # Set color map
        cmap = kwargs.pop('cmap', plt.cm.viridis)
        data = np.ma.array(data, mask=np.isnan(data))
        
        # Create PolyCollection
        coll = PolyCollection(verts, array=data.flatten()[valid_indices], cmap=cmap, edgecolors='none', **kwargs)
        if crange:
            coll.set_clim(*crange)

        # Add background if specified
        if bgcolor:
            radius = 2 * ((90 - self.minlat) / (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder=0, facecolor=bgcolor)
            self.ax.add_artist(bg)

        # Add collection to the plot
        return self.ax.add_collection(coll)


    def coastlines(self, time=None, mag=None, north=True, resolution='50m', **kwargs):
        """
        Plot coastlines in either geographic or magnetic coordinates.

        Parameters
        ----------
        time : datetime, optional
            If provided, converts longitude to local time for plotting.
        mag : apexpy.Apex, optional
            If provided, converts geographic coordinates to magnetic apex coordinates.
        north : bool, optional
            If False, plots coastlines for the southern hemisphere. Default is True.
        resolution : str, optional
            Resolution of the coastline data, either '50m' or '110m'. Default is '50m'.
        **kwargs : dict, optional
            Additional keyword arguments passed to matplotlib's LineCollection.

        Returns
        -------
        matplotlib.collections.LineCollection
            The LineCollection object of the plotted coastlines.

        Raises
        ------
        FileNotFoundError
            If coastline data files are not found.
        ValueError
            If an unsupported resolution is provided.
        """

        # Validate resolution
        if resolution not in ['50m', '110m']:
            raise ValueError("Unsupported resolution. Choose either '50m' or '110m'.")

        # Load coastline data
        try:
            coastlines = np.load(datapath + 'coastlines_' + resolution + '.npz')
        except FileNotFoundError:
            raise FileNotFoundError(f"Coastline data file for resolution '{resolution}' not found.")

        segments = []
        for cl in coastlines:
            lat, lon = coastlines[cl]

            # Convert to magnetic coordinates if provided
            if mag is not None:
                lat, lon = mag.geo2apex(lat, lon, mag.refh)

            # Adjust longitude for local time if time is given
            if time is not None:
                if mag is None:  # Geographic local time
                    sslat, sslon = subsol(time)  # Replace with your method to get subsolar lat, lon
                    londiff = (lon - sslon + 180) % 360 - 180
                    lon = (180. + londiff)
                else:  # Magnetic local time
                    lon = mag.mlon2mlt(lon, time) * 15

            # Adjust for southern hemisphere
            if not north:
                lat = -lat

            # Filter and prepare segments
            valid = lat > self.minlat
            lat[~valid], lon[~valid] = np.nan, np.nan
            x, y = self._latlt2xy(lat, lon / 15)
            segments.append(np.vstack((x, y)).T)

        # Create and add LineCollection
        collection = LineCollection(segments, **kwargs)
        return self.ax.add_collection(collection)


    def _latlt2xy(self, lat, lt, ignore_plot_limits=False):
        """
        Convert latitude and local time to x, y coordinates in a polar plot.

        Parameters
        ----------
        lat : array_like
            Array of latitudes in degrees. Latitudes are mirrored and absolute valued.
        lt : array_like
            Array of local times in hours. The local time is wrapped to fit within a 24-hour range.
        ignore_plot_limits : bool, optional
            If True, allows plotting outside the predefined plot limits. Useful for labels or special markers.
            Default is False.

        Raises
        ------
        ValueError
            Raised if the input arrays `lat` and `lt` do not have the same size.

        Returns
        -------
        x : np.ndarray
            The x-coordinate(s) in the polar plot.
        y : np.ndarray
            The y-coordinate(s) in the polar plot. Coordinates are masked based on plot limits unless 
            `ignore_plot_limits` is True.
        """
        np.seterr(invalid='ignore', divide='ignore')
        lt = np.array(lt) % 24
        lat = np.abs(np.array(lat))

        if lt.size != lat.size:
            raise ValueError('Latitude and local time arrays must be the same size')

        r = (90. - np.abs(lat)) / (90. - self.minlat)
        a = (lt - 6.) / 12. * np.pi

        x, y = r * np.cos(a), r * np.sin(a)

        if ignore_plot_limits:
            return x, y
        else:
            mask = np.ones_like(x)
            mask[(~eval(self.ltlims)) | (lat < self.minlat)] = np.nan
            return x * mask, y * mask


    def _xy2latlt(self, x, y, ignore_plot_limits=False):
        """
        Convert x, y coordinates in a polar plot to latitude and local time.

        Parameters
        ----------
        x : float or array_like
            X-coordinate(s) in the polar plot.
        y : float or array_like
            Y-coordinate(s) in the polar plot.
        ignore_plot_limits : bool, optional
            If True, allows conversion without considering the predefined plot limits. Default is False.

        Returns
        -------
        lat : np.ndarray
            The calculated latitude(s) in degrees.
        lt : np.ndarray
            The calculated local time(s) in hours. Coordinates are masked based on plot limits unless 
            `ignore_plot_limits` is True.
        """
        x, y = np.array(x, ndmin=1, dtype='float64'), np.array(y, ndmin=1, dtype='float64')

        lat = 90 - np.sqrt(x**2 + y**2) * (90. - self.minlat)
        lt = np.arctan2(y, x) * 12 / np.pi + 6
        lt = lt % 24

        if ignore_plot_limits:
            return lat, lt
        else:
            mask = np.ones_like(lt)
            mask[(~eval(self.ltlims)) | (lat < self.minlat)] = np.nan
            return lat * mask, lt * mask


    def _northEastToCartesian(self, north, east, lt):
        """
        Convert northward and eastward components to Cartesian coordinates.

        This function is used for converting directional data (e.g., wind or current vectors) from
        northward and eastward components to Cartesian coordinates in a polar plot.

        Parameters
        ----------
        north : array_like
            Northward component(s) of the vector.
        east : array_like
            Eastward component(s) of the vector.
        lt : array_like
            Local time(s) corresponding to each vector, used for the directional conversion.

        Returns
        -------
        np.ndarray
            Cartesian coordinates (x, y) of the input vectors in the polar plot.
        """
        a = (lt - 6) / 12 * np.pi  # convert LT to angle with x axis (pointing from pole towards dawn)

        x1 = np.array([-north * np.cos(a), -north * np.sin(a)])  # arrow direction towards origin (northward)
        x2 = np.array([-east * np.sin(a), east * np.cos(a)])     # arrow direction eastward

        return x1 + x2


    def _create_coordinate_formatter(self, lt_label='lt', lat_label='lat'):
        """
        Creates a formatter function for displaying both Cartesian and transformed coordinates in a plot.

        This method is useful for enhancing interactivity in plots, especially when working with
        data in different coordinate systems. It shows both the original Cartesian coordinates and
        their corresponding transformed coordinates (like local time and latitude).

        Parameters
        ----------
        lt_label : str, optional
            Label for the transformed x-coordinate (default is 'lt' for local time).
        lat_label : str, optional
            Label for the transformed y-coordinate (default is 'lat' for latitude).

        Returns
        -------
        function
            A function that takes x, y coordinates and returns a formatted string with both
            the original and transformed coordinates.

        """

        def format_coord(x, y):
            """
            Formatter function for coordinates.

            Parameters
            ----------
            x : float
                The x-coordinate in the plot's data space.
            y : float
                The y-coordinate in the plot's data space.

            Returns
            -------
            str
                A string that combines both original and transformed coordinates.
            """
            transformed_coord = self._xy2latlt(x, y)
            lat, lt = transformed_coord[::-1]
            lat = lat[0] if isinstance(lat, np.ndarray) else lat
            lt = lt[0] if isinstance(lt, np.ndarray) else lt
            string_original = f'x={x:.2f}, y={y:.2f}'
            string_transformed = f'{lt_label}={lt:.2f}, {lat_label}={lat:.2f}'

            return f"{string_original.ljust(20)}{string_transformed}"

        return format_coord




# Helper functions
# ----------------
def date_to_doy(month, day, leapyear = False):
    """ return day of year (DOY) at given month, day
    """

    # check that month in [1, 12]
    if month < 1 or month > 12:
        raise ValueError('month not in [1, 12]')

    # check if day < 1
    if day < 1:
        raise ValueError('date2doy: day must not be less than 1')

    # check if day exceeds days in months
    days_in_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    if day > days_in_month[month]:
        if not (leapyear & (month == 2) & (day == 29)):
            raise ValueError('date2doy: day must not exceed number of days in month')

    cumdaysmonth = np.cumsum(days_in_month[:-1])

    # day of year minus possibly leap day:
    doy = cumdaysmonth[month - 1] + day

    # add 1 if leapyear and month >= 3:
    return doy + leapyear if month >= 3 else doy


def is_leapyear(year):
    """ Check for leapyear

    """
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def subsol(time):
    """
    calculate subsolar point at given datetime(s)

    returns:
      subsol_lat  -- latitude of the subsolar point
      subsol_lon  -- longiutde of the subsolar point

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac,
    results are good to at least 0.01 degree latitude and 0.025 degree
    longitude between years 1950 and 2050.  Accuracy for other years
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.

    """

    year = time.year
    # day of year:
    doy  = date_to_doy(time.month, time.day, is_leapyear(year))
    # seconds since start of day:
    ut   = time.hour * 60.**2 + time.minute*60. + time.second

    yr = year - 2000

    if year >= 2100 or year <= 1600:
        raise ValueError('subsol.py: subsol invalid after 2100 and before 1600')

    nleap = np.floor((year-1601)/4.)
    nleap = nleap - 99

    # exception for years <= 1900:
    ncent = np.floor((year-1601)/100.)
    ncent = 3 - ncent
    if year <= 1900: nleap = nleap + ncent

    l0 = -79.549 + (-.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400. - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = .9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = .9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*np.pi/180.

    # Ecliptic longitude:
    lmbda = l + 1.915*np.sin(grad) + .020*np.sin(2.*grad)
    lmrad = lmbda*np.pi/180.
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.e-7*n
    epsrad  = epsilon*np.pi/180.

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad)*sinlm, np.cos(lmrad)) * 180./np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad)*sinlm) * 180./np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg/360.)
    etdeg = etdeg - 360.*nrot

    # Apparent time (degrees):
    aptime = ut/240. + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180. - aptime
    nrot = np.round(sbsllon/360.)
    sbsllon = sbsllon - 360.*nrot

    return sbsllat, sbsllon
