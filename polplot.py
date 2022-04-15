from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import apexpy
from matplotlib import rc
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
from pysymmetry.sunlight import terminator


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class Polarplot(object):
    def __init__(self, ax, minlat = 50, plotgrid = True, sector = 'all', **kwargs):
        """ pax = Polarsubplot(axis, minlat = 50, plotgrid = True, **kwargs)

            **kwargs are the plot parameters for the grid

            this is a class which handles plotting in polar coordinates, specifically
            an MLT/MLAT grid or similar

            Example:
            --------
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            pax = Polarsubplot(ax)
            pax.MEMBERFUNCTION()
            plt.show()


            where memberfunctions include:
            plotgrid()                           - called by __init__
            plot(mlat, mlt, **kwargs)            - works like plt.plot
            write(mlat, mlt, text, **kwargs)     - works like plt.text
            scatter(mlat, mlt, **kwargs)         - works like plt.scatter
            writeMLTlabels(mlat = 48, **kwargs)  - writes MLT at given mlat - **kwargs to plt.text
            plotarrows(mlats, mlts, north, east) - works like plt.arrow (accepts **kwargs too)
            contour(mlat, mlt, f)                - works like plt.contour
            contourf(mlat, mlt, f)               - works like plt.contourf

        """
        self.minlat = minlat # the lower latitude boundary of the plot
        self.ax = ax
        self.ax.axis('equal')
        self.minlat = minlat

        self.sector = sector

        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = .5

        if 'color' not in kwargs.keys():
            kwargs['color'] = 'lightgrey'

        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'


        if sector == 'all':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-1.1, 1.1)
        if sector == 'dusk':
            self.ax.set_xlim(-1.1, 0.1)
            self.ax.set_ylim(-1.1, 1.1)
        if sector == 'dawn':
            self.ax.set_xlim(-0.1, 1.1)
            self.ax.set_ylim(-1.1, 1.1)
        if sector == 'night':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-1.1, 0.1)
        if sector == 'day':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_axis_off()

        self.ax.format_coord= self.make_format()
        if plotgrid:
            self.plotgrid(**kwargs)

    def plot(self, mlat, mlt, **kwargs):
        """ plot curve based on mlat, mlt. Calls matplotlib.plot, so any keywords accepted by this is also accepted here """

        x, y = self._mltMlatToXY(mlt, mlat)
        return self.ax.plot(x, y, **kwargs)

    def write(self, mlat, mlt, text, **kwargs):
        """ write text on specified mlat, mlt. **kwargs go to matplotlib.pyplot.text"""
        x, y = self._mltMlatToXY(mlt, mlat)

        self.ax.text(x, y, text, **kwargs)

    def scatter(self, mlat, mlt, **kwargs):
        """ scatterplot on the polar grid. **kwargs go to matplotlib.pyplot.scatter """

        x, y = self._mltMlatToXY(mlt, mlat)
        c = self.ax.scatter(x, y, **kwargs)
        return c

    def plotgrid(self, **kwargs):
        """ plot mlt, mlat-grid on self.ax """

        if self.sector == 'all':
            self.ax.plot([-1, 1], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [-1, 1], **kwargs)
            angles = np.linspace(0, 2*np.pi, 360)
        if self.sector == 'dawn':
            self.ax.plot([0, 1], [0 , 0], **kwargs)
            self.ax.plot([0, 0], [-1, 1], **kwargs)
            angles = np.linspace(-np.pi/2, np.pi/2, 180)
        if self.sector == 'dusk':
            self.ax.plot([-1, 0], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [-1, 1], **kwargs)
            angles = np.linspace( np.pi/2, 3*np.pi/2, 180)
        if self.sector == 'night':
            self.ax.plot([-1, 1], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [-1, 0], **kwargs)
            angles = np.linspace(np.pi, 2*np.pi, 180)
        if self.sector == 'day':
            self.ax.plot([-1, 1], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [ 0, 1], **kwargs)
            angles = np.linspace(0, np.pi, 180)


        latgrid = (90 - np.r_[self.minlat:90:10])/(90. - self.minlat)

        for lat in latgrid:
            self.ax.plot(lat*np.cos(angles), lat*np.sin(angles), **kwargs)

    def writeMLTlabels(self, mlat = None, degrees = False, **kwargs):
        """ write MLT labels at given latitude (default 48)
            if degrees is true, the longitude will be written instead of hour (with 0 at midnight)
        """
        if mlat is None:
            mlat = self.minlat - 2

        if degrees:
            if self.sector in ['all', 'night', 'dawn', 'dusk']:
                self.write(mlat, 0,    '0$^\circ$', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dawn', 'day']:
                self.write(mlat, 6,   '90$^\circ$', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs)
            if self.sector in ['all', 'dusk', 'dawn', 'day']:
                self.write(mlat, 12, '180$^\circ$', verticalalignment = 'bottom', horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dusk', 'day']:
                self.write(mlat, 18, '-90$^\circ$', verticalalignment = 'center', horizontalalignment = 'right' , **kwargs)
        else:
            if self.sector in ['all', 'night', 'dawn', 'dusk']:
                self.write(mlat, 0, '00', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dawn', 'day']:
                self.write(mlat, 6, '06', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs)
            if self.sector in ['all', 'dusk', 'dawn', 'day']:
                self.write(mlat, 12, '12', verticalalignment = 'bottom', horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dusk', 'day']:
                self.write(mlat, 18, '18', verticalalignment = 'center', horizontalalignment = 'right' , **kwargs)

    def plotpins(self, mlats, mlts, north, east, rotation = 0, SCALE = None, size = 10, unit = '', colors = 'black', markercolor = 'black', marker = 'o', markersize = 20, **kwargs):
        """ like plotarrows, only it's not arrows but a dot with a line pointing in the arrow direction

            kwargs go to ax.plot

            the markers at each pin can be modified by the following keywords, that go to ax.scatter:
            marker (default 'o')
            markersize (defult 20 - size in points^2)
            markercolor (default black)

        """

        mlts = mlts.flatten()
        mlats = mlats.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))

        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                self.ax.plot([0.9, 1], [0.95, 0.95], color = colors, linestyle = '-', linewidth = 2)
                self.ax.text(0.9, 0.95, ('%.1f ' + unit) % SCALE, horizontalalignment = 'right', verticalalignment = 'center', size = size)

            scale = 0.1/SCALE

        segments = []
        for i in range(len(mlats)):

            mlt = mlts[i]
            mlat = mlats[i]

            x, y = self._mltMlatToXY(mlt, mlat)
            dx, dy = R.dot(self._northEastToCartesian(north[i], east[i], mlt).reshape((2, 1))).flatten()

            segments.append([(x, y), (x + dx*scale, y + dy*scale)])

                
        self.ax.add_collection(LineCollection(segments, colors = colors, **kwargs))

        if markersize != 0:
            self.scatter(mlats, mlts, marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)


    def quiver(self, mlat, mlt, north, east, rotation = 0, qkeyproperties = None, **kwargs):
        """ wrapper for matplotlib's quiver function, juts for mlat/mlt coordinates and
            east/north components. Rotation specifies a rotation angle for the vectors,
            which could be convenient if you want to rotate magnetic field measurements
            to an equivalent current direction

            rotation in radians

            qkeyproperties: set to dict to pass to matplotlib quiverkey. The dict must contain
            'label': the label written next to arrow
            'U': length of the arrow key in same unit as north/east components

        """

        mlt = mlt.flatten()
        mlat = mlat.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))

        x, y = self._mltMlatToXY(mlt, mlat)
        dx, dy = R.dot(self._northEastToCartesian(north, east, mlt))

        q = self.ax.quiver(x, y, dx, dy, **kwargs)

        if qkeyproperties is not None:
            qkeykwargs = {'X':.8, 'Y':.9, 'labelpos':'E'}
            qkeykwargs.update(qkeyproperties)

            self.ax.quiverkey(q, **qkeykwargs)

        return q



    def plottrack(self, mlats, mlts, north, east, rotation = 0, SCALE = None, size = 10, unit = '', color = 'black', markercolor = 'black', marker = 'o', markersize = 20, **kwargs):
        """ like plotpins, only it's not a line pointing in the arrow direction but a dot at the tip of the arrow

            kwargs go to ax.plot

            the markers at each pin can be modified by the following keywords, that go to ax.scatter:
            marker (default 'o')
            markersize (defult 20 - size in points^2)
            markercolor (default black)

        """

        mlts = mlts.flatten()
        mlats = mlats.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))

        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                self.ax.plot([0.9, 1], [0.95, 0.95], color = color, linestyle = '-', linewidth = 2)
                self.ax.text(0.9, 0.95, ('%.1f ' + unit) % SCALE, horizontalalignment = 'right', verticalalignment = 'center', size = size)

            #self.ax.set_xlim(-1.1, 1.1)
            #self.ax.set_ylim(-1.1, 1.1)
            scale = 0.1/SCALE

        xs, ys, dxs, dys = [],[],[],[]
        for i in range(len(mlats)):#mlt, mlat in zip(mlts, mlats):#mlatenumerate(means.index):

            mlt = mlts[i]
            mlat = mlats[i]

            x, y = self._mltMlatToXY(mlt, mlat)
            dx, dy = R.dot(self._northEastToCartesian(north[i], east[i], mlt).reshape((2, 1))).flatten()
            xs.append(x)
            ys.append(y)
            dxs.append(dx)
            dys.append(dy)

        xs, ys, dxs, dys = np.array(xs),np.array(ys),np.array(dxs)*scale,np.array(dys)*scale

        self.ax.scatter(xs+dxs,ys+dys,marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)


    def contour(self, mlat, mlt, f, **kwargs):
        """ plot contour on grid, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xea, yea = self._mltMlatToXY(mlt.flatten(), mlat.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contour(xx, yy, gridf, **kwargs)


    def contourf(self, mlat, mlt, f, **kwargs):
        """ plot contour on grid, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xea, yea = self._mltMlatToXY(mlt.flatten(), mlat.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contourf(xx, yy, gridf, **kwargs)

    def fill(self, mlat, mlt, **kwargs):
        """ Fill polygon defined in mlat/mlt, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xx, yy = self._mltMlatToXY(mlt.flatten(), mlat.flatten())


        # plot
        return self.ax.fill(xx, yy, **kwargs)


    def plot_terminator(self, position, sza = 90, north = True, shadecolor = None, terminatorcolor = 'black', terminatorlinewidth = 1, shadelinewidth = 0, **kwargs):
        """ shade the area antisunward of the terminator

            position            -- either a scalar or a datetime object
                                  if scalar: interpreted as the signed magnetic colatitude of the terminator, positive on dayside, negative on night side
                                  if datetime: terminator is calculated, and converted to magnetic apex coordinates (refh = 0, height = 0)
            sza                 -- sza to locate terminator, used if position is datetime
            north               -- True if northern hemisphere, south if not (only matters if position is datetime)
            shadecolor               -- color of the shaded area - default None
            shadelinewidth           -- width of the contour surrounding the shaded area - default 0 (invisible)
            terminatorcolor     -- color of the terminator - default black
            terminatorlinewidth -- width of the terminator contour
            **kwargs            -- passed to Polygon


            EXAMPLE: to only plot the terminator (no shade):
            plot_terminator(position, color = 'white') <- sets the shade to white (or something different if the plot background is different)


            useful extensions:
            - height dependence...
        """

        if np.isscalar(position): # set the terminator as a horizontal bar
            if position >= 0: # dayside
                position = np.min([90 - self.minlat, position])
                x0, y0 = self._mltMlatToXY(12, 90 - np.abs(position))
            else: #nightside
                x0, y0 = self._mltMlatToXY(24, 90 - np.abs(position))

            xr = np.sqrt(1 - y0**2)
            xl = -xr
            lat, left_mlt  = self._XYtomltMlat(xl, y0)
            lat, right_mlt = self._XYtomltMlat(xr, y0)
            if position > -(90 - self.minlat):
                right_mlt += 24

            x = np.array([xl, xr])
            y = np.array([y0, y0])

        else: # calculate the terminator trajectory
            a = apexpy.Apex(date = position)

            t_glat, t_glon = terminator(position, sza = sza, resolution = 3600)
            t_mlat, t_mlon = a.geo2apex(t_glat, t_glon, 0)
            t_mlt          = a.mlon2mlt(t_mlon, position)

            # limit contour to correct hemisphere:
            iii = (t_mlat >= self.minlat) if north else (t_mlat <= -self.minlat)
            if len(iii) == 0:
                return 0 # terminator is outside plot
            t_mlat = t_mlat[iii]
            t_mlt = t_mlt[iii]

            x, y = self._mltMlatToXY(t_mlt, t_mlat)

            # find the points which are closest to minlat, and use these as edgepoints for the rest of the contour:
            xmin = np.argmin(x)
            xmax = np.argmax(x)
            left_mlt = t_mlt[xmin]
            right_mlt = t_mlt[xmax]
            if right_mlt < left_mlt:
                right_mlt += 24

        mlat_b = np.full(100, self.minlat)
        mlt_b  = np.linspace(left_mlt, right_mlt, 100)
        xb, yb = self._mltMlatToXY(mlt_b, mlat_b)

        # sort x and y to be in ascending order
        iii = np.argsort(x)
        x = x[iii[::-1]]
        y = y[iii[::-1]]

        if terminatorcolor is not None:
            self.ax.plot(x, y, color = terminatorcolor, linewidth = terminatorlinewidth)
        if shadecolor is not None:
            kwargs['color'] = shadecolor
            kwargs['linewidth'] = shadelinewidth
            shade = Polygon(np.vstack((np.hstack((x, xb)), np.hstack((y, yb)))).T, closed = True, **kwargs)
            self.ax.add_patch(shade)



    def filled_cells(self, mlat, mlt, mlatres, mltres, data, resolution = 10, crange = None, levels = None, bgcolor = None, verbose = False, **kwargs):
        """ specify a set of cells in mlat and mlt, along with a data array,
            and make a color plot of the cells

        """

        if not all([len(mlat) == len(q) for q in [mlt,mltres,data]]):
            print("WARNING: Input arrays to filled_cells of unequal length! Your plot is probably incorrect")

        mlat, mlt, mlatres, mltres, data = map(np.ravel, [mlat, mlt, mlatres, mltres, data])

        if verbose:
            print(mlt.shape, mlat.shape, mlatres.shape, mltres.shape)

        la = np.vstack(((mlt - 6) / 12. * np.pi + i * mltres / (resolution - 1.) / 12. * np.pi for i in range(resolution))).T
        if verbose:
            print (la.shape)
        ua = la[:, ::-1]

        vertslo = np.dstack(((90 - mlat          )[:, np.newaxis] / (90. - self.minlat) * np.cos(la),
                             (90 - mlat          )[:, np.newaxis] / (90. - self.minlat) * np.sin(la)))
        vertshi = np.dstack(((90 - mlat - mlatres)[:, np.newaxis] / (90. - self.minlat) * np.cos(ua),
                             (90 - mlat - mlatres)[:, np.newaxis] / (90. - self.minlat) * np.sin(ua)))
        verts = np.concatenate((vertslo, vertshi), axis = 1)

        if verbose:
            print( verts.shape, vertslo.shape, vertshi.shape)


        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
            kwargs.pop('cmap')
        else:
            cmap = plt.cm.viridis

        if levels is not None:
            # set up a function that maps data values to color levels:
            nlevels = len(levels)
            lmin, lmax = levels.min(), levels.max()
            self.colornorm = lambda x: plt.Normalize(lmin, lmax)(np.floor((x - lmin) / (lmax - lmin) * (nlevels - 1)) / (nlevels - 1) * (lmax - lmin) + lmin)
            coll = PolyCollection(verts, facecolors = cmap(self.colornorm(data.flatten())), **kwargs)

        else:
            coll = PolyCollection(verts, array = data, cmap = cmap, **kwargs)
            if crange is not None:
                coll.set_clim(crange[0], crange[1])



        if bgcolor != None:
            radius = 2*((90 - self.minlat)/ (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder = 0, facecolor = bgcolor)
            self.ax.add_artist(bg)


        self.ax.add_collection(coll)

    def showFUVimage(self,img,inImg='image', crange = None, bgcolor = None, **kwargs):
        """ show FUV image """

        mlat = img['mlat'].values.copy()
        mlt = img['mlt'].values.copy()
        image = img[inImg].values.copy()

        ll = self._mltMlatToXY(mlt[1:,  :-1].flatten(), mlat[1:,  :-1].flatten())
        lr = self._mltMlatToXY(mlt[1:,   1:].flatten(), mlat[1:,   1:].flatten())
        ul = self._mltMlatToXY(mlt[:-1, :-1].flatten(), mlat[:-1, :-1].flatten())
        ur = self._mltMlatToXY(mlt[:-1,  1:].flatten(), mlat[:-1,  1:].flatten())

        vertsx = np.vstack((ll[0], lr[0], ur[0], ul[0])).T
        vertsy = np.vstack((ll[1], lr[1], ur[1], ul[1])).T

        iii = np.where(vertsx**2 + vertsy**2 <=1, True, False)
        iii = np.any(iii, axis = 1).nonzero()[0]

        vertsx = vertsx[iii]
        vertsy = vertsy[iii]

        verts = np.dstack((vertsx, vertsy))

        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
            kwargs.pop('cmap')
        else:
            cmap = plt.cm.viridis

        coll = PolyCollection(verts, array=image[1:, :-1].flatten()[iii], cmap = cmap, edgecolors='none', **kwargs)
        if crange is not None:
            coll.set_clim(crange[0], crange[1])

        if bgcolor is not None:
            radius = 2*((90 - self.minlat)/ (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder = 0, facecolor = bgcolor)
            self.ax.add_artist(bg)

        self.ax.add_collection(coll)


    def ampereplot(self, keys, data, contour = False, crange = None, nlevels = 100, cbar = False, **kwargs):
        """ plot ampere data. keys is the columns of the ampere dataframes.
            data is an array of equal size, that will be plotted in grids of 1 degree by 1 hour size

            set cbar True to draw a color bar

            2021-7-1: Kalle: Added this function from an old version of polarsublot. It worked in 2016, but I have
                             no clue if it still works...
        """
        data = np.array(data)

        # MAKE VERTICES
        colats, mlts = np.array(list(keys)).T
        mlats = np.abs(90. - colats)

        mltres, mlatres = 1, 1

        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
            kwargs.pop('cmap')
        else:
            cmap = plt.cm.RdBu_r

        if not contour: # plot so that the grid cells are seen
            mlatl, mltl = mlats, mlts
            mlatu = mlatl + mlatres
            mltr  = mltl + mltres
            verts = np.empty((len(keys), 20, 2))

            for i in range(len(mlts)):
                # convert corners to cartesian (mlat 90 at origin, mlat 0 at distance 1 from origin)
                la = np.linspace((mltl[i]-6)/12.*np.pi, (mltr[i]-6)/12.*np.pi, 10)
                ua = np.linspace((mltl[i]-6)/12.*np.pi, (mltr[i]-6)/12.*np.pi, 10)[::-1]
                verts[i, :10, :]  = (90 - mlatl[i])/(90 - self.minlat) * np.array([np.cos(la), np.sin(la)]).T
                verts[i,  10:, :] = (90 - mlatu[i])/(90 - self.minlat) * np.array([np.cos(ua), np.sin(ua)]).T

            # the sign of the data is changed to make the color scale consistent with AMPERE summary plots
            coll = PolyCollection(verts[np.isfinite(data)], array = data[np.isfinite(data)], cmap = cmap, edgecolors='none', **kwargs)
            if crange is not None:
                coll.set_clim(crange[0], crange[1])
            self.ax.add_collection(coll)

        else: # plot a smooth contour
            coll = self.contourf(mlats, mlts, data, cmap = cmap, levels = np.linspace(crange[0], crange[1], nlevels), extend = 'both', **kwargs)






        if cbar: 

            clim = coll.get_clim()
            axpos = self.ax.get_position().bounds # position of current plot
            cbax = self.ax.figure.add_subplot(111)
            cbax.set_position([axpos[0] + axpos[2], axpos[1], 0.02, axpos[3]])
            #cbax.get_xaxis().set_visible(False)
            #cbax.get_yaxis().set_visible(False)
            RESOLUTION = 100
            colors = np.linspace(clim[0], clim[1], RESOLUTION + 1)
            colors -= colors.min()
            cbax.set_ylim(colors.min(), colors.max())
            verts = np.empty((RESOLUTION, 4, 2))
            for i in range(RESOLUTION):
                verts[i, :, :] = np.array([[0, 0, 1, 1], [colors[i], colors[i+1], colors[i+1], colors[i]]]).T
            
            cbarcoll = PolyCollection(verts, array = np.linspace(clim[0], clim[1], RESOLUTION), cmap = cmap, edgecolors='none')
            cbarcoll.set_clim(clim)
            cbax.add_collection(cbarcoll)

            cbax.set_xlabel(r'$\mu$A/m$^2$', labelpad = 1)
            cbax.set_yticks([colors.min(), 0 - clim[0], colors.max()])
            cbax.set_yticklabels([str(clim[0]), '0', str(clim[1])])
            cbax.set_xlim(0, 1)
            cbax.set_xticks([0, 1])
            cbax.set_xticklabels(['', ''])





        return coll


    def _mltMlatToXY(self, mlt, mlat):
        mlt = np.asarray(mlt)
        mlat = np.asarray(mlat)
        r = (90. - np.abs(mlat))/(90. - self.minlat)
        a = (mlt - 6.)/12.*np.pi

        return r*np.cos(a), r*np.sin(a)

    def _XYtomltMlat(self, x, y):
        """ convert x, y to mlt, mlat, where x**2 + y**2 = 1 corresponds to self.minlat """
        x, y = np.array(x, ndmin = 1), np.array(y, ndmin = 1) # conver to array to allow item assignment

        lat = 90 - np.sqrt(x**2 + y**2)*(90. - self.minlat)
        mlt = np.arctan2(y, x)*12/np.pi + 6
        mlt[mlt < 0] += 24
        mlt[mlt > 24] -= 24

        return lat.squeeze()[()], mlt.squeeze()[()]


    def _northEastToCartesian(self, north, east, mlt):
        a = (mlt - 6)/12*np.pi # convert MLT to angle with x axis (pointing from pole towards dawn)

        x1 = np.array([-north*np.cos(a), -north*np.sin(a)]) # arrow direction towards origin (northward)
        x2 = np.array([-east*np.sin(a),  east*np.cos(a)])   # arrow direction eastward

        return x1 + x2


    def make_format(current):
    # current and other are axes
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current._XYtomltMlat(x, y)[::-1]
            ax_coord= (float(i) for i in display_coord)
            string_original= 'x={:.2f}, y={:.2f}'.format(x, y)
            string_magnetic= 'mlt={:.2f}, mlat={:.2f}'.format(*ax_coord)
            return (string_original.ljust(20)+string_magnetic)
            # return ('Left: {:<40}    Right: {:<}'
            #         .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
        return format_coord
