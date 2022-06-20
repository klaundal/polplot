from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib import rc
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection

# Added by AØH 17/06/2022 for Lompe, should be changed when merged with Lompe 
from lompe.utils.sunlight import terminator


# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Verdana']})
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
        

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.
        minlat : TYPE, optional
            DESCRIPTION. The default is 50.
        plotgrid : TYPE, optional
            DESCRIPTION. The default is True.
        sector : string, optional
            Used to generate portions of polar plot.
            Can either use one of the folling strings: 'All', 'dusk',
            'dawn', 'night', 'day'. Or can be defined using numbers for example '18-6' will produce
            the samem as night while '6-18' will produce the same as day. The default is 'all'.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.minlat = minlat # the lower latitude boundary of the plot
        self.ax = ax
        self.ax.axis('equal')
        self.in_axes= ax.in_axes
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
            self.mltlims= 'mlt>-9.999e3'
        elif sector == 'dusk':
            self.ax.set_xlim(-1.1, 0.1)
            self.ax.set_ylim(-1.1, 1.1)
            self.mltlims= '(mlt>=12) | (mlt<=0)'
        elif sector == 'dawn':
            self.ax.set_xlim(-0.1, 1.1)
            self.ax.set_ylim(-1.1, 1.1)
            self.mltlims= '(mlt>=24) | (mlt<=12)'
        elif sector == 'night':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-1.1, 0.1)
            self.mltlims= '(mlt>=18) | (mlt<=6)'
        elif sector == 'day':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-0.1, 1.1)
            self.mltlims= '(mlt>=6) & (mlt<=18)'
        else:
            sector= [float(s) for s in sector.split('-')]
            if sector[0]> sector[-1]:
                self.mltlims= f'(mlt>={sector[0]})|(mlt<={sector[-1]})'
                mlts= [sector[0]]+list(range(int(np.ceil(sector[0])), 24))+ list(range(0, int(np.floor(sector[-1]))))+[sector[-1]]
                xlims, ylims= self._mltMlatToXY(mlts, [self.minlat]*len(mlts))
                xlims, ylims= self.ax.set_xlim(min([xlims.min()-.1, -0.1]), max([xlims.max()+.1, 0.1])), self.ax.set_ylim(min([ylims.min()-.1, -0.1]), max([ylims.max()+.1, 0.1]))
            else:
                self.mltlims= f'(mlt>={sector[0]})&(mlt<={sector[-1]})'
                mlts= [sector[0]]+ list(range(int(np.ceil(sector[0])), int(np.floor(sector[-1]))))+ [sector[-1]]
                xlims, ylims= self._mltMlatToXY(mlts, [self.minlat]*len(mlts))
                xlims, ylims= self.ax.set_xlim(min([xlims.min()-.1, -0.1]), max([xlims.max()+.1, 0.1])), self.ax.set_ylim(min([ylims.min()-.1, -0.1]), max([ylims.max()+.1, 0.1]))

        self.ax.set_axis_off()

        self.ax.format_coord= self.make_format()
        if plotgrid:
            self.plotgrid(**kwargs)

    def plot(self, mlat, mlt, **kwargs):
        """ plot curve based on mlat, mlt. Calls matplotlib.plot, so any keywords accepted by this is also accepted here """

        x, y = self._mltMlatToXY(mlt, mlat)
        return self.ax.plot(x, y, **kwargs)
    
    # added for consistency with pyplot
    def text(self, mlat, mlt, text, bypass=False, **kwargs):
        """ calls write() - write text on specified mlat, mlt.

        """
        self.write(mlat, mlt, text, bypass, **kwargs)
    
    def write(self, mlat, mlt, text, bypass=False, **kwargs):
        """
        write text on specified mlat, mlt. **kwargs go to matplotlib.pyplot.text

        Parameters
        ----------
        mlat : TYPE
            DESCRIPTION.
        mlt : TYPE
            DESCRIPTION.
        text : TYPE
            DESCRIPTION.
        bypass : boolean, optional
            When True the conversion will ignore the limits of the plot allowing plotting outside
            the limits of the subplot, for example this is used in writeMLTlabels. The default is False.

        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        x, y = self._mltMlatToXY(mlt, mlat, bypass=bypass)

        return self.ax.text(x, y, text, **kwargs)

    def scatter(self, mlat, mlt, **kwargs):
        """ scatterplot on the polar grid. **kwargs go to matplotlib.pyplot.scatter """

        x, y = self._mltMlatToXY(mlt, mlat)
        c = self.ax.scatter(x, y, **kwargs)
        return c

    def plotgrid(self, labels=False, **kwargs):
        """ plot mlt, mlat-grid on self.ax """
        mlat, mlt= self._XYtomltMlat(np.linspace(-1, 1, 101), [0]*101)
        mlat= mlat[np.isfinite(mlt)]
        mlt= mlt[np.isfinite(mlt)]
        if len(mlat) and len(mlt):
       	 self.plot(mlat[[0, -1]], mlt[[0, -1]], **kwargs)
        mlat, mlt= self._XYtomltMlat([0]*100, np.linspace(-1, 1, 100))
        mlat= mlat[np.isfinite(mlt)]
        mlt= mlt[np.isfinite(mlt)]
        if len(mlat) and len(mlt):
    	    self.plot(mlat[[0, -1]], mlt[[0, -1]], **kwargs)
        angles = np.linspace(0, 2*np.pi, 360)

        latgrid = (90 - np.r_[self.minlat:90:10])/(90. - self.minlat)

        for lat in latgrid:
            mlat, mlt= self._XYtomltMlat(lat*np.cos(angles), lat*np.sin(angles))
            self.plot(mlat, mlt, **kwargs)
        
        # add MLAT and MLT labels to axis
        if labels:
            self.writeMLATlabels()
            self.writeMLTlabels()

    def writeMLTlabels(self, mlat = None, degrees = False, **kwargs):
        """ write MLT labels at given latitude (default 48)
            if degrees is true, the longitude will be written instead of hour (with 0 at midnight)
        """
        if mlat is None:
            mlat = self.minlat - 2
        labels=[]
        if degrees:
            if self.sector in ['all', 'night', 'dawn', 'dusk']:
                labels.append(self.write(mlat, 0,    '0$^\circ$', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs))
            if self.sector in ['all', 'night', 'dawn', 'day']:
                labels.append(self.write(mlat, 6,   '90$^\circ$', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs))
            if self.sector in ['all', 'dusk', 'dawn', 'day']:
                labels.append(self.write(mlat, 12, '180$^\circ$', verticalalignment = 'bottom', horizontalalignment = 'center', **kwargs))
            if self.sector in ['all', 'night', 'dusk', 'day']:
                labels.append(self.write(mlat, 18, '-90$^\circ$', verticalalignment = 'center', horizontalalignment = 'right' , **kwargs))
        else:
            mlt=np.array([0, 24])
            if any(eval(self.mltlims)):
                labels.append(self.write(mlat, 0, '00', verticalalignment = 'top'    , horizontalalignment = 'center', bypass=True, **kwargs))
            mlt=6
            if eval(self.mltlims):
                labels.append(self.write(mlat, 6, '06', verticalalignment = 'center' , horizontalalignment = 'left'  , bypass=True, **kwargs))
            mlt=12
            if eval(self.mltlims):
                labels.append(self.write(mlat, 12, '12', verticalalignment = 'bottom', horizontalalignment = 'center', bypass=True, **kwargs))
            mlt=18
            if eval(self.mltlims):
                labels.append(self.write(mlat, 18, '18', verticalalignment = 'center', horizontalalignment = 'right' , bypass=True, **kwargs))

            return labels
    
    # added by AØH 20/06/2022 to plot magnetic latitude labels
    def writeMLATlabels(self, mlt = None, **kwargs):
        """ write magnetic latitude labels """
        if mlt == None:
            mlt = 3
        if kwargs is not None:
            mlatkwargs = {'rotation':45, 'color':'lightgrey', 'backgroundcolor':'white', 'zorder':2, 'alpha':1.}
            mlatkwargs.update(kwargs)
        labels = []
        for mlat in np.r_[self.minlat:81:10]:
            labels.append(self.write(mlat, mlt, str(mlat)+'$^{\circ}$', bypass = False, **mlatkwargs))
        
        return labels

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
        if 'key_coords' in kwargs:
            key_x, key_y= kwargs['key_coords']
            kwargs.pop('key_coords', None)
        else:
            key_x, key_y= 0.9, 0.95
        key=[]
        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                key.append(self.ax.plot([key_x, key_x+.1], [key_y]*2, color = color, linestyle = '-', linewidth = 2))
                key.append(self.ax.text(key_x, key_y, ('%.1f ' + unit) % SCALE, horizontalalignment = 'right', verticalalignment = 'center', size = size))

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
        if len(key)!=0:
            return self.ax.scatter(xs+dxs,ys+dys,marker = marker, c = markercolor, s = markersize, edgecolors = markercolor), key
        else:
            return self.ax.scatter(xs+dxs,ys+dys,marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)


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

    # TODO: This needs work, but check out utils.terminator() first! (AØH 17/06/2022)
    # def plot_terminator(self, position, sza = 90, north = True, shadecolor = None, terminatorcolor = 'black', terminatorlinewidth = 1, shadelinewidth = 0, **kwargs):
    #      """ shade the area antisunward of the terminator
    
    #           position            -- either a scalar or a datetime object
    #                                if scalar: interpreted as the signed magnetic colatitude of the terminator, positive on dayside, negative on night side
    #                                if datetime: terminator is calculated, and converted to magnetic apex coordinates (refh = 0, height = 0)
    #          sza                 -- sza to locate terminator, used if position is datetime
    #          north               -- True if northern hemisphere, south if not (only matters if position is datetime)
    #          shadecolor               -- color of the shaded area - default None
    #          shadelinewidth           -- width of the contour surrounding the shaded area - default 0 (invisible)
    #          terminatorcolor     -- color of the terminator - default black
    #          terminatorlinewidth -- width of the terminator contour
    #          **kwargs            -- passed to Polygon
    
    
    #          EXAMPLE: to only plot the terminator (no shade):
    #          plot_terminator(position, color = 'white') <- sets the shade to white (or something different if the plot background is different)
    
    
    #          useful extensions:
    #          - height dependence...
    #      """
    
    #      if np.isscalar(position): # set the terminator as a horizontal bar
    #          if position >= 0: # dayside
    #              position = np.min([90 - self.minlat, position])
    #              x0, y0 = self._mltMlatToXY(12, 90 - np.abs(position))
    #          else: #nightside
    #              x0, y0 = self._mltMlatToXY(24, 90 - np.abs(position))
    
    #          xr = np.sqrt(1 - y0**2)
    #          xl = -xr
    #          lat, left_mlt  = self._XYtomltMlat(xl, y0)
    #          lat, right_mlt = self._XYtomltMlat(xr, y0)
    #          if position > -(90 - self.minlat):
    #              right_mlt += 24
    
    #          x = np.array([xl, xr])
    #          y = np.array([y0, y0])
    
    #      else: # calculate the terminator trajectory
             
    #          # local added by Amalie 17/06/2022
    #          try:
    #              import apexpy
    #          except ModuleNotFoundError:
    #              raise ModuleNotFoundError('plot_terminator requires apexpy module.')
                 
    #          a = apexpy.Apex(date = position)
    
    #          t_glat, t_glon = terminator(position, sza = sza, resolution = 3600)
    #          t_mlat, t_mlon = a.geo2apex(t_glat, t_glon, 0)
    #          t_mlt          = a.mlon2mlt(t_mlon, position)
    
    #          # limit contour to correct hemisphere:
    #          iii = (t_mlat >= self.minlat) if north else (t_mlat <= -self.minlat)
    #          if len(iii) == 0:
    #              return 0 # terminator is outside plot
    #          t_mlat = t_mlat[iii]
    #          t_mlt = t_mlt[iii]
    
    #          x, y = self._mltMlatToXY(t_mlt, t_mlat)
    
    #          # find the points which are closest to minlat, and use these as edgepoints for the rest of the contour:
    #          xmin = np.argmin(x)
    #          xmax = np.argmax(x)
    #          left_mlt = t_mlt[xmin]
    #          right_mlt = t_mlt[xmax]
    #          if right_mlt < left_mlt:
    #              right_mlt += 24
    
    #      mlat_b = np.full(100, self.minlat)
    #      mlt_b  = np.linspace(left_mlt, right_mlt, 100)
    #      xb, yb = self._mltMlatToXY(mlt_b, mlat_b)
    
    #      # sort x and y to be in ascending order
    #      iii = np.argsort(x)
    #      x = x[iii[::-1]]
    #      y = y[iii[::-1]]
    
    #      if terminatorcolor is not None:
    #          self.ax.plot(x, y, color = terminatorcolor, linewidth = terminatorlinewidth)
    #      if shadecolor is not None:
    #          kwargs['color'] = shadecolor
    #          kwargs['linewidth'] = shadelinewidth
    #          shade = Polygon(np.vstack((np.hstack((x, xb)), np.hstack((y, yb)))).T, closed = True, **kwargs)
    #          self.ax.add_patch(shade)



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

    def plotimg(self,mlat,mlt,image,corr=True , crange = None, bgcolor = None, **kwargs):
        """
        Displays an image in polar coordinates.
        mlat,mlt,image are m x n arrays.
        If corr == True, mlat and mlt are corrected to edge coordinates. The plotted image is (m-2) x (n-2)
        If corr == False, mlat and mlat are assumed to be edge coordinates (plotted with a small offset). The plotted image is (m-1) x (n-1).
        crange is a tuple giving lower and upper colormap limits.
        bgcolor is a str giving background color inside the polar region.
        """

        x,y =self._mltMlatToXY(mlt,mlat)
        if corr:
            xe = np.full(np.subtract(x.shape,(1,1)),np.nan)
            ye = np.full(np.subtract(y.shape,(1,1)),np.nan)
            for i in range(xe.shape[0]):
                for j in range(xe.shape[1]):
                    xe[i,j]=np.mean(x[i:i+2,j:j+2])
                    ye[i,j]=np.mean(y[i:i+2,j:j+2])
            x=xe
            y=ye
            data = image[1:-1,1 :-1]
        else:
            data = image[1:, :-1]

        ll = (x[1:,  :-1].flatten(), y[1:,  :-1].flatten())
        lr = (x[1:,   1:].flatten(), y[1:,   1:].flatten())
        ul = (x[:-1, :-1].flatten(), y[:-1, :-1].flatten())
        ur = (x[:-1,  1:].flatten(), y[:-1,  1:].flatten())

        vertsx = np.vstack((ll[0], lr[0], ur[0], ul[0])).T
        vertsy = np.vstack((ll[1], lr[1], ur[1], ul[1])).T

        iii = np.where(vertsx**2 + vertsy**2 <=1, True, False)
        iii = np.any(iii, axis = 1).nonzero()[0]

        vertsx = vertsx[iii]
        vertsy = vertsy[iii]

        verts = np.dstack((vertsx, vertsy))
        if 'cmap' in kwargs.keys():
            cmap = kwargs.pop('cmap')
        else:
            cmap = plt.cm.viridis
        coll = PolyCollection(verts, array=data.flatten()[iii], cmap = cmap, edgecolors='none', **kwargs)
        if crange is not None:
            coll.set_clim(crange[0], crange[1])

        if bgcolor is not None:
            radius = 2*((90 - self.minlat)/ (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder = 0, facecolor = bgcolor)
            self.ax.add_artist(bg)

        return self.ax.add_collection(coll)


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
    def get_projected_coastlines(self, datetime, height=0, **kwargs):
        try:
            import cartopy.io.shapereader as shpreader
            from apexpy import Apex
        except ModuleNotFoundError:
            ModuleNotFoundError('Package missing. cartopy and apexpy are required for producing coastlines')
        """ generate coastlines in projected coordinates """

        if 'resolution' not in kwargs.keys():
            kwargs['resolution'] = '50m'
        if 'category' not in kwargs.keys():
            kwargs['category'] = 'physical'
        if 'name' not in kwargs.keys():
            kwargs['name'] = 'coastline'

        shpfilename = shpreader.natural_earth(**kwargs)
        reader = shpreader.Reader(shpfilename)
        coastlines = reader.records()
        multilinestrings = []
        A = Apex(date=datetime)
        for coastline in coastlines:
            if coastline.geometry.geom_type == 'MultiLineString':
                multilinestrings.append(coastline.geometry)
                continue
            lon, lat = np.array(coastline.geometry.coords[:]).T 
            mlat, mlon= A.geo2apex(lat, lon, height)
            mlt= A.mlon2mlt(mlon, datetime)
            yield mlat, mlt

        for mls in multilinestrings:
            for ls in mls:
                lon, lat = np.array(ls.coords[:]).T
                mlat, mlon= A.geo2apex(lat, lon, height)
                ind= mlat<self.minlat
                mlt= A.mlon2mlt(mlon, datetime)
                mlat[ind], mlt[ind]= np.nan, np.nan
                yield mlat, mlt
    def coastlines(self, datetime, height=0, map_kwargs=None, plot_kwargs=None):
        if (plot_kwargs is None):
            plot_kwargs= {'color':'k'}
        elif not('color' in plot_kwargs.keys()):
            plot_kwargs.update({'color':'k'})
        plots=[]
        if map_kwargs is None:
            for line in self.get_projected_coastlines(datetime,height=height):
                plots.extend(self.plot(line[0], line[1], **plot_kwargs))
        else:
            for line in self.get_projected_coastlines(datetime,height=height, **map_kwargs):
                plots.extend(self.plot(line[0], line[1], **plot_kwargs))
            
        return plots

    def _mltMlatToXY(self, mlt, mlat, bypass=False):
        """

        Parameters
        ----------
        mlt : TYPE
            DESCRIPTION.
        mlat : TYPE
            DESCRIPTION.
        bypass : boolean, optional
            When True the conversion will ignore the limits of the plot allowing plotting outside
            the limits of the subplot, for example this is used in writeMLTlabels. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        mlt = np.asarray(mlt, dtype='float64').copy()
        mlat = np.asarray(mlat, dtype='float64').copy()
        if not bypass:
            nan_index= (~eval(self.mltlims)) | (mlat<self.minlat)
        else:
            nan_index= np.bool8(np.zeros_like(mlat))
        if mlt.shape!= mlat.shape:
            raise ValueError('x and y must be the same size')
        mlt[nan_index]=np.nan
        mlat[nan_index]=np.nan
        r = (90. - np.abs(mlat))/(90. - self.minlat)
        a = (mlt - 6.)/12.*np.pi

        return r*np.cos(a), r*np.sin(a)

    def _XYtomltMlat(self, x, y, bypass=False):
        """
        convert x, y to mlt, mlat, where x**2 + y**2 = 1 corresponds to self.minlat

        Parameters
        ----------
        x : float/integer/list/array
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        bypass : boolean, optional
            When True the conversion will ignore the limits of the plot allowing plotting outside
            the limits of the subplot, for example this is used in writeMLTlabels. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        x, y = np.array(x, ndmin = 1, dtype='float64'), np.array(y, ndmin = 1, dtype='float64') # convert to array to allow item assignment

        lat = 90 - np.sqrt(x**2 + y**2)*(90. - self.minlat)
        mlt = np.arctan2(y, x)*12/np.pi + 6
        mlt[mlt < 0] += 24
        mlt[mlt > 24] -= 24
        if not bypass:
            nan_index= (~eval(self.mltlims)) | (lat<self.minlat)
        else:
            nan_index= np.bool8(np.zeros_like(x))
        mlt[nan_index]=np.nan
        lat[nan_index]=np.nan
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
        return format_coord
 
