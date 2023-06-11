import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import numpy as np
from tobac_flow.utils import weighted_correlation


def get_goes_ccrs(goes_ds):
    return ccrs.Geostationary(
        satellite_height=goes_ds.goes_imager_projection.perspective_point_height,
        central_longitude=goes_ds.goes_imager_projection.longitude_of_projection_origin,
        sweep_axis=goes_ds.goes_imager_projection.sweep_angle_axis,
    )


def get_goes_extent(goes_ds):
    h = goes_ds.goes_imager_projection.perspective_point_height
    img_extent = (
        goes_ds.x.data[0] * h,
        goes_ds.x.data[-1] * h,
        goes_ds.y.data[-1] * h,
        goes_ds.y.data[0] * h,
    )
    return img_extent


def goes_subplot(goes_ds, *args, fig=None, cbar_size="5%", cbar_pad=0.1, **kwargs):
    if fig is None:
        fig = plt.gcf()
    crs = get_goes_ccrs(goes_ds)
    img_extent = get_goes_extent(goes_ds)
    h = goes_ds.goes_imager_projection.perspective_point_height

    ax = fig.add_subplot(*args, projection=crs, **kwargs)
    try:
        ax.set_extent(img_extent, ax.projection)
    except ValueError:
        ax.set_global()
    if not np.allclose(np.array(img_extent), np.array(ax.get_extent())):
        print(img_extent, ax.get_extent())
        ax.set_global()
    ax_divider = make_axes_locatable(ax)
    cax_r = ax_divider.new_horizontal(size=cbar_size, pad=cbar_pad, axes_class=plt.Axes)
    cax_l = ax_divider.new_horizontal(
        size=cbar_size, pad=cbar_pad, pack_start=True, axes_class=plt.Axes
    )
    cax_t = ax_divider.new_vertical(size=cbar_size, pad=cbar_pad, axes_class=plt.Axes)
    cax_b = ax_divider.new_vertical(
        size=cbar_size, pad=cbar_pad, pack_start=True, axes_class=plt.Axes
    )

    ax._imshow = ax.imshow.__get__(ax)
    ax._contour = ax.contour.__get__(ax)
    ax._contourf = ax.contourf.__get__(ax)
    ax._quiver = ax.quiver.__get__(ax)

    # def colorbar(self, *args, orientation='vertical', **kwargs):
    #     if orientation == 'vertical':
    #         fig.add_axes(cax_r)
    #         cbar = plt.colorbar(*args, cax=cax_r, orientation='vertical', **kwargs)
    #     elif orientation == 'horizontal':
    #         fig.add_axes(cax_b)
    #         cbar = plt.colorbar(*args, cax=cax_b, orientation='horizontal', **kwargs)
    #     else:
    #         raise ValueError("orientation keyword must be 'vertical' or 'horizontal'")
    #     return cbar
    # ax.colorbar = colorbar.__get__(ax)

    def colorbar(self, *args, location="right", **kwargs):
        if location == "right":
            fig.add_axes(cax_r)
            cbar = plt.colorbar(*args, cax=cax_r, orientation="vertical", **kwargs)
        elif location == "left":
            fig.add_axes(cax_l)
            cbar = plt.colorbar(*args, cax=cax_l, orientation="vertical", **kwargs)
            cbar.yaxis.set_ticks_position("left")
            cbar.yaxis.set_label_position("left")
        elif location == "bottom":
            fig.add_axes(cax_b)
            cbar = plt.colorbar(*args, cax=cax_b, orientation="horizontal", **kwargs)
        elif location == "top":
            fig.add_axes(cax_t)
            cbar = plt.colorbar(*args, cax=cax_t, orientation="horizontal", **kwargs)
            cbar.xaxis.set_ticks_position("top")
            cbar.xaxis.set_label_position("top")
        else:
            raise ValueError(
                "Location keyword must be 'left', 'right', 'bottom' or 'top'"
            )
        return cbar

    ax.colorbar = colorbar.__get__(ax)

    def imshow(self, *args, extent=img_extent, **kwargs):
        img = self._imshow(*args, extent=extent, **kwargs)
        return img

    ax.imshow = imshow.__get__(ax)

    def contour(self, data, *args, **kwargs):
        cntr = self._contour(goes_ds.x * h, goes_ds.y * h, data, *args, **kwargs)
        self.set_extent(img_extent, self.projection)
        return cntr

    ax.contour = contour.__get__(ax)

    def contourf(self, data, *args, **kwargs):
        cntr = self._contourf(goes_ds.x * h, goes_ds.y * h, data, *args, **kwargs)
        self.set_extent(img_extent, self.projection)
        return cntr

    ax.contourf = contourf.__get__(ax)

    def quiver(self, u, v, *args, spacing=1, block_method="slice", **kwargs):
        if block_method == "slice":
            slc = slice(spacing // 2, None, spacing)
            quiv = self._quiver(
                goes_ds.x[slc] * h,
                goes_ds.y[slc] * h,
                u[slc, slc],
                v[slc, slc],
                *args,
                **kwargs,
            )
        elif block_method == "reduce":
            from skimage.measure import block_reduce

            new_x, new_y = block_reduce(
                goes_ds.x.data * h, (spacing,), np.nanmean
            ), block_reduce(goes_ds.y.data * h, (spacing,), np.nanmean)
            new_u, new_v = block_reduce(
                u, (spacing, spacing), np.nanmean
            ), block_reduce(v, (spacing, spacing), np.nanmean)
            quiv = self._quiver(new_x, new_y, new_u, new_v, *args, **kwargs)
        else:
            raise ValueError("invalid input for 'block_method'")
        return quiv

    ax.quiver = quiver.__get__(ax)

    return ax


def goes_figure(goes_ds, *args, **kwargs):
    fig = plt.figure(*args, **kwargs)

    def subplot(self, *args, **kwargs):
        ax = goes_subplot(goes_ds, *args, fig=fig, **kwargs)
        return ax

    fig.subplot = subplot.__get__(fig)
    return fig


def add_gl_ticks(ax, gl):
    x_tick_locs = []
    if gl.bottom_labels:
        x_tick_locs += [
            artist.properties()["unitless_position"][0]
            for artist in gl.bottom_label_artists
            if artist.properties()["visible"]
        ]
    if gl.top_labels:
        x_tick_locs += [
            artist.properties()["unitless_position"][0]
            for artist in gl.top_label_artists
            if artist.properties()["visible"]
        ]
    x_tick_labels = [""] * len(x_tick_locs)
    ax.set_xticks(x_tick_locs, crs=ax.projection)
    ax.set_xticklabels(x_tick_labels)

    y_tick_locs = []
    if gl.left_labels:
        y_tick_locs += [
            artist.properties()["unitless_position"][1]
            for artist in gl.left_label_artists
            if artist.properties()["visible"]
        ]
    if gl.right_labels:
        y_tick_locs += [
            artist.properties()["unitless_position"][1]
            for artist in gl.right_label_artists
            if artist.properties()["visible"]
        ]
    y_tick_labels = [""] * len(y_tick_locs)
    ax.set_yticks(y_tick_locs, crs=ax.projection)
    ax.set_yticklabels(y_tick_labels)

    ax.tick_params(
        top=gl.top_labels,
        bottom=gl.bottom_labels,
        left=gl.left_labels,
        right=gl.right_labels,
    )


def bias_plot(ax4, obs, truths, weights):
    from scipy.stats import linregress
    from sklearn.linear_model import LinearRegression
    from tobac_flow.utils import weighted_average_and_std

    # Plot points
    ax4.scatter(truths.ravel(), obs.ravel(), alpha=0.05, c="b")
    # 1-1 line
    ax4.plot([-1e4, 1e4], [-1e4, 1e4], "k--")

    # Find linear fit and plot
    wh = np.isfinite(obs.ravel())
    linear_fit = linregress(truths.ravel()[wh], obs.ravel()[wh])
    print(f"All points -- Slope: {linear_fit.slope}, Intercept:{linear_fit.intercept}")
    fit_equation = lambda x: linear_fit.slope * x + linear_fit.intercept
    plt.plot(np.array([-1e4, 1e4]), fit_equation(np.array([-1e4, 1e4])), "b")

    # Get info about fit/bias
    wh = np.isfinite(obs)
    bias = (obs[wh] - truths[wh]).mean()
    std = (obs[wh] - truths[wh]).std()
    std *= wh.sum() / (wh.sum() - 1)
    rmse = ((obs[wh] - truths[wh]) ** 2).mean() ** 0.5
    r_value = np.corrcoef(truths[wh], obs[wh])[0, 1]
    n_points = wh.sum()
    units = "$Wm^{-2}$"
    print(f"Bias: {bias:.02f} {units}")
    print(f"Std: {std:.02f} {units}")
    print(f"RMSE: {rmse:.02f} {units}")
    print(f"R: {r_value:.02f}")
    print(f"N: {n_points}")
    ax4.text(
        0.975,
        0.025,
        f"All locations:\nBias: {bias:.02f} {units}\nStd: {std:.02f} {units}\nRMSE: {rmse:.02f} {units}\nR: {r_value:.02f}\nN: {n_points}",
        color="b",
        ha="right",
        va="bottom",
        transform=ax4.transAxes,
    )

    # Now weight by DCC anvil locations
    wh = weights > 0
    ax4.scatter(truths[wh], obs[wh], alpha=0.05, c="r")

    regr = LinearRegression()
    regr.fit(truths[wh].reshape(-1, 1), obs[wh].reshape(-1, 1), weights[wh])
    print(f"DCC weighted -- Slope: {regr.coef_[0][0]}, Intercept:{regr.intercept_[0]}")
    plt.plot(
        np.array([-1e4, 1e4]), regr.predict(np.array([-1e4, 1e4]).reshape(-1, 1)), "r"
    )

    wh = weights > 0
    weighted_bias, weighted_std = weighted_average_and_std(
        (obs - truths)[wh], weights=weights[wh]
    )
    weighted_rmse = np.average((obs - truths)[wh] ** 2, weights=weights[wh]) ** 0.5
    weighted_r = weighted_correlation(truths[wh], obs[wh], weights[wh])
    weighted_n = wh.sum()
    units = "$Wm^{-2}$"
    print(f"Bias: {weighted_bias:.02f} {units}")
    print(f"Std: {weighted_std:.02f} {units}")
    print(f"RMSE: {weighted_rmse:.02f} {units}")
    print(f"R: {weighted_r:.02f}")
    print(f"N: {weighted_n}")
    ax4.text(
        0.025,
        0.975,
        f"DCC observation weighted:\nBias: {weighted_bias:.02f} {units}\nStd: {weighted_std:.02f} {units}\nRMSE: {weighted_rmse:.02f} {units}\nR: {weighted_r:.02f}\nN: {weighted_n}",
        color="r",
        ha="left",
        va="top",
        transform=ax4.transAxes,
    )
