"""Functions to create create dummies data and compute spatial indexes.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Point, Polygon, MultiPolygon
import rtree as rt


def generate_data(**kwargs):
    """Generate dummies data.

    Function to generate grid and random points. The dimensions default of grid
    are 10x10 and resolution equals 5, the lower is boundary 0 and upper
    boundary 100. By default, 100 random points samples are generated within the
    grid dimensions.

    :param kwargs: The following parameters can be passed optionally:
    `lower_limite` is lower coordinate for the rows and columns (default is 0).
    `upper_limite` is upper coordinate for the rows and columns
    (default is 100). `resolution` is the width and height of the cell grid
    (default is 10). `n_samples` is the number of random points generated, the
    distribution functions used is uniform (default is 100).

    :return `grid`, `points`: The return of function are two geo data frames. The
    first geo data frame `grid` contains $resolution^2$ rows indicating the
    boundaries of each grid cell. The second geo data frame `points` is the
    coordinates of each point.
    """

    lower_limit = kwargs.get('lower_limit', 0)
    upper_limit = kwargs.get('upper_limit', 100)
    n_samples = kwargs.get('n_samples', 100)
    resolution = kwargs.get('resolution', None)

    if resolution:
        width = resolution
        height = resolution
    else:
        width = kwargs.get('width', 10)
        height = kwargs.get('height', 10)

    grid = gpd.GeoDataFrame()
    for row in range(lower_limit + height, upper_limit + height, height):
        for col in range(lower_limit + width, upper_limit + width, width):
            poly = Polygon(((col - width, row), (col, row), (col, row - height),
                            (col - width, row - height), (col - width, row)))
            temp = gpd.GeoDataFrame({'geometry': [poly]})
            grid = grid.append(temp)
    grid.reset_index(drop=True, inplace=True)

    x = np.random.uniform(lower_limit, upper_limit, n_samples)
    y = np.random.uniform(lower_limit, upper_limit, n_samples)
    points = gpd.GeoDataFrame(data={'x': x, 'y': y})
    points['geometry'] = points.apply(lambda row: Point((row['x'], row['y'])),
                                      axis=1)

    return grid, points


def compute_hits(grid, points):
    """Compute hits of points in polygons.

    Function to compute total points in each grid cell. The function is
    optimized using R-tree as a spatial index.

    :param grid: The `grid` is a geo data frame that represents the grid cells.
    :param points: The `points` is a geo data frame that represents all points
    to compare.

    :return `df_ref`: The return is a data frame that contain the reference in
    the grid of each point.
    """
    rtree = rt.index.Index()
    grid_hits = np.zeros(grid.size)

    for i, point in points['geometry'].iteritems():
        rtree.insert(i, point.bounds)

    df_ref = gpd.GeoDataFrame()
    for i, row in grid.iterrows():
        poly = row['geometry']
        sindex_matches = list(rtree.intersection(poly.bounds))
        points_matches = points.iloc[sindex_matches]
        points_matches = points_matches[points_matches.intersects(poly)]
        points_matches['grid'] = i
        df_ref = df_ref.append(points_matches)
    return df_ref


def run():
    res = 10
    samples = 1000
    ngrid = 22
    idx = ngrid//res
    idy = ngrid%res
    grid, points = generate_data(resolution=res, n_samples=samples)
    west, south, east, north = points.unary_union.bounds

    df_ref = compute_hits(grid, points)
    within = df_ref.query('grid == %d'%(ngrid))
    outside = df_ref[~df_ref.isin(within)]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    ax[0].set_title('Grid %dx%d'%(res, res))
    ax[1].set_title('Grid %dx%d and %d points\npoints of index [%d][%d] '
                    'selected'%(res, res, samples, idx, idy))
    oldtick = np.arange(5, 100, 10)
    newtick = np.arange(10)
    ax[0].set_xticks(oldtick)
    ax[0].set_xticklabels(newtick)
    ax[1].set_xticks(oldtick)
    ax[1].set_xticklabels(newtick)
    ax[0].set_yticks(oldtick)
    ax[0].set_yticklabels(newtick)

    for i, g in grid['geometry'].iteritems():
        patch = PolygonPatch(g, fc='#dcdcdc', ec='r', zorder=2)
        patch2 = PolygonPatch(g, fc='#dcdcdc', ec='r', zorder=2)
        ax[0].add_patch(patch)
        ax[1].add_patch(patch2)

    ax[0].set_xlim(west, east)
    ax[0].set_ylim(south, north)
    ax[1].set_xlim(west, east)
    ax[1].set_ylim(south, north)

    # ax[0].scatter(points.iloc[:, 0].values, points.iloc[:, 1].values,
    #               zorder=3, marker='.', color='b')

    ax[1].scatter(within['x'].values, within['y'].values, zorder=3, marker='.',
                  color='r')
    ax[1].scatter(outside['x'].values, outside['y'].values, zorder=3,
                  marker='.', color='b')

    plt.show()


if __name__ == "__main__":
    run()


