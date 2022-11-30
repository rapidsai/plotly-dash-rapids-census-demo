import os
import requests
import dask_cudf
import cudf
from bokeh import palettes
import numpy as np
from pyproj import Transformer
import datashader as ds
import cupy as cp
import pickle
import datashader.transfer_functions as tf
import pandas as pd
import dask.dataframe as dd

# Colors
bgcolor = "#000000"  # mapbox dark map land color
text_color = "#cfd8dc"  # Material blue-grey 100
mapbox_land_color = "#000000"
c = 9200
# Colors for categories
colors = {}
colors["race"] = [
    "aqua",
    "lime",
    "yellow",
    "orange",
    "blue",
    "fuchsia",
    "saddlebrown",
]
race2color = {
    "White": "aqua",
    "African American": "lime",
    "American Indian": "yellow",
    "Asian alone": "orange",
    "Native Hawaiian": "blue",
    "Other Race alone": "fuchsia",
    "Two or More": "saddlebrown",
}
colors["net"] = [
    palettes.RdPu9[2],
    palettes.Greens9[4],
    palettes.PuBu9[2],
]  # '#32CD32'


# Figure template
row_heights = [150, 440, 250, 75]
template = {
    "layout": {
        "paper_bgcolor": bgcolor,
        "plot_bgcolor": bgcolor,
        "font": {"color": text_color},
        "margin": {"r": 0, "t": 0, "l": 0, "b": 0},
        "bargap": 0.05,
        "xaxis": {"showgrid": False, "automargin": True},
        "yaxis": {"showgrid": True, "automargin": True},
        #   'gridwidth': 0.5, 'gridcolor': mapbox_land_color},
    }
}

id2county = pickle.load(open("../id2county.pkl", "rb"))
county2id = {v: k for k, v in id2county.items()}
id2race = {
    0: "All",
    1: "White",
    2: "African American",
    3: "American Indian",
    4: "Asian alone",
    5: "Native Hawaiian",
    6: "Other Race alone",
    7: "Two or More",
}
race2id = {v: k for k, v in id2race.items()}

mappings = {}
mappings_hover = {}
# Load mapbox token from environment variable or file
token = os.getenv("MAPBOX_TOKEN")
mapbox_style = "carto-darkmatter"
if not token:
    try:
        token = "pk.eyJ1IjoibmlzaGFudGoiLCJhIjoiY2w1aXpwMXlkMDEyaDNjczBkZDVjY2l6dyJ9.7oLijsue-xOICmTqNInrBQ"

    except Exception as e:
        print("mapbox token not found, using open-street-maps")
        mapbox_style = "carto-darkmatter"

def set_projection_bounds(df_d):
    transformer_4326_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857")

    def epsg_4326_to_3857(coords):
        return [
            transformer_4326_to_3857.transform(*reversed(row))
            for row in coords
        ]

    transformer_3857_to_4326 = Transformer.from_crs("epsg:3857", "epsg:4326")

    def epsg_3857_to_4326(coords):
        return [
            list(reversed(transformer_3857_to_4326.transform(*row)))
            for row in coords
        ]

    data_3857 = (
        [df_d.easting.min(), df_d.northing.min()],
        [df_d.easting.max(), df_d.northing.max()],
    )
    data_center_3857 = [
        [
            (data_3857[0][0] + data_3857[1][0]) / 2.0,
            (data_3857[0][1] + data_3857[1][1]) / 2.0,
        ]
    ]

    data_4326 = epsg_3857_to_4326(data_3857)
    data_center_4326 = epsg_3857_to_4326(data_center_3857)

    return data_3857, data_center_3857, data_4326, data_center_4326


# Build Dash app and initial layout
def blank_fig(height):
    """
    Build blank figure with the requested height
    Args:
        height: height of blank figure in pixels
    Returns:
        Figure dict
    """
    return {
        "data": [],
        "layout": {
            "height": height,
            "template": template,
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        },
    }

# Plot functions
def build_colorscale(colorscale_name, transform):
    """
    Build plotly colorscale
    Args:
        colorscale_name: Name of a colorscale from the plotly.colors.sequential module
        transform: Transform to apply to colors scale. One of 'linear', 'sqrt', 'cbrt',
        or 'log'
    Returns:
        Plotly color scale list
    """
    global colors, mappings

    colors_temp = getattr(palettes, colorscale_name)
    if transform == "linear":
        scale_values = np.linspace(0, 1, len(colors_temp))
    elif transform == "sqrt":
        scale_values = np.linspace(0, 1, len(colors_temp)) ** 2
    elif transform == "cbrt":
        scale_values = np.linspace(0, 1, len(colors_temp)) ** 3
    elif transform == "log":
        scale_values = (10 ** np.linspace(0, 1, len(colors_temp)) - 1) / 9
    else:
        raise ValueError("Unexpected colorscale transform")
    return [(v, clr) for v, clr in zip(scale_values, colors_temp)]

def get_min_max(df, col):
    if isinstance(df, dask_cudf.core.DataFrame):
        return (df[col].min().compute(), df[col].max().compute())
    return (df[col].min(), df[col].max())

def build_datashader_plot(
    df,
    colorscale_name,
    colorscale_transform,
    new_coordinates,
    position,
    x_range,
    y_range,
    view_name,
):
    # global data_3857, data_center_3857, data_4326, data_center_4326

    x0, x1 = x_range
    y0, y1 = y_range

    datashader_color_scale = {}

    cvs = ds.Canvas(
        plot_width=3840, plot_height=2160, x_range=x_range, y_range=y_range
    )

    colorscale_transform = "linear"

    if view_name == "race":
        aggregate_column = "race"
        aggregate = "mean"
    elif view_name == "total":
        aggregate_column = "net"
        aggregate = "count"
        colorscale_name = "Viridis10"
    elif view_name == "in":
        aggregate_column = "net"
        aggregate = "count"
        colorscale_name = "PuBu9"
    elif view_name == "stationary":
        aggregate_column = "net"
        aggregate = "count"
        colorscale_name = "Greens9"
    elif view_name == "out":
        aggregate_column = "net"
        aggregate = "count"
        colorscale_name = "RdPu9"
    else:  # net
        aggregate_column = "net"
        aggregate = "mean"

    if aggregate == "mean":
        datashader_color_scale["cmap"] = colors[aggregate_column]
        datashader_color_scale["how"] = "linear"
        datashader_color_scale["span"] = get_min_max(df, aggregate_column)
    else:
        datashader_color_scale["cmap"] = [
            i[1]
            for i in build_colorscale(colorscale_name, colorscale_transform)
        ]
        datashader_color_scale["how"] = "log"

    agg = cvs.points(
        df,
        x="easting",
        y="northing",
        agg=getattr(ds, aggregate)(aggregate_column),
    )

    cmin = cp.asnumpy(agg.min().data)
    cmax = cp.asnumpy(agg.max().data)

    # Count the number of selected towers
    temp = agg.sum()
    temp.data = cp.asnumpy(temp.data)
    n_selected = int(temp)

    if n_selected == 0:
        # Nothing to display
        lat = [None]
        lon = [None]
        customdata = [None]
        marker = {}
        layers = []
    else:

        img = tf.shade(
            tf.dynspread(agg, threshold=0.7),
            **datashader_color_scale,
        ).to_pil()
        # img = tf.shade(agg,how='log',**datashader_color_scale).to_pil()

        # Add image as mapbox image layer. Note that as of version 4.4, plotly will
        # automatically convert the PIL image object into a base64 encoded png string
        layers = [
            {
                "sourcetype": "image",
                "source": img,
                "coordinates": new_coordinates,
            }
        ]

        # Do not display any mapbox markers
        lat = [None]
        lon = [None]
        customdata = [None]
        marker = {}

    # Build map figure
    map_graph = {
        "data": [],
        "layout": {
            "template": template,
            "uirevision": True,
            "zoom": 10,
            "mapbox": {
                "style": mapbox_style,
                "accesstoken": token,
                "layers": layers,
                "center": {
                    "lon": -78.81063494489342,
                    "lat": 37.471878534555074,
                },
            },
            "margin": {"r": 140, "t": 0, "l": 0, "b": 0},
            "height": 700,
            "shapes": [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0,
                    "y0": 0,
                    "x1": 1,
                    "y1": 1,
                    "line": {
                        "width": 1,
                        "color": "#191a1a",
                    },
                }
            ],
        },
    }

    if aggregate == "mean":
        # for `Age By PurBlue` category
        if view_name == "race":
            colorscale = [0, 1]

            marker = dict(
                size=0,
                showscale=True,
                colorbar={
                    "title": {
                        "text": "Race",
                        "side": "right",
                        "font": {"size": 14},
                    },
                    "tickvals": [
                        (0 + 0.5) / 7,
                        (1 + 0.5) / 7,
                        (2 + 0.5) / 7,
                        (3 + 0.5) / 7,
                        (4 + 0.5) / 7,
                        (5 + 0.5) / 7,
                        (6 + 0.5) / 7,
                    ],
                    "ticktext": [
                        "White",
                        "African American",
                        "American Indian",
                        "Asian alone",
                        "Native Hawaiian",
                        "Other Race alone",
                        "Two or More",
                    ],
                    "ypad": 30,
                },
                # colorscale=[(1, colors['race'][1]),(2, colors['race'][2]), (3, colors['race'][3]),(4, colors['race'][4]), (5, colors['race'][5]),(6, colors['race'][6]),(7, colors['race'][7])],
                colorscale=[
                    (0 / 7, colors["race"][0]),
                    (1 / 7, colors["race"][0]),
                    (1 / 7, colors["race"][1]),
                    (2 / 7, colors["race"][1]),
                    (2 / 7, colors["race"][2]),
                    (3 / 7, colors["race"][2]),
                    (3 / 7, colors["race"][3]),
                    (4 / 7, colors["race"][3]),
                    (4 / 7, colors["race"][4]),
                    (5 / 7, colors["race"][4]),
                    (5 / 7, colors["race"][5]),
                    (6 / 7, colors["race"][5]),
                    (6 / 7, colors["race"][6]),
                    (7 / 7, colors["race"][6]),
                ],
                cmin=0,
                cmax=1,
            )  # end of marker
        else:
            colorscale = [0, 1]

            marker = dict(
                size=0,
                showscale=True,
                colorbar={
                    "title": {
                        "text": "Migration",
                        "side": "right",
                        "font": {"size": 14},
                    },
                    "tickvals": [(0 + 0.5) / 3, (1 + 0.5) / 3, (2 + 0.5) / 3],
                    "ticktext": ["Out", "Stationary", "In"],
                    "ypad": 30,
                },
                colorscale=[
                    (0 / 3, colors["net"][0]),
                    (1 / 3, colors["net"][0]),
                    (1 / 3, colors["net"][1]),
                    (2 / 3, colors["net"][1]),
                    (2 / 3, colors["net"][2]),
                    (3 / 3, colors["net"][2]),
                ],
                cmin=0,
                cmax=1,
            )  # end of marker

        map_graph["data"].append(
            {
                "type": "scattermapbox",
                "lat": lat,
                "lon": lon,
                "customdata": customdata,
                "marker": marker,
                "hoverinfo": "none",
            }
        )
        map_graph["layout"]["annotations"] = []

    else:
        marker = dict(
            size=0,
            showscale=True,
            colorbar={
                "title": {
                    "text": "Population",
                    "side": "right",
                    "font": {"size": 14},
                },
                "ypad": 30,
            },
            colorscale=build_colorscale(colorscale_name, colorscale_transform),
            cmin=cmin,
            cmax=cmax,
        )  # end of marker

        map_graph["data"].append(
            {
                "type": "scattermapbox",
                "lat": lat,
                "lon": lon,
                "customdata": customdata,
                "marker": marker,
                "hoverinfo": "none",
            }
        )

    map_graph["layout"]["mapbox"].update(position)

    return map_graph


def query_df_range_lat_lon(df, x0, x1, y0, y1, x, y):
    mask_ = (df[x] >= x0) & (df[x] <= x1) & (df[y] <= y0) & (df[y] >= y1)
    if mask_.sum() != len(df):
        df = df[mask_]
        if isinstance(df, cudf.DataFrame):
            df.index = cudf.RangeIndex(0, len(df))
        else:
            df.index = pd.RangeIndex(0, len(df))
    del mask_
    return df


def bar_selected_ids(selection, column):  # select ids for each column

    if (column == "county_top") | (column == "county_bottom"):
        selected_ids = [county2id[p["label"]] for p in selection["points"]]
    else:
        selected_ids = [race2id[p["label"]] for p in selection["points"]]

    return selected_ids


def query_df_selected_ids(df, col, selected_ids):
    if (col == "county_top") | (col == "county_bottom"):
        col = "county"
    return df[df[col].isin(selected_ids)]


def no_data_figure():
    return {
        "data": [
            {
                "title": {"text": "Query Result"},
                "text": "SOME RANDOM",
                "marker": {"text": "NO"},
            }
        ],
        "layout": {
            "height": 250,
            "template": template,
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        },
    }


def build_histogram_default_bins(
    df,
    column,
    selections,
    orientation,
    colorscale_name,
    colorscale_transform,
    view_name,
    flag,
):

    if (view_name == "out") & (column == "race"):
        return no_data_figure()

    global race2color

    if (column == "county_top") | (column == "county_bottom"):
        column = "county"

    if isinstance(df, dask_cudf.core.DataFrame):
        df = df[[column, "net"]].groupby(column)["net"].count().compute().to_pandas()
    elif isinstance(df, cudf.DataFrame):
        df = df[[column, "net"]].groupby(column)["net"].count().to_pandas()
    else:
        df = df[[column, "net"]].groupby(column)["net"].count()

    print("Grouping done")

    df = df.sort_values(ascending=False)  # sorted grouped ids by counts

    if (flag == "top") | (flag == "bottom"):
        if flag == "top":
            view = df[:15]
        else:
            view = df[-15:]
        names = [id2county[cid] for cid in view.index.values]
    else:
        view = df
        names = [id2race[rid] for rid in view.index.values]

    bin_edges = names
    counts = view.values

    mapping_options = {}
    xaxis_labels = {}
    if column in mappings:
        if column in mappings_hover:
            mapping_options = {
                "text": list(mappings_hover[column].values()),
                "hovertemplate": "%{text}: %{y} <extra></extra>",
            }
        else:
            mapping_options = {
                "text": list(mappings[column].values()),
                "hovertemplate": "%{text} : %{y} <extra></extra>",
            }
        xaxis_labels = {
            "tickvals": list(mappings[column].keys()),
            "ticktext": list(mappings[column].values()),
        }

    if view_name == "total":
        bar_color = counts
        bar_scale = build_colorscale("Viridis10", colorscale_transform)
    elif view_name == "in":
        bar_color = counts
        bar_scale = build_colorscale("PuBu9", colorscale_transform)
    elif view_name == "stationary":
        bar_color = counts
        bar_scale = build_colorscale("Greens9", colorscale_transform)
    elif view_name == "out":
        bar_color = counts
        bar_scale = build_colorscale("RdPu9", colorscale_transform)
    elif view_name == "race":
        if column == "race":
            bar_color = [race2color[race] for race in names]
        else:
            bar_color = "#2C718E"
        bar_scale = None
    else:  # net
        bar_color = "#2C718E"
        bar_scale = None

    # print(bar_scale)

    fig = {
        "data": [
            {
                "type": "bar",
                "x": bin_edges,
                "y": counts,
                "marker": {"color": bar_color, "colorscale": bar_scale},
                **mapping_options,
            }
        ],
        "layout": {
            "yaxis": {
                "type": "linear",
                "title": {"text": "Count"},
            },
            "xaxis": {**xaxis_labels},
            "selectdirection": "h",
            "dragmode": "select",
            "template": template,
            "uirevision": True,
            "hovermode": "closest",
        },
    }

    if column not in selections:
        for i in range(len(fig["data"])):
            fig["data"][i]["selectedpoints"] = False

    return fig

def cull_empty_partitions(df):
    ll = list(df.map_partitions(len).compute())
    df_delayed = df.to_delayed()
    df_delayed_new = list()
    pempty = None
    for ix, n in enumerate(ll):
        if 0 == n:
            pempty = df.get_partition(ix)
        else:
            df_delayed_new.append(df_delayed[ix])
    if pempty is not None:
        df = dd.from_delayed(df_delayed_new, meta=pempty)
    return df

def build_updated_figures_dask(
    df,
    relayout_data,
    selected_map,
    selected_race,
    selected_county_top,
    selected_county_bottom,
    colorscale_name,
    data_3857,
    data_center_3857,
    data_4326,
    data_center_4326,
    coordinates_4326_backup,
    position_backup,
    view_name,
):
    colorscale_transform = "linear"
    selected = {}

    selected = {
        col: bar_selected_ids(sel, col)
        for col, sel in zip(
            ["race", "county_top", "county_bottom"],
            [selected_race, selected_county_top, selected_county_bottom],
        )
        if sel and sel.get("points", [])
    }

    if relayout_data is not None:
        transformer_4326_to_3857 = Transformer.from_crs(
            "epsg:4326", "epsg:3857"
        )

    def epsg_4326_to_3857(coords):
        return [
            transformer_4326_to_3857.transform(*reversed(row))
            for row in coords
        ]

    coordinates_4326 = relayout_data and relayout_data.get(
        "mapbox._derived", {}
    ).get("coordinates", None)
    dragmode = (
        relayout_data
        and "dragmode" in relayout_data
        and coordinates_4326_backup is not None
    )

    if dragmode:
        coordinates_4326 = coordinates_4326_backup
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        position = position_backup
    elif coordinates_4326:
        lons, lats = zip(*coordinates_4326)
        lon0, lon1 = max(min(lons), data_4326[0][0]), min(
            max(lons), data_4326[1][0]
        )
        lat0, lat1 = max(min(lats), data_4326[0][1]), min(
            max(lats), data_4326[1][1]
        )
        coordinates_4326 = [
            [lon0, lat0],
            [lon1, lat1],
        ]
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        coordinates_4326_backup = coordinates_4326

        position = {
            "zoom": relayout_data.get("mapbox.zoom", None),
            "center": relayout_data.get("mapbox.center", None),
        }
        position_backup = position

    else:
        position = {
            "zoom": 3.3350828189345934,
            "pitch": 0,
            "bearing": 0,
            "center": {
                "lon": -100.55828959790324,
                "lat": 38.68323453274175,
            },  # {'lon': data_center_4326[0][0]-100, 'lat': data_center_4326[0][1]-10}
        }
        coordinates_3857 = data_3857
        coordinates_4326 = data_4326

    new_coordinates = [
        [coordinates_4326[0][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[0][1]],
        [coordinates_4326[0][0], coordinates_4326[0][1]],
    ]

    # print(new_coordinates)

    x_range, y_range = zip(*coordinates_3857)
    x0, x1 = x_range
    y0, y1 = y_range
    
    if selected_map is not None:
        coordinates_4326 = selected_map["range"]["mapbox"]
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        x_range_t, y_range_t = zip(*coordinates_3857)
        x0, x1 = x_range_t
        y0, y1 = y_range_t
        df = df.map_partitions(query_df_range_lat_lon, x0, x1, y0, y1, "easting", "northing").persist()

    # Select points as per view

    if (view_name == "total") | (view_name == "race"):
        df = df[(df["net"] == 0) | (df["net"] == 1)]
        # df['race'] = df['race'].astype('category')
    elif view_name == "in":
        df = df[df["net"] == 1]
        df["net"] = df["net"].astype("int8")
    elif view_name == "stationary":
        df = df[df["net"] == 0]
        df["net"] = df["net"].astype("int8")
    elif view_name == "out":
        df = df[df["net"] == -1]
        df["net"] = df["net"].astype("int8")
    else:  # net migration condition
        df = df
        # df["net"] = df["net"].astype("category")

    for col in selected:
        df = df.map_partitions(query_df_selected_ids, col, selected[col])

    # cull empty partitions
    df = cull_empty_partitions(df).persist()

    datashader_plot = build_datashader_plot(
        df,
        colorscale_name,
        colorscale_transform,
        new_coordinates,
        position,
        x_range,
        y_range,
        view_name,
    )

    # Build indicator figure
    n_selected_indicator = {
        "data": [
            {
                "domain": {"x": [0.31, 0.41], "y": [0, 0.5]},
                "title": {"text": "Query Result"},
                "type": "indicator",
                "value": len(df),
                "number": {
                    "font": {"color": text_color, "size": "50px"},
                    "valueformat": ",",
                    "suffix": " people",
                },
            },
            {
                "domain": {"x": [0.71, 0.81], "y": [0, 0.5]},
                "title": {"text": "Data Size"},
                "type": "indicator",
                "value": len(df),
                "number": {
                    "font": {"color": text_color, "size": "50px"},
                    "valueformat": ",",
                    "suffix": " rows",
                },
            },
        ],
        "layout": {
            "template": template,
            "height": row_heights[3],
            # 'margin': {'l': 0, 'r': 0,'t': 5, 'b': 5}
        },
    }

    # print("DATASHADER done")

    race_histogram = build_histogram_default_bins(
        df,
        "race",
        selected,
        "v",
        colorscale_name,
        colorscale_transform,
        view_name,
        flag="All",
    )

    # # print("RACE done")

    # # print("INSIDE UPDATE")
    county_top_histogram = build_histogram_default_bins(
        df,
        "county",
        selected,
        "v",
        colorscale_name,
        colorscale_transform,
        view_name,
        flag="top",
    )

    # # print("COUNTY TOP done")

    county_bottom_histogram = build_histogram_default_bins(
        df,
        "county",
        selected,
        "v",
        colorscale_name,
        colorscale_transform,
        view_name,
        flag="bottom",
    )

    # print("COUNTY BOTTOM done")

    del (df)
    return (
        datashader_plot,
        county_top_histogram,
        county_bottom_histogram,
        race_histogram,
        n_selected_indicator,
        coordinates_4326_backup,
        position_backup,
    )

def build_updated_figures(
    df,
    relayout_data,
    selected_map,
    selected_race,
    selected_county_top,
    selected_county_bottom,
    colorscale_name,
    data_3857,
    data_center_3857,
    data_4326,
    data_center_4326,
    coordinates_4326_backup,
    position_backup,
    view_name,
):

    colorscale_transform = "linear"
    selected = {}

    selected = {
        col: bar_selected_ids(sel, col)
        for col, sel in zip(
            ["race", "county_top", "county_bottom"],
            [selected_race, selected_county_top, selected_county_bottom],
        )
        if sel and sel.get("points", [])
    }

    if relayout_data is not None:
        transformer_4326_to_3857 = Transformer.from_crs(
            "epsg:4326", "epsg:3857"
        )

    def epsg_4326_to_3857(coords):
        return [
            transformer_4326_to_3857.transform(*reversed(row))
            for row in coords
        ]

    coordinates_4326 = relayout_data and relayout_data.get(
        "mapbox._derived", {}
    ).get("coordinates", None)
    dragmode = (
        relayout_data
        and "dragmode" in relayout_data
        and coordinates_4326_backup is not None
    )

    if dragmode:
        coordinates_4326 = coordinates_4326_backup
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        position = position_backup
    elif coordinates_4326:
        lons, lats = zip(*coordinates_4326)
        lon0, lon1 = max(min(lons), data_4326[0][0]), min(
            max(lons), data_4326[1][0]
        )
        lat0, lat1 = max(min(lats), data_4326[0][1]), min(
            max(lats), data_4326[1][1]
        )
        coordinates_4326 = [
            [lon0, lat0],
            [lon1, lat1],
        ]
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        coordinates_4326_backup = coordinates_4326

        position = {
            "zoom": relayout_data.get("mapbox.zoom", None),
            "center": relayout_data.get("mapbox.center", None),
        }
        position_backup = position

    else:
        position = {
            "zoom": 3.3350828189345934,
            "pitch": 0,
            "bearing": 0,
            "center": {
                "lon": -100.55828959790324,
                "lat": 38.68323453274175,
            },  # {'lon': data_center_4326[0][0]-100, 'lat': data_center_4326[0][1]-10}
        }
        coordinates_3857 = data_3857
        coordinates_4326 = data_4326

    new_coordinates = [
        [coordinates_4326[0][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[1][1]],
        [coordinates_4326[1][0], coordinates_4326[0][1]],
        [coordinates_4326[0][0], coordinates_4326[0][1]],
    ]

    # print(new_coordinates)

    x_range, y_range = zip(*coordinates_3857)
    x0, x1 = x_range
    y0, y1 = y_range

    if selected_map is not None:
        coordinates_4326 = selected_map["range"]["mapbox"]
        coordinates_3857 = epsg_4326_to_3857(coordinates_4326)
        x_range_t, y_range_t = zip(*coordinates_3857)
        x0, x1 = x_range_t
        y0, y1 = y_range_t
        df = query_df_range_lat_lon(df, x0, x1, y0, y1, "easting", "northing")

    # Select points as per view

    if (view_name == "total") | (view_name == "race"):
        df = df[(df["net"] == 0) | (df["net"] == 1)]
        df["net"] = df["net"].astype("int8")
        # df['race'] = df['race'].astype('category')
    elif view_name == "in":
        df = df[df["net"] == 1]
        df["net"] = df["net"].astype("int8")
    elif view_name == "stationary":
        df = df[df["net"] == 0]
        df["net"] = df["net"].astype("int8")
    elif view_name == "out":
        df = df[df["net"] == -1]
        df["net"] = df["net"].astype("int8")
    else:  # net migration condition
        df = df
        # df["net"] = df["net"].astype("category")

    for col in selected:
        df = query_df_selected_ids(df, col, selected[col])

    datashader_plot = build_datashader_plot(
        df,
        colorscale_name,
        colorscale_transform,
        new_coordinates,
        position,
        x_range,
        y_range,
        view_name,
    )

    # Build indicator figure
    n_selected_indicator = {
        "data": [
            {
                "domain": {"x": [0.31, 0.41], "y": [0, 0.5]},
                "title": {"text": "Query Result"},
                "type": "indicator",
                "value": len(df),
                "number": {
                    "font": {"color": text_color, "size": "50px"},
                    "valueformat": ",",
                    "suffix": " people",
                },
            },
            {
                "domain": {"x": [0.71, 0.81], "y": [0, 0.5]},
                "title": {"text": "Data Size"},
                "type": "indicator",
                "value": len(df),
                "number": {
                    "font": {"color": text_color, "size": "50px"},
                    "valueformat": ",",
                    "suffix": " rows",
                },
            },
        ],
        "layout": {
            "template": template,
            "height": row_heights[3],
            # 'margin': {'l': 0, 'r': 0,'t': 5, 'b': 5}
        },
    }

    # print("DATASHADER done")

    race_histogram = build_histogram_default_bins(
        df,
        "race",
        selected,
        "v",
        colorscale_name,
        colorscale_transform,
        view_name,
        flag="All",
    )

    # # print("RACE done")

    # # print("INSIDE UPDATE")
    county_top_histogram = build_histogram_default_bins(
        df,
        "county",
        selected,
        "v",
        colorscale_name,
        colorscale_transform,
        view_name,
        flag="top",
    )

    # # print("COUNTY TOP done")

    county_bottom_histogram = build_histogram_default_bins(
        df,
        "county",
        selected,
        "v",
        colorscale_name,
        colorscale_transform,
        view_name,
        flag="bottom",
    )

    # print("COUNTY BOTTOM done")

    del (df)
    return (
        datashader_plot,
        county_top_histogram,
        county_bottom_histogram,
        race_histogram,
        n_selected_indicator,
        coordinates_4326_backup,
        position_backup,
    )

def check_dataset(dataset_url, data_path):
    if not os.path.exists(data_path):
        print(f"Dataset not found at "+data_path+".\n"
              f"Downloading from {dataset_url}")
        # Download dataset to data directory
        os.makedirs('../data', exist_ok=True)
        with requests.get(dataset_url, stream=True) as r:
            r.raise_for_status()
            with open(data_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print('Download completed!')
    else:
        print(f"Found dataset at {data_path}")


def load_dataset(path, dtype="dask_cudf"):
    """
    Args:
        path: Path to arrow file containing mortgage dataset
    Returns:
        pandas DataFrame
    """
    if os.path.isdir(path):
        path = path + '/*'
    if dtype == "dask":
        return dd.read_parquet(path, split_row_groups=True)
    return dask_cudf.read_parquet(path, split_row_groups=True)
