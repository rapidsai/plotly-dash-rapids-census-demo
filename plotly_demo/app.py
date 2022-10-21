import os
import pandas as pd
import numpy as np
import cudf
import cupy
from dash import Dash, dcc, html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import datashader.transfer_functions as tf
import datashader as ds
from dash import dcc
import dash_bootstrap_components as dbc
import time
import dash_daq as daq
import pickle
from bokeh import palettes
import dash
from pyproj import Transformer
import dask_cudf

# from dask.distributed import Client, wait
# from dask_cuda import LocalCUDACluster
# client = Client(LocalCUDACluster())

df = cudf.read_parquet(
    "./data/total_population_dataset.parquet", columns=["easting", "northing", "race", "net", "county"]
)

# ### Dashboards start here

# Colors
bgcolor = "#000000"  # mapbox dark map land color
text_color = "#cfd8dc"  # Material blue-grey 100
mapbox_land_color = "#000000"
c = 9200
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

id2county = pickle.load(open("./id2county.pkl", "rb"))
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

DATA_SIZE = len(df)

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
        datashader_color_scale["span"] = (
            df[aggregate_column].min(), df[aggregate_column].max())
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

    cmin = cupy.asnumpy(agg.min().data)
    cmax = cupy.asnumpy(agg.max().data)

    # Count the number of selected towers
    temp = agg.sum()
    temp.data = cupy.asnumpy(temp.data)
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
    # print(col,selected_ids)
    if (col == "county_top") | (col == "county_bottom"):
        col = "county"
    queried_df = df[df[col].isin(selected_ids)]
    return queried_df


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


df["race"].value_counts()


df.groupby("race")["net"].count()


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

    print(column)

    if isinstance(df, cudf.DataFrame):
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


def build_updated_figures(
    df,
    relayout_data,
    selected_map,
    # selected_race,
    # selected_county_top,
    # selected_county_bottom,
    colorscale_name,
    data_3857,
    data_center_3857,
    data_4326,
    data_center_4326,
    coordinates_4326_backup,
    position_backup,
    view_name,
):
    global DATA_SIZE

    colorscale_transform = "linear"
    selected = {}

    # selected = {
    #     col: bar_selected_ids(sel, col)
    #     for col, sel in zip(
    #         ["race", "county_top", "county_bottom"],
    #         [selected_race, selected_county_top, selected_county_bottom],
    #     )
    #     if sel and sel.get("points", [])
    # }

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
        df_hists = df[(df["net"] == 0) | (df["net"] == 1)]
        df_hists["net"] = df_hists["net"].astype("int8")
        # df_hists['race'] = df_hists['race'].astype('category')
    elif view_name == "in":
        df_hists = df[df["net"] == 1]
        df_hists["net"] = df_hists["net"].astype("int8")
    elif view_name == "stationary":
        df_hists = df[df["net"] == 0]
        df_hists["net"] = df_hists["net"].astype("int8")
    elif view_name == "out":
        df_hists = df[df["net"] == -1]
        df_hists["net"] = df_hists["net"].astype("int8")
    else:  # net migration condition
        df_hists = df
        # df_hists["net"] = df_hists["net"].astype("category")

    for col in selected:
        df_hists = query_df_selected_ids(df_hists, col, selected[col])

    datashader_plot = build_datashader_plot(
        df_hists,
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
                "value": len(df_hists),
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
                "value": DATA_SIZE,
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

    # race_histogram = build_histogram_default_bins(
    #     df_hists,
    #     "race",
    #     selected,
    #     "v",
    #     colorscale_name,
    #     colorscale_transform,
    #     view_name,
    #     flag="All",
    # )

    # # print("RACE done")

    # # print("INSIDE UPDATE")
    # county_top_histogram = build_histogram_default_bins(
    #     df_hists,
    #     "county",
    #     selected,
    #     "v",
    #     colorscale_name,
    #     colorscale_transform,
    #     view_name,
    #     flag="top",
    # )

    # # print("COUNTY TOP done")

    # county_bottom_histogram = build_histogram_default_bins(
    #     df_hists,
    #     "county",
    #     selected,
    #     "v",
    #     colorscale_name,
    #     colorscale_transform,
    #     view_name,
    #     flag="bottom",
    # )

    # print("COUNTY BOTTOM done")

    return (
        datashader_plot,
        # county_top_histogram,
        # county_bottom_histogram,
        # race_histogram,
        n_selected_indicator,
        coordinates_4326_backup,
        position_backup,
    )


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    children=[
        ################# Title Bar ##############
        html.Div(
            [
                html.H1(
                    children=[
                        "Census 2020 Net Migration Visualization",
                        html.A(
                            html.Img(
                                src="assets/rapids-logo.png",
                                style={
                                    "float": "right",
                                    "height": "45px",
                                    "margin-right": "1%",
                                    "margin-top": "-7px",
                                },
                            ),
                            href="https://rapids.ai/",
                        ),
                        html.A(
                            html.Img(
                                src="assets/dash-logo.png",
                                style={"float": "right", "height": "30px"},
                            ),
                            href="https://dash.plot.ly/",
                        ),
                    ],
                    style={
                        "text-align": "left",
                        "heights": "30px",
                        "margin-left": "20px",
                    },
                ),
            ]
        ),
        ###################### Options Bar ######################
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Table(
                            [
                                html.Tr(
                                    [
                                        html.Td(
                                            html.Div("CPU"),
                                            style={
                                                "font-size": "20px",
                                                "padding-left": "1.3rem",
                                            },  # className="config-label"
                                        ),
                                        html.Td(
                                            html.Div(
                                                [
                                                    daq.DarkThemeProvider(
                                                        daq.BooleanSwitch(
                                                            on=True,  # Turn on CPU/GPU
                                                            color="#00cc96",
                                                            id="gpu-toggle",
                                                        )
                                                    ),
                                                    dbc.Tooltip(
                                                        "Caution: Using CPU compute for more than 50 million points is not recommended.",
                                                        target="gpu-toggle",
                                                        placement="bottom",
                                                        autohide=True,
                                                        style={
                                                            "textAlign": "left",
                                                            "font-size": "15px",
                                                            "color": "white",
                                                            "width": "350px",
                                                            "padding": "15px",
                                                            "border-radius": "5px",
                                                            "background-color": "#2a2a2e",
                                                        },
                                                    ),
                                                ]
                                            )
                                        ),
                                        html.Td(
                                            html.Div("GPU + RAPIDS"),
                                            style={
                                                "font-size": "20px"
                                            },  # , className="config-label"
                                        ),
                                        #######  Indicator graph ######
                                        html.Td(
                                            [
                                                dcc.Loading(
                                                    dcc.Graph(
                                                        id="indicator-graph",
                                                        figure=blank_fig(50),
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                        style={"width": "95%"},
                                                    ),
                                                    color="#b0bec5",
                                                    # style={'height': f'{50}px', 'width':'10px'}
                                                ),  # style={'width': '50%'},
                                            ]
                                        ),
                                        ###### VIEWS ARE HERE ###########
                                        html.Td(
                                            html.Div("Data-Selection"),
                                            style={"font-size": "20px"},
                                        ),  # className="config-label"
                                        html.Td(
                                            dcc.Dropdown(
                                                id="view-dropdown",
                                                options=[
                                                    {
                                                        "label": "Total Population",
                                                        "value": "total",
                                                    },
                                                    {
                                                        "label": "Migrating In",
                                                        "value": "in",
                                                    },
                                                    {
                                                        "label": "Stationary",
                                                        "value": "stationary",
                                                    },
                                                    {
                                                        "label": "Migrating Out",
                                                        "value": "out",
                                                    },
                                                    {
                                                        "label": "Net Migration",
                                                        "value": "net",
                                                    },
                                                    {
                                                        "label": "Population with Race",
                                                        "value": "race",
                                                    },
                                                ],
                                                value="total",
                                                searchable=False,
                                                clearable=False,
                                            ),
                                            style={
                                                "width": "10%",
                                                "height": "15px",
                                            },
                                        ),
                                        html.Td(
                                            html.Div(
                                                children=[
                                                    html.Button(
                                                        "Clear All Selections",
                                                        id="clear-all",
                                                        className="reset-button",
                                                    ),
                                                ]
                                            ),
                                            style={
                                                "width": "10%",
                                                "height": "15px",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={"width": "100%", "margin-top": "0px"},
                        ),
                        # Hidden div inside the app that stores the intermediate value
                        html.Div(
                            id="datapoints-state-value",
                            style={"display": "none"},
                        ),
                    ],
                    className="columns pretty_container",
                ),  # className='columns pretty_container', id="config-div"),
            ]
        ),
        ########## End of options bar #######################################
        html.Hr(
            id="line1", style={"border": "1px solid grey", "margin": "0px"}
        ),
        # html.Div( html.Hr(id='line',style={'border': '1px solid red'}) ),
        ##################### Map starts  ###################################
        html.Div(
            children=[
                html.Button(
                    "Clear Selection", id="reset-map", className="reset-button"
                ),
                html.H4(
                    [
                        "Individual Distribution",
                    ],
                    className="container_title",
                ),
                dcc.Graph(
                    id="map-graph",
                    config={"displayModeBar": False},
                    figure=blank_fig(440),
                ),
                # Hidden div inside the app that stores the intermediate value
                html.Div(
                    id="intermediate-state-value", style={"display": "none"}
                ),
            ],
            className="columns pretty_container",
            style={"width": "100%", "margin-right": "0", "height": "66%"},
            id="map-div",
        ),
        html.Hr(
            id="line2", style={"border": "1px solid grey", "margin": "0px"}
        ),
        ################# Bars start #########################
        # Race start
        html.Div(
            children=[
                html.Button(
                    "Clear Selection",
                    id="clear-race",
                    className="reset-button",
                ),
                html.H4(
                    [
                        "Race Distribution",
                    ],
                    className="container_title",
                ),
                dcc.Graph(
                    id="race-histogram",
                    config={"displayModeBar": False},
                    figure=blank_fig(row_heights[2]),
                    animate=False,
                ),
            ],
            className="columns  pretty_container",
            id="race-div",
            style={"width": "33.33%", "height": "20%"},
        ),
        # County top starts
        html.Div(
            children=[
                html.Button(
                    "Clear Selection",
                    id="clear-county-top",
                    className="reset-button",
                ),
                html.H4(
                    [
                        "County-wise Top 15",
                    ],
                    className="container_title",
                ),
                dcc.Graph(
                    id="county-histogram-top",
                    config={"displayModeBar": False},
                    figure=blank_fig(row_heights[2]),
                    animate=False,
                ),
            ],
            className="columns  pretty_container",
            id="county-div-top",
            style={"width": "33.33%", "height": "20%"},
        ),
        # County bottom starts
        html.Div(
            children=[
                html.Button(
                    "Clear Selection",
                    id="clear-county-bottom",
                    className="reset-button",
                ),
                html.H4(
                    [
                        "County-wise Bottom 15",
                    ],
                    className="container_title",
                ),
                dcc.Graph(
                    id="county-histogram-bottom",
                    config={"displayModeBar": False},
                    figure=blank_fig(row_heights[2]),
                    animate=False,
                ),
            ],
            className="columns  pretty_container",
            id="county-div-bottom",
            style={"width": "33.33%", "height": "20%"},
        ),
        ############## End of  Bars #####################
        html.Hr(
            id="line3", style={"border": "1px solid grey", "margin": "0px"}
        ),
        html.Div(
            [
                html.H4(
                    "Acknowledgements and Data Sources",
                    style={"margin-top": "0"},
                ),
                dcc.Markdown(
                    """\
**Important Data Caveats:** Geospatially filtered data will show accurate distribution, but due to anonymized, multiple cross filtered distributions will not return meaningful results. See [FAQ](https://github.com/rapidsai/plotly-dash-rapids-census-demo/tree/master#faq-and-known-issues) fore details.
- 2010 Population Census and 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org/) ( not for redistribution ).
- Base map layer provided by [Mapbox](https://www.mapbox.com/).
- Dashboard developed with [Plotly Dash](https://plotly.com/dash/).
- Geospatial point rendering developed with [Datashader](https://datashader.org/).
- GPU toggle accelerated with [RAPIDS cudf](https://rapids.ai/) and [cupy](https://cupy.chainer.org/), CPU toggle with [pandas](https://pandas.pydata.org/).
- For source code and data workflow, visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo/tree/master).
"""
                ),
            ],
            style={"width": "100%"},
            className="columns pretty_container",
        ),
    ]
)

# Clear/reset button callbacks


@app.callback(
    Output("map-graph", "selectedData"),
    [Input("reset-map", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_map(*args):
    return None


@app.callback(
    Output("race-histogram", "selectedData"),
    [Input("clear-race", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_race_hist_selections(*args):
    return None


@app.callback(
    Output("county-histogram-top", "selectedData"),
    [Input("clear-county-top", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_county_hist_top_selections(*args):
    return None


@app.callback(
    Output("county-histogram-bottom", "selectedData"),
    [Input("clear-county-bottom", "n_clicks"), Input("clear-all", "n_clicks")],
)
def clear_county_hist_bottom_selections(*args):
    return None


# # Query string helpers

# def register_update_plots_callback(client):
#     """
#     Register Dash callback that updates all plots in response to selection events
#     Args:
#         df_d: Dask.delayed pandas or cudf DataFrame
#     """
@app.callback(
    [
        Output("indicator-graph", "figure"),
        Output("map-graph", "figure"),
        Output("map-graph", "config"),
        # Output("county-histogram-top", "figure"),
        # Output("county-histogram-top", "config"),
        # Output("county-histogram-bottom", "figure"),
        # Output("county-histogram-bottom", "config"),
        # Output("race-histogram", "figure"),
        # Output("race-histogram", "config"),
        Output("intermediate-state-value", "children"),
    ],
    [
        Input("map-graph", "relayoutData"),
        Input("map-graph", "selectedData"),
        # Input("race-histogram", "selectedData"),
        # Input("county-histogram-top", "selectedData"),
        # Input("county-histogram-bottom", "selectedData"),
        Input("view-dropdown", "value"),
        Input("gpu-toggle", "on"),
    ],
    [State("intermediate-state-value", "children")],
)
def update_plots(
    relayout_data,
    selected_map,
    # selected_race,
    # selected_county_top,
    # selected_county_bottom,
    view_name,
    gpu_enabled,
    coordinates_backup,
):
    global df, data_3857, data_center_3857, data_4326, data_center_4326

    t0 = time.time()

    if coordinates_backup is not None:
        coordinates_4326_backup, position_backup = coordinates_backup
    else:
        coordinates_4326_backup, position_backup = None, None

    # print(df)

    if gpu_enabled:
        if isinstance(df, pd.DataFrame):
            df = cudf.from_pandas(df)
    else:
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()

    colorscale_name = "Viridis"

    (
        data_3857,
        data_center_3857,
        data_4326,
        data_center_4326,
    ) = set_projection_bounds(df)

    figures = build_updated_figures(
        df,
        relayout_data,
        selected_map,
        # selected_race,
        # selected_county_top,
        # selected_county_bottom,
        colorscale_name,
        data_3857,
        data_center_3857,
        data_4326,
        data_center_4326,
        coordinates_4326_backup,
        position_backup,
        view_name,
    )

    # figures = figures_d.compute()

    (
        datashader_plot,
        # race_histogram,
        # county_top_histogram,
        # county_bottom_histogram,
        n_selected_indicator,
        coordinates_4326_backup,
        position_backup,
    ) = figures

    barchart_config = {
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ],
    }
    compute_time = time.time() - t0
    # print(f"Update time: {compute_time}")
    n_selected_indicator["data"].append(
        {
            "title": {"text": "Query Time"},
            "type": "indicator",
            "value": round(compute_time, 4),
            "domain": {"x": [0.53, 0.61], "y": [0, 0.5]},
            "number": {
                "font": {
                    "color": text_color,
                    "size": "50px",
                },
                "suffix": " seconds",
            },
        }
    )
    return (
        n_selected_indicator,
        datashader_plot,
        {
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "zoomInMapbox",
                "zoomOutMapbox",
                "toggleHover",
            ],
        },
        # race_histogram,
        # barchart_config,
        # county_top_histogram,
        # barchart_config,
        # county_bottom_histogram,
        # barchart_config,
        (coordinates_4326_backup, position_backup),
    )


if __name__ == '__main__':
    # development entry point
    # publish_dataset_to_cluster()

    # Launch dashboard
    app.run_server(
        debug=False, dev_tools_silence_routes_logging=True, host='0.0.0.0')
