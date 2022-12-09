import pandas as pd
import cudf
from dash import dcc, html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
import time
import dash_daq as daq
import dash
from dask import delayed
from distributed import Client
from dask_cuda import LocalCUDACluster
from utils.utils import *
import argparse


# ### Dashboards start here
text_color = "#cfd8dc"  # Material blue-grey 100

(
    data_center_3857,
    data_3857,
    data_4326,
    data_center_4326,
    selected_map_backup,
    selected_race_backup,
    selected_county_top_backup,
    selected_county_bt_backup,
    view_name_backup,
    gpu_enabled_backup,
    dragmode_backup,
) = ([], [], [], [], None, None, None, None, None, None, "pan")


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
        html.Hr(id="line1", style={"border": "1px solid grey", "margin": "0px"}),
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
                html.Div(id="intermediate-state-value", style={"display": "none"}),
            ],
            className="columns pretty_container",
            style={"width": "100%", "margin-right": "0", "height": "66%"},
            id="map-div",
        ),
        html.Hr(id="line2", style={"border": "1px solid grey", "margin": "0px"}),
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
        html.Hr(id="line3", style={"border": "1px solid grey", "margin": "0px"}),
        html.Div(
            [
                html.H4(
                    "Acknowledgements and Data Sources",
                    style={"margin-top": "0"},
                ),
                dcc.Markdown(
                    """\
- 2010 Population Census and 2020 Population Census Datasets.
- Base map layer provided by [Mapbox](https://www.mapbox.com/).
- Dashboard developed with [Plotly Dash](https://plotly.com/dash/).
- Geospatial point rendering developed with [Datashader](https://datashader.org/).
- GPU toggle accelerated with [RAPIDS cudf and dask_cudf](https://rapids.ai/) and [cupy](https://cupy.chainer.org/), CPU toggle with [pandas](https://pandas.pydata.org/).
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


def register_update_plots_callback(client):
    """
    Register Dash callback that updates all plots in response to selection events
    Args:
        df_d: Dask.delayed pandas or cudf DataFrame
    """

    @app.callback(
        [
            Output("indicator-graph", "figure"),
            Output("map-graph", "figure"),
            Output("map-graph", "config"),
            Output("map-graph", "relayoutData"),
            Output("county-histogram-top", "figure"),
            Output("county-histogram-top", "config"),
            Output("county-histogram-bottom", "figure"),
            Output("county-histogram-bottom", "config"),
            Output("race-histogram", "figure"),
            Output("race-histogram", "config"),
            Output("intermediate-state-value", "children"),
        ],
        [
            Input("map-graph", "relayoutData"),
            Input("map-graph", "selectedData"),
            Input("race-histogram", "selectedData"),
            Input("county-histogram-top", "selectedData"),
            Input("county-histogram-bottom", "selectedData"),
            Input("view-dropdown", "value"),
            Input("gpu-toggle", "on"),
        ],
        [
            State("intermediate-state-value", "children"),
            State("indicator-graph", "figure"),
            State("map-graph", "figure"),
            State("map-graph", "config"),
            State("map-graph", "relayoutData"),
            State("county-histogram-top", "figure"),
            State("county-histogram-top", "config"),
            State("county-histogram-bottom", "figure"),
            State("county-histogram-bottom", "config"),
            State("race-histogram", "figure"),
            State("race-histogram", "config"),
            State("intermediate-state-value", "children"),
        ],
    )
    def update_plots(
        relayout_data,
        selected_map,
        selected_race,
        selected_county_top,
        selected_county_bottom,
        view_name,
        gpu_enabled,
        coordinates_backup,
        *backup_args,
    ):
        global data_3857, data_center_3857, data_4326, data_center_4326, selected_map_backup, selected_race_backup, selected_county_top_backup, selected_county_bt_backup, view_name_backup, gpu_enabled_backup, dragmode_backup

        # condition to avoid reloading on tool update
        if (
            type(relayout_data) == dict
            and list(relayout_data.keys()) == ["dragmode"]
            and selected_map == selected_map_backup
            and selected_race_backup == selected_race
            and selected_county_top_backup == selected_county_top
            and selected_county_bt_backup == selected_county_bottom
            and view_name_backup == view_name
            and gpu_enabled_backup == gpu_enabled
        ):
            backup_args[1]["layout"]["dragmode"] = relayout_data["dragmode"]
            dragmode_backup = relayout_data["dragmode"]
            return backup_args

        selected_map_backup = selected_map
        selected_race_backup = selected_race
        selected_county_top_backup = selected_county_top
        selected_county_bt_backup = selected_county_bottom
        view_name_backup = view_name
        gpu_enabled_backup = gpu_enabled

        t0 = time.time()

        if coordinates_backup is not None:
            coordinates_4326_backup, position_backup = coordinates_backup
        else:
            coordinates_4326_backup, position_backup = None, None

        # Get delayed dataset from client
        if gpu_enabled:
            df = client.get_dataset("c_df_d")
        else:
            df = client.get_dataset("pd_df_d")

        colorscale_name = "Viridis"

        if data_3857 == []:
            projections = delayed(set_projection_bounds)(df)
            (
                data_3857,
                data_center_3857,
                data_4326,
                data_center_4326,
            ) = projections.compute()

        figures = build_updated_figures_dask(
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
        )

        (
            datashader_plot,
            race_histogram,
            county_top_histogram,
            county_bottom_histogram,
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
        print(f"Query time: {compute_time}")
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
        datashader_plot["layout"]["dragmode"] = (
            relayout_data["dragmode"]
            if "dragmode" in relayout_data
            else dragmode_backup
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
            relayout_data,
            race_histogram,
            barchart_config,
            county_top_histogram,
            barchart_config,
            county_bottom_histogram,
            barchart_config,
            (coordinates_4326_backup, position_backup),
        )


def publish_dataset_to_cluster(cuda_visible_devices):

    census_data_url = "https://rapidsai-data.s3.us-east-2.amazonaws.com/viz-data/total_population_dataset.parquet"
    data_path = "../data/total_population_dataset.parquet"
    check_dataset(census_data_url, data_path)

    # Note: The creation of a Dask LocalCluster must happen inside the `__main__` block,
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=cuda_visible_devices)
    client = Client(cluster)
    print(f"Dask status: {cluster.dashboard_link}")

    # Load dataset and persist dataset on cluster
    def load_and_publish_dataset():
        # dask_cudf DataFrame
        c_df_d = load_dataset(data_path, "dask_cudf").persist()
        # pandas DataFrame
        pd_df_d = load_dataset(data_path, "dask").persist()

        # Unpublish datasets if present
        for ds_name in ["pd_df_d", "c_df_d"]:
            if ds_name in client.datasets:
                client.unpublish_dataset(ds_name)

        # Publish datasets to the cluster
        client.publish_dataset(pd_df_d=pd_df_d)
        client.publish_dataset(c_df_d=c_df_d)

    load_and_publish_dataset()

    # Precompute field bounds
    c_df_d = client.get_dataset("c_df_d")

    # Register top-level callback that updates plots
    register_update_plots_callback(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda_visible_devices",
        help="supply the value of CUDA_VISIBLE_DEVICES as a comma separated string (e.g: --cuda_visible_devices=0,1)",
        default="0,1",
    )

    args = parser.parse_args()
    # development entry point
    publish_dataset_to_cluster(args.cuda_visible_devices)

    # Launch dashboard
    app.run_server(debug=False, dev_tools_silence_routes_logging=True, host="0.0.0.0")
