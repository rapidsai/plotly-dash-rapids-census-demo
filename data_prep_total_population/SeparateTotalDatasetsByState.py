import cudf
import cuspatial
import geopandas as gpd
import os
from shapely.geometry import Polygon, MultiPolygon

DATA_PATH = "../data"
DATA_PATH_STATE = f"{DATA_PATH}/state-wise-population"

# create DATA_PATH if it does not exist
if not os.path.exists(DATA_PATH_STATE):
    os.makedirs(DATA_PATH_STATE)

# Read the total population dataset as a cudf dataframe from the parquet file
df = cudf.read_parquet(f"{DATA_PATH}/total_population_dataset.parquet")

# Read the shapefile as a cuspatial dataframe and get the state names and geometries
# downloaded from https://hub.arcgis.com/datasets/1b02c87f62d24508970dc1a6df80c98e/explore
shapefile_path = f"{DATA_PATH}/States_shapefile/States_shapefile.shp"
states_data = gpd.read_file(shapefile_path)[
    ["State_Code", "State_Name", "geometry"]
].to_crs(3857)

print("Number of states to process: ", len(states_data))
print("Number of points in total population dataset: ", len(df))
print("Processing states with Polygon geometries...")

processed_states = 0
# Loop through the states and get the points in each state and save as a separate dataframe
# process all Polygon geometries in the shapefile
for index, row in states_data.iterrows():
    if isinstance(row["geometry"], MultiPolygon):
        # skip MultiPolygon geometries
        continue

    state_name = row["State_Name"]
    processed_states += 1
    print(
        "Processing state: ",
        state_name,
        " (",
        processed_states,
        "/",
        len(states_data),
        ")",
    )

    if os.path.exists(f"{DATA_PATH_STATE}/{state_name}.parquet"):
        print("State already processed. Skipping...")
        continue

    # process all MultiPolygon geometries in the shapefile
    # Use cuspatial point_in_polygon to get the points in the state from the total population dataset
    state_geometry = cuspatial.GeoSeries(
        gpd.GeoSeries(row["geometry"]), index=["selection"]
    )

    # Loop through the total population dataset in batches of 50 million points to avoid OOM issues
    batch_size = 50_000_000
    points_in_state = cudf.DataFrame({"selection": []})
    for i in range(0, len(df), batch_size):
        # get the batch of points
        batch = df[i : i + batch_size][["easting", "northing"]]
        # convert to GeoSeries
        points = cuspatial.GeoSeries.from_points_xy(
            batch.interleave_columns().astype("float64")
        )
        # get the points in the state from the batch
        points_in_state_current_batch = cuspatial.point_in_polygon(
            points, state_geometry
        )
        # append the points in the state from the batch to the points_in_state dataframe
        points_in_state = cudf.concat([points_in_state, points_in_state_current_batch])
        # free up memory
        del batch

    print(
        f"Number of points in {state_name}: ",
        df[points_in_state["selection"]].shape[0],
    )

    # save the points in the state as a separate dataframe
    df[points_in_state["selection"]].to_parquet(
        f"{DATA_PATH_STATE}/{state_name}.parquet"
    )

print("Processing states with MultiPolygon geometries...")
# process all MultiPolygon geometries in the shapefile
for index, row in states_data.iterrows():
    if isinstance(row["geometry"], Polygon):
        # skip Polygon geometries
        continue

    state_name = row["State_Name"]
    processed_states += 1
    print(
        "Processing state: ",
        state_name,
        " (",
        processed_states,
        "/",
        len(states_data),
        ")",
    )
    if os.path.exists(f"{DATA_PATH_STATE}/{state_name}.parquet"):
        print("State already processed. Skipping...")
        continue

    # process all MultiPolygon geometries in the shapefile
    points_in_state = None
    for polygon in list(row["geometry"].geoms):
        # process each polygon in the MultiPolygon
        state_geometry = cuspatial.GeoSeries(
            gpd.GeoSeries(polygon), index=["selection"]
        )

        # Loop through the total population dataset in batches of 50 million points to avoid OOM issues
        batch_size = 50_000_000
        points_in_state_current_polygon = cudf.DataFrame({"selection": []})
        for i in range(0, len(df), batch_size):
            # get the batch of points
            batch = df[i : i + batch_size][["easting", "northing"]]
            # convert to GeoSeries
            points = cuspatial.GeoSeries.from_points_xy(
                batch.interleave_columns().astype("float64")
            )
            # get the points in the state from the batch
            points_in_state_current_batch = cuspatial.point_in_polygon(
                points, state_geometry
            )
            # append the points in the state from the batch to the points_in_state_current_polygon dataframe
            points_in_state_current_polygon = cudf.concat(
                [points_in_state_current_polygon, points_in_state_current_batch]
            )
            # free up memory
            del batch

        points_in_state = (
            points_in_state_current_polygon
            if points_in_state is None
            else points_in_state | points_in_state_current_polygon
        )

    print(
        f"Number of points in {state_name}: ",
        df[points_in_state["selection"]].shape[0],
    )

    # save the points in the state as a separate dataframe
    df[points_in_state["selection"]].to_parquet(
        f"{DATA_PATH_STATE}/{state_name}.parquet"
    )
