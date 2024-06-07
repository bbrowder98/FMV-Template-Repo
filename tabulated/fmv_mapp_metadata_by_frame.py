import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import datetime
import sys
import pandas

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def explode_by_frame(df):
    '''Explodes the dataframe so that each row represents individual frames of a sequence, since mapp_metadata is
    stored on a frame-by-frame basis.'''
    json_schema = F.schema_of_json(df.select(F.col("src_json")).first()[0])  # noqa
    df = df.withColumn("src_json", F.from_json("src_json", json_schema))
    df = df.withColumn("src_json", F.explode("src_json"))
    return df


def expand_nested_structs(df):
    '''Expands out the inner feed struct and platform struct into their individual columnar data.'''
    df = df.select(
        "*",
        F.col("src_json.seq_frame").alias("sequence_frame").cast("integer"),
        "src_json.all_fields",  # not expanded into individual columns outright, since its keys are inconsistent
        "src_json.feed.*",
        "src_json.platform.*",
        F.col("src_json.all_fields.slant_range").alias("slant_range"),
        F.col("src_json.all_fields.platform_heading_angle").alias("platform_heading_angle"),
        F.col("src_json.all_fields.uas_datalink_ls_version_number").cast("string").alias("uas_datalink_ls_version_number"),  # noqa
        F.col("src_json.all_fields.image_source_sensor").alias("image_source_sensor"),
        F.lit(None).cast("string").alias("area_code"),
        F.col("src_json.all_fields.solar_angles.image_plane.theta_degrees").alias("image_plane_theta"),
        F.col("src_json.all_fields.solar_angles.image_plane.phi_degrees").alias("image_plane_phi"),
        F.col("src_json.all_fields.solar_angles.ned.theta_degrees").alias("solar_ned_theta"),
        F.col("src_json.all_fields.solar_angles.ned.phi_degrees").alias("solar_ned_phi")
    )
    df = df.drop("id")  # remove columns that could cause confusion
    return df


def convert_nans_to_nones(df):
    '''Iterates through all LongType and DoubleType columns and searches for NaN values, then reaplces them with
    null-values to prevent issues downstream.'''
    for col in df.columns:
        if (df.schema[col].dataType in (T.LongType(), T.DoubleType())):
            df = df.withColumn(col, F.when(F.isnan(F.col(col)), None).otherwise(F.col(col)))
    return df


def compute_global_averages(df):
    '''Computes global averages as new columns in the dataframe, which represents average values for all frames across
    all sequences.'''
    COLUMNS = ["altitude_ft", "elevation_angle_degree", "avg_center_ires_ft"]
    grouped_df = df.groupBy().agg(*[F.avg(F.col(col)).alias(f"global_average_{col}") for col in COLUMNS])

    df = df.withColumn("group_on", F.lit("temp"))  # create a temporary column to propagate the grouped df's values
    grouped_df = grouped_df.withColumn("group_on", F.lit("temp"))
    df = df.join(grouped_df, "group_on", "left")
    df = df.drop("group_on")
    return df


def compute_starts_and_ends(df):
    '''Compute start and end frame bounds for specific columns as new columns in the dataframe.'''
    FIRST_FRAME = 1
    LAST_FRAME = 150
    COLUMNS = ["altitude_ft", "slant_range", "elevation_angle_degree", "avg_center_ires_ft"]
    window = Window.partitionBy(F.col("sequence_id"))
    for col in COLUMNS:
        df = (
            df
            .withColumn(f"start_{col}", F.when(F.col("sequence_frame") == FIRST_FRAME, F.col(col)))
            .withColumn(f"end_{col}", F.when(F.col("sequence_frame") == LAST_FRAME, F.col(col)))
            .withColumn(f"start_{col}", F.max(F.col(f"start_{col}")).over(window))
            .withColumn(f"end_{col}", F.max(F.col(f"end_{col}")).over(window))
        )
    return df


def elevation_angle_degree():
    '''Calculates the elevation angle degree.'''
    # return F.col("center_elevation_angle_deg") + F.col("sensor_relative_elevation_angle")
    return F.col("center_elevation_angle_deg")


def avg_center_ires_ft(ires_col1, ires_col2):
    '''Calculates the average center ires using the center horizontal and center vertical ires columns.'''
    return (ires_col1 + ires_col2) / 2

spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()

df = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\fmv_mapp_metadata_tabulated.json')
metadata_df = spark.createDataFrame(df)
metadata_df = metadata_df.drop("modified", "filesize_bytes")
metadata_df = metadata_df.transform(explode_by_frame)
metadata_df = metadata_df.transform(expand_nested_structs)
metadata_df = metadata_df.transform(convert_nans_to_nones)

metadata_df = metadata_df.withColumn("elevation_angle_degree", elevation_angle_degree())
metadata_df = metadata_df.withColumn("avg_center_ires_ft", avg_center_ires_ft(F.col("center_horz_ires_ft"), F.col("center_vert_ires_ft")))  # noqa
metadata_df = metadata_df.transform(compute_starts_and_ends)
metadata_df = metadata_df.transform(compute_global_averages)
metadata_df = metadata_df.withColumn("src_json", F.to_json(F.col("src_json"))) \
                        .withColumn("all_fields", F.to_json(F.col("all_fields")))                        
metadata_df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mapp_metadata_by_frame.csv', index=False)

