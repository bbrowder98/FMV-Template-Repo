import os
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession
import datetime
import sys
import pandas
from collections import Counter
from functools import reduce

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()

def derive_and_bucket_telemetry(df):
    '''Derives frame telemetry (e.g., time of day, look angle, etc.) into new columns. Additionally buckets those
    values based on set ranges in the telemetry_utils module and creates new columns for those as well.'''
    df = df.withColumn("time_of_day", time_of_day_bucket(F.col("solar_ned_phi")))
    df = df.withColumn("look_angle_deg", look_angle())
    df = df.withColumn("look_angle", look_angle_legible_bucket(F.col("look_angle_deg")))
    df = df.withColumn("look_angle_legacy", look_angle_bucket(F.col("look_angle_deg")))
    return df


def derive_and_bucket_gsd(df):
    '''Derives ground sample distancing information from the metadata and creates columns for vertical and horizontal
    top, center, and bottom planes. Then, buckets them on fixed ranges.'''
    df = (
        df
        .withColumn("gsd_v_top", ires_to_gsd__v("top_middle_vert_ires_ft"))
        .withColumn("gsd_h_top", ires_to_gsd__h("top_middle_horz_ires_ft"))
        .withColumn("gsd_v_center", ires_to_gsd__v("center_vert_ires_ft"))
        .withColumn("gsd_h_center", ires_to_gsd__h("center_horz_ires_ft"))
        .withColumn("gsd_v_bottom", ires_to_gsd__v("bottom_middle_vert_ires_ft"))
        .withColumn("gsd_h_bottom", ires_to_gsd__h("bottom_middle_horz_ires_ft"))
        .withColumn("gsd", F.col("center_horz_ires_ft") * F.lit(12.0))
    )
    return df


def compute_modes_and_medians(df):
    '''Goes through the dataframe and aggregates individual frames by sequence ID. Numerical columns are aggregated by
    their median and non-numerical columns are aggregated by their mode.'''
    # force the resolution columns to aggregate by mode instead of mean by casting them as strings, then uncasting them
    df = df.withColumn("height_pix", F.col("height_pix").cast("string"))
    df = df.withColumn("width_pix", F.col("width_pix").cast("string"))

    mode_cols = []
    median_cols = []
    for col in df.drop("sequence_id", "sequence_frame").schema.fields:
        col_type = col.dataType.typeName()
        if col_type == "string":
            mode_cols.append(col.name)
        if col_type in ("float", "double"):
            median_cols.append(col.name)

    aggs = [F.count("*").alias("frame_count")]
    aggs += [mode(F.collect_list(col)).alias(col) for col in mode_cols]
    aggs += [median(F.collect_list(col)).alias(col) for col in median_cols]

    df = df.groupBy("sequence_id").agg(*aggs)
    df = df.withColumn("height_pix", F.col("height_pix").cast("integer"))
    df = df.withColumn("width_pix", F.col("width_pix").cast("integer"))
    return df


def metadata_quality(col_names):
    '''Compute the "quality" of the metadata per sequence by computing the ratio of null column values to the total
    number of columns provided in col_names.'''
    metadata_quality = [F.col(col).isNotNull().cast("integer").cast("double") / len(col_names) for col in col_names]
    return F.round(reduce((lambda a, b: a+b), metadata_quality), 2)


@F.udf(returnType=T.StringType())
def mode(li):
    co = Counter(_ for _ in li if _ not in (None, '', 'Missing'))
    return None if not co else co.most_common(1)[0][0]


@F.udf(returnType=T.DoubleType())
def median(li):
    vals = sorted(_ for _ in li if _ not in (None, 0, 0.0))
    return vals[int(len(vals)/2)] if vals else None

def ires_to_gsd__h(ires):
    if isinstance(ires, str):
        ires = F.col(ires)
    return F.when(ires.isNull(), 'Missing') \
            .when(ires < 0.0, '-') \
            .when(ires < 0.0208, F.lit('hGSD0.25')) \
            .when(ires < 0.0416, F.lit('hGSD0.5')) \
            .when(ires < 0.0833, F.lit('hGSD1')) \
            .when(ires < 0.1667, F.lit('hGSD2')) \
            .when(ires < 0.3333, F.lit('hGSD4')) \
            .when(ires < 0.6667, F.lit('hGSD8')) \
            .when(ires < 1.3333, F.lit('hGSD16')) \
            .when(ires < 2.6667, F.lit('hGSD32')) \
            .otherwise(F.lit('hGSDinf'))


def ires_to_gsd__v(ires):
    if isinstance(ires, str):
        ires = F.col(ires)
    return F.when(ires.isNull(), 'Missing') \
            .when(ires < 0.0, '-') \
            .when(ires < 0.0208, F.lit('vGSD0.25')) \
            .when(ires < 0.0416, F.lit('vGSD0.5')) \
            .when(ires < 0.0833, F.lit('vGSD1')) \
            .when(ires < 0.1667, F.lit('vGSD2')) \
            .when(ires < 0.3333, F.lit('vGSD4')) \
            .when(ires < 0.6667, F.lit('vGSD8')) \
            .when(ires < 1.3333, F.lit('vGSD16')) \
            .when(ires < 2.6667, F.lit('vGSD32')) \
            .otherwise(F.lit('vGSDinf'))


def time_of_day_bucket(phi):
    phi = F.col(phi) if isinstance(phi, str) else phi
    return F.when(phi.isNull(), 'Missing') \
            .when(phi < 0.0, '-') \
            .when(phi < 41.25, 'Night1') \
            .when(phi < 82.5, 'Night2') \
            .when(phi < 90, 'Twilight') \
            .when(phi < 97.5, 'Golden Hour') \
            .when(phi < 138.75, 'Day1') \
            .when(phi < 180.0, 'Day2') \
            .otherwise('-')


def look_angle():
    import math
    eb = F.col('bottom_middle_elevation_angle_deg')
    ec = F.col('center_elevation_angle_deg')
    Sb = F.col('bottom_middle_slant_range_nmi')
    Sc = F.col('center_slant_range_nmi')

    rads_per_deg = math.pi / 180.0
    spread_rads = (ec-eb) * rads_per_deg
    phi_rads = F.lit(math.pi/2) - F.asin(Sb * F.sin(spread_rads) / F.sqrt(Sb*Sb + Sc*Sc - 2*Sb*Sc * F.cos(spread_rads)))
    return phi_rads / rads_per_deg


def look_angle_legible_bucket(phi):
    phi = F.col(phi) if isinstance(phi, str) else phi
    return F.when(phi.isNull(), 'Missing') \
            .when(phi < 0.0, 'Missing') \
            .when(phi < 15, '00-15') \
            .when(phi < 30, '15-30') \
            .when(phi < 45, '30-45') \
            .when(phi < 60, '45-60') \
            .when(phi < 75, '60-75') \
            .when(phi < 90, '75-90') \
            .otherwise('Missing')


def look_angle_bucket(phi):
    phi = F.col(phi) if isinstance(phi, str) else phi
    return F.when(phi.isNull(), 'Missing') \
            .when(phi < 0.0, 'Missing') \
            .when(phi < 15, 'LA1') \
            .when(phi < 30, 'LA2') \
            .when(phi < 45, 'LA3') \
            .when(phi < 60, 'LA4') \
            .when(phi < 75, 'LA5') \
            .when(phi < 90, 'LA6') \
            .otherwise('Missing')

df = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mapp_metadata_by_frame.csv')
frames_df = spark.createDataFrame(df)
frames_df = frames_df.replace({"SLT_NONE": ""})
frames_df = frames_df.transform(derive_and_bucket_telemetry)
frames_df = frames_df.transform(derive_and_bucket_gsd)
frames_df = frames_df.transform(compute_modes_and_medians)

METADATA_COLUMNS = ["sensor_band", "gsd", "look_angle_deg", "solar_ned_phi", "platform_heading_angle"]
frames_df = frames_df.withColumn("metadata_quality", metadata_quality(METADATA_COLUMNS))
frames_df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mapp_metadata_parsed.csv', index=False)