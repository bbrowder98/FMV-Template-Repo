import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import sys
import pandas
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("test").master("local[1]").config("spark.driver.memory", "15g").getOrCreate()

def to_timestamp(time_col):
    '''Converts the value of the modified column into a proper Timestamp column.'''
    return F.to_timestamp(F.from_unixtime(time_col / 1000, format="yyyy-MM-dd HH:mm:ss"))


def explode_by_object(df):
    '''Reorders the dataframe so that each row represents a unique labeled object, alongside its type and
    position in the scene.'''
    df = df.withColumn("objects", F.explode(F.col("objects")))
    df = df.select(
        "*",
        F.col("objects.id").alias("object_id"),
        F.col("objects.labels").alias("labels"),
        F.col("objects.tracks").alias("tracks")
    )
    df = df.drop("objects")
    return df


def explode_by_label(df):
    '''Reorders the dataframe so that each row represenets an individual frame of a label.'''
    df = df.withColumn("tracks", F.explode(F.col("tracks")))
    df = df.select("*", F.col("tracks.annotations").alias("annotations"))
    df = df.withColumn("annotations", F.explode(F.col("annotations")))
    df = df.select(
        "*",
        F.col("annotations.type").alias("type"),
        F.col("annotations.position").alias("position"),
        F.col("annotations.coords").alias("coords")
    )
    df = df.drop("annotations", "tracks")
    return df


def get_versioning_info(df):
    '''Assigns label versions for duplicate labels on the same sequence ID and marks the latest label version
    (i.e., the duplicate label with the most recent download date).'''
    label_version = F.row_number().over(Window.partitionBy(F.col("sequence_id")).orderBy(F.col("download_date")))
    df = df.withColumn("label_version", label_version)

    latest_label_version = F.max(F.col("label_version")).over(Window.partitionBy(F.col("sequence_id")))
    df = df.withColumn("latest_label_version", latest_label_version)
    return df


def keep_only_latest_label(df):
    '''De-duplicates labels by only keeping sequences' latest label version.'''
    df = df.where(F.col("label_version") == F.col("latest_label_version"))
    return df

def explode_by_sequence(df):
    '''Reorders the dataframe so that each row represents a sequence and its associated labeled objects.'''
    df = df.withColumn("data", F.explode(F.col("data")))
    df = df.select(
        "*",
        F.col("data.id").alias("task_id"),
        F.col("data.name").alias("filename"),
        F.col("data.url").alias("url"),
        F.col("data.objectOccurrences").alias("objects")
    )
    df = df.drop("data")
    df = df.select(F.regexp_replace(F.col("filename"), "\\.mp4", "").alias("sequence_id"), "*")
    return df

@F.udf(T.ArrayType(T.StringType()))
def load_array(array):
    return json.loads(array)

df = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels_tabulated.json')
labels_df = spark.createDataFrame(df)
labels_df = labels_df.transform(get_versioning_info)
labels_df = labels_df.transform(keep_only_latest_label)
labels_df = labels_df.drop("latest_label_version")  # now redundant
labels_df = labels_df.withColumn("ingest_date", to_timestamp(F.col("modified")))
labels_df = labels_df.drop("modified")
labels_df = labels_df.withColumn("labeling_fps", F.lit(30))
labels_df = labels_df.withColumn("legacy", F.lit(False))
labels_df = labels_df.withColumn("label_source", F.lit("Anno.Ai"))
labels_df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_label_reports.txt', index=False)