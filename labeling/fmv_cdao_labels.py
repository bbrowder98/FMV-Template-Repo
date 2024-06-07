import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import sys
import pandas
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("test").master("local[1]").config("spark.driver.memory", "15g").getOrCreate()
spark_ontology = SparkSession.builder.appName("ontology").master("local[2]").getOrCreate()

def explode_by_object(df):
    '''Reorders the dataframe so that each row represents a unique labeled object, alongside its type and
    position in the scene.'''
    df = df.withColumn("objects", F.explode(F.col("objects")))
    df = df.select(
        F.col("objects.id").alias("object_id"),
        F.col("objects.labels").alias("labels"),
        F.col("objects.tracks").alias("tracks"),
        "*"
    )
    df = df.drop("objects")
    return df


def explode_by_label(df):
    '''Reorders the dataframe so that each row represents an individual frame of a label.'''
    df = df.withColumn("tracks", F.explode(F.col("tracks")))
    df = df.select("*", F.col("tracks.annotations").alias("annotations"))
    df = df.withColumn("annotations", F.explode(F.col("annotations")))
    df = df.select(
        "*",
        F.col("annotations.type").alias("type"),
        F.col("annotations.position").alias("frame"),
        F.col("annotations.coords").alias("coords")
    )
    df = df.drop("annotations", "tracks")
    return df


def primary_key(*cols):
    '''Creates a label's primary key by concatenating other ID values and the frame of the label in the sequence.'''
    return F.concat_ws("-", *cols)


def explode_coordinates(df):
    '''Explodes the coordinate struct column into individual coordinate columns. Note that depending on the type of
    label, the coordinate values need to be modified.'''
    df = df.select("*", F.col("coords.*"))  # {"points":null,"xMax":592,"xMin":576,"yMax":256,"yMin":250}
    rectangular_df = df.transform(convert_rectangular_coords)
    polygonal_df = df.transform(convert_polygonal_coords)
    return rectangular_df.unionByName(polygonal_df)


def convert_rectangular_coords(df):
    '''Leverages the coordinates of the bounding box to compute its location, width, and height.'''
    rectangular_df = df.where(F.col("type") == "AnnoRect")
    rectangular_df = rectangular_df.select(
        "*",
        F.col("xMin").alias("x"),
        F.col("yMin").alias("y"),
        (F.col("xMax") - F.col("xMin")).alias("width"),
        (F.col("yMax") - F.col("yMin")).alias("height")
    )
    rectangular_df = rectangular_df.drop("xMin", "xMax", "yMin", "yMax", "coords", "points")
    return rectangular_df


def convert_polygonal_coords(df):
    '''The coords of labels with polygonal bounding boxes need to have their coordinates re-computed to be in line with
    rectangular bounding boxes.'''
    init_df = df.where(F.col("type") == "AnnoPolygon")
    polygon_df = init_df.select("primary_key", "points")
    polygon_df = polygon_df.withColumn("points", F.explode(F.col("points")))
    polygon_df = (
        polygon_df
        .groupBy("primary_key")
        .agg(
            F.min("points.x").alias("min_x"),
            F.min("points.y").alias("min_y"),
            F.max("points.x").alias("max_x"),
            F.max("points.y").alias("max_y")
        )
    )
    polygon_df = polygon_df.select(
        "primary_key",
        F.col("min_x").alias("x"),
        F.col("min_y").alias("y"),
        (F.col("max_x") - F.col("min_x")).alias("width"),
        (F.col("max_y") - F.col("min_y")).alias("height")
    )
    init_df = init_df.join(polygon_df, "primary_key", "left")
    init_df = init_df.drop("xMin", "xMax", "yMin", "yMax", "coords", "points")
    return init_df


def add_ontology_fields(df, ontology_df):
    '''Adds ontology hierarchy names and iris from the ontology dataframe.'''
    ontology_df = ontology_df.select(
        F.col("ontology_iri").alias("object_iri"),
        F.col("ontology_hierarchy_names").alias("category_hierarchy"),
        F.col("ontology_hierarchy_iris").alias("object_iri_hierarchy")
    )
    df = df.join(F.broadcast(ontology_df), "object_iri", "left")
    return df

df = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels_tabulated.csv')
ontology_df = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\ontology\ontology.csv')
labels_df = spark.createDataFrame(df)
ontology_df = spark_ontology.createDataFrame(ontology_df)
labels_df = add_ontology_fields(labels_df, ontology_df)

labels_df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels.csv', index=False)