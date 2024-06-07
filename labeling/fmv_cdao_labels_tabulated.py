import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import datetime
import sys
import pandas
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("test").master("local[1]").config("spark.driver.memory", "8g").config("spark.driver.memory", "15g").getOrCreate()

def to_timestamp(time_col):
    '''Converts the value of the modified column into a proper Timestamp column.'''
    return F.to_timestamp(F.from_unixtime(time_col / 1000, format="yyyy-MM-dd HH:mm:ss"))

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

def rename_columns(df):
    '''Renames columns that originated from the nested JSON files.'''
    return df.select(
        F.col("filepath"),
        F.col("modified"),
        F.col("filesize_bytes"),
        F.col("src_json"),
        F.col("version"),
        F.col("releaseName").alias("release_name"),
        F.col("releaseVersion").alias("release_version"),
        F.col("projectId").alias("project_id"),
        F.col("projectName").alias("project_name"),
        F.col("datasetId").alias("dataset_id"),
        F.col("datasetName").alias("dataset_name"),
        F.col("applicationName").alias("application_name"),
        F.col("applicationVersion").alias("application_version"),
        F.col("modelName").alias("model_name"),
        F.col("modelVersion").alias("model_version"),
        F.col("downloadedDate").alias("download_date"),
        F.col("traits"),
        F.col("data")
    )

def convert_json_to_struct(df, json_col_name):
    '''Converts a column containing a JSON string into a StructType column and returns it as a PySpark column object.'''
    json_schema = SparkSession.builder.appName("json").master("local[2]").getOrCreate().read.json(df.rdd.map(lambda row: row[json_col_name])).schema
    return F.from_json(F.col(json_col_name), json_schema)


def expand_json_column(df, json_col_name, *nested_keys):
    '''Expands StringType column containing formatted JSON data into columnar data. Note that all of the resulting
    columns will be StringType. Can also optionally pass in the names of keys whose values are nested keys, which will
    ensure that they are also flattened into individual columns.'''
    df = df.withColumn("struct_json", convert_json_to_struct(df, json_col_name))
    df = df.select("*", "struct_json.*")
    for col_name in nested_keys:
        df = df.select("*", f"{col_name}.*")
        df = df.drop(col_name)
    df = df.drop("struct_json")
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

@F.udf(T.StringType())
def json_to_string(arr):
    return json.dumps(arr)


SCHEMA = T.StructType([
    T.StructField("filepath", T.StringType(), False),
    T.StructField("modified", T.StringType(), False),
    T.StructField("filesize_bytes", T.IntegerType(), False),
    T.StructField("src_json", T.StringType(), True)
])

folder = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\raw\Avalanche_USCG_20220930_json'
output_path = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels_tabulated.csv'
incremental = os.path.isfile(output_path)
directory = os.listdir(folder)
if incremental == True:
    output_spark = SparkSession.builder.appName("output").master("local[2]").getOrCreate()
    output = pandas.read_csv(output_path)
    output_df = spark.createDataFrame(output)
    output_list = output_df.withColumn("dataset_name", F.concat(F.col("dataset_name"), F.lit(".json"))).select('dataset_name').toPandas()['dataset_name'].tolist()
    directory = [x for x in directory if x not in output_list]
    print(directory)

rows = []
for name in directory:
    # Open file
    file_path = os.path.join(folder, name)
    with open(file_path, encoding="utf-8") as json_file:
        json_data = json_file.read()
    # file modification timestamp of a file
    m_time = os.path.getmtime(file_path)
    size = os.path.getsize(file_path)
    # convert timestamp into DateTime object
    modified = datetime.datetime.fromtimestamp(m_time)
    row_contents = [file_path, str(modified), int(size), json_data]
    rows.append(row_contents)

labels_df = spark.createDataFrame(rows, SCHEMA)
if directory != []:
    labels_df = expand_json_column(labels_df, "src_json")
    labels_df = labels_df.transform(rename_columns)
    labels_df = labels_df.drop("src_json")  # drop already parsed columns
    labels_df = labels_df.transform(explode_by_sequence)
    labels_df = labels_df.transform(get_versioning_info)
    labels_df = labels_df.transform(keep_only_latest_label)
    labels_df = labels_df.drop("latest_label_version")  # now redundant
    labels_df = labels_df.withColumn("ingest_date", to_timestamp(F.col("modified")))
    labels_df = labels_df.drop("modified")
    labels_df = labels_df.withColumn("labeling_fps", F.lit(30))
    labels_df = labels_df.withColumn("legacy", F.lit(False))
    labels_df = labels_df.withColumn("label_source", F.lit("Anno.Ai"))
    labels_df = labels_df.transform(explode_by_object)
    labels_df = labels_df.transform(explode_by_label)
    labels_df = labels_df.select(
            primary_key(F.col("sequence_id"), F.col("frame")).alias("primary_key"),
            F.element_at(F.col("labels.id"), 1).alias("label_id"),
            F.element_at(F.col("labels.name"), 1).alias("category"),
            F.element_at(F.col("labels.key"), 1).alias("object_iri"),
            "*"
        )
    labels_df = labels_df.drop("labels")
    labels_df = labels_df.transform(explode_coordinates)
    labels_df = labels_df.withColumn("traits", F.to_json(F.col("traits")))
if incremental == True:
    labels_df = output_df.unionByName(labels_df)
labels_df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels_tabulated.csv', index=False)