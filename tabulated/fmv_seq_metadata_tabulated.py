import os
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession
import datetime
import sys
import pandas

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()


def get_sequence_id(filepath):
    #Parses out the sequence_id from the filepath of the raw file.
    return str(os.path.splitext(os.path.basename(filepath))[0])

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

SCHEMA = T.StructType([
    T.StructField("sequence_id", T.StringType(), False),
    T.StructField("filepath", T.StringType(), False),
    T.StructField("modified", T.StringType(), False),
    T.StructField("filesize_bytes", T.IntegerType(), False),
    T.StructField("src_json", T.StringType(), True)
])

folder = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\raw\template_fmv_cdao_seq_metadata'
output_path = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv'
incremental = os.path.isfile(output_path)
directory = os.listdir(folder)

if incremental == True:
    output_spark = SparkSession.builder.appName("output").master("local[2]").getOrCreate()
    output = pandas.read_csv(output_path)
    output_df = spark.createDataFrame(output)
    output_list = output_df.withColumn("sequence_id", F.concat(F.col("sequence_id"), F.lit(".json"))).select('sequence_id').toPandas()['sequence_id'].tolist()
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
    sequence_id = get_sequence_id(file_path)
    row_contents = [sequence_id, file_path, str(modified), int(size), json_data]
    rows.append(row_contents)

df = spark.createDataFrame(rows, SCHEMA)
df = expand_json_column(df, "src_json", "metadata", "src_metadata")
df = df.withColumnRenamed("heigth", "height")
df = df.orderBy(df.modified.desc(), df.sequence_id)
df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_seq_metadata_tabulated.csv', index=False)