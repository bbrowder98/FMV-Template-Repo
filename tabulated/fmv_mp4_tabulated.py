import os
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession
import datetime
import sys
import pandas
import hashlib

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()

def get_sequence_id(filepath):
    #Parses out the sequence_id from the filepath of the raw file.
    return str(os.path.splitext(os.path.basename(filepath))[0])

SCHEMA = T.StructType([
    T.StructField("sequence_id", T.StringType(), False),
    T.StructField("filepath", T.StringType(), False),
    T.StructField("modified", T.StringType(), False),
    T.StructField("filesize_bytes", T.IntegerType(), False),
    T.StructField("md5_hash", T.StringType(), True)
])

directory = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\raw\template_fmv_cdao_mp4s'
rows = []
for name in os.listdir(directory):
    # Open file
    file_path = os.path.join(directory, name)
    with open(file_path, "rb") as mp4_file:
        md5_hash = str(hashlib.md5(mp4_file.read()).hexdigest())
    # file modification timestamp of a file
    m_time = os.path.getmtime(file_path)
    size = os.path.getsize(file_path)
    # convert timestamp into DateTime object
    modified = datetime.datetime.fromtimestamp(m_time)
    sequence_id = get_sequence_id(file_path)
    row_contents = [sequence_id, file_path, str(modified), int(size), md5_hash]
    rows.append(row_contents)

df = spark.createDataFrame(rows, SCHEMA)
df = df.orderBy(df.modified.desc(), df.sequence_id)
df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv', index=False)