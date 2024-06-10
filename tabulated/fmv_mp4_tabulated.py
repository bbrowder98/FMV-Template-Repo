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

folder = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\raw\template_fmv_cdao_mp4s'
output_path = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv'
incremental = os.path.isfile(output_path)
directory = os.listdir(folder)

#Comment out incremental in order to run through all files
if incremental == True:
    output_spark = SparkSession.builder.appName("output").master("local[2]").getOrCreate()
    output = pandas.read_csv(output_path)
    output_df = spark.createDataFrame(output)
    output_list = output_df.withColumn("sequence_id", F.concat(F.col("sequence_id"), F.lit(".mp4"))).select('sequence_id').toPandas()['sequence_id'].tolist()
    directory = [x for x in directory if x not in output_list]

rows = []
for name in directory:
    # Open file
    file_path = os.path.join(folder, name)
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

#Comment out incremental in order to run through all files
if incremental == True:
    df = output_df.unionByName(df)
df = df.orderBy(df.modified.desc(), df.sequence_id)
df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv', index=False)