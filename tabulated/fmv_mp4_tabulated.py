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
    T.StructField("md5_hash", T.StringType(), True),
    T.StructField("batch_path", T.StringType(), False),
])

folder = r'C:\Users\ecs\Desktop\FMV Data Processing\raw\AI-DP\FMV'
output_path = r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv'
incremental = os.path.isfile(output_path)
directory = os.listdir(folder)

#To run through all files uncomment incremental == False
#incremental = False


if incremental == True:
    #Creates spark session for output dataset
    output_spark = SparkSession.builder.appName("output").master("local[2]").getOrCreate()
    output = pandas.read_csv(output_path)
    output_df = output_spark.createDataFrame(output)
    output_list = output_df.withColumn("batch_path", F.concat(F.col("batch_path"))).select('batch_path').toPandas()['batch_path'].tolist()
    directory = [x for x in directory if x not in output_list]

rows = []
for batch in directory:
    batch_name = batch
    #Identifies batch filepath
    batch_path = os.path.join(folder, batch)
    #Identifies batch directory
    batch = os.listdir(batch_path)
    #Runs through each image
    for files in batch:
        #Identifies image filepath
        image_path = os.path.join(batch_path, files, "seq_mp4")
        #Identifies image directory
        image = os.listdir(image_path)[0]
        #Identifies mp4 filepath
        file_path = os.path.join(image_path, image)
        with open(file_path, "rb") as mp4_file:
            md5_hash = str(hashlib.md5(mp4_file.read()).hexdigest())
        # file modification timestamp of a file
        m_time = os.path.getmtime(file_path)
        size = os.path.getsize(file_path)
        # convert timestamp into DateTime object
        modified = datetime.datetime.fromtimestamp(m_time)
        sequence_id = get_sequence_id(file_path)
        row_contents = [sequence_id, file_path, str(modified), int(size), md5_hash, batch_path]
        rows.append(row_contents)

df = spark.createDataFrame(rows, SCHEMA)
if incremental == True:
    df = output_df.unionByName(df)
df = df.orderBy(df.modified.desc(), df.sequence_id)
df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv', index=False)