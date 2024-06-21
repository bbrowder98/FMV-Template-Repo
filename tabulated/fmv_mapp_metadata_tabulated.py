import os
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession
import datetime
import sys
import pandas

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def winapi_path(dos_path, encoding=None):
    if (not isinstance(dos_path, str) and encoding is not None): 
        dos_path = dos_path.decode(encoding)
    path = os.path.abspath(dos_path)
    if path.startswith(u"\\\\"):
        return u"\\\\?\\UNC\\" + path[2:]
    return u"\\\\?\\" + path

spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()


def get_sequence_id(filepath):
    #Parses out the sequence_id from the filepath of the raw file.
    return str(os.path.splitext(os.path.basename(filepath))[0])

SCHEMA = T.StructType([
    T.StructField("sequence_id", T.StringType(), False),
    T.StructField("filepath", T.StringType(), False),
    T.StructField("modified", T.StringType(), False),
    T.StructField("filesize_bytes", T.IntegerType(), False),
    T.StructField("src_json", T.StringType(), True),
    T.StructField("batch_path", T.StringType(), False),
])

folder = r'C:\Users\ecs\Desktop\FMV Data Processing\raw\AI-DP\FMV'
output_path = r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\fmv_mapp_metadata_tabulated.csv'
#Checks if output path exists
incremental = os.path.isfile(output_path)
directory = os.listdir(folder)

#To run through all files uncomment incremental = False
incremental = False

if incremental == True:
    output_spark = SparkSession.builder.appName("output").master("local[2]").getOrCreate()
    output = pandas.read_csv(output_path)
    output_df = output_spark.createDataFrame(output)
    output_list = output_df.withColumn("batch_path", F.concat(F.col("batch_path"))).select('batch_path').toPandas()['batch_path'].tolist()
    directory = [x for x in directory if x not in output_list]

rows = []
for batch in directory:
    batch_name = batch
    batch_path = os.path.join(folder, batch)
    batch = os.listdir(batch_path)
    for files in batch:
        image_path = os.path.join(batch_path, files, "mapp_metadata")
        image = os.listdir(image_path)[0]
        file_path = os.path.join(image_path, image)
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

if incremental == True:
    df = output_df.unionByName(df)
df = df.orderBy(df.sequence_id.desc(), df.modified.desc())
#Convert to .json file if dictionary value is over csv limit of 32767 characters per cell
df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\fmv_mapp_metadata_tabulated.csv', index=False)