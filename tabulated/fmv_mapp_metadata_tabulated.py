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
    T.StructField("src_json", T.StringType(), True)
])

folder = 'C:\\Users\\benedict.browder\\Desktop\\FMV Data Processing\\raw\\template_fmv_cdao_mapp_metadata'
output_path = r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\fmv_mapp_metadata_tabulated.json'
incremental = os.path.isfile(output_path)
directory = os.listdir(folder)

#Comment out incremental in order to run through all files
if incremental == True:
    output_spark = SparkSession.builder.appName("output").master("local[2]").getOrCreate()
    output = pandas.read_csv(output_path)
    output_df = spark.createDataFrame(output)
    output_list = output_df.withColumn("sequence_id", F.concat(F.col("sequence_id"), F.lit(".json"))).select('sequence_id').toPandas()['sequence_id'].tolist()
    directory = [x for x in directory if x not in output_list]

rows = []
for name in directory:
    # Open file
    file_path = os.path.join(folder, name)
    file_path = winapi_path(file_path)
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

#Comment out incremental in order to run through all files
if incremental == True:
    df = output_df.unionByName(df)
df = df.orderBy(df.sequence_id.desc(), df.modified.desc())
#saved to json in order to avoid csv cell limit for large json columns
df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\fmv_mapp_metadata_tabulated.json', index=False)