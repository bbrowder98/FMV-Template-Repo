import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import sys
import pandas
import shutil

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#adjust memory if needed
spark_sequences = SparkSession.builder.appName("sequences").master("local[1]").config("spark.driver.memory", "15g").getOrCreate()
spark_mp4 = SparkSession.builder.appName("mp4").master("local[2]").getOrCreate()
spark_jpeg = SparkSession.builder.appName("jpeg").master("local[3]").getOrCreate()
spark_mapp = SparkSession.builder.appName("mapp_metadata").master("local[4]").getOrCreate()
spark_seq = SparkSession.builder.appName("seq_metadata").master("local[5]").getOrCreate()
df = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_fmv_sequences_vendor.csv')
sequences_df = spark_sequences.createDataFrame(df)
mp4_pd = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv')
mp4_df = spark_mp4.createDataFrame(mp4_pd)
jpeg_pd = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_jpegs_tabulated.csv')
jpegs_df = spark_jpeg.createDataFrame(jpeg_pd)
mapp_pd = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\fmv_mapp_metadata_tabulated.csv')
mapp_df = spark_mapp.createDataFrame(mapp_pd)
seq_pd = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_seq_metadata_tabulated.csv')
seq_df = spark_seq.createDataFrame(seq_pd)

output_path = r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_fmv_vendor_test.csv'

def columns_as_struct(df):
    '''Converts all of the columns of the dataframe as a single struct.'''
    BLACKLIST = ["_filepath", "_rid", "_bytes"]  # remove internal use columns
    return F.to_json(F.struct(*[F.col(col) for col in df.columns if not any(map(col.__contains__, BLACKLIST))]))

@F.udf(T.StringType())
def filename(id_col):
    '''Generates the name of the output JSON file when it is created downstream.'''
    return r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_vendor_data\\' + id_col + ".json"

@F.udf(T.StringType())
def jpeg_filename(id_col):
    '''Generates the name of the output JSON file when it is created downstream.'''
    return r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_vendor_data\\' + id_col + "_jpeg.zip"

@F.udf(T.StringType())
def mapp_filename(id_col):
    '''Generates the name of the output JSON file when it is created downstream.'''
    return r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_vendor_data\\' + id_col + "_mapp_metadata.json"

@F.udf(T.StringType())
def mp4_filename(id_col):
    '''Generates the name of the output JSON file when it is created downstream.'''
    return r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_vendor_data\\' + id_col + ".mp4"

@F.udf(T.StringType())
def seq_filename(id_col):
    '''Generates the name of the output JSON file when it is created downstream.'''
    return r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_vendor_data\\' + id_col + "_seq_metadata.json"

sequences_df = sequences_df.withColumn("output_json", F.encode(columns_as_struct(sequences_df), "UTF-8"))
sequences_df = sequences_df.withColumn("filename", filename(F.col("sequence_id")))

def write_json(row):
    '''Writes the contents of the output_json column as a JSON file.'''
    with open(row.filename, "wb") as json_file:
        json_file.write(row.output_json)

sequences_df.foreach(write_json)
def process_filesystem(file):
        '''Moves the contents of one filesystem to the vendor data filesystem.'''
        try:
            with open(file.filepath, "rb") as fsrc:
                with open(file.filename, "wb") as fdest:
                    shutil.copyfileobj(fsrc, fdest)
        except: #noqa
            pass

filesystems = {
        "mp4s_filepath": mp4_df.withColumn("filename", mp4_filename(F.col("sequence_id"))),
        "jpegs_filepath": jpegs_df.withColumn("filename", jpeg_filename(F.col("sequence_id"))),
        "mapp_metadata_filepath": mapp_df.withColumn("filename", mapp_filename(F.col("sequence_id"))),
        "seq_metadata_filepath": seq_df.withColumn("filename", seq_filename(F.col("sequence_id")))
    }
for file_path, file_system in filesystems.items():
    sequences_df = sequences_df.drop("filename", "filepath")
    sequences_df = sequences_df.join(file_system.select("sequence_id", "filename", "filepath"), "sequence_id", "left")
    sequences_df.foreach(process_filesystem)

#Convert to .json file if dictionary value is over csv limit of 32767 characters per cell
sequences_df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_fmv_vendor_test.csv', index=False)