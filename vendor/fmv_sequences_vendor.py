import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import sys
import pandas
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#adjust memory if needed
spark = SparkSession.builder.appName("test").master("local[1]").config("spark.driver.memory", "15g").getOrCreate()

df = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\enrichments\template_fmv_sequences_full.csv', dtype=str, keep_default_na=False)
sequences_df = spark.createDataFrame(df)
VENDOR_ONLY_SPLITS = ["training", "validation"]
sequences_df = sequences_df.where(F.col("is_labeled") == True)
sequences_df = sequences_df.where(F.col("split").isin(VENDOR_ONLY_SPLITS))
#Convert to .json file if dictionary value is over csv limit of 32767 characters per cell
sequences_df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_fmv_sequences_vendor.csv', index=False)