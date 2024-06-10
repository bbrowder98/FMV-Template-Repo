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

df = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\enrichments\template_fmv_sequences_full.json', dtype=str, keep_default_na=False)
sequences_df = spark.createDataFrame(df)
VENDOR_ONLY_SPLITS = ["training", "validation"]
sequences_df = sequences_df.where(F.col("is_labeled") == True)
sequences_df = sequences_df.where(F.col("split").isin(VENDOR_ONLY_SPLITS))
#saved to json in order to avoid csv cell limit for large json columns
sequences_df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\vendor\template_fmv_sequences_vendor.json', index=False)