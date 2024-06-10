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
spark_ontology = SparkSession.builder.appName("ontology").master("local[2]").getOrCreate()

def add_ontology_fields(df, ontology_df):
    '''Adds ontology hierarchy names and iris from the ontology dataframe.'''
    ontology_df = ontology_df.select(
        F.col("ontology_iri").alias("object_iri"),
        F.col("ontology_hierarchy_names").alias("category_hierarchy"),
        F.col("ontology_hierarchy_iris").alias("object_iri_hierarchy")
    )
    df = df.join(F.broadcast(ontology_df), "object_iri", "left")
    return df

df = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels_tabulated.csv')
ontology_df = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\ontology\ontology.csv')
labels_df = spark.createDataFrame(df)
ontology_df = spark_ontology.createDataFrame(ontology_df)
labels_df = add_ontology_fields(labels_df, ontology_df)

labels_df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels.csv', index=False)