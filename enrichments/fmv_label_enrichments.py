import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import sys
import pandas
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("test").master("local[1]").config("spark.driver.memory", "15g").getOrCreate()
spark_sequences = SparkSession.builder.appName("ontology").master("local[2]").getOrCreate()

def annotation_struct():
    '''Creates a new StructType object that contains labeling data.'''
    return F.struct(
        F.lit(None).cast("string").alias("annotated_by"),
        F.lit("visible").alias("visibility"),
        F.col("type"),
        F.col("x"),
        F.col("y"),
        F.col("object_iri"),
        F.col("category").alias("object_category"),
        F.col("category_hierarchy").alias("object_category_hierarchy"),
        F.col("height"),
        F.col("width"),
        F.col("label_id").alias("id"),
        F.lit(None).cast("string").alias("is_visible"),
        F.lit(None).cast("string").alias("is_truncated")
    )


def group_labels_by_sequence(df):
    '''Groups labeling data by sequence and returns a new dataframe grouped by sequence ID and columns for all distinct
    labeled object categories collected together as a list and the total number of labels.'''
    df = df.select("label_id", "sequence_id", "annotations", "category",).distinct()
    df = (
        df
        .groupBy("sequence_id")
        .agg(
            F.collect_set("category").alias("categories"),
            F.collect_list("annotations").alias("annotations"),
            F.count("label_id").alias("label_count")
        )
    )
    return df


def labeled_objects(categories_col):
    '''Returns the number of distinct, labeled objects found in a sequence.'''
    # we wrap the size function in a when clause, since an empty list will return a size of -1.
    return F.when(categories_col.isNotNull(), F.size(categories_col)).otherwise(F.lit(None))


def rename_label_report_columns(df):
    '''Renames some columns from the label reports dataframe to prevent collisions or clarify their source once they
    are joined to the main sequences dataframe.'''
    return df.withColumn("label_filepath", F.col("filepath")) \
        .withColumn("label_filesize_bytes", F.col("filesize_bytes")) \
        .withColumn("label_ingest_date", F.col("ingest_date")) \
        .withColumn("label_legacy", F.col("legacy")) \
        .select(
        "sequence_id",
        "label_filepath",
        "label_filesize_bytes",
        "label_version",
        "label_ingest_date",
        "label_legacy",
        "label_source"
    )


def is_labeled(col):
    '''Check any of the columns that originated from the label reports dataframe and create a BooleanType value that
    describes whether or not the sequence has labeling data available or not.'''
    return F.when(col.isNotNull(), F.lit(True)).otherwise(F.lit(False))


def add_label_size_to_filesize(df):
    '''Adds the filesize of the originating label file to the running filesize count for each sequence and drops it
    from the dataframe.'''
    df = df.withColumn("total_filesize_bytes", F.col("total_filesize_bytes") + F.col("label_filesize_bytes"))
    df = df.drop("label_filesize_bytes")
    return df

df = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\labeling\template_fmv_cdao_labels.csv')
sequences = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\enrichments\template_fmv_sequences_split.csv')
labels_df = spark.createDataFrame(df)
sequences_df = spark_sequences.createDataFrame(sequences)
labels_df = labels_df.withColumn("annotations", annotation_struct())
grouped_labels_df = labels_df.transform(group_labels_by_sequence)
sequences_df = sequences_df.join(grouped_labels_df, "sequence_id", "left")
sequences_df = sequences_df.withColumn("labeled_objects", labeled_objects(F.col("categories")))

label_reports_df = labels_df.transform(rename_label_report_columns)
sequences_df = sequences_df.join(label_reports_df, "sequence_id", "left")
sequences_df = sequences_df.withColumn("is_labeled", is_labeled(F.col("label_version")))
sequences_df = sequences_df.transform(add_label_size_to_filesize)

#saved to json in order to avoid csv cell limit for large json columns
sequences_df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\enrichments\template_fmv_sequences_full.json', index=False)
sequences_df = spark.createDataFrame(df)