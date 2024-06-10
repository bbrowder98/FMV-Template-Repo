import os
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession
import sys
import pandas

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def split_key(srk_col, batch_col):
    '''Generates the split_key for the sequence. If the sequence does not have a source record key assigned to it,
    fallback to the batch name.'''
    return F.coalesce(srk_col, F.concat(F.lit("Batch:"), batch_col))


def split_key_hash(split_key_col):
    '''Uses the split key as an input to a hashing function and returns the output. This also ensures that we are
    splitting based on the partition of each src_record_key.'''
    return (F.abs(F.hash(split_key_col)) % 1000).cast("double")/1000.0


def split(hash_col):
    '''Assigns each sequence a split based on the sequences' split key's random hash.'''
    TRAINING_THRESHOLD = 0.8
    TEST_THRESHOLD = 0.1
    return (
        F
        .when(hash_col < TRAINING_THRESHOLD, F.lit("training"))
        .when(hash_col < TRAINING_THRESHOLD + TEST_THRESHOLD, F.lit("test"))
        .otherwise(F.lit("validation"))
    )


def split_strategy():
    '''Typically, sequences are split as 80% training, 10% test, and 80% validation based on the split_key, which is
    classified as "split_key_split". However, sequences could also be split using alternative strategies, such as
    assigning all sequences of a particular source record key as the same split. In this case, we are simply splitting
    all sequences using the split_key_split strategy.'''
    return F.lit("split_key_split")


def split_status():
    '''The split status of a sequence can be one of three different statuses:
      - unsplit: the sequence has not been split yet
      - ready to split: the split process is being staged
      - finalized: the sequence has been split
    Because we are accomplishing this programatically, we just assume that its split has been finalized.'''
    return (F.lit("finalized"))


def split_timestamp(ingest_date_col):
    '''When sequences are split via Object Explorer, the exact time when the split was finalized is automatically
    recorded. In this case, we'll just add an arbitrary amount of time to the ingest date.'''
    TIMESTAMP_OFFSET = F.expr("INTERVAL 2 HOURS")
    return ingest_date_col + TIMESTAMP_OFFSET


def split_assigned_by():
    '''When sequences are split via Object Explorer, the user ID of the person that finalized the split is
    automatically recorded. In this case, we'll just assign the split assignee as an arbitrary string.'''
    USER_ID = "john_doe"
    return F.lit(USER_ID)

spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()
df = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\enrichments\template_fmv_sequences.csv')
sequences_df = spark.createDataFrame(df)
sequences_df = sequences_df.withColumn("split_key", split_key(F.col("src_record_key"), F.col("batch")))
sequences_df = sequences_df.withColumn("split_key_hash", split_key_hash(F.col("split_key")))
sequences_df = sequences_df.withColumn("split", split(F.col("split_key_hash")))
sequences_df = sequences_df.drop("split_key_hash")

sequences_df = sequences_df.withColumn("split_strategy", split_strategy())
sequences_df = sequences_df.withColumn("split_status", split_status())
sequences_df = sequences_df.withColumn("ingest_date", F.col("ingest_date").cast("timestamp"))
sequences_df = sequences_df.withColumn("split_timestamp", split_timestamp(F.col("ingest_date")))
sequences_df = sequences_df.withColumn("split_assigned_by", split_assigned_by())

sequences_df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\enrichments\template_fmv_sequences_split.csv', index=False)