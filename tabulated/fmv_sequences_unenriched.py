import os
from pyspark.sql import functions as F, types as T, Window
from pyspark.sql import SparkSession
import datetime
import sys
import pandas

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def to_timestamp(time_col):
    '''Converts the value of the modified column into a proper Timestamp column.'''
    return F.to_timestamp(F.from_unixtime(time_col / 1000, format="yyyy-MM-dd HH:mm:ss"))

def filter_to_cdao_only(df):
    '''Filters the dataframe to only include sequences from a subset of data sources.'''
    CDAO_SOURCES = [
        "12092021", "USCG-CORPUS", "USCG-SANDIEGO", "ECITY-CG", "USCG-MIAMI-12222021",
        "USCG-BARBERS", "USCG-ECITY", "CG-S-And-R", "USCG-KEYWEST", "USCG-MIAMI"
    ]
    return df.where(F.col("data_source").isin(CDAO_SOURCES))

def deduplicate_jpegs(df):
    '''Deduplicates jpegs by ensuring that only the latest ingest of a sequence's jpegs are kept.'''
    df = df.withColumn("version", F.row_number().over(Window.partitionBy(F.col("sequence_id")).orderBy("modified")))
    df = df.withColumn("latest", F.max(F.col("version")).over(Window.partitionBy(F.col("sequence_id"))))
    df = df.where(F.col("version") == F.col("latest")).drop("version", "latest", "modified")
    return df


def prepare_schema_for_merge(df, prefix, drop_modified: bool = True):
    '''Renames DMP metadata columns (e.g., filepath, filesize, batch_path, etc.) to prevent collisions and optionally
    drops the modified column.'''
    df = df.withColumnRenamed("filepath", f"{prefix}_filepath")
    df = df.withColumnRenamed("filesize_bytes", f"{prefix}_filesize_bytes")
    df = df.withColumnRenamed("batch_path", f"{prefix}_batch_path")
    if drop_modified:
        df = df.drop("modified")
    return df


def potential_dupes(sequence_id_col, md5_hash_col):
    '''Uses the md5 hash generated in the raw mp4 dataset to identify sequences with duplicate video files.'''
    duplicate_sequences = F.collect_set(sequence_id_col).over(Window.partitionBy(md5_hash_col))
    # the following F.array_except makes sure that the sequence isn't being marked as a duplicate of itself
    duplicate_sequences = F.array_except(duplicate_sequences, F.array(sequence_id_col))
    duplicate_sequences = F.array_sort(duplicate_sequences)
    return duplicate_sequences


def coalesce_frame_counts(df):
    '''There are a few different possible sources of frame count, so coalesece them by prioritizing the most
    accurate and present source. Does the same for the fps of the sequence.'''
    df = df.withColumn("fps", F.coalesce(F.col("seq_fps"), F.col("fps")))
    df = df.withColumn("frame_count", F.coalesce(F.col("seq_nb_frames"), F.col("frame_count")))
    df = df.drop("seq_fps", "seq_nb_frames")
    df = df.withColumnRenamed("nb_frames", "srk_frame_count")  # reduce ambiguity
    return df


def get_dimension_col(col_name):
    '''Coalesces a number of different columns that describe the dimensions of a sequence. If certain column data is
    unavailable for a sequence, attempt to find this data from other sequences of the same SRK, since all sequences
    of the same SRK should have the same video structure.'''
    srk_window = Window.partitionBy(F.col("src_record_key"))
    return F.coalesce(
        F.col(col_name),
        F.col(f"{col_name}_pix"),
        F.max(col_name).over(srk_window),
        F.max(f"{col_name}_pix").over(srk_window)
    ).cast(T.IntegerType())


def aggregate_filesizes(df):
    '''Computes the sum of a sequence's mp4, jpegs, and metadata files' filesizes together as a new column, then
    drops the original filesize columns.'''
    filesize_cols = [col for col in df.columns if col.endswith("_filesize_bytes")]
    df = df.withColumn("total_filesize_bytes", sum(F.col(col) for col in filesize_cols))
    df = df.drop(*filesize_cols)
    return df


def recast_columns(df):
    '''Recast certain columns as other types, since derived from the metadata are interpreted as StringType by
    default.'''
    COLUMNS = [
        "seq_nb_frames", "time_offset_s", "bit_rate", "duration", "fps", "nb_frames", "nb_streams", "size",
        "bits_per_raw_sample"
    ]
    return df.select(
        *[col for col in df.columns if col not in COLUMNS],
        *(F.col(col).cast("double").alias(col) for col in COLUMNS)
    )

mapp_metadata_spark = SparkSession.builder.appName("mapp_metadata").master("local[1]").getOrCreate()
sequence_metadata_spark = SparkSession.builder.appName("sequence_metadata").master("local[2]").getOrCreate()
mp4s_spark = SparkSession.builder.appName("mp4_metadata").master("local[3]").getOrCreate()
jpegs_spark = SparkSession.builder.appName("sequence_metadata").master("local[4]").getOrCreate()

mapp_metadata = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mapp_metadata_parsed.csv')
sequence_metadata = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_seq_metadata_tabulated.csv')
mp4s = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_mp4s_tabulated.csv')
jpegs = pandas.read_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_jpegs_tabulated.csv')

mapp_metadata_df = mapp_metadata_spark.createDataFrame(mapp_metadata)
sequence_metadata_df = sequence_metadata_spark.createDataFrame(sequence_metadata)
mp4s_df = mp4s_spark.createDataFrame(mp4s)
jpegs_df = jpegs_spark.createDataFrame(jpegs)

df = sequence_metadata_df  # use the seq_metadata dataframe as the base for the final output dataframe
df = prepare_schema_for_merge(sequence_metadata_df, "seq_metadata", False)
df = df.drop("src_json")

mapp_metadata_df = prepare_schema_for_merge(mapp_metadata_df, "mapp_metadata")
df = df.join(mapp_metadata_df, "sequence_id", "left")

jpegs_df = jpegs_df.transform(deduplicate_jpegs)
jpegs_df = prepare_schema_for_merge(jpegs_df, "jpegs")
df = df.join(jpegs_df, "sequence_id", "left")

mp4s_df = mp4s_df.cache().distinct()
mp4s_df = mp4s_df.withColumn("potential_dupes", potential_dupes(F.col("sequence_id"), F.col("md5_hash")))
mp4s_df = mp4s_df.drop("md5_hash")
mp4s_df = prepare_schema_for_merge(mp4s_df, "mp4s")
df = df.join(mp4s_df, "sequence_id", "left")

df = df.transform(aggregate_filesizes)
df = df.withColumn("fps", F.col("fps").cast(T.IntegerType()))
df = df.withColumn("width", get_dimension_col("width"))
df = df.withColumn("height", get_dimension_col("height"))
df = df.drop("width_pix", "height_pix")

df = df.transform(recast_columns)
df = df.transform(coalesce_frame_counts)
df = df.withColumn("ingest_date", to_timestamp(F.col("modified")))
df = df.drop("modified")

df = df.transform(filter_to_cdao_only)  # ensure we are only using CDAO data going forward
df = df.repartition(5, "src_record_key")
df.toPandas().to_csv(r'C:\Users\ecs\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_sequences_unenriched.csv', index=False)