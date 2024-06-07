import os
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession
import sys
import pandas

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

COASTGUARD_BATCHES = [
    "976-JWO-USCG-MIAMI_12092021_BATCH-2",
    "899-JWO_USCG-SANDIEGO_20211122",
    "922-JWO_USCG-CORPUS_10242021_BATCH-3",
    "603-USCG_ECITY-CG_BATCH-2",
    "921-JWO_USCG-CORPUS_10242021_BATCH-2",
    "968-JWO-USCG-MIAMI_12092021_BATCH-1",
    "1035-JWO_USCG-MIAMI_12222021_Batch-24",
    "1038-JWO_USCG-MIAMI_04182022_Batch-25",
    "1043-JWO_USCG-MIAMI_04182022_Batch-26",
    "1029-JWO_USCG-MIAMI-12222021_Batch-22",
    "959-JWO_USCG-BARBERS_BATCH-1",
    "1046-JWO_USCG-MIAMI_04202022_Batch-28",
    "1044-JWO_USCG-MIAMI_04202022_Batch-27",
    "1034-JWO_USCG-MIAMI_12222021_Batch-23",
    "1049-JWO_USCG-MIAMI_20220422_Batch-29",
    "592-JW_CG-S-And-R_Batch-1",
    "769-JWO_USCG-ECITY_05-26_27-2021",
    "770-JWO_USCG-ECITY_12-2020_05-2021",
    "682-JW_USCG-KEYWEST_04012021"
]

EDGE_BATCHES = ["1123-MAVEN_EDGE_20220920"]

EDGE_FM_SRKS = ["198613752", "198613764", "198613756", "198613750", "198613760", "198613763", "198613755",
                "198613753", "198613759", "198613746", "198613747", "198613748", "198613745", "198613749",
                "198613743", "198613741", "198613761"]

EDGE_FT_SRKS = ["198613766", "198613751", "198613742", "198613754", "198613744", "198613757", "198613758"]

def batch(filepath_col, index):
    '''Extracts the name of the batch using one of the sequence's raw file's filepaths.'''
    return F.split(filepath_col, "/")[index]

def locator(rid_col, filepath_col, branch: str = "master", suffix=None):
    '''Creates a locator string that describes the originating database, its branch, and filepath of a sequence. If
    for some reason the rid dataset is not provided, then the locator string is simply null.'''
    locator = F.concat_ws(":", F.element_at(F.split(rid_col, "\\."), -1), F.lit(branch), filepath_col)
    if suffix is not None:
        locator = F.concat(locator, F.lit(suffix))
    return (F.when(rid_col.isNull(), F.lit(None)).otherwise(locator))


def is_maritime(filepath_col):
    '''Describes whether or not a sequence contains maritime data by checking the sequence's filepath for particular
    keywords. As new FMV maritime batches are added, they should be added to the logic here.'''
    MARITIME_SUBSTRINGS = [
        "MARITIME", "Maritime", "maritime", "FMV-MAR", "MAVEN_HMI-MH", "MAVEN_HMI-FMH", "CCFLIR2",
        "TDO_MedAlt-MVC_20190923", "380-TDO_Triton_Job1_Batch-I", "USCG", "579-TDO_Trition_20210112",
        "592-JW_CG-S-And-R_Batch-1", "FEWSHOT_DESTROYERS"
    ]
    has_maritime_substring = filepath_col.rlike(f".*({'|'.join(MARITIME_SUBSTRINGS)}).*")
    is_edge_maritime = filepath_col.contains("EDGE_") & ~(F.col("src_record_key").isin(EDGE_FT_SRKS))
    return F.when((has_maritime_substring) | (is_edge_maritime), F.lit(True)).otherwise(F.lit(False))


def mission_type(batch_col, srk_col, is_maritime_col):
    '''Maps a mission_type based on keywords in the batch name, specific source record keys, and the is_maritime
    column. If no mapping criteria are satisified, the sequence is assigned a mission_type of "CT/COIN" by default.'''
    return (
        F
        .when(batch_col.isin(COASTGUARD_BATCHES), "Coastguard-CDAO")
        .when(batch_col.isin(EDGE_BATCHES) & srk_col.isin(EDGE_FM_SRKS), "Fixed Maritime Horizontal")
        .when(batch_col.isin(EDGE_BATCHES) & srk_col.isin(EDGE_FT_SRKS), "Fixed Terrestrial Horizontal")
        .when(batch_col.isin(EDGE_BATCHES), "Maritime Horizontal")
        .when(batch_col.contains("MAVEN_HMI-MH"), "Maritime Horizontal")
        .when(batch_col.contains("MAVEN_HMI-FMH"), "Fixed Maritime Horizontal")
        .when(is_maritime_col, "Maritime")
        .when(batch_col.contains("CCTV"), "Fixed Terrestrial Horizontal")
        .when(batch_col.contains("MAVEN_HMI-TH"), "Terrestrial Horizontal")
        .when(batch_col.contains("NDS"), "NDS")
        .when(batch_col.contains("SpecialFMV"), "NDS")
        .when(batch_col.contains("SCALE_ANNOS"), "CDAO-Scale-Coastguard")
        .otherwise("CT/COIN")
    )


def loe(mission_type_col):
    '''Determines whether or not the sequence is an HMI sequence or an FMV sequence.'''
    return F.when(mission_type_col.contains("Horizontal"), "HMI").otherwise("FMV")


def find_max_framecount_per_batch(df):
    '''Partitions sequences by batch, finds the largest frame count value from the sequence metadata across each
    partition and stores it as a new as a new column.'''
    frames_per_batch_df = df.groupBy("batch").agg(F.max(F.col("frame_count")).alias("max_frame_count"))
    return df.join(frames_per_batch_df, "batch", "left")


def compute_contiguous_sequences(df):
    '''Finds the following or previous sequential sequence ID for each sequence ID and stores them a new column. The
    direction value determines whether or to find the following or previous sequences. This value is only computed for
    5, 8, and 30 second sequences that have mapp metadata, otherwise a literal none is stored.'''
    next_df = compute_contiguous_sequence_in_direction(df, "next")
    prev_df = compute_contiguous_sequence_in_direction(df, "prev")

    df = df.join(next_df, ["src_record_key", "time_offset_s"], "left")
    df = df.join(prev_df, ["src_record_key", "time_offset_s"], "left")
    return df


def compute_contiguous_sequence_in_direction(df, direction: str):
    '''Finds sequential sequence IDs in a given direction.'''
    df = df.select("sequence_id", "src_record_key", "time_offset_s", "max_frame_count")
    df = df.withColumnRenamed("sequence_id", f"{direction}_sequence_id")

    offset = 1 if direction == "next" else -1
    df = df.withColumn("time_offset_s", time_offset_s(F.col("max_frame_count"), F.col("time_offset_s"), offset))
    df = df.drop("max_frame_count")
    return df


def time_offset_s(max_frame_count_col, time_offset_s_col, offset):
    '''Computes a time_offset_s value based on the value of max_frame_count. If max_frame_count does not equal 900, 240,
    or 150 (i.e., the batch does not consist of 5, 8, or 30 second sequences), a literal none is returned instead.'''
    return (
        F
        .when(max_frame_count_col == 900, time_offset_s_col - 30 * offset)
        .when(max_frame_count_col == 240, time_offset_s_col - 8 * offset)
        # .when(max_frame_count_col == 150, time_offset_s_col - 5 * offset)
        # .otherwise(F.lit(None))
        .otherwise(time_offset_s_col - 5 * offset)
    )

spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()
df = pandas.read_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\tabulated\template_fmv_sequences_unenriched.csv')
sequences_df = spark.createDataFrame(df)
sequences_df = sequences_df.withColumn("batch", batch(F.col("seq_metadata_filepath"), 2))
sequences_df = sequences_df.withColumn("is_maritime", is_maritime(F.col("seq_metadata_filepath")))
sequences_df = sequences_df.withColumn("mission_type", mission_type(F.col("batch"), F.col("src_record_key"), F.col("is_maritime")))  # noqa
sequences_df = sequences_df.withColumn("loe", loe(F.col("mission_type")))
sequences_df = sequences_df.transform(find_max_framecount_per_batch)
sequences_df = sequences_df.transform(compute_contiguous_sequences)
sequences_df = sequences_df.drop("max_frame_count")

sequences_df = sequences_df.repartition(5, "src_record_key")
sequences_df = sequences_df.sortWithinPartitions("src_record_key", "time_offset_s")
sequences_df = sequences_df.dropDuplicates()

sequences_df.toPandas().to_csv(r'C:\Users\benedict.browder\Desktop\FMV Data Processing\datasets\enrichments\template_fmv_sequences.csv', index=False)
