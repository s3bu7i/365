from pyspark.sql import SparkSession  # type: ignore
from pyspark.sql.functions import from_json, col  # type: ignore
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType  # type: ignore
from pyspark.sql import SQLContext  # type: ignore
from pyspark.sql import SparkSession  # type: ignore

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("LogDataProcessing") \
    .getOrCreate()

# Define schema for the log data
schema = StructType([
    StructField("time", TimestampType(), True),
    StructField("host", StringType(), True),
    StructField("plugin", StringType(), True),
    StructField("plugin_instance", StringType(), True),
    StructField("type", StringType(), True),
    StructField("type_instance", StringType(), True),
    StructField("value", DoubleType(), True)
])

# Read from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "log_data") \
    .load()

# Parse the value field of Kafka message
log_df = df.selectExpr("CAST(value AS STRING)")

# Convert JSON string to DataFrame
log_df = log_df.withColumn("value", from_json(
    col("value"), schema)).select(col("value.*"))

# Write data to console for debugging (optional)
query = log_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
