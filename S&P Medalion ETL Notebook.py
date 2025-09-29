# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - Ingestion & Data Validation
# MAGIC First, we will load the raw S&P500 companies dataset from DBFS into a Spark DataFram so that we create a controlled enty point before validation.

# COMMAND ----------

# DBTITLE 1,Reading and Type Enforcing
# Creating project schemas
spark.sql("CREATE SCHEMA IF NOT EXISTS sp500.bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS sp500.silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS sp500.gold")

# Reading the uploaded csv into the bronze schema
src_tbl = "hive_metastore.default.sp_500_companies"
bronze_df = spark.read.table(src_tbl)

# Enforcing data types for controlled entry
from pyspark.sql.types import *
import pyspark.sql.functions as F


bronze_df = (bronze_df
    .withColumn("Exchange", F.col("Exchange").cast(StringType()))
    .withColumn("Symbol", F.col("Symbol").cast(StringType()))
    .withColumn("Shortname", F.col("Shortname").cast(StringType()))
    .withColumn("Longname", F.col("Longname").cast(StringType()))
    .withColumn("Sector", F.col("Sector").cast(StringType()))
    .withColumn("Industry", F.col("Industry").cast(StringType()))
    .withColumn("Currentprice", F.col("Currentprice").cast(DoubleType()))
    .withColumn("Marketcap", F.col("Marketcap").cast(DoubleType()))
    .withColumn("Ebitda", F.col("Ebitda").cast(DoubleType()))
    .withColumn("Revenuegrowth", F.col("Revenuegrowth").cast(DoubleType()))
    .withColumn("City", F.col("City").cast(StringType()))
    .withColumn("State", F.col("State").cast(StringType()))
    .withColumn("Country", F.col("Country").cast(StringType()))
    .withColumn("Fulltimeemployees", F.col("Fulltimeemployees").cast(IntegerType()))
    .withColumn("Longbusinesssummary", F.col("Longbusinesssummary").cast(StringType()))
    .withColumn("Weight", F.col("Weight").cast(DoubleType()))
)

# Moving the read table and type enforced table into the bronze schema
target_tbl = "sp500.bronze.sp_500_companies"

(bronze_df.write
 .mode("overwrite")
 .format("delta")
 .saveAsTable(target_tbl))


# Verification
print("Bronze table:", target_tbl)
print("Row count:", spark.table(target_tbl).count())
display(spark.table(target_tbl).limit(10))

# COMMAND ----------

# DBTITLE 1,Load & Type
bronze = spark.table("sp500.bronze.sp_500_companies")

print("Row count:", bronze.count())
bronze.printSchema()
display(bronze.limit(5))

# COMMAND ----------

# DBTITLE 1,Null Values
# Null values check
from pyspark.sql import functions as F

nulls = bronze.select(
    F.count(F.when(F.col("Symbol").isNull(), 1)).alias("Symbol_nulls"),
    F.count(F.when(F.col("Sector").isNull(), 1)).alias("Sector_nulls"),
    F.count(F.when(F.col("Industry").isNull(), 1)).alias("Industry_nulls"),
    F.count(F.when(F.col("Marketcap").isNull(), 1)).alias("Marketcap_nulls")
)

display(bronze.filter(F.col("Marketcap").isNull()))

# 1 market cap is missing, so we drop the row
bronze = bronze.filter(F.col("Marketcap").isNotNull())

# then we save it to the bronze table
(bronze.write
 .mode("overwrite")
 .format("delta")
 .saveAsTable(target_tbl))

display(nulls)

# COMMAND ----------

# DBTITLE 1,Duplicates
# Check and display duplicate tickers for exmaple
dups = bronze.groupBy("Symbol").count().filter(F.col("count") > 1)
display(dups)

# COMMAND ----------

# DBTITLE 1,Distribution
# Sector distribution check for obvious errors
sector_counts = bronze.groupBy("Sector").count().orderBy(F.desc("count"))
display(sector_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver - Transformation
# MAGIC Our goal here is to make a clean, analysis-ready table with consistent names and types

# COMMAND ----------

# DBTITLE 1,Loading
# Loading bronze output and setting baseline count
bronze = spark.table("sp500.bronze.sp_500_companies")
bronze_count = bronze.count()
print("Bronze table row count:", bronze_count)

# COMMAND ----------

# DBTITLE 1,Standardizing column names
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Renaming
silver_df = (bronze
    .withColumnRenamed("Symbol", "ticker")
    .withColumnRenamed("Shortname", "short")
    .withColumnRenamed("Longname", "long")
    .withColumnRenamed("Sector", "sector")
    .withColumnRenamed("Industry", "industry")
    .withColumnRenamed("Currentprice", "current_price")
    .withColumnRenamed("Marketcap", "market_cap")
    .withColumnRenamed("Ebitda", "ebitda")
    .withColumnRenamed("Revenuegrowth", "rev_growth")
    .withColumnRenamed("City", "city")
    .withColumnRenamed("State", "state")
    .withColumnRenamed("Country", "country")
    .withColumnRenamed("Fulltimeemployees", "employees")
    .withColumnRenamed("Longbusinesssummary", "summary")
    .withColumnRenamed("Exchange", "exchange")
    .withColumnRenamed("Weight", "weight")
)

# COMMAND ----------

# DBTITLE 1,Saving Silver Table in Catalog
target_silver = "sp500.silver.sp_500_companies"

(silver_df.write
 .mode("overwrite")
 .format("delta")
 .saveAsTable(target_silver))

# Show all tables in bronze and silver schemas
print("Bronze tables:")
spark.sql("SHOW TABLES IN sp500.bronze").show(truncate=False)

print("\nSilver tables:")
spark.sql("SHOW TABLES IN sp500.silver").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Gold - Modeling 
# MAGIC Our goal now is to create a simple fact and dimension tables (sector, industry, exchange, location) from the curated Silver dataset to support analysis and PowerBI resporting.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/2: Dimension Tables
# MAGIC

# COMMAND ----------

# DBTITLE 1,Loading
# Loading Silver 
silver = spark.table("sp500.silver.sp_500_companies")

print("Silver rows:", silver.count())

# COMMAND ----------

# DBTITLE 1,dim_sector
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Creating a stable and small lookup for sectors
dim_sector = (silver
               .select("sector")
               .distinct()
               .orderBy("sector")
               .withColumn("sector_id", F.row_number().over(Window.orderBy("sector")))
              ).select("sector", "sector_id")

# Saving it as Gold
dim_sector.write.mode("overwrite").format("delta").saveAsTable("sp500.gold.dim_sector")

# Check
display(dim_sector)

# COMMAND ----------

# DBTITLE 1,dim_industry
# Build industry list with the sector it belongs to, then attach sector_id
dim_industry_base = (silver
    .select("industry", "sector").distinct()
)

# Join to dim_sector to get sector_id (foreign key)
dim_industry = (dim_industry_base
    .join(dim_sector, on="sector", how="left")
    .select("industry", "sector_id")               
    .orderBy("industry")                           
    .withColumn("industry_id",
                F.row_number().over(Window.orderBy(F.col("industry"))))
    .select("industry_id", "industry", "sector_id") 
)

# Save
dim_industry.write.mode("overwrite").format("delta").saveAsTable("sp500.gold.dim_industry")

print("dim_industry rows:", spark.table("sp500.gold.dim_industry").count())
display(spark.table("sp500.gold.dim_industry").limit(20))

# COMMAND ----------

# DBTITLE 1,dim_exchange
# Simple lookup for exchanges
dim_exchange = (silver
    .select("exchange").distinct().orderBy("exchange")
    .withColumn("exchange_id", F.row_number().over(Window.orderBy(F.col("exchange"))))
    .select("exchange_id", "exchange")
)

dim_exchange.write.mode("overwrite").format("delta").saveAsTable("sp500.gold.dim_exchange")

print("dim_exchange rows:", spark.table("sp500.gold.dim_exchange").count())
display(spark.table("sp500.gold.dim_exchange").limit(20))


# COMMAND ----------

# DBTITLE 1,dim_location
# Now since our silver table contains a few entires with null as their state for example, we have to normalize all these nulls by giving them an "unknown" attribute.
from pyspark.sql import functions as F

silver = spark.table("sp500.silver.sp_500_companies")

silver = (silver
    .withColumn("country", F.coalesce(F.col("country"), F.lit("Unknown")))
    .withColumn("state",   F.coalesce(F.col("state"),   F.lit("Unknown")))
    .withColumn("city",    F.coalesce(F.col("city"),    F.lit("Unknown")))
)

# Group location fields to avoid repeating large text in the fact table
dim_location = (silver
    .select("country","state","city").distinct().orderBy("country","state","city")
    .withColumn("location_id",
                F.row_number().over(Window.orderBy("country","state","city")))
    .select("location_id","country","state","city")
)

dim_location.write.mode("overwrite").format("delta").saveAsTable("sp500.gold.dim_location")

# Also we save the new normlized fact table over the orignial silver fact table
silver.write.mode("overwrite").format("delta").saveAsTable("sp500.silver.sp_500_companies")




# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/2: Fact Table
# MAGIC Now we join our Silver Table to our 4 dimensions and keep the key business measures.

# COMMAND ----------

# DBTITLE 1,Load Dimensions
# Loading Silver and Gold (Dimensions)
silver = spark.table("sp500.silver.sp_500_companies")
dim_industry = spark.table("sp500.gold.dim_industry")
dim_exchange = spark.table("sp500.gold.dim_exchange")
dim_location = spark.table("sp500.gold.dim_location")

print("Silver rows:", silver.count())

# COMMAND ----------

# DBTITLE 1,Joining
from pyspark.sql import functions as F

# Join silver with dim tables so that we link each company to it's respective industry, sector, exchange and location. By doing so we link also their respective id's as a byproduct.
joined = (silver
    .join(dim_industry, on="industry", how="left")
    .join(dim_exchange, on="exchange", how="left")
    .join(dim_location, on=["country", "state", "city"], how="left")
)

display(joined)


# COMMAND ----------

# DBTITLE 1,Refining Fact Table
# Now we drop the names and keep the IDs
fact_company = (joined.select(
    "ticker",
    "industry_id", "sector_id", "exchange_id", "location_id",
    "short", "long",
    "market_cap", "current_price", "ebitda", "rev_growth", "employees", "weight"
))

display(fact_company.limit(10))

                


# COMMAND ----------

# DBTITLE 1,Save as new Gold
target_fact = "sp500.gold.fact_company"
fact_company.write.mode("overwrite").format("delta").saveAsTable(target_fact)

# Quick comparasion between rows in silver and new fact
print("Silver rows:", silver.count(), "| Fact rows:", fact_company.count())

fk_nulls = fact_company.select(
    F.sum(F.col("industry_id").isNull().cast("int")).alias("industry_id_nulls"),
    F.sum(F.col("sector_id").isNull().cast("int")).alias("sector_id_nulls"),
    F.sum(F.col("exchange_id").isNull().cast("int")).alias("exchange_id_nulls"),
    F.sum(F.col("location_id").isNull().cast("int")).alias("location_id_nulls")
)
display(fk_nulls)

# COMMAND ----------

# MAGIC %md
# MAGIC