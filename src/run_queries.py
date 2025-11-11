# -*- coding: utf-8 -*-
"""Runner skripta – učitava CSV fajlove i izvršava sve upite.

Primer pokretanja:
    python .\src\run_queries.py --results .\data\results.csv --goals .\data\goalscorers.csv

Rezultati se upisuju u folder `output/`.
"""
import argparse
import os
import sys
from pathlib import Path
import csv



JAVA_HOME = r"C:\Program Files\Java\jdk1.8.0_202"
if os.environ.get("JAVA_HOME") != JAVA_HOME:
    os.environ["JAVA_HOME"] = JAVA_HOME
    os.environ["PATH"] = JAVA_HOME + r"\bin;" + os.environ.get("PATH", "")


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print(f"[INFO] PYSPARK_PYTHON = {os.environ['PYSPARK_PYTHON']}")
print(f"[INFO] PYSPARK_DRIVER_PYTHON = {os.environ['PYSPARK_DRIVER_PYTHON']}")
print(f"[INFO] JAVA_HOME = {os.environ.get('JAVA_HOME')}")

from pyspark.sql import SparkSession 
from pyspark.sql import functions as F  

from queries import (  # noqa: E402
    load_results,
    load_goalscorers,
    q1_serbia_best_worst_opponent_rdd,
    q2_euro_hat_tricks,
    q3_world_cup_top_scorers,
    q4_longest_unbeaten_streaks,
    _q1_serbia_best_worst_opponent_df,
)


def get_spark(app_name: str = "FootballSpark") -> SparkSession:
   
    local_tmp = Path(__file__).resolve().parents[1] / "tmp" / "spark-local"
    local_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
       
        .config("spark.pyspark.driver.python", __import__('sys').executable)
        .config("spark.pyspark.python", __import__('sys').executable)
       
        .config("spark.python.use.daemon", "false")
        .config("spark.local.dir", str(local_tmp))
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    return spark


def save_df(df, path: Path):
   
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        (df.coalesce(1)
           .write.mode("overwrite")
           .option("header", True)
           .csv(str(path)))
    except Exception as e:
        msg = str(e)
        if "HADOOP_HOME" in msg or "winutils" in msg:
            print("[WARN] Spark CSV write failed due to missing winutils/HADOOP_HOME. Using local CSV fallback.")
           
            path.mkdir(parents=True, exist_ok=True)
            outfile = path / "result.csv"
            cols = df.columns
            with open(outfile, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                for row in df.toLocalIterator():
                    writer.writerow([row[c] for c in cols])
            print(f"[INFO] Fallback CSV written: {outfile}")
        else:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Putanja do results.csv")
    parser.add_argument("--goals", required=True, help="Putanja do goalscorers.csv")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "output"), help="Izlazni folder")
    args = parser.parse_args()

    out_dir = Path(args.out)

    spark = get_spark()

    
    results = load_results(spark, args.results)
    scorers = load_goalscorers(spark, args.goals)

   
    try:
        q1_df = q1_serbia_best_worst_opponent_rdd(results)
    except Exception as e:
        print(f"[WARN] Q1 RDD failed: {e}. Falling back to DataFrame implementation.")
        q1_df = _q1_serbia_best_worst_opponent_df(results, "Serbia")
    q1_df.show(truncate=False)
    save_df(q1_df, out_dir / "q1_serbia_best_worst")

   
    print("[INFO] Running Q2: EURO hat-tricks…")
    try:
        q2_df = q2_euro_hat_tricks(results, scorers)
        q2_df.show(20, truncate=False)
        save_df(q2_df, out_dir / "q2_euro_hat_tricks")
    except Exception as e:
        print(f"[ERROR] Q2 failed: {e}")

   
    print("[INFO] Running Q3: World Cup top scorers…")
    try:
        q3_df = q3_world_cup_top_scorers(results, scorers, from_year=1990)
        q3_df.show(50, truncate=False)
        save_df(q3_df, out_dir / "q3_wc_top_scorers")
    except Exception as e:
        print(f"[ERROR] Q3 failed: {e}")

    
    print("[INFO] Running Q4: Longest unbeaten streaks…")
    try:
        q4_df = q4_longest_unbeaten_streaks(results)
        
        q4_named = (
            q4_df.select(
                F.col("team").alias("Reprezentacija"),
                F.col("loss_to").alias("Poraz od"),
                F.col("loss_date").alias("Poraz datum"),
                F.col("previous_loss_to").alias("Prethodni poraz od"),
                F.col("previous_loss_date").alias("Prethodni poraz datum"),
                F.col("period_days").alias("Period-dana"),
            )
        )
        q4_named.show(20, truncate=False)
        save_df(q4_named, out_dir / "q4_longest_unbeaten")
    except Exception as e:
        print(f"[ERROR] Q4 failed: {e}")

    spark.stop()


if __name__ == "__main__":
    main()
