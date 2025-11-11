# -*- coding: utf-8 -*-

from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


#učitavanje podataka, kreiranje DataFrame-a
def _read_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("escape", "\"")
        .csv(path)
    )


def load_results(spark: SparkSession, path: str) -> DataFrame:
    df = _read_csv(spark, path)
    
    df = (
        df
        .withColumn("date", F.to_date(F.col("date")))
        .withColumn("home_score", F.col("home_score").cast("int"))
        .withColumn("away_score", F.col("away_score").cast("int"))

        .withColumn("neutral_norm", F.lower(F.col("neutral").cast("string")))
        .withColumn("neutral", F.when(F.col("neutral_norm").isin("true", "t", "yes", "1"), F.lit(True)).otherwise(F.lit(False)))
        .drop("neutral_norm")
    )
    return df


def load_goalscorers(spark: SparkSession, path: str) -> DataFrame:
    df = _read_csv(spark, path)
    df = (
        df
        .withColumn("date", F.to_date(F.col("date")))
        .withColumn("minute", F.col("minute").cast("int"))  
    )
    return df



def _q1_serbia_best_worst_opponent_df(results_df: DataFrame, team_name: str) -> DataFrame:
    base = results_df.select(
        "home_team", "away_team", "home_score", "away_score"
    ).where((F.col("home_team") == team_name) | (F.col("away_team") == team_name))


    df_calc = (
        base.withColumn(
            "opponent",
            F.when(F.col("home_team") == team_name, F.col("away_team")).otherwise(F.col("home_team"))
        ).withColumn(
            "goal_diff",  
            F.when(
                F.col("home_team") == team_name,
                F.col("home_score") - F.col("away_score")
            ).otherwise(F.col("away_score") - F.col("home_score"))
        ).select("opponent", "goal_diff")
    )

    agg = df_calc.groupBy("opponent").agg(F.sum("goal_diff").alias("goal_diff"))

    if agg.rdd.isEmpty():  
        return results_df.sparkSession.createDataFrame([], schema=T.StructType([
            T.StructField("opponent", T.StringType(), True),
            T.StructField("goal_diff", T.IntegerType(), True),
            T.StructField("kind", T.StringType(), True),
        ]))

    w_best = agg.orderBy(F.col("goal_diff").desc()).limit(1).withColumn("kind", F.lit("best"))
    w_worst = agg.orderBy(F.col("goal_diff").asc()).limit(1).withColumn("kind", F.lit("worst"))
    return w_best.unionByName(w_worst)


def q1_serbia_best_worst_opponent_rdd(results_df: DataFrame, team_name: str = "Serbia") -> DataFrame:
   
    try:
        cols = ["home_team", "away_team", "home_score", "away_score"]
        base = results_df.select(*cols).where((F.col("home_team") == team_name) | (F.col("away_team") == team_name))

        def map_row(row) -> Tuple[str, int]:
            if row.home_team == team_name:
                return (row.away_team, int(row.home_score) - int(row.away_score))
            else:
                return (row.home_team, int(row.away_score) - int(row.home_score))

        pair_rdd = base.rdd.map(map_row)
        agg = pair_rdd.reduceByKey(lambda a, b: a + b)

        if agg.isEmpty():
            return base.sparkSession.createDataFrame([], schema=T.StructType([
                T.StructField("opponent", T.StringType(), True),
                T.StructField("goal_diff", T.IntegerType(), True),
                T.StructField("kind", T.StringType(), True),
            ]))

        best = agg.max(key=lambda t: t[1])
        worst = agg.min(key=lambda t: t[1])
        out_df = base.sparkSession.createDataFrame(
            [(best[0], int(best[1]), "Najbolja"), (worst[0], int(worst[1]), "Najgora")],
            schema=["Protivnik", "Gol razlika", "Najbolja / Najgora"],
        )
        return out_df
    except Exception as e:  
        spark = results_df.sparkSession
        print(f"[WARN] RDD verzija Q1 je pala: {e}. Koristim DataFrame fallback.")
        return _q1_serbia_best_worst_opponent_df(results_df, team_name)



def q2_euro_hat_tricks(results_df: DataFrame, scorers_df: DataFrame) -> DataFrame:
    

    euro_aliases = ["uefa euro", "uefa european championship", "european championship"]
    results_norm = results_df.withColumn("tournament_l", F.lower(F.col("tournament")))
    euro_results = results_norm.where(F.col("tournament_l").isin(*euro_aliases))

    
    hats = (
        scorers_df
        .groupBy("date", "home_team", "away_team", "scorer")
        .agg(F.count(F.lit(1)).alias("total_scorer_goals"))
        .where(F.col("total_scorer_goals") >= 3)
    )

    joined = (
        hats.join(
            euro_results.select(
                "date", "home_team", "away_team", "home_score", "away_score",
                F.col("tournament"), "country"
            ),
            on=["date", "home_team", "away_team"],
            how="inner",
        )
        .withColumn("score", F.concat_ws(":", F.col("home_score"), F.col("away_score")))
        .select(F.col("date").alias("Datum"), F.col("home_team").alias("Domacin"), F.col("away_team").alias("Gost"), F.col("score").alias("Rezultat"), F.col("scorer").alias("Strelac"), F.col("total_scorer_goals").alias("Ukupno golova"), F.col("tournament").alias("Turnir"), F.col("country").alias("Drzava"))
        .orderBy("Datum")
    )
    return joined


def q3_world_cup_top_scorers(results_df: DataFrame, scorers_df: DataFrame, from_year: int = 1990) -> DataFrame:
    

    wc_aliases = ["fifa world cup", "world cup"]

    
    wc_results = (
        results_df
        .withColumn("tournament_l", F.lower(F.col("tournament")))
        .where(F.col("tournament_l").isin(*wc_aliases))
        .withColumn("year", F.year(F.col("date")))
        .where(F.col("year") >= F.lit(from_year))
        .select("date", "home_team", "away_team", "country", "year")
    )

    
    hosts_counts = (
        wc_results
        .groupBy("year", "country")
        .agg(F.count(F.lit(1)).alias("matches"))
    )
    host_rank_w = Window.partitionBy("year").orderBy(F.col("matches").desc(), F.col("country"))
    hosts_top = (
        hosts_counts
        .withColumn("r", F.dense_rank().over(host_rank_w))
        .where(F.col("r") == 1)
        .groupBy("year")
        .agg(F.array_join(F.array_sort(F.collect_set("country")), "/").alias("country"))
    )

    
    goals = (
        scorers_df
        .join(wc_results.select("date", "home_team", "away_team", "year"),
              on=["date", "home_team", "away_team"], how="inner")
        .groupBy("year", "scorer", "team")
        .agg(F.count(F.lit(1)).alias("goals"))
    )

    w = Window.partitionBy("year").orderBy(F.col("goals").desc(), F.col("scorer"))

    ranked = (
        goals
        .withColumn("rank", F.dense_rank().over(w))
        .where(F.col("rank") <= 2)
        .withColumn("tournament", F.lit("FIFA World Cup"))
    )

    
    ranked_with_host = ranked.join(hosts_top, on="year", how="left")

    out = (
        ranked_with_host
        .select(
            F.col("scorer").alias("Strelac"),
            F.col("team").alias("Reprezentacija"),
            F.col("goals").alias("Broj golova"),
            F.col("rank").alias("Rang"),
            F.col("year").alias("Godina"),
            F.col("tournament").alias("Turnir"),
            F.col("country").alias("Drzava"),
        )
        .orderBy("Godina", F.col("Broj golova").desc(), "Strelac")
    )
    return out




def q4_longest_unbeaten_streaks(results_df: DataFrame) -> DataFrame:
    results_df = results_df.withColumn("date", F.to_date(F.col("date")))


    wc_teams = (
        results_df.where(F.col("tournament") == F.lit("FIFA World Cup"))
        .select(F.col("home_team").alias("team")).unionByName(
            results_df.where(F.col("tournament") == F.lit("FIFA World Cup")).select(F.col("away_team").alias("team"))
        )
        .distinct()
    )

    base = results_df.select("date", "home_team", "away_team", "home_score", "away_score")

    home = base.select(
        F.col("date"),
        F.col("home_team").alias("team"),
        F.col("away_team").alias("opponent"),
        F.col("home_score").alias("gf"),
        F.col("away_score").alias("ga"),
    )
    away = base.select(
        F.col("date"),
        F.col("away_team").alias("team"),
        F.col("home_team").alias("opponent"),
        F.col("away_score").alias("gf"),
        F.col("home_score").alias("ga"),
    )

    matches = home.unionByName(away)

    matches = matches.where(F.col("date") >= F.lit("1980-01-01"))

    matches = matches.join(wc_teams, on="team", how="inner")

    losses = (
        matches
        .withColumn("is_loss", F.col("ga") > F.col("gf"))
        .where(F.col("is_loss"))
        .select("team", "opponent", "date")
    )

    w = Window.partitionBy("team").orderBy("date")
    losses2 = (
        losses
        .withColumn("prev_loss_date", F.lag("date").over(w))
        .withColumn("prev_loss_opponent", F.lag("opponent").over(w))
        .where(F.col("prev_loss_date").isNotNull())
        .withColumn("period_days", F.datediff(F.col("date"), F.col("prev_loss_date")))
    )

    top20 = (
        losses2
        .orderBy(F.col("period_days").desc())
        .limit(20)
        .select(
            F.col("team").alias("team"),
            F.col("opponent").alias("loss_to"),
            F.col("date").alias("loss_date"),
            F.col("prev_loss_opponent").alias("previous_loss_to"),
            F.col("prev_loss_date").alias("previous_loss_date"),
            F.col("period_days")
        )
    )
    return top20
