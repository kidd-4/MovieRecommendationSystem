import csv
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import coalesce,first
from pyspark.mllib.stat import Statistics
import math

spark = SparkSession \
    .builder \
    .appName("Item Based Recommendation System") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sc = spark.sparkContext


def main():
    # ratingDF = spark.read.format("csv").option("header","true").load("/Users/grey/Documents/Big Data/project/files/ratings_small.csv")
    ratingFile = sc.textFile("/Users/grey/Documents/Big Data/project/files/ratings_small.csv")
    ratingrdd = ratingFile.mapPartitions(lambda x: csv.reader(x))
    ratingheader = ratingrdd.first()
    ratingrdd = ratingrdd.filter(lambda x: x != ratingheader)
    ratings = ratingrdd.map(lambda metadata: Row(userId= int(metadata[0]),movieId = int(metadata[1]),rating = float(metadata[2])))
    ratingDF = spark.createDataFrame(ratings)
    # (smallRatingDF, largeRatingDF) = ratingDF.randomSplit([0.01, 0.99],123)
    smallRatingDF = ratingDF.limit(5000)
    # smallRatingDF.show()
    # print(smallRatingDF.count())
    # ratings.show()

    # linkDataFrame = spark.read.format("csv").option("header","true").load("/Users/grey/Documents/Big Data/project/files/links_small.csv")
    linkFile = sc.textFile("/Users/grey/Documents/Big Data/project/files/links_small.csv")
    linkrdd = linkFile.mapPartitions(lambda x: csv.reader(x))
    linkheader = linkrdd.first()
    linkrdd = linkrdd.filter(lambda x: x != linkheader)
    linkrdd = linkrdd.filter(lambda x: x[0] != '' and x[2] != '')
    linkMovieId = linkrdd.map(lambda metadata: Row(movieId = int(metadata[0]),tmdbId = int(metadata[2])))
    filterMovieId = linkrdd.map(lambda x: int(x[2])).collect()
    linkDataFrame = spark.createDataFrame(linkMovieId)
    # linkDataFrame.show()

    metaFile = sc.textFile("/Users/grey/Documents/Big Data/project/files/movies_metadata.csv")
    rdd = metaFile.mapPartitions(lambda x: csv.reader(x))
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)
    rdd = rdd.filter(lambda x: x[5] != '')
    rdd = rdd.filter(lambda x: "-" not in x[5])
    moviesMetadata = rdd.map(lambda metadata: Row(tmdbId= int(metadata[5]),title = metadata[20]))
    moviesMetadata = moviesMetadata.filter(lambda x: x[1] in filterMovieId)
    moviesMetadataDF = spark.createDataFrame(moviesMetadata)
    # moviesMetadataDF.show()

    intermediateDF = linkDataFrame.join(moviesMetadataDF,linkDataFrame.tmdbId == moviesMetadataDF.tmdbId, 'inner')
    intermediateDF = intermediateDF.sort('movieId', ascending=True)
    intermediateDF = intermediateDF.drop('tmdbId')
    # intermediateDF.show()

    finalDF = smallRatingDF.join(intermediateDF, smallRatingDF.movieId == intermediateDF.movieId, 'inner')
    finalDF = finalDF.drop('movieId')
    # finalDF.show()

    ratings_pivot = finalDF.groupBy("userId").pivot("title").agg(coalesce(first("rating")))
    ratings_pivot = ratings_pivot.na.fill(0)
    # ratings_pivot.show()
    ratings_pivot_RDD = ratings_pivot.rdd.map(lambda x: (x[1:]))
    # print(len(ratings_pivot_RDD.collect()))

    # print(len(movieDict))
    trainingRDD = ratings_pivot_RDD.zipWithIndex().filter(lambda x: x[1]<=26).map(lambda x: x[0])
    # print(len(trainingRDD.collect()))
    matrix = Statistics.corr(trainingRDD, method="pearson")
    # print(matrix[0])

    testRDD = ratings_pivot_RDD.zipWithIndex().filter(lambda x: x[1] == 27).flatMap(lambda x:x[0])
    # print(len(testRDD.collect()))
    testRDD = testRDD.zipWithIndex().filter(lambda x: x[0] != 0)
    testRating = testRDD.collect()
    length = len(testRating)
    testResult = list()
    for i in range(5):
        sims = matrix[testRating[length-1-i][1]]
        # print(sims)
        similarity = 0
        wightedSimilarity = 0
        for j in range(16):
            if math.isnan(sims[testRating[j][1]]):
                continue
            # else:
            #     print(sims[testRating[j][1]])
            similarity = similarity + sims[testRating[j][1]]
            wightedSimilarity = wightedSimilarity + sims[testRating[j][1]] * testRating[j][0]
        # print("--------------")
        # print(wightedSimilarity)
        # print(similarity)
        # print("--------------")
        testResult.append(wightedSimilarity / similarity)

    RMSE = 0
    for x in range(len(testResult)):
        # print(testResult[x])
        # print(testRating[length - 1 - x][0])
        RMSE = RMSE + (testResult[x] - testRating[length - 1 - x][0]) ** 2

    # print(RMSE)
    print(math.sqrt(RMSE))

# 3.591362615487066
# 3.583256710777442
# 3.9837358030886243
# 3.605477107922382
# 3.680190942939259







main()
