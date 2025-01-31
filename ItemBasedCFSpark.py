import csv
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import coalesce,first
from pyspark.mllib.stat import Statistics
from pyspark.ml.linalg import Vectors

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
    # print(ratings_pivot_RDD.collect())
    ratings_pivot_head = ratings_pivot.schema.names[1:]
    # print(ratings_pivot_head)
    movieDict = dict()
    # movieRatingDict = dict()
    movieNameRDD = sc.parallelize(ratings_pivot_head)
    movieRatingRDD = movieNameRDD.map(lambda x: (x,0))
    for i in range(0,len(ratings_pivot_head)):
        movieDict[ratings_pivot_head[i]] = i
        # movieRatingDict[i-1] = ratings_pivot_head[i]
    # print(len(movieDict))
    matrix = Statistics.corr(ratings_pivot_RDD, method="pearson")
    # print(matrix[0])

    userProfile = [['Harry Potter and the Chamber of Secrets', 5.0]]


    userMovies = list()
    for i in range(0,len(userProfile)):
        userMovies.append(userProfile[i][0])

    for i in range(0,len(userProfile)):
        print("add similarity for "+ userProfile[i][0] +" .....")
        # print("index is " + str(movieDict[userProfile[i][0]]))
        sims = matrix[movieDict[userProfile[i][0]]]
        simsRDD = sc.parallelize(sims)
        simsRDD = simsRDD.map(lambda x: x * userProfile[i][1])
        # print(userProfile[i][1])
        # print(simsRDD.collect())
        TempRDD = movieNameRDD.zip(simsRDD)
        # print(TempRDD.collect())
        movieRatingRDD = movieRatingRDD.union(TempRDD).reduceByKey(lambda a,b:a+b)
        # if i==2:
        # break
        # print(movieRatingRDD.collect())
        # print(len(sims))
        # print(sims)
        # break
    # print(movieRatingRDD.collect())
    movieRatingRDD = movieRatingRDD.filter(lambda x: x[0] not in userMovies)
    recomMovies = movieRatingRDD.map(lambda metadata: Row(MovieTitle = metadata[0],similarity = float(metadata[1])))
    recomMoviesDF = spark.createDataFrame(recomMovies)
    recomMoviesDF.createOrReplaceTempView("recomMovies")
    result = spark.sql("select MovieTitle,similarity from recomMovies order by similarity desc")
    result.show(truncate=False)






main()
