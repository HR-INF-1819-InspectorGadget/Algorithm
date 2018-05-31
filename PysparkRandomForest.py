from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, TimestampType, StringType, datetime
import pandas as pd
import numpy as np
import functools
from datetime import date
print("[-] -> Application Start")

from pyspark import SparkContext
print("[v] -> PySpark")

from pyspark.sql import SparkSession
print("[v] -> PySpark SQL")
SPARK_URL = "local[*]"
spark = SparkSession.builder.appName("SimpleApp").master(SPARK_URL).getOrCreate()
sc = spark.sparkContext

#Hier onze data importeren
#collomen benoemd gebasseerd op query
cols_select = ['itvOwnerId', 'itvInterventieOptieId', 'sjGender', 'sjDateOfBirth', 'sjMaritalStatusId', 'sjWoonplaatsId', 'lgscoreScore', 'itvGoalReached']

csvpath = "C:\InterventiecsvNONULL.csv"
df = spark.read.options(header = "true", inferschema = "true").csv(csvpath)
print("[v] -> csv")


df.printSchema()
df.count()
df.show()
def calculate_age(born):
    born = datetime.datetime.strptime(born, "%Y-%m-%d %H:%M:%S").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

calculate_age_udf = udf(calculate_age, IntegerType())
df = df.withColumn("sjAge", calculate_age_udf(df.sjDateOfBirth.cast("string")))

binary_to_bool_udf = udf(lambda x: 'reached' if x ==1  else 'notreached', StringType())
df = df.withColumn("itvGoalReachedString", binary_to_bool_udf(df.itvGoalReached))

print("[v] -> ageconvertion")

#df.printSchema()

from matplotlib import pyplot as plt

#print("Histograms below: ")

#responses = df.groupBy('sjWoonplaatsId').count().collect() # list of Rows
#categories = [i[0] for i in responses]
#counts = [i[1] for i in responses]

#ind = np.array(range(len(categories)))
#width = 0.35
#plt.bar(ind, counts, width=width, color='r')

#plt.ylabel('counts')
#plt.title('Woonplaats distribution')
#plt.xticks(ind + width/2., categories)
#plt.show()

#onehot encoding zorgt ervoor dat dingen die niet int zijn worden omgezet naar getallen die onderscheiden kunnen worden
cols_select = ['itvOwnerId', 'itvInterventieOptieId', 'sjGender', 'sjMaritalStatusId', 'sjWoonplaatsId', 'lgscoreScore', 'sjAge', 'itvGoalReachedString']

column_vec_in = ['itvOwnerId', 'itvInterventieOptieId', 'sjGender', 'sjMaritalStatusId', 'sjWoonplaatsId']

column_vec_out = ['itvOwnerIdvec', 'itvInterventieOptieIdvec', 'sjGendervec', 'sjMaritalStatusIdvec', 'sjWoonplaatsIdvec']
 
indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp') for x in column_vec_in ]
 
encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y)
            for x,y in zip(column_vec_in, column_vec_out)]

tmp = [[i,j] for i,j in zip(indexers, encoders)]
tmp = [i for sublist in tmp for i in sublist]

print("[v] -> onehotencoding")


#finalize with pipeline
cols_now = ['itvOwnerIdvec', 'itvInterventieOptieIdvec', 'sjGendervec', 'sjMaritalStatusIdvec', 'sjWoonplaatsIdvec', 'lgscoreScore', 'sjAge']

assembler_features = VectorAssembler(inputCols=cols_now, outputCol='variables')
labelIndexer = StringIndexer(inputCol='itvGoalReachedString', outputCol="resultintervention")
tmp += [assembler_features, labelIndexer]
pipeline = Pipeline(stages=tmp)
print("adf")


allData = pipeline.fit(df).transform(df)
allData.show()
allData = allData.select(allData['itvOwnerIdvec'], allData['itvInterventieOptieIdvec'], allData['sjGendervec'], allData['sjMaritalStatusIdvec'], allData['sjWoonplaatsIdvec'], allData['lgscoreScore'], allData['sjAge'], allData['itvGoalReachedString'], allData['variables'], allData["resultintervention"])
allData.show()
print("[v] -> pipeline")
allData.cache()
print("[v] -> cashe")
trainingData, testData = allData.randomSplit([0.8,0.2], seed=69) # need to ensure same split for each time
print("traindata amount" + str(trainingData.count()))
print("testdata amount" + str(testData.count()))
print("trainingdata")
trainingData.describe().show()
print("trainingdata2")
trainingData.show()

print("testdata")
testData.show()
print("alldata")
allData.printSchema()

rf = RF(labelCol='resultintervention', featuresCol='variables', numTrees=200)
fit = rf.fit(trainingData)
transformed = fit.transform(testData)


results = transformed.select(['probability', 'resultintervention'])
 
## prepare score-label set
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
print("result list")
scoreAndLabels = sc.parallelize(results_list)

print("metrics") 
metrics = metric(scoreAndLabels)
print("The ROC score is (@numTrees=200): ", metrics.areaUnderROC)