from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
import numpy as np
import functools
from pyspark.ml.feature import OneHotEncoder
print("[-] -> Application Start")

from pyspark import SparkContext
print("[v] -> PySpark")

from pyspark.sql import SparkSession
print("[v] -> PySpark SQL")
SPARK_URL = "local[*]"
spark = SparkSession.builder.appName("SimpleApp").master(SPARK_URL).getOrCreate()

#Hier onze data importeren
#collomen benoemd gebasseerd op query
cols_select = ['itvOwnerId', 'itvInterventieOptieId', 'sjGender', 'sjDateOfBirth', 'sjMaritalStatusId', 'sjWoonplaatsId', 'lgscoreScore', 'itvGoalReached']

csvpath = "C:\InterventiecsvNONULL.csv"
df = spark.read.options(header = "true", inferschema = "true").csv(csvpath)
print("[v] -> csv")


df.printSchema()
df.select('sjDateOfBirth').show()



# geen idee of dit werkt
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from datetime import date


## dit werkt dus niet, wil hier date omrekenen naar tijd en dan in colomn age zetten
#def calculate_age():
    
#    today = date.today()
#    return 1 #today.year - born.year - ((today.month, today.day) < (born.month, born.day))
  
#udfValueToCategory = udf(calculate_age, IntegerType())
#df = df.withColumn("sjAge", udfValueToCategory("sjDateOfBirth"))
#print("[v] -> ageconvertion")
#df.printSchema()
#df.select('sjDateOfBirth').show()
#df.select('sjAge').show()


#onehot encoding zorgt ervoor dat dingen die niet int zijn worden omgezet naar getallen die onderscheiden kunnen worden
cols_select = ['itvOwnerId', 'itvInterventieOptieId', 'sjGender', 'sjMaritalStatusId', 'sjWoonplaatsId', 'lgscoreScore', 'itvGoalReached']

column_vec_in = ['itvOwnerId', 'itvInterventieOptieId', 'sjGender', 'sjMaritalStatusId', 'sjWoonplaatsId']

column_vec_out = ['itvOwnerIdvec', 'itvInterventieOptieIdvec', 'sjGendervec', 'sjMaritalStatusIdvec', 'sjWoonplaatsIdvec']
 
indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp') for x in column_vec_in ]
 
encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y)
            for x,y in zip(column_vec_in, column_vec_out)]

tmp = [[i,j] for i,j in zip(indexers, encoders)]
tmp = [i for sublist in tmp for i in sublist]

print("[v] -> onehotencoding")


#finalize with pipeline
cols_now = ['itvOwnerIdvec', 'itvInterventieOptieIdvec', 'sjGendervec', 'sjMaritalStatusIdvec', 'sjWoonplaatsIdvec', 'lgscoreScore', 'itvGoalReached']

assembler_features = VectorAssembler(inputCols=cols_now, outputCol='variables')
labelIndexer = StringIndexer(inputCol='itvGoalReached', outputCol="resultintervention")
tmp += [assembler_features, labelIndexer]
pipeline = Pipeline(stages=tmp)

allData = pipeline.fit(df).transform(df)
print("[v] -> pipeline")
allData.cache()
print("[v] -> cashe")
trainingData, testData = allData.randomSplit([0.8,0.2], seed=0) # need to ensure same split for each time
print("Distribution of Pos and Neg in trainingData is: ", trainingData.groupBy("variables").count().take(3))
allData.printSchema()