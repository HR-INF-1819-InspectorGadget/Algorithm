from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, datetime
from datetime import date
from pyspark import SparkContext
from pyspark.sql import SparkSession


def SuggestIntervention(subjectid):

    print("[v] -> PySpark SQL")
    SPARK_URL = "local[*]"
    spark = SparkSession.builder.appName("Kapper").master(SPARK_URL).getOrCreate()
    sc = spark.sparkContext
    #Hier onze data importeren
    #collomen benoemd gebasseerd op query

    #['itvId','itvInterventieOptieId','itvRegieParentId','sjId','sjGender','sjDateOfBirth','sjMaritalStatusId','sjWoonplaatsId','casId','casClassification','casThemaGebiedId','lgscoreRegieParentId','lgscoreScore','probProbleemOptieId','itvGoalReached','itvGeresidiveerd']


    # DATA OPHALEN VAN ITV ID
    csvpath = "C:\FinalCSVnoNull.csv"
    df = spark.read.options(header = "true", inferschema = "true").csv(csvpath)
    print("[v] -> csv")

    def calculate_age(born):
        born = datetime.datetime.strptime(born, "%Y-%m-%d %H:%M:%S").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    calculate_age_udf = udf(calculate_age, IntegerType())
    df = df.withColumn("sjAge", calculate_age_udf(df.sjDateOfBirth.cast("string")))
    print("[v] -> ageconvertion")

    #onehot encoding zorgt ervoor dat dingen die niet int zijn worden omgezet naar getallen die onderscheiden kunnen worden
    column_vec_in = ['itvInterventieOptieId','itvRegieParentId','sjGender','sjMaritalStatusId','sjWoonplaatsId','casClassification','casThemaGebiedId','lgscoreRegieParentId','lgscoreScore','probProbleemOptieId']
    column_vec_out = ['itvInterventieOptieIdvec','itvRegieParentIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec','probProbleemOptieIdvec']
    indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp') for x in column_vec_in ]
    encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y)
                for x,y in zip(column_vec_in, column_vec_out)]
    tmp = [[i,j] for i,j in zip(indexers, encoders)]
    tmp = [i for sublist in tmp for i in sublist]
    print("[v] -> onehotencoding")


    #finalize with pipeline
    cols_now = ['itvInterventieOptieIdvec','itvRegieParentIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec','probProbleemOptieIdvec','itvGoalReached','sjAge']
    assembler_features = VectorAssembler(inputCols=cols_now, outputCol='parameters')
    labelIndexer = StringIndexer(inputCol='itvGeresidiveerd', outputCol="resultintervention")
    tmp += [assembler_features, labelIndexer]
    pipeline = Pipeline(stages=tmp)


    allData = pipeline.fit(df).transform(df)
    allData = allData.select(['itvInterventieOptieId', 'itvInterventieOptieIdvec','itvRegieParentIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec','probProbleemOptieIdvec','itvGoalReached','sjAge','parameters',"resultintervention"])
    print("[v] -> pipeline")
    allData.cache()
    print("[v] -> cashe")
    trainingData, testData = allData.randomSplit([0.8,0.2], seed=42069) # need to ensure same split for each time
    print("traindata amount" + str(trainingData.count()))
    print("testdata amount" + str(testData.count()))

    rf = RF(labelCol='resultintervention', featuresCol='parameters', numTrees=200)
    rfit = rf.fit(trainingData)
    transformed = rfit.transform(testData)
    print("hierzo")
    transformed.select(['itvInterventieOptieId', 'itvInterventieOptieIdvec']).show()

    #dit moet gereturnend worden!!!!!!!!!!
    results = transformed.select(['itvInterventieOptieId', 'probability', 'resultintervention'])
    #dit ^^^^^^^^^^^^^^^^^^^
    return results.show()

SuggestIntervention("asdfasdfadsfa")
