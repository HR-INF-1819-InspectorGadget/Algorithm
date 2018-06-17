from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import IntegerType, datetime
from datetime import date
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
import igDatabaseModule
import pandas


def SuggestIntervention(subjectid):

    print("[v] -> PySpark SQL")
    SPARK_URL = "local[*]"
    spark = SparkSession.builder.appName("").master(SPARK_URL).getOrCreate()
    sc = spark.sparkContext
    #Hier onze data importeren
    #collomen benoemd gebasseerd op query
    sqlcontext = SQLContext(sc)
    columns = ['itvInterventieOptieId','itvRegieParentId','sjId','sjGender','sjDateOfBirth','sjMaritalStatusId','sjWoonplaatsId','casId','casClassification','casThemaGebiedId','lgscoreRegieParentId','lgscoreScore','probProbleemOptieId','itvGoalReached','itvGeresidiveerd']


    # DATA OPHALEN VAN ITV ID
    database = igDatabaseModule.Database("PGA_HRO")
    subjectdata = database.get_data_from_the_database('SELECT distinct intoptid, sjId ,sjGender,sjDateOfBirth,sjMaritalStatusId,sjWoonplaatsId,casId,casClassification,casThemaGebiedId,lgscoreRegieParentId,lgscoreScore FROM tblSubject, tblMeldingZSMPlus, tblCasus, tblLeefgebiedScore, (select * from tblInterventieOptie where intoptIsInactive = 0) as interventies where zsmSubjectId = sjid and zsmCasusId = casId and lgscoreRegieParentId = casId and casThemaGebiedId is not null and sjMaritalStatusId is not null and sjWoonplaatsId is not Null and sjid = {};'.format(subjectid))
    pandadf = pandas.DataFrame(subjectdata, columns=['itvInterventieOptieId', 'sjId', 'sjGender','sjDateOfBirth','sjMaritalStatusId','sjWoonplaatsId','casId','casClassification', 'casThemaGebiedId','lgscoreRegieParentId','lgscoreScore'])
    
    subjectDF = sqlcontext.createDataFrame(pandadf)

    goalreacheudf = udf(lambda x: x, IntegerType())
    subjectDF = subjectDF.withColumn("itvGoalReached", lit(1))
    subjectDF = subjectDF.withColumn("itvGeresidiveerd", lit("Nee"))
    print("[v] -> parallelize")

   
    csvpath = "c:\FinalCSVnoNull.csv"
    dataDF = spark.read.options(header = "true", inferschema = "true").csv(csvpath)
    print("[v] -> csv")
    dataDF = dataDF.select(['itvInterventieOptieId', 'sjId', 'sjGender','sjDateOfBirth','sjMaritalStatusId','sjWoonplaatsId','casId','casClassification', 'casThemaGebiedId','lgscoreRegieParentId','lgscoreScore','itvGoalReached','itvGeresidiveerd'])


    def calculate_age(born):
        born = datetime.datetime.strptime(born, "%Y-%m-%d %H:%M:%S").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def calculate_age_subject(born):
        born = datetime.datetime.strptime(born, "%Y-%m-%d").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    calculate_age_udf = udf(calculate_age, IntegerType())
    calculate_age_udf_subject = udf(calculate_age_subject, IntegerType())
    
    dataDF = dataDF.withColumn("sjAge", calculate_age_udf(dataDF.sjDateOfBirth.cast("string")))
    
    subjectDF = subjectDF.withColumn("sjAge", calculate_age_udf_subject(subjectDF.sjDateOfBirth.cast("string")))
    print("[v] -> ageconvertion")

    df = dataDF.union(subjectDF)

    #onehot encoding zorgt ervoor dat dingen die niet int zijn worden omgezet naar getallen die onderscheiden kunnen worden
    column_vec_in = ['itvInterventieOptieId','sjGender','sjMaritalStatusId','sjWoonplaatsId','casClassification','casThemaGebiedId','lgscoreRegieParentId','lgscoreScore']
    column_vec_out = ['itvInterventieOptieIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec']
    indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp') for x in column_vec_in ]
    encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y)
                for x,y in zip(column_vec_in, column_vec_out)]
    tmp = [[i,j] for i,j in zip(indexers, encoders)]
    tmp = [i for sublist in tmp for i in sublist]
    print("[v] -> onehotencoding")


    #finalize with pipeline
    cols_now = ['itvInterventieOptieIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec','itvGoalReached','sjAge']
    assembler_features = VectorAssembler(inputCols=cols_now, outputCol='parameters')
    labelIndexer = StringIndexer(inputCol='itvGeresidiveerd', outputCol="resultintervention")
    tmp += [assembler_features, labelIndexer]
    pipeline = Pipeline(stages=tmp)


    allData = pipeline.fit(df).transform(df)
    allData = allData.select(['itvInterventieOptieId', 'sjId', 'itvInterventieOptieIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec','itvGoalReached','sjAge','parameters',"resultintervention"])
    print("[v] -> pipeline")
    allData.cache()
    print(str(allData.count()))
    print("[v] -> trainingcashe")

    trainingData = allData.filter("not sjId = {}".format(subjectid))
    testData = allData.filter("sjId = {}".format(subjectid))

    trainingData = trainingData.select(['itvInterventieOptieIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec','itvGoalReached','sjAge','parameters',"resultintervention"])
    testc = testData.select(['itvInterventieOptieIdvec','sjGendervec','sjMaritalStatusIdvec','sjWoonplaatsIdvec','casClassificationvec','casThemaGebiedIdvec','lgscoreRegieParentIdvec','lgscoreScorevec','itvGoalReached','sjAge','parameters',"resultintervention"])
    
    print("traindata amount" + str(trainingData.count()))
    print("testdata amount" + str(testData.count()))

    rf = RF(labelCol='resultintervention', featuresCol='parameters', numTrees=200)
    rfit = rf.fit(trainingData)
    transformed = rfit.transform(testData)
    print("hierzo")
    transformed.select(['itvInterventieOptieId', 'itvInterventieOptieIdvec'])

    #dit moet gereturnend worden!!!!!!!!!!
    results = transformed.select(['itvInterventieOptieId','probability', 'prediction'])
    #dit ^^^^^^^^^^^^^^^^^^^
    resultslist = results.toJSON()
    return resultslist

SuggestIntervention("6")
