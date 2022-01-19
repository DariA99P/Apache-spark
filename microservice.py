from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from logistic_regression import logistic_regression
from random_forest_classifier import random_forest_classifier
from decision_tree_classifier import decision_tree_classifier
from gradient_boosted_tree_classifier import gradient_boosted_tree_classifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def microservice():
    spark = SparkSession.builder.appName('ml-bank').getOrCreate()
    df = spark.read.csv('bank.csv', header=True, inferSchema=True)
    print('-----Schema-----')
    df.printSchema()
    cols = df.schema.names
    print('-----First 5 users-----')
    print(pd.DataFrame(df.take(5), columns=df.columns).transpose())

    print('-----Summary statistics for numeric variables-----')
    numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
    df.select(numeric_features).describe().toPandas().transpose()
    print(df)

    print('-----Preparing Data for Machine Learning-----')
    categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    label_stringIdx = StringIndexer(inputCol='deposit', outputCol='label')
    stages += [label_stringIdx]
    numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['label', 'features'] + cols
    df = df.select(selectedCols)
    df.printSchema()

    print(pd.DataFrame(df.take(5), columns=df.columns).transpose())

    # Розподіл на тестові та навчальні вибірки
    train, test = df.randomSplit([0.7, 0.3], seed=2018)
    print("Кількість набору навчальних даних: " + str(train.count()))
    print("Кількість тестових наборів даних: " + str(test.count()))

    # Метод логістичної регресії
    lr, lrModel, evaluatorLr = logistic_regression(train, test)

    # Класифікатор випадкових лісів
    rf, rfModel, evaluatorRf = random_forest_classifier(train, test)

    # Класифікатор дерева рішень
    dt, dtModel, evaluatorDt = decision_tree_classifier(train, test)

    # Класифікатор дерев із підсиленим градієнтом
    gbt, gbtModel, evaluatorGbt = gradient_boosted_tree_classifier(train, test)

    # Налаштування моделі за допомогою ParamGridBuilder і CrossValidator
    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [2, 4, 6])
                 .addGrid(gbt.maxBins, [20, 60])
                 .addGrid(gbt.maxIter, [10, 20])
                 .build())
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluatorGbt, numFolds=5)
    cvModel = cv.fit(train)
    predictions = cvModel.transform(test)
    evaluatorGbt.evaluate(predictions)