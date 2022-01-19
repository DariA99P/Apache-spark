from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def random_forest_classifier(train, test):
    rf = RandomForestClassifier(featuresCol='features', labelCol='label')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    print("Random Forest Classifier: Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    return rf, rfModel, evaluator
