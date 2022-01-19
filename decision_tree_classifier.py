from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def decision_tree_classifier(train, test):
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=3)
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    print("Decision Tree: Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    return dt, dtModel, evaluator
