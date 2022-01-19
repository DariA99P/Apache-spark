from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def gradient_boosted_tree_classifier(train, test):
    gbt = GBTClassifier(maxIter=10)
    gbtModel = gbt.fit(train)
    predictions = gbtModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    print("Gradient-Boosted Tree Classifier: Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    return gbt, gbtModel, evaluator
