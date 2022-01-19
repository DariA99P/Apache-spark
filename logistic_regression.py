from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import numpy as np

def logistic_regression(train, test):
    lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
    lrModel = lr.fit(train)

    beta = np.sort(lrModel.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    plt.show()

    trainingSummary = lrModel.summary
    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'], roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

    pr = trainingSummary.pr.toPandas()
    plt.plot(pr['recall'], pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

    predictions = lrModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

    evaluator = BinaryClassificationEvaluator()
    print('Logistic Regression: Test Area Under ROC', evaluator.evaluate(predictions))
    return lr, lrModel, evaluator

