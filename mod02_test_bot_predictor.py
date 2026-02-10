# packages
import pandas as pd
from mod02_build_bot_predictor import train_model
#takes in the building the bot file


def predict_bot(df, model=None):
    #set to None so happens every time?
    """
    Predict whether each account is a bot (1) or human (0).
    #so this is where the predicting happens
    """
    if model is None:
        model = train_model()

    preds = model.predict(df)
    #uses model used for training, to now make these predictions
    return pd.Series(preds, index=df.index)

def confusion_matrix_and_metrics(y_true, y_pred):
    """
    Computes confusion matrix and common error rates for binary classification.

    Assumes labels:
      0 = negative class
      1 = positive class

    Returns:
      dict with:
        tn, fp, fn, tp
        misclassification_rate
        #tp means bot that is also predicted as bot

        #tn means person that is also predicted as person

        
        #both fp and fn bad, fp = thought human, actually bot
        #fn = thought bot actually human
        false_positive_rate
        false_negative_rate
    """
    tn = fp = fn = tp = 0

    for yt, yp in zip(y_true, y_pred):
        #y_true ='s if bot or if human
        #y_pred ='s what machine predicted as
        if yt == 0 and yp == 0:
            #correct, found human
            tn += 1

        elif yt == 0 and yp == 1:
            #incorrect, is human 0, thought bot 1
            fp += 1
        elif yt == 1 and yp == 0:
            #incorrect, is bot 1, thought human 0
            fn += 1
        elif yt == 1 and yp == 1:
            #correct, found bot
            tp += 1
        else:
            raise ValueError("Labels must be 0 or 1")

    total = tn + fp + fn + tp

    #percent, as decm, of predictions did wrong
    misclassification_rate = (fp + fn) / total if total > 0 else 0.0

    #how many humans wrong
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    #how many bots did not get
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "misclassification_rate": misclassification_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


TRAIN_PATH = "mod02_data/train.csv"
#should be mod02 data???

#around training data
train = pd.read_csv(TRAIN_PATH)

TEST_PATH = "mod02_data/test.csv"
#should be mod02 data???

#evals how correct model was
test = pd.read_csv(TEST_PATH)

#takes out all columns that say they are a bot in the file used for 
#training
X_train = train.drop(columns=["is_bot"])
y_train = train['is_bot']

##takes out all columns that say they are a bot
#for TESTING with NEW tests
X_test = test.drop(columns=["is_bot"])
y_test = test['is_bot']

model = train_model(X_train, y_train)

#used to train the machine when making guesses on what is bot or 
#human
y_pred_train = predict_bot(X_train, model)

#tests the actual bot
y_pred_test = predict_bot(X_test, model)


#goes back to function above in this file
print( confusion_matrix_and_metrics(y_train, y_pred_train) )

print( confusion_matrix_and_metrics(y_test, y_pred_test) )










