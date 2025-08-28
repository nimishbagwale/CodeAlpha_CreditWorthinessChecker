import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelBinarizer

dataset = pd.read_csv("statlog+german+credit+data/german.data",header=None,delim_whitespace=True)

dataset.loc[(dataset[3] == "A48") | (dataset[3] == "A44") | (dataset[3] == "A45"),3] = "A410"
dataset.loc[dataset[8] == "A94", 8] = "A91"

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

categories = [
    ["A11","A12","A13","A14"] ,
    ["A34","A33","A32","A31","A30"] ,
    ["A65","A61","A62","A63","A64"] ,
    ["A71","A72","A73","A74","A75"] ,
    ["A101","A102","A103"] ,
    ["A124","A123","A122","A121"] ,
    ["A141","A142","A143"] ,
    ["A151","A152","A153"] ,
    ["A171","A172","A173","A174"]
]

le = LabelEncoder()
dataset.loc[:,18] = le.fit_transform(dataset.loc[:,18])
dataset.loc[:,19] = le.fit_transform(dataset.loc[:,19])

col_trans = ColumnTransformer(
    transformers=[
        ('ordinal',OrdinalEncoder(categories=categories),[0, 2, 5, 6, 9, 11, 13, 14, 16]) ,
        ('encoder',OneHotEncoder(),[3,8])
    ] , 
    remainder='passthrough'
)
dataset = col_trans.fit_transform(dataset)

no_columns = len(col_trans.named_transformers_['encoder'].get_feature_names_out())
print(no_columns)

dataset = pd.DataFrame(dataset)
dataset = dataset.astype(int)

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

y[y == 2] = 0

def train_predict_linear(x_train,y_train,x_test):
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    return classifier.predict(x_test)

def train_predict_tree(x_train, y_train, x_test, type="decision"):
    if type=="decision" or type=="d" :
        from sklearn.tree import DecisionTreeClassifier

        tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
        tree.fit(x_train, y_train)
    elif type=="random" or type=="r" :
        from sklearn.ensemble import RandomForestClassifier

        tree = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
        tree.fit(x_train, y_train)
        
    return tree.predict(x_test)

def train_ann(x_train, y_train, x_test, hidden_units=6, phs=100):
    import tensorflow as tf

    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.Dense(units=hidden_units, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=hidden_units, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=hidden_units, activation='relu'))

    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    ann.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    ann.fit(x_train, y_train, batch_size=40, epochs=phs)

    return ann.predict(x_test)

def train_naive_bayes(x_train, y_train, x_test):
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    
    return classifier.predict(x_test)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

def show_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)

    # ✅ Precision, Recall, F1
    precision = precision_score(y_test, y_pred, average="weighted")  
    recall = recall_score(y_test, y_pred, average="weighted")  
    f1 = f1_score(y_test, y_pred, average="weighted")

    # ✅ ROC-AUC (needs binary or one-vs-rest for multi-class)
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_pred_binarized = lb.transform(y_pred)

    roc_auc = roc_auc_score(y_test_binarized, y_pred_binarized, average="weighted")

    # ✅ Print results
    print("Model Performance Metrics:")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy , precision , recall , f1

y_pred = train_predict_linear(x_train, y_train, x_test)

lr_accuracy , lr_precision , lr_recall , lr_f1 = show_metrics(y_test,y_pred)

y_pred = train_naive_bayes(x_train, y_train, x_test)

nvb_accuracy , nvb_precision , nvb_recall , nvb_f1 = show_metrics(y_test,y_pred)

y_pred = train_predict_tree(x_train, y_train, x_test, type='r')

tr_accuracy ,tr_precision ,tr_recall ,tr_f1 = show_metrics(y_test,y_pred)

y_pred = train_ann(x_train, y_train, x_test, 11, 100)

y_pred = y_pred > 0.5
y_pred = y_pred.astype('int')
y_test = y_test.reshape(200,1)

an_accuracy ,an_precision ,an_recall ,an_f1 = show_metrics(y_test,y_pred)