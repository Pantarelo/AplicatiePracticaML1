import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data['Data'] = pd.to_datetime(data['Data'])
    data = data.dropna()

    numeric_columns = ["Consum[MW]", "Carbune[MW]", "Hidrocarburi[MW]", "Ape[MW]",
                       "Nuclear[MW]", "Eolian[MW]", "Foto[MW]", "Biomasa[MW]", "Sold[MW]"]
    for col in numeric_columns:
        data[col] = data[col].replace(r'[^\d.-]', '', regex=True).astype(float)

    return data


file_path = 'Grafic_SEN.xlsx'
data = preprocess_data(file_path)

#date antrenare (ianuarie-noiembrie) si testare (decembrie)
train_data = data[data['Data'].dt.month.isin(range(1, 12))]
test_data = data[data['Data'].dt.month == 12]

X_train = train_data[["Consum[MW]", "Carbune[MW]", "Hidrocarburi[MW]", "Ape[MW]",
                      "Nuclear[MW]", "Eolian[MW]", "Foto[MW]", "Biomasa[MW]"]]
y_train = train_data["Sold[MW]"]

X_test = test_data[["Consum[MW]", "Carbune[MW]", "Hidrocarburi[MW]", "Ape[MW]",
                    "Nuclear[MW]", "Eolian[MW]", "Foto[MW]", "Biomasa[MW]"]]
y_test = test_data["Sold[MW]"]

#ID3 (Arbore de Decizie pentru regresie)
id3_regressor = DecisionTreeRegressor(max_depth=5)
id3_regressor.fit(X_train, y_train)

#decembrie ID3
id3_december_predictions = id3_regressor.predict(X_test)

id3_december_mean = id3_december_predictions.mean()


#Model Bayesian
def naive_bayes_regression(X_train, y_train, X_test):
    buckets = pd.cut(y_train, bins=10, labels=False)
    means = y_train.groupby(buckets).mean()

    predictions = []
    for _, row in X_test.iterrows():
        predicted_bucket = np.random.choice(means.index)
        predictions.append(means[predicted_bucket])

    return np.array(predictions)


bayes_december_predictions = naive_bayes_regression(X_train, y_train, X_test)

#Bayes
bayes_december_mean = bayes_december_predictions.mean()

real_december_mean = y_test.mean()


print("Media valorilor pentru luna decembrie:")
print(f"Valori reale: {real_december_mean}")
print(f"Predictii ID3: {id3_december_mean}")
print(f"Predictii Bayes: {bayes_december_mean}")

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Valori reale", color="blue")
#plt.plot(id3_december_predictions, label="Predictii ID3", color="green")
plt.plot(bayes_december_predictions, label="Predictii Bayes", color="orange")
plt.legend()
plt.title("Comparatie Predictii vs Valori Reale pentru Decembrie")
plt.show()
