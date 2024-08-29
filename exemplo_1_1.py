import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# Dados de exemplo
X = np.array([[-3,-3], [-2,-2], [-1,-1], [0,0], [1,1], [2,2], [3,3],
[4,4], [5,5], [6,6]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size
=0.2, random_state=42)
# Criar o modelo de regressão logística
model = LogisticRegression()
# Treinar o modelo
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix,
index=["Actual 0", "Actual 1"],
columns=["Predicted 0", "Predicted 1"
])
# Gerar o relatório de classificação
class_report = classification_report(y_test, y_pred, output_dict=
True)
class_report_df = pd.DataFrame(class_report).transpose()
print("\nMétricas do Modelo ")
print(f"Acurácia do modelo: {accuracy:.2f}")
print("Matriz de Confusão:")
print(conf_matrix_df)
print("Relatório de Classificação:")
print(class_report_df)
print("\nPrevisão do Modelo:")
print("Previsão para a observação (X1=8 e X2=8)\n")
print(f"Resultado y = {model.predict([[8,8]])[0]}")