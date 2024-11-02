import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Pré-processamento: Preencher valores nulos e codificar variáveis categóricas
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Converter colunas categóricas em numéricas
le = LabelEncoder()
for column in ['Sex', 'Embarked']:
    data[column] = le.fit_transform(data[column])

# Selecionar as features e o alvo
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configurar a árvore de decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Fazer previsões e calcular a acurácia
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived'])

# Exibir no Streamlit
st.write("Acurácia da Árvore de Decisão:", accuracy)
st.write("Relatório de Classificação:", report)
st.write("Importância das Features:", clf.feature_importances_)
