import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle 

def create_model(data):
    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 42, test_size= 0.2)

    # train the data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model 
    y_pred  = model.predict(X_test)

    # Accuracy Score
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    print("Classification Report: \n ", classification_report(y_test, y_pred))

    return model, scaler 

def get_clean_data():
    # Load dataset
    data = pd.read_csv('Projects/breastCancerPredictor/data/data.csv', engine = 'python')
    # Drop unwanted columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    # Map benign/ malignant to 0 and 1 respectively
    data['diagnosis'] = data['diagnosis'].map({'M': 1,'B':0})

    return data 

def export_model(model, scaler):
    # Export ML Model
    with open('Projects/breastCancerPredictor/model/model.pkl','wb') as file:
        pickle.dump(model, file)
    
    # Export Scaler 
    with open('Projects/breastCancerPredictor/model/scaler.pkl','wb') as file:
        pickle.dump(scaler, file)

def main():
    data = get_clean_data()
    model, scaler = create_model(data)
    export_model(model, scaler)

if __name__ == '__main__':
    main()