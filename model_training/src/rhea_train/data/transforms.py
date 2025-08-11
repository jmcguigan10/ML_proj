from sklearn.preprocessing import StandardScaler

def fit_standard_scaler(X_train):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_train)
    return sc, Xs

def apply_standard_scaler(scaler, X):
    return scaler.transform(X)

