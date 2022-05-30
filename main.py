import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score, davies_bouldin_score
from sklearn.model_selection import KFold


def Vizualiziáció(x:np.ndarray, labels:np.ndarray, cl_type:str,n_clusters:int):

    u_labels = np.unique(labels)
    colors = ['r','b','g','y','c','m','k','w']
    for idx_lab in range( min(len(u_labels),n_clusters)):
        for idx in range(len(x)):
            if u_labels[idx_lab] == labels[idx]:
                plt.scatter(x[idx][0],x[idx][1],c=colors[idx_lab])

    plt.suptitle(cl_type)
    plt.show()



def klasszifikáció(x_tr:pd.DataFrame,x_ts:pd.DataFrame,y_tr:np.ndarray,tp:str,n_neighbors:int=5,
             c:float =0.5,kernel:str='linear'):
    if tp=='LogReg':
        model=LogisticRegression(penalty='l2')
    elif tp == 'SVC':
        model=SVC(C=c,kernel=kernel)
    elif tp == 'KNN':
        model=KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        print(f"Not valid type: {tp}")
        exit(1)

    model.fit(X=x_tr,y=y_tr)
    y_pred = model.predict(X=x_ts)
    return y_pred

def eval_cl(tp:str,x:pd.DataFrame,y_pred:np.ndarray, y_true:np.ndarray=None):
    if tp == 'Silhouette':
        return silhouette_score(X=x,labels=y_pred)
    elif tp == 'Davis-Bouldin':
        return davies_bouldin_score(X=x,labels=y_pred)
    elif tp == 'AUC':
        return roc_auc_score(y_true=y_true,y_score=y_pred)
    elif tp == 'f1':
        return f1_score(y_true=y_true,y_pred=y_pred)
    elif tp == 'accuracy':
        return accuracy_score(y_true=y_true,y_pred=y_pred)
    else:
        print(f"Not valid type: {tp}")
        exit(2)




if __name__ == "__main__":

    data_header = ["Folyoszámla", "időtartam", "hiteltörténet", "jov_cél", "összeg", "kötvényei", "foglalk", "arány",
                   "nem+állapot", "hitelei", "lakóhelye", "ingatlanok", "kor", "tervek", "lakh_helyzet", "jóváírások",
                   "munkakörülmény", "Karbantartók", "telefon", "külföldi-e", "cimke"]

    data = pd.read_csv(filepath_or_buffer="project_data.csv", sep=";", header=None, skiprows=1, names=data_header)
    # Kiiratom hány sorból és oszlopból áll az adathalmazom.
    print(data.shape)
    # Kiiratom az eredeti adat első 5 sorát.
    print(data.head(n=5))
    # kiiratom az adatok tipusait
    data.info()
    data.isnull().sum()

    # 1. részfeladat . . . Nominális adatok LabelEncoderrel való adatfeldolgozás
    le = LabelEncoder()

    tmp = le.fit_transform(data["hiteltörténet"])
    data.insert(21, "le_hiteltört.", tmp)
    data.drop(columns=["hiteltörténet"], inplace=True)

    tmp = le.fit_transform(data["jov_cél"])
    # hozzáaduk a 21.sorba
    data.insert(21, "le_jov_cél", tmp)
    # kitölröm az eredeti sima jov_cél oszlopot
    data.drop(columns=["jov_cél"], inplace=True)

    tmp = le.fit_transform(data["nem+állapot"])
    data.insert(21, "le_nem+áll.", tmp)
    data.drop(columns=["nem+állapot"], inplace=True)

    tmp = le.fit_transform(data["hitelei"])
    data.insert(21, "le_hitelei", tmp)
    data.drop(columns=["hitelei"], inplace=True)

    tmp = le.fit_transform(data["ingatlanok"])
    data.insert(21, "le_ingatlanok", tmp)
    data.drop(columns=["ingatlanok"], inplace=True)

    tmp = le.fit_transform(data["tervek"])
    data.insert(21, "le_tervek", tmp)
    data.drop(columns=["tervek"], inplace=True)

    tmp = le.fit_transform(data["lakh_helyzet"])
    data.insert(21, "le_lakh_helyzet", tmp)
    data.drop(columns=["lakh_helyzet"], inplace=True)

    # 1.b részfeladat . . . Nominális adatok (melyeknek kevés a változója) One hot encoderrel való adatfeldolgozás
    tmp2 = pd.DataFrame(OneHotEncoder(categories=[["A191", "A192"]]).fit_transform(data[["telefon"]]).toarray(),
                        columns=["nincs_telefon", "van_telefon"])
    data = data.join(tmp2)
    data.drop(columns=["telefon"], inplace=True)

    tmp2 = pd.DataFrame(OneHotEncoder(categories=[["A201", "A202"]]).fit_transform(data[["külföldi-e"]]).toarray(),
                        columns=["külföldi", "Nem_külföldi"])
    data = data.join(tmp2)
    data.drop(columns=["külföldi-e"], inplace=True)

    # 1.c részfeladat . . . Nem Nominális adatok feldolgozása Ordinal Encoderrel
    oe = OrdinalEncoder(categories=[["A14", "A11", "A12", "A13"]])
    data.insert(1, "oe_Folyoszám", oe.fit_transform(data.loc[:, ["Folyoszámla"]]))
    data.drop(columns=["Folyoszámla"], inplace=True)

    oe = OrdinalEncoder(categories=[["A65", "A61", "A62", "A63", "A64"]])
    data.insert(1, "oe_kötvény", oe.fit_transform(data.loc[:, ["kötvényei"]]))
    data.drop(columns=["kötvényei"], inplace=True)

    oe = OrdinalEncoder(categories=[["A71", "A72", "A73", "A74", "A75"]])
    data.insert(1, "oe_foglalk", oe.fit_transform(data.loc[:, ["foglalk"]]))
    data.drop(columns=["foglalk"], inplace=True)

    oe = OrdinalEncoder(categories=[["A171", "A172", "A173", "A174"]])
    data.insert(1, "oe_munkakö.", oe.fit_transform(data.loc[:, ["munkakörülmény"]]))
    data.drop(columns=["munkakörülmény"], inplace=True)

    # 1.d Skálázom a számokból álló adatsoraimat
    data.insert(0, "min_max_összeg", MinMaxScaler().fit_transform(data["összeg"].values.reshape(-1, 1)))
    data.drop(columns=["összeg"], inplace=True)
    data.insert(0, "min_max_idő", MinMaxScaler().fit_transform(data["időtartam"].values.reshape(-1, 1)))
    data.drop(columns=["időtartam"], inplace=True)

    print(data.shape)

    # 2.a Túlillesztés miatt kidobálom a nem kívánatos oszlopokat
    data.drop(columns=["le_tervek"], inplace=True)
    data.drop(columns=["lakóhelye"], inplace=True)

    # kiírom hánny sorból áll az anyag
    print(data.shape)
    # kiirom ez első 5 sort ellenőrzés szempontból.
    print(data.head(n=5))
    data.info()
    data.isnull().sum()

    # 3. Grafikus ábrázolása különböző adatoknak pl. kor szerinti eloszlása a hitelfelvétel összegének(arányáról)
    # sns.scatterplot(data=data, x='kor', y='min_max_összeg')
    # sns.lineplot(data=data, x='kor', y='min_max_összeg')
    # plt.show()

    # sns.lineplot(data=data, x='le_nem+áll.', y='kor')
    # plt.show()

    # 3.b
    cl = ['LogReg', 'SVC', 'KNN']
    eval_metrics = ['Silhouette', 'Davis-Bouldin', 'AUC', 'f1', 'accuracy']
    kernels = ['linear', 'poly', 'rbf']

    n_dims = 13
    x_prep = data
    y = data['cimke']
    c = 2
    n_neighbors = 4
    kernel = kernels[1]
    
    #eredeti adathalmaz
    x_orig = pd.read_csv(filepath_or_buffer="project_data.csv", sep=";", header=None, skiprows=1, names=data_header)

    pca = PCA(n_components=n_dims)
    x_tr_2D = pca.fit_transform(x_prep)

    #kirajzolom az eredeti adathalmazt
    #Vizualiziáció(x_tr_2D,y,'ORIG',n_dims)

    test = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, ts_idx in test.split(X=x_tr_2D):
        x_tr, x_ts = x_tr_2D[tr_idx], x_tr_2D[ts_idx]
        y_tr, y_ts = y[tr_idx], y[ts_idx]

    tp = 'KNN'
    for n_neighbors in [2,4,5,8]:
        y_pred = klasszifikáció(x_tr=x_tr, x_ts=x_ts, y_tr=y_tr, tp=tp,
                          n_neighbors=n_neighbors, c=c, kernel=kernel)
        #Vizualiziáció(x_ts, y_pred, f"{tp}: n_n: {n_neighbors}", n_dims)
        print(f"alg: {tp}, n_n: {n_neighbors} AUC: {eval_cl('AUC', x_tr_2D, y_pred, y_ts)}, "
              f"f1: {eval_cl('f1', x_tr_2D, y_pred, y_ts)}, "
              f"acc: {eval_cl('accuracy', x_tr_2D, y_pred, y_ts)}")

    tp = 'SVC'
    for c in [0.1, 0.5, 2,]:
        for kernel in kernels:
            y_pred = klasszifikáció(x_tr=x_tr, x_ts=x_ts, y_tr=y_tr, tp=tp,
                              n_neighbors=n_neighbors, c=c, kernel=kernel)
            #Vizualiziáció(x_ts, y_pred, f"{tp}: c: {c}, kernel: {kernel}", n_dims)
            print(f"alg: {tp}, c: {c}, kernel: {kernel}, "
                  f"AUC: {eval_cl('AUC', x_tr_2D, y_pred, y_ts)}, "
                  f"f1: {eval_cl('f1', x_tr_2D, y_pred, y_ts)}, "
                  f"acc: {eval_cl('accuracy', x_tr_2D, y_pred, y_ts)}")

    tp = 'LogReg'
    y_pred = klasszifikáció(x_tr=x_tr, x_ts=x_ts, y_tr=y_tr, tp=tp,
                      n_neighbors=n_neighbors, c=c, kernel=kernel)
    #Vizualiziáció(x_ts, y_pred, f"{tp} ", n_dims)
    print(f"alg: {tp}, "
          f"AUC: {eval_cl('AUC', x_tr_2D, y_pred, y_ts)}, "
          f"f1: {eval_cl('f1', x_tr_2D, y_pred, y_ts)}, "
          f"acc: {eval_cl('accuracy', x_tr_2D, y_pred, y_ts)}")



    print("Done")
