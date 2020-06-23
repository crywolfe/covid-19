import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#################################### LOAD Dataset (meta_app_path csv file) ####################################
def loadDataset(meta_app_path):
    try:
        # load dataset
        df = pd.read_csv(meta_app_path, low_memory=False)

        nRow, nCol = df.shape
        print(f'There are {nRow} rows and {nCol} columns')
        
        return df
    except Exception as e:
        print("loadDataset error")
        print(str(e))
        return df

#---------------------------------display data frame column ----------------------------------#
def dataframe_columns(df):
    try:
        # iterating the columns 
        for col in df.columns: 
            print(col)

    except Exception as e:
        print("displayDataset error")
        print(str(e))
    return

#---------------------------------display dataframe info / shape / describe ----------------------------------#
def dataframe_info(df):
    try:
         print(df.shape)
         print(df.info())
        #On classification problems you need to know how balanced the class values are.( This is an example)
        # positive is the class or target name
#          print(df.groupby(targetVariable).size() )
        # We can analyze all the data set
         print(df.describe())
    except Exception as e:
        print("Preprocess dataframe info errors 2")
        print (str(e))

#################################### display data types / meta data ####################################
def displayDataTypes(df):
    try:
        print(df.dtypes.value_counts())
    except Exception as e:
        print("displayDataTypes error")
        print(str(e))

#---------------------------------display columns by type----------------------------------#
def displayColumnsByType(df, dataType):
    try:
        print(df.dtypes.loc[df.dtypes==dataType].index)
    except Exception as e:
        print("displayColumnsByType error")
        print(str(e))

#--------------------------------- Get the count,size and unique values in each column of application data----------------------------------#
def displayDataTypesAggregate(df):
    try:
        print(df.agg(['count','size','nunique']))
    except Exception as e:
        print("displayDataTypesAggregate error")
        print(str(e))
        
#-------------------------------------------- calculate meta-data-------------------------------------------#
#Function to calculate meta-data to identify % of data is missing in each column
def meta_data(df):
    try:
        total = df.isnull().sum()
        percent = (df.isnull().sum()/df.isnull().count()*100)
        unique = df.nunique()
        datatypes = df.dtypes
        return pd.concat([total, percent, unique, datatypes], axis=1, keys=['Total', 'Percent', 'Unique',\
                                                                            'Data_Type']).sort_values(by="Percent", ascending=False)
    except Exception as e:
        print("meta_data error")
        print(str(e))

        
        
        
#################################### plot data frame ####################################
def boxplot(df):
    try:
        df.plot(kind='box', subplots=True, layout=(54,54), sharex=False, sharey=False, fontsize=8, figsize=(10,15))
        plt.savefig('boxplot.png')
#         plt.show()
    except Exception as e:
        print("errorUnivariate")
        print (str(e))

#-------------------------------------------- scatter Plot Matrix-------------------------------------------#
def scatter_Plot_Matrix(df):
    try:
        scatter_matrix(df,figsize=(20,20))
        plt.show()
    except Exception as e:
        print("errorUnivariate")
        print (str(e))

#-------------------------------------------- function to plot charts-------------------------------------------#
def plot_charts(var, defaulters, nondefaulters, label_rotation,horizontal_layout):
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,20))
    
    s1=sns.countplot(ax=ax1,x=defaulters[var], data=defaulters, order= defaulters[var].value_counts().index,)
    ax1.set_title('Distribution of '+ '%s' %var +' for Defaulters', fontsize=10)
    ax1.set_xlabel('%s' %var)
    ax1.set_ylabel("Count of Loans")
    if(label_rotation):
        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
    s2=sns.countplot(ax=ax2,x=nondefaulters[var], data=nondefaulters, order= nondefaulters[var].value_counts().index,)
    if(label_rotation):
        s2.set_xticklabels(s2.get_xticklabels(),rotation=90)
    ax2.set_xlabel('%s' %var)
    ax2.set_ylabel("Count of Loans")
    ax2.set_title('Distribution of '+ '%s' %var +' for Non-Defaulters', fontsize=10)
    plt.show()
    return        

#-------------------------------------------- function for plotting box plot and histogram -------------------------------------------#
import matplotlib.pyplot as plt # plotting
import seaborn as sns
plt.style.use('ggplot')
def plot_boxhist(var, defaulters):
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    s=sns.boxplot(y=defaulters[var]);
    plt.title('Box Plot of '+ '%s' %var +' for Defaulters', fontsize=10)
    plt.xlabel('%s' %var)
    plt.ylabel("Count of Loans")
    plt.subplot(1, 2, 2)
    s=plt.hist(x=defaulters[var]);
    plt.xlabel('%s' %var)
    plt.ylabel("Count of Loans")
    plt.title('Histogram of '+ '%s' %var +' for Defaulters', fontsize=10)
plt.show()

#################################### data preprocessing ####################################
def LabelEncoder_categorical(df):
    from sklearn.preprocessing import LabelEncoder
    categorical = list(df.select_dtypes(include=['object']).columns.values)
    le=LabelEncoder()
    try:
     for cat in categorical:
            print(cat)
            #df[cat].fillna(targetVariable, inplace=True)
            #df[cat] = le.fit_transform(df[cat])
            df[cat] = le.fit_transform(df[cat].astype(str))
     #for i in cv:
      #  df[i]=le.fit_transform(df[i])
       # test[i]=le.fit_transform(test[i])
     return df
    except Exception as e:
        print("LabelEncoder_categorical error")
        print (str(e))

#--------------------------------------------  drop constant column  -------------------------------------------#
def drop_constant_columns(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    result = dataframe.copy()
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result = result.drop(column,axis=1)
    return result

#--------------------------------------------  drop column with low variance -------------------------------------------#
def drop_column_low_variance(df):
  try:
    dropped_columns = set()
   # temp=df.loc[:, "NAME"].var()
    eps = 1e-6
    C = df.columns
    print('Identifing low-variance columns...', end=' ')
    for c in C:
#        print(df[c])
       temp = df.loc[:, c]
       variance=df.loc[:, c].var()
#        print(variance)
      # temp =  c.var()
       if variance < eps:
            # print('.. %-30s: too low variance ... column ignored'%(c))
            dropped_columns.add(c)
            df.drop(c)
#     print('done!')
    return df
  except Exception as e:
      print("drop no variance")
      print(str(e))

#-------------------------------------------- display missing data-------------------------------------------#
def display_missing_data(df):
    try:
        missing_data=df.isnull()
        # check infinite value
        print("check infinite value, you should get false then true")
        print("now all false")
        print(np.isnan(df.any())) #you should get False
        print("now all true")
        print(np.isfinite(df.all())) #You should get  True
        print("end check infinite values")
        print(missing_data.head(5))
        print(missing_data.sum())
        print(df.head(10))
    except Exception as e:
        print("meta_data error")
        print(str(e))

#-------------------------------------------- Evaluation for Missing Data  -------------------------------------------#
def evaluate_missing_data(df):
    try:
     missing_data= df.isnull()
     missing_data.head(5)
     print(missing_data.sum())
     for column in missing_data.columns.values.tolist():
        print(column)
        print(missing_data[column].value_counts())
        print("--------------------------------")
    #Finding the porcentage of  missing data
     print(round(((missing_data.sum()/len(missing_data))*100), 4))
    #Lets see our outliers in the data set

     q = df.quantile(0.99)
     print(df [df > q])
    #housing.dropna(subset=["total_bedrooms"]) # option 1

    #housing.drop("total_bedrooms", axis=1) # option 2

    #median = housing["total_bedrooms"].median()

    #housing["total_bedrooms"].fillna(median) # option 3

    #Lets check the types
     print(df.dtypes)
    except Exception as e:
        print("error_Missing Data")
        print (str(e))
     
#-------------------------------------------- drop missing data  -------------------------------------------#
def dropMissingData(df, meta_data):
    #dropping columns with more than 57% missing values 
    cols_to_keep=list(meta_data[(meta_data.Percent<57)].index)
    df=df[cols_to_keep]
    df.describe()

    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')
    
    #Checking columns with very less missing values
    low_missing=pd.DataFrame(meta_data[(meta_data.Percent>0)&(meta_data.Percent<15)])
    low_missing
    
    return df

#--------------------------------------------  Fill NaNs By Zero / mean -------------------------------------------#
def FillNaNsByZero(df):
    try:
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        print("FillNAs")
        print (str(e))

#--------------------------------------------  Fill NaNs By Mean -------------------------------------------#
def FillNaNsByMean(df):
    try:
        df.fillna(df.mean(), inplace=True)
        return df
    except Exception as e:
        print("FillNAs")
        print (str(e))

#--------------------------------------------  clean the data set from nulls, nan, and infitinity values -------------------------------------------#
def clean_dataset(df):
    try:
        df.dropna(axis=1,inplace=True)
        return df
    except Exception as e:
        print("FillNAs")
        print (str(e))

#-------------------------------------------- removing outliers -------------------------------------------#
def removingOutliers(df):
    for col in df.select_dtypes('float64').columns[:35]: 
        #Removing all entries above 99 percentile
        print(col)
        df = df[df[col]<np.nanpercentile(df[col], 99)]   
    return df
        
#################################### Correlation Between Attributes ####################################
def attributes_correlation(df):
    try:
     print(df.corr())
     corr_matrix= df.corr()
    #To check a correlation with our target

     print(corr_matrix[targetVariable].sort_values(ascending=False))
     plt.figure()
     sns.heatmap(df.corr(), vmin=-1, vmax=1.0, fontsize=8, annot=True)
     plt.savefig('corr_matrix.png')
     #plt.show()
    except Exception as e:
        print("attributes_correlation")
        print (str(e))

#--------------------------------------------  Skew of Univariate Distributions -------------------------------------------#
#The skew result can show us a positive(right) or negative (left) skew. Values closer to zero show less skew
def skew_dataframe(df):
    try:
     print(df.skew())
    #It is better to use some data visualization to get a better idea.
     from matplotlib import pyplot as plt
     df.hist(bins=10, figsize=(20,15))
     plt.savefig('Univariate_Distributions.png')
     #plt.show()
    except Exception as e:
        print("errorUnivariate")
        print (str(e))

#################################### feature selection ####################################
#Feature Selection For Machine Learning
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn import utils
from sklearn.feature_selection import chi2

def select_k_best_features(df, targetVariable):
    try:
     X=df.drop(targetVariable,axis=1)
     y=df[targetVariable]

     lab_enc = preprocessing.LabelEncoder()
     training_scores_encoded = lab_enc.fit_transform(y)

     print(utils.multiclass.type_of_target(y))
     print(utils.multiclass.type_of_target(training_scores_encoded.astype('int')))
     print(utils.multiclass.type_of_target(training_scores_encoded))
     print(training_scores_encoded)
    #1-Univariete Selection

    #apply SelectKBest class to extract top 10 best features
     bestfeatures = SelectKBest(score_func=chi2, k='all')
     fit = bestfeatures.fit(X,training_scores_encoded)
     dfscores = pd.DataFrame(fit.scores_)
     dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization

     featureScores = pd.concat([dfcolumns,dfscores],axis=1)
     featureScores.columns = ['Specs','Score']  #naming the dataframe columns
     print(featureScores.nlargest(10,'Score'))  #print 10 best features
     return X, y, training_scores_encoded
    except Exception as e:
        print("error Feature Selection")
        print (str(e))

#------------------------------- Feature Extraction - unsupervised to supervised ----------------------------#
def get_synonym_count(df, feature, dict):
    length = len(df[feature])
    
    for key, value in dict.items():
        
        for i in range(length):
            count_all = 0
            for synonym in value:
                count_all += df[feature][i].count(synonym)
            df[key][i] = count_all
    return df 
        
#-------------------------------------------- Feature Importance -------------------------------------------#
from sklearn.ensemble import ExtraTreesClassifier

def feature_importance(X, training_scores_encoded):
    try:
     model = ExtraTreesClassifier()
     model.fit(X,training_scores_encoded)
     print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
     feat_importances = pd.Series(model.feature_importances_, index=X.columns)
     feat_importances.nlargest(10).plot(kind='barh')
     plt.savefig('important_features.png')
     #plt.show()
    except Exception as e:
        print("error Feature Importance")
        print (str(e))
        
#-------------------------------------------- Principal Components Analysis-------------------------------------------#
def principal_components(X):
    try:
        # Normalization of the dataset
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=[0, 1])
        data_rescaled = scaler.fit_transform(X)
        
        # PCA
        from sklearn.decomposition import PCA
        """
        It can be seen that first principal component is responsible for 10.95% variance. Similarly, 
        the second principal component causes 8.2% variance in the dataset. 
        Collectively we can say that (10.95 + 8.2) 19.69% percent of the classification information 
        contained in the feature set is captured by the first two principal components.
        ref: https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
        """
        pca = PCA(n_components=len(X.columns), whiten = False, random_state = 2019)

        principalComponents_credit = pca.fit_transform(data_rescaled)
        principalComponents_credit_Df = pd.DataFrame(data = principalComponents_credit)

        principalComponents_credit_Df.tail()
        print("Length of the principal component: "+str(len(pca.explained_variance_ratio_)))
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        
        print('sum pca variance ratio: {}'.format(pca.explained_variance_ratio_.sum()))
        
    except Exception as e:
        print("error principal_components")
        print (str(e))
    

#################################### Model, predict and solve ####################################
#Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#-------------------------------------------- Linear Regression-------------------------------------------#
from sklearn.linear_model import LinearRegression
def linear_regression(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42,)
     lin_reg= LinearRegression()
     results_linreg= cross_val_score(lin_reg, X, training_scores_encoded, cv=kfold)
     print('Estimate accuracy',results_linreg.mean())
     return results_linreg, lin_reg
    except Exception as e:
        print("Linear Regression error")
        print (str(e))

#-------------------------------------------- Logistic Regression-------------------------------------------#
from sklearn.linear_model import LogisticRegression

def logistic_regression(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     logreg = LogisticRegression(solver='lbfgs',max_iter=10000)
     results_logreg = cross_val_score(logreg, X, training_scores_encoded, cv=kfold,scoring='accuracy')
     print('Estimate accuracy',results_logreg.mean())
     return results_logreg, logreg
    except Exception as e:
        print("error Logistic regression")
        print (str(e))
    
#-------------------------------------------- Support Vector Machines-------------------------------------------#
from sklearn.svm import SVC, LinearSVC

def support_vector_machines(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     svc = SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
       decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',
       max_iter=-1, probability=False, random_state=None, shrinking=True,
       tol=0.001, verbose=False)
     results_svc = cross_val_score(svc, X, training_scores_encoded, cv=kfold,scoring='accuracy')
     print('Estimate accuracy',results_svc.mean())
     return results_svc, svc
    except Exception as e:
        print("errorSVM")
        print (str(e))

#-------------------------------------------- KNN-------------------------------------------#
from sklearn.neighbors import KNeighborsClassifier

def KNN(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     knn = KNeighborsClassifier(n_neighbors = 3)
     results_knn = cross_val_score(knn, X, training_scores_encoded, cv=kfold)
     print('Estimate accuracy',results_knn.mean())
     return results_knn, knn
    except Exception as e:
        print("errorSVM")
        print (str(e))

#-------------------------------------------- Gaussian Naive Bayes-------------------------------------------#
from sklearn.naive_bayes import GaussianNB

def gaussian_naive_bayes(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     gaussian = GaussianNB()
     results_gaussian = cross_val_score(gaussian, X, training_scores_encoded, cv=kfold)
     print('Estimate accuracy',results_gaussian.mean())
     return results_gaussian, gaussian
    except Exception as e:
        print("error GNB")
        print (str(e))

#-------------------------------------------- Perceptron-------------------------------------------#
from sklearn.linear_model import Perceptron

def perceptron(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     perceptron = Perceptron(max_iter=1000,tol=1e-3)
     results_perceptron = cross_val_score(perceptron, X, training_scores_encoded, cv=kfold,scoring='accuracy')
     print('Estimate accuracy',results_perceptron.mean())
     return results_perceptron, perceptron
    except Exception as e:
        print("error Perceptron")
        print (str(e))
    
#-------------------------------------------- Linear SVC-------------------------------------------#
from sklearn.svm import LinearSVC

def linear_SVC(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     linear_svc = LinearSVC(max_iter=1000)
     results_linearsvc= cross_val_score(linear_svc, X, training_scores_encoded, cv=kfold, scoring='accuracy')
     print('Estimate accuracy',results_linearsvc.mean())
     return results_linearsvc, linear_svc
    except Exception as e:
        print("error Linear SVC")
        print (str(e))

#-------------------------------------------- Stochastic Gradient Descent-------------------------------------------#
from sklearn.linear_model import SGDClassifier

def stochastic_gradient_descent(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     sgd = SGDClassifier(max_iter=1000,tol=1e-3)
     results_sgd = cross_val_score(sgd, X, training_scores_encoded, cv=kfold, scoring='accuracy')
     print('Estimate accuracy',results_sgd.mean())
     return results_sgd, sgd
    except Exception as e:
        print("error Stochastic Gradient Descent")
        print (str(e))

#-------------------------------------------- Decision Tree-------------------------------------------#
from sklearn.tree import DecisionTreeClassifier

def decision_tree(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     decision_tree = DecisionTreeClassifier()
     results_decisiontree = cross_val_score(decision_tree, X, training_scores_encoded, cv=kfold,scoring='accuracy')
     print('Estimate accuracy',results_decisiontree.mean())
     return results_decisiontree, decision_tree
    except Exception as e:
        print("error Decision Tree")
        print (str(e))

#-------------------------------------------- Random Forest-------------------------------------------#
from sklearn.ensemble import RandomForestClassifier

def random_forest(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     random_forest = RandomForestClassifier(n_estimators=100)
     results_randomforest = cross_val_score(random_forest, X, training_scores_encoded, cv=kfold,scoring='accuracy')
     print('Estimate accuracy',results_randomforest.mean())
     return results_randomforest, random_forest
    except Exception as e:
        print("error Random Forest")
        print (str(e))

#-------------------------------------------- Linear Discriminant Analysis-------------------------------------------#
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis        

def linear_discriminant_analysis(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     clf = LinearDiscriminantAnalysis()
     results_clf = cross_val_score(clf, X, training_scores_encoded, cv=kfold, scoring='accuracy')
     print('Estimate accuracy',results_clf.mean())
     return results_clf, clf
    except Exception as e:
        print("error Linear Discriminant Analysis")
        print (str(e))
        
        
#-------------------------------------------- Ada Boost Classifier-------------------------------------------#
from sklearn.ensemble import AdaBoostClassifier      

def ada_boost_classifier(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     AB = AdaBoostClassifier()
     results_AB = cross_val_score(AB, X, training_scores_encoded, cv=kfold, scoring='accuracy')
     print('Estimate accuracy',results_AB.mean())
     return results_AB, AB
    except Exception as e:
        print("error  Ada Boost Classifier")
        print (str(e))
        
        
#-------------------------------------------- Gradient Boosting Classifier-------------------------------------------#
from sklearn.ensemble import GradientBoostingClassifier        

def gradient_boosting_classifier(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     GBC = GradientBoostingClassifier()
     results_GBC = cross_val_score(GBC, X, training_scores_encoded, cv=kfold, scoring='accuracy')
     print('Estimate accuracy',results_GBC.mean())
     return results_GBC, GBC
    #GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    except Exception as e:
        print("error Gradient Boosting Classifier")
        print (str(e))

#-------------------------------------------- Extra Trees Classifier-------------------------------------------#
from sklearn.ensemble import ExtraTreesClassifier

def extra_trees_classifier(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     ETC=ExtraTreesClassifier(n_estimators=100)
     results_ETC = cross_val_score(ETC, X, training_scores_encoded, cv=kfold, scoring='accuracy')
     print('Estimate accuracy',results_ETC.mean())
     return results_ETC, ETC
    #ExtraTreesClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
    except Exception as e:
        print("error ExtraTreesClassifier")
        print (str(e))

#-------------------------------------------- XGB Classifier-------------------------------------------#
from xgboost.sklearn import XGBClassifier

def XGB_classifier(X, training_scores_encoded):
    try:
     kfold = KFold(n_splits=10, random_state=42)
     xgbs = XGBClassifier(objective="binary:logistic", random_state=42)
     results_xgbs = cross_val_score(xgbs, X, training_scores_encoded, cv=kfold, scoring='accuracy')
     print('Estimate accuracy',results_xgbs.mean())
     return results_xgbs, xgbs
    except Exception as e:
        print("error XGBClassifier")
        print (str(e))

#-------------------------------------------- Voting Ensemble for Classification-------------------------------------------#
import pandas
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

def voting_ensemble_classification(df, logreg, svc, knn, gaussian, perceptron, linearsvc, sgd, decisiontree, randomforest, clf, AB, GBC, ETC, xgbs):
    try:
        array = df.values
        X = array[:,0:8]
        Y = array[:,8]

        seed = 7
        kfold = KFold(n_splits=10, random_state=seed)

        # create the sub models
        estimators = []

        estimators.append(('logistic', logreg))
        estimators.append(('svc', svc))
        estimators.append(('knn', knn))
        estimators.append(('gaussian', gaussian))
        estimators.append(('perceptron', perceptron))
        estimators.append(('linearsvc', linearsvc))
        estimators.append(('sgd', sgd))
        estimators.append(('decisiontree', decisiontree))
        estimators.append(('randomforest', randomforest))
        estimators.append(('clf', clf))
        estimators.append(('AB', AB))
        
        estimators.append(('GBC', GBC))
        estimators.append(('ETC', ETC))
        estimators.append(('xgbs', xgbs))

        # create the ensemble model
        ensemble = VotingClassifier(estimators)
        results_ensemble = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
        print(results_ensemble.mean())
        return results_ensemble
    except Exception as e:
        print("error ensemble")
        print (str(e))

#-------------------------------------------- neural network-------------------------------------------#
import tensorflow as tf
from tensorflow import keras

def neural_network_classification(X,y):
    try:
        X1 = X.values.astype(float)
        y1 = y.values.astype(int)
        # define the keras model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(12, input_dim=7, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.summary()
        print(model.summary())
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X1, y1, epochs=35, batch_size=10)
        # evaluate the keras model
        _, accuracy = model.evaluate(X1, y1)
        print('Accuracy: %.2f' % (accuracy*100))
        return accuracy
    except Exception as e:
        print("error neural network")
        print (str(e))
        
#-------------------------------------------- model evaluation-------------------------------------------#
def model_evaluation(results_linreg, results_logreg, results_svc, results_knn, results_gaussian, results_perceptron, results_linearsvc, results_sgd, results_decisiontree, results_randomforest, results_clf, results_AB, results_GBC, results_ETC, results_xgbs, results_ensemble, results_neural_network):
    try:
    #Model evaluation
     print("Model evaluation")
     models = pd.DataFrame({
        'Model': ['Linear Regression','Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree','Linear Discriminant Analysis','Ada Boost Classifier','Gradient Boosting Classifier','Extra TreesClassifier','XGB Classifier', 'Ensemble', 'Neural Network'],
        'Score': [results_linreg.mean(),results_logreg.mean(),results_svc.mean(),results_knn.mean(),results_gaussian.mean(),results_perceptron.mean(),results_linearsvc.mean(),results_sgd.mean(),results_decisiontree.mean(),results_randomforest.mean(),results_clf.mean(),results_AB.mean(),results_GBC.mean(),results_ETC.mean(),results_xgbs.mean(), results_ensemble.mean(), results_neural_network]})
     print(models.sort_values(by='Score', ascending=False))
    #Model	Score
    except Exception as e:
        print("error Model evaluation")
        print (str(e))
        
#################################### TEXT ANALYSIS ####################################
def get_wordcount(df, feature, is_unique_words=False):
    if is_unique_words == False:
        return df[feature].apply(lambda x: len(x.strip().split()))
    else:
        return df[feature].apply(lambda x:len(set(str(x).split())))

#-------------------------------------------- text pre-processing -------------------------------------------#
# This function drops duplicates and drops na.
# *args shall be features in the form of strings e.g 'feature_1', 'feature_2'
def text_preprocessing(df, *args):
    arg_len = len(args)
    for i in range(arg_len):
        df.drop_duplicates(args[i], inplace=True)
    df.dropna(inplace=True)
    return df

#-------------------------------------------- language selection -------------------------------------------#
from tqdm import tqdm
from langdetect import detect, DetectorFactory

def language_selection(df, feature, language):
    # set seed
    DetectorFactory.seed = 0

    # hold label - language
    languages = []

    # go through each text
    for ii in tqdm(range(0,len(df))):
        # split by space into list, take the first x intex, join with space
        text = df.iloc[ii][feature].split(" ")

        lang = language
        try:
            if len(text) > 50:
                lang = detect(" ".join(text[:50]))
            elif len(text) > 0:
                lang = detect(" ".join(text[:len(text)]))
        # ught... beginning of the document was not in a good format
        except Exception as e:
            all_words = set(text)
            try:
                lang = detect(" ".join(all_words))
            # what!! :( let's see if we can find any text in abstract...
            except Exception as e:

                try:
                    # let's try to label it through the abstract then
                    lang = detect(df.iloc[ii][feature])
                except Exception as e:
                    lang = "unknown"
                    pass
        languages.append(lang)
    return languages

#-------------------------------------------- text parser -------------------------------------------#
# The parser function will convert text to lower case, remove punctuation, and find and remove stopwords including custom stopwords. 

#NLP 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg
import string

# Parser
parser = en_core_sci_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000

# Stop Words and Custom Stop Words
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
stopwords[:10]

custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]

for w in custom_stop_words:
    if w not in stopwords:
        stopwords.append(w)
#-------------------------------------------- tokenization -------------------------------------------#
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

#-------------------------------------------- token similarity -------------------------------------------#
def similarize_tokens(words, df, feature):
    nlp = en_core_sci_lg.load()
    processed_text = nlp(df[feature])
    word_tokens = nlp(' '.join(words))
    similarities = []

    for token1 in word_tokens:
        for token2 in processed_text:
            if token1 != token2:
                similarity = token1.similarity(token2)

            if (similarity is not None and similarity > 0.5 and similarity < 1):
                similarities.append(
                    {'token1': token1.text, 'token2': token2.text, 'similarity': similarity})
    return similarities
#-------------------------------------------- vectorization -------------------------------------------#
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(text, maxx_features):
    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X

#-------------------------------------------- dependency parser visualization -------------------------------------------#
import scispacy

def dependency_parser_visualizer(text):

    nlp = en_core_sci_lg.load()
    doc = nlp(text)

    print(list(doc.sents))

    # Examine the entities extracted by the mention detector.
    print(doc.ents)

    from spacy import displacy
    displacy.render(next(doc.sents), style='dep', jupyter=True)

#-------------------------------------------- filtering df on abstract + title -------------------------------------------#
# Helper function for filtering df on abstract + title substring
def abstract_title_filter(df, search_string):
    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |
            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))

#-------------------------------------------- Cleveland dot plot visualisation -------------------------------------------#
# Helper function for Cleveland dot plot visualisation of count data
import plotly
import plotly.graph_objects as go

def dotplot(input_series, title, x_label='Count', y_label='Regex'):
    subtitle = '<br><i>Hover over dots for exact values</i>'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=input_series.sort_values(),
    y=input_series.sort_values().index.values,
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="Count",
    ))
    fig.update_layout(title=f'{title}{subtitle}',
                  xaxis_title=x_label,
                  yaxis_title=y_label)
#     fig.show()
    plotly.offline.plot(fig)
    
#-------------------------------------------- counts synonyms and adds tag column -------------------------------------------#
# Helper function which counts synonyms and adds tag column to DF
def count_and_tag(df: pd.DataFrame,
                  synonym_list: list,
                  tag_suffix: str) -> (pd.DataFrame, pd.Series):
    counts = {}
    df[f'tag_{tag_suffix}'] = False
    for s in synonym_list:
        synonym_filter = abstract_title_filter(df, s)
        counts[s] = sum(synonym_filter)
        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True
    return df, pd.Series(counts)

#-------------------------------------------- printing out key passage of abstract -------------------------------------------#
# Function for printing out key passage of abstract based on key terms
def print_key_phrases(df, key_terms, n=5, chars=300):
    for ind, item in enumerate(df[:n].itertuples()):
        print(f'{ind+1} of {len(df)}')
        print(item.title)
        print('[ ' + item.doi + ' ]')
        try:
            i = len(item.abstract)
            for kt in key_terms:
                kt = kt.replace(r'\b', '')
                term_loc = item.abstract.lower().find(kt)
                if term_loc != -1:
                    i = min(i, term_loc)
            if i < len(item.abstract):
                print('    "' + item.abstract[i-30:i+chars-30] + '"')
            else:
                print('    "' + item.abstract[:chars] + '"')
        except:
            print('NO ABSTRACT')
        print('---')
        
#################################### PICKLES ####################################
import pickle5 as pickle
import datetime
#-------------------------------------------- read from pickle -------------------------------------------#
def read_pickles(file_name, date): # date is 2020-05-30 format
    fileObject = open(f'pickled_{file_name}{date}.pickle', 'rb')
    name = pickle.load(fileObject)
    fileObject.close()
    return name
#-------------------------------------------- write to pickle -------------------------------------------#
def write_pickles(file_name):
    date_of_launch = str(datetime.date.today())
    fileObject = open(f'pickled_{file_name}{date_of_launch}.pickle', 'wb')
    pickle.dump(file_name, fileObject)
    fileObject.close()