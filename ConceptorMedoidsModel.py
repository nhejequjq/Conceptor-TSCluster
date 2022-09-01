from phmlearner.conceptor import Reservoir
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from collections import OrderedDict
import pandas as pd
import pickle


class ConceptorMedoidsModel(object):
    """
    A machine learning Model using Conceptor Network to obtain the center of Conceptor Matrix with a uniform class label, then measure the distance between the conceptor converted by test time series and the conceptor center.

    Parameters
    ----------
    n_features : int
        the input feature dimensions of time series
    scaler : str, optional
        scaling method of time series, by default 'StandardScaler', ('StandardScaler', 'MinMaxScaler')
    n_reservoir : int, optional
        Reservoir neuron number, by default 32
    spectral_radius : float, optional
        Spectral Radius, by default 0.95
    connectivity : float, optional
        the connectivity of Reservoir neurons, by default 0.15
    random_state : int, optional
        random state, by default 2

    References
    ----------
    Xu, M., P. Baraldi, and E. Zio. "Fault diagnostics by conceptors-aided clustering." 
    In 30th European Safety and Reliability Conference, ESREL 2020 and 15th Probabilistic
    Safety Assessment and Management Conference, PSAM 2020, pp. 3656-3663. 
    Research Publishing Services, 2020.

    Notes
    -----
    The setting of parameters spectral_radius and connectivity affect the 
    dynamic of the reservoir. The proper setting can obtain reservoir with 
    long-term memory. Larger spectral_radius brings longer memory, proper 
    connectivity, e.g. 5 / n_reservoir, usually derives better dynamic for reservoir.

    """

    def __init__(
            self,
            n_features,
            scaler='StandardScaler',
            n_reservoir=32,
            spectral_radius=0.95,
            connectivity=0.15,
            random_state=2
    ):
        super(ConceptorMedoidsModel, self).__init__()

        if scaler == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scaler == 'MinMaxScaler':
            self.scaler = MinMaxScaler()  
        else:
            pass  

        self.reservoir = Reservoir(
            n_features=n_features,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            connectivity=connectivity,
            random_state=random_state,
        )

        self.conceptor = np.zeros((n_reservoir, n_reservoir))

    @staticmethod
    def get_normalized_df(df, std_scaler, selected_columns):
        df[selected_columns] = std_scaler.transform(df[selected_columns].values)
        return df

    @staticmethod
    def get_conceptor(df, reservoir, selected_columns):
        X = df[selected_columns].values
        X = X[np.newaxis, :, :]
        C = reservoir.transform_conceptor(X)
        df_res = pd.DataFrame([], columns=['conceptor_matrix'])
        df_res['conceptor_matrix'] = list(C)
        return df_res

    @staticmethod
    def get_conceptor_center(df):
        conceptor_center = np.zeros(df.iloc[0].conceptor_matrix.shape)
        conceptor_matrixes = df.loc[:, 'conceptor_matrix']
        for i, c in enumerate(conceptor_matrixes):
            conceptor_center += c
        conceptor_center = conceptor_center / conceptor_matrixes.shape[0]
        return conceptor_center

    @staticmethod
    def get_distance_to_conceptor_centers(df, conceptor_center, output_column):
        df_res = pd.DataFrame([], columns=[output_column])
        df_res[output_column] = [np.linalg.norm(df['conceptor_matrix'].values[0] - conceptor_center, ord='fro')]
        return df_res

    @staticmethod
    def get_dist_to_medoid(df, reservoir, conceptor_center, selected_columns, output_column):
        df_res = (
            df
                .pipe(ConceptorMedoidsModel.get_conceptor, reservoir, selected_columns)
                .pipe(ConceptorMedoidsModel.get_distance_to_conceptor_centers, conceptor_center, output_column)
        )
        return df_res

    def fit(self, df, selected_columns, index):
        """
        train ConceptorMedoidsModel 

        Parameters
        ----------
        df : pandas dataframe
            input pandas dataframe used to train the ConceptorMedoidsModel, the input samples contain only a uniform class
        selected_columns : list of str
            the columns of pandas dataframe used to input into the model 
        index : str or list of str
            the index column or columns used to group and split the series in the pandas dataframe 

        Returns
        -------
        ConceptorMedoidsModel() object
            contain the model parameters of Reservoir() object, StandardScaler() object and conceptor arrays of

        Example
        --------
        >>> model = ConceptorMedoidsModel(n_features=1)
        >>> model.fit(df, selected_columns=['torque'], index='service_id')
        """

        self.scaler = self.scaler.fit(df.loc[:, selected_columns].values)

        df_conceptor = (
            df
                .groupby(index).apply(
                lambda x: ConceptorMedoidsModel.get_normalized_df(x, self.scaler, selected_columns))
                .reset_index(drop=True)
                .groupby(index).apply(
                lambda x: ConceptorMedoidsModel.get_conceptor(x, self.reservoir, selected_columns))
                .reset_index(index)
        )
        self.conceptor = ConceptorMedoidsModel.get_conceptor_center(df_conceptor)

        return self

    def predict(self, df, selected_columns, index, output_column):
        """
        Use ConceptorMedoidsModel to inference and make the predictions

        Parameters
        ----------
        df : pandas dataframe
            the pandas dataframe used to obtain the predictions  
        selected_columns : list of str
            the columns of pandas dataframe used to input into the model 
        index : str or list of str
            the index column or columns used to group and split the series in the pandas dataframe 
        output_column : str
            the output columns names

        Returns
        -------
        pandas dataframe
            the output pandas dataframe contain the output columns, which means the distance between the conceptor centor representing a class of time series and conceptors representing test time series.

        Example
        --------
        >>> model = ConceptorMedoidsModel(n_features=1)
        >>> model.fit(df1, selected_columns=['torque'], index='service_id')
        >>> model.predict(df2, selected_columns=['torque'], index='service_id', output_column='dist_to_normal')
        """

        df_res = (
            df
                .groupby(index).apply(
                lambda x: ConceptorMedoidsModel.get_normalized_df(x, self.scaler, selected_columns))
                .reset_index(drop=True)
                .groupby(index).apply(
                lambda x: ConceptorMedoidsModel.get_dist_to_medoid(x, self.reservoir, self.conceptor, selected_columns, output_column))
                .reset_index(index)
        )

        return df_res

    def save_to_file(self, path):
        """
        save all the ConceptorMedoidsModel parameters.

        Parameters
        ----------
        path : str
            the path to save the model file.
        
        Example
        -------
        >>> model.save_to_file(path='my_conceptor_model.pkl')

        """
        model_dict = OrderedDict()
        model_dict['reservoir'] = self.reservoir
        model_dict['scaler'] = self.scaler
        model_dict['conceptor'] = self.conceptor

        with open(path, 'wb') as file:
            pickle.dump(model_dict, file)

    def extract_model_dict(self):
        model_dict = OrderedDict()
        model_dict['reservoir'] = self.reservoir
        model_dict['scaler'] = self.scaler
        model_dict['conceptor'] = self.conceptor
        return model_dict


def load_from_file(path):
    """
    load parameters from model file and return the ConceptorMedoidsModel() object. 

    Parameters
    ----------
    path : str
        the path to read model file.

    Returns
    -------
    ConceptorMedoidsModel() object 

    Example
    --------
    >>> from ConceptorMedoidsModel import load
    >>> model = load(path='my_conceptor_model.pkl')

    """
    model = ConceptorMedoidsModel(n_features=1)
    with open(path, 'rb') as file:
        model_dict = pickle.load(file)
    model.reservoir = model_dict['reservoir']
    model.scaler = model_dict['scaler']
    model.conceptor = model_dict['conceptor']
    return model


def load_from_model_dict(model_dict):
    """
    load parameters from parameter dict and return the ConceptorMedoidsModel() object. 

    Parameters
    ----------
    model_dict : OrderedDict()
        parameter dict

    Returns
    -------
    ConceptorMedoidsModel() object

    Example
    -------
    >>> from ConceptorMedoidsModel import load
    >>> model = load(my_model_dict)

    """
    model = ConceptorMedoidsModel(n_features=1)
    model.reservoir = model_dict['reservoir']
    model.scaler = model_dict['scaler']
    model.conceptor = model_dict['conceptor']
    return model
