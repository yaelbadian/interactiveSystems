from utils import features, string_fields
import numpy as np
import pandas as pd

def create_bins(data):
        """
        Calculate the percentiles for every feature.

        Parameters
        ----------
        data : pd.DataFrame
            The 34 features extracted from the MSD (with song IDS and some more).
        """

        percentiles = {}
        for feature in features:
            percentiles[feature] = np.quantile(data[feature], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        return percentiles

class DJ_MC:

    def __init__(self, data, song_pref, transition_pref):
        self.data_ = data
        self.percentiles_ = create_bins(data)
        self.song_pref_ = song_pref
        self.transition_pref_ = transition_pref
        self.ks_ = len(song_pref)
        self.kt_ = len(transition_pref)
        self.phi_s_ = np.zeros(340) + 1 / (self.ks_ + 10)
        self.phi_t_ = np.zeros(3400) + 1 / (self.kt_ + 10)

    def feature_percentile(self, song_vec):
        """
        Given a song vector (from data.csv) return a list of
        the percentile range each features fall under.
        e.g. if the 5th feature is between percentile
        30 and 40, the 5th entry in the return value of the
        function will contain the value 3.

        Parameters
        ----------
        song_vec : pd.Series
            A row in self.data_. Contains the 34 features
            of a song.

        Returns
        -------
        feature_vec : np.array
            See description
        """

        song_percentile = []
        for feature in features:
            percentile = np.searchsorted(self.percentiles_[feature], song_vec[feature])
            song_percentile.append(percentile)
        return np.array(song_percentile)

    def theta_t(self, song_vec_from, song_vec_to):
        """
        Returns the binarized feature vector for a
        transition between two songs.

        Parameters
        ----------
        song_vec_from : pd.Series
            A row in self.data_. Contains the 34 features
            of a song. The song to transition from.

        song_vec_to : pd.Series
            A row in self.data_. Contains the 34 features
            of a song. The sonog to transition to.

        Returns
        -------
        feature_vec : np.array
            Binary vector of length 3400, sums up to 34.
        """

        feature_vec = np.zeros(3400)
        song_percentiles_from = self.feature_percentile(song_vec_from)
        song_percentiles_to = self.feature_percentile(song_vec_to)
        idx = song_percentiles_from * 10 + song_percentiles_to + 100 * np.arange(0, len(features))
        feature_vec[idx] += 1
        return feature_vec

    def theta_s(self, song_vec):
        """
        Returns the binarized feature vector for a song.

        Parameters
        ----------
        song_vec : pd.Series
            A row in self.data_. Contains the 34 features
            of a song.

        Returns
        -------
        feature_vec : np.array
            Binary vector of length 340, sums up to 34.
        """

        feature_vec = np.zeros(340)
        song_percentiles = self.feature_percentile(song_vec)
        idx = song_percentiles + 10 * np.arange(0, len(features))
        feature_vec[idx] += 1
        return feature_vec

    def Algorithm_1(self):
        for song in self.song_pref_:
            self.phi_s_ += self.theta_s(self.data_[song]) * (1 / (self.ks_ + 1))

    def Algorithm_2(self):
        pass

    def Algorithm_3(self):
        pass

    def Algorithm_4(self):
        pass

    def Algorithm_5(self):
        pass
