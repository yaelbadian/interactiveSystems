from utils import features, string_fields
import numpy as np
import pandas as pd
import time


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

    def __init__(self, data, song_pref, K):
        self.data_ = data
        self.percentiles_ = create_bins(data)
        self.song_pref_ = song_pref
        self.ks_ = len(song_pref)
        self.kt_ = self.ks_
        self.phi_s_ = np.zeros(340) + 1 / (self.ks_ + 10)
        self.phi_t_ = np.zeros(3400) + 1 / (self.kt_ + 100)
        self.K_ = K

    def feature_percentile(self, song):
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
            percentile = np.searchsorted(self.percentiles_[feature], self.data_.loc[song].loc[feature])
            song_percentile.append(percentile)
        return np.array(song_percentile)

    def theta_t(self, song_from, song_to):
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
        song_percentiles_from = self.feature_percentile(song_from)
        song_percentiles_to = self.feature_percentile(song_to)
        idx = song_percentiles_from * 10 + song_percentiles_to + 100 * np.arange(0, len(features))
        feature_vec[idx] += 1
        return feature_vec

    def theta_s(self, song):
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
        song_percentiles = self.feature_percentile(song)
        idx = song_percentiles + 10 * np.arange(0, len(features))
        feature_vec[idx] += 1
        return feature_vec

    def M_star(self):
        self.data_["R_s"] = self.data_.index.to_series().apply(lambda x: self.R_s(x))
        return self.data_[self.data_["R_s"] >= np.median(self.data_["R_s"])]

    def R_s(self, song):
        return self.theta_s(song) @ self.phi_s_

    def R_t(self, song_from, song_to):
        return self.theta_t(song_from, song_to) @ self.phi_t_

    def algorithm_1(self):
        for song in self.song_pref_:
            self.phi_s_ += self.theta_s(song) * (1 / (self.ks_ + 1))

    def algorithm_2(self):
        song_prev = self.song_pref_[0]
        for song_curr in self.song_pref_[1:]:
            self.phi_t_ += self.theta_t(song_prev, song_curr) * (1 / (self.kt_ + 1))

    def algorithm_3(self):
        print("Alg 3")
        rewards = []
        songs = []
        for k in range(self.K_):
            start_time = time.time()
            curr_song = self.algorithm_4(self.K_ - k)
            song_reward = self.R_s(curr_song)
            trans_reward = self.R_t(songs[-1], curr_song) if k else 0
            r_i = song_reward + trans_reward

            songs.append(curr_song)
            rewards.append(r_i)

            r_avg = np.mean(rewards)
            r_inc = np.log(r_i / r_avg)

            w_s = song_reward / (song_reward + trans_reward)
            w_t = 1 - w_s
            self.phi_s_ = (1 / (k + 2)) * self.phi_s_ + (1 / (k + 2)) * self.theta_s(curr_song) * w_s * r_inc
            if k:
                self.phi_t_ = (1 / (k + 2)) * self.phi_t_ + (1 / (k + 2)) * self.theta_t(songs[-1], curr_song) * w_t * r_inc
            else:
                self.phi_t_ = (1 / (k + 2)) * self.phi_t_
            
            for i in range(34):  # 34 = number of features
                self.phi_s_[i * 10: (i + 1) * 10] /= sum(self.phi_s_[i * 10: (i + 1) * 10])
                self.phi_t_[i * 100: (i + 1) * 100] /= sum(self.phi_t_[i * 100: (i + 1) * 100])
            print("--- %s seconds ---" % (time.time() - start_time))

    def algorithm_4(self, horizon, T=3000):
        print("Alg 4")
        best_trajectory = []
        highest_expected_payoff = -float("inf")
        start_time = time.time()
        M_star = self.M_star()
        print("--- %s seconds ---" % (time.time() - start_time))
        for _ in range(T):
            trajectory = M_star.index.to_series().sample(horizon)
            expected_payoff = self.R_s(trajectory[0]) + sum(self.R_t(x[0], x[1]) + self.R_s(x[1]) for x in zip(trajectory[:-1], trajectory[1:]))
            if expected_payoff > highest_expected_payoff:
                highest_expected_payoff = expected_payoff
                best_trajectory = trajectory
        return best_trajectory[0]

    def algorithm_5(self):
        self.algorithm_1()
        self.algorithm_2()
        self.algorithm_3()
