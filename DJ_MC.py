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

    def __init__(self, data, song_pref, K, song_info):
        self.data_ = data
        self.percentiles_ = create_bins(data)
        self.song_pref_ = song_pref
        self.ks_ = len(song_pref)
        self.kt_ = self.ks_
        self.phi_s_ = np.zeros(340) + 1 / (self.ks_ + 10)
        self.phi_t_ = np.zeros(3400) + 1 / (self.kt_ + 100)
        self.K_ = K
        self.song_info_ = song_info

    def __getstate__(self):
        return {"data": self.data_,
                "percentiles": self.percentiles_,
                "song_pref": self.song_pref_,
                "ks": self.ks_,
                "kt": self.kt_,
                "phi_s": self.phi_s_,
                "phi_t": self.phi_t_,
                "K": self.K_,
                "song_info": self.song_info_}

    def __setstate__(self, state_dict):
        self.data_ = state_dict["data"]
        self.percentiles_ = state_dict["percentiles"]
        self.song_pref_ = state_dict["song_pref"]
        self.ks_ = state_dict["ks"]
        self.kt_ = state_dict["kt"]
        self.phi_s_ = state_dict["phi_s"]
        self.phi_t_ = state_dict["phi_t"]
        self.K_ = state_dict["K"]
        self.song_info_ = state_dict["song_info"]
        return self

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
        R_s = self.data_.index.to_series().apply(lambda x: self.R_s(x))
        return self.data_[R_s >= np.median(R_s)]

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
            self.phi_s_ = ((k + 1) / (k + 2)) * self.phi_s_ + (1 / (k + 2)) * self.theta_s(curr_song) * w_s * r_inc
            if k:
                self.phi_t_ = ((k + 1) / (k + 2)) * self.phi_t_ + (1 / (k + 2)) * self.theta_t(songs[-1], curr_song) * w_t * r_inc
            else:
                self.phi_t_ = ((k + 1) / (k + 2)) * self.phi_t_
            
            for i in range(34):  # 34 = number of features
                self.phi_s_[i * 10: (i + 1) * 10] /= sum(self.phi_s_[i * 10: (i + 1) * 10])
                self.phi_t_[i * 100: (i + 1) * 100] /= sum(self.phi_t_[i * 100: (i + 1) * 100])
            print("--- %s seconds ---" % (time.time() - start_time))

    def algorithm_4(self, horizon, T=300):
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

    def algorithm_3_with_human(self):
        rewards = []
        songs = []
        for k in range(self.K_):
            start_time = time.time()
            curr_song = self.algorithm_4(self.K_ - k)

            while True:
                enjoyed_song = input(f"Do you like the song {self.song_info_.loc[curr_song, 'title']} by {self.song_info_.loc[curr_song, 'artist']} (y/n)?")
                if enjoyed_song in ["y", "n"]:
                    break
            enjoyed_song = 1 if enjoyed_song == "y" else -1

            song_reward = self.R_s(curr_song)
            trans_reward = self.R_t(songs[-1], curr_song) if k else 0
            r_i = song_reward + trans_reward

            songs.append(curr_song)
            rewards.append(r_i)

            r_avg = np.mean(rewards)
            r_inc = np.log(r_i / r_avg)

            w_s = song_reward / (song_reward + trans_reward)
            w_t = 1 - w_s
            self.phi_s_ = ((k + 1) / (k + 2)) * self.phi_s_ + enjoyed_song * (1 / (k + 2)) * self.theta_s(curr_song) * w_s * r_inc
            if k:
                enjoyed_transition = 1 if enjoyed_transition == "y" else -1
                while True:
                    enjoyed_transition = input(f"Did you think the transition between songs was smooth (y/n)?")
                    if enjoyed_transition in ["y", "n"]:
                        break
                self.phi_t_ = ((k + 1) / (k + 2)) * self.phi_t_ + enjoyed_transition * (1 / (k + 2)) * self.theta_t(songs[-1], curr_song) * w_t * r_inc
            else:
                self.phi_t_ = ((k + 1) / (k + 2)) * self.phi_t_
            
            for i in range(34):  # 34 = number of features
                self.phi_s_[i * 10: (i + 1) * 10] /= sum(self.phi_s_[i * 10: (i + 1) * 10])
                self.phi_t_[i * 100: (i + 1) * 100] /= sum(self.phi_t_[i * 100: (i + 1) * 100])

    def algorithm_4_with_human(self, horizon, T=300):
        best_trajectory = []
        highest_expected_payoff = -float("inf")
        for _ in range(T):
            trajectory = self.data_.index.to_series().sample(horizon)
            expected_payoff = self.R_s(trajectory[0]) + sum(self.R_t(x[0], x[1]) + self.R_s(x[1]) for x in zip(trajectory[:-1], trajectory[1:]))
            if expected_payoff > highest_expected_payoff:
                highest_expected_payoff = expected_payoff
                best_trajectory = trajectory
        return best_trajectory[0]

    # fit
    def algorithm_5(self):
        self.algorithm_1()
        self.algorithm_2()
        self.algorithm_3()

    def random_predict(self):
        songs = self.data_.index.to_series().sample(self.K_).values
        R_s = self.data_.index.to_series().apply(lambda x: self.R_s(x))
        rewards = [R_s[songs[0]]]
        for i in range(1, self.K_):
            rewards.append(rewards[-1] + R_s[songs[i]] + self.R_t(songs[i - 1], songs[i]))
        return songs, rewards
    
    def greedy_predict(self):
        R_s = self.data_.index.to_series().apply(lambda x: self.R_s(x))
        songs = []
        rewards = []
        for i in range(self.K_):
            best_song = R_s.idxmax()
            songs.append(best_song)
            rewards.append(R_s[best_song] + (0 if i == 0 else rewards[-1] + self.R_t(songs[i - 1], best_song)))
            R_s = R_s.drop(best_song)
        return songs, rewards

    def DJ_MC_predict(self):
        songs = []
        R_s = self.data_.index.to_series().apply(lambda x: self.R_s(x))
        first_song = R_s.idxmax()
        songs.append(first_song)
        rewards = [R_s[first_song]]
        for _ in range(1, self.K_):
            start_time = time.time()
            R = self.data_.index.to_series().apply(lambda x: self.R_s(x) + self.R_t(songs[-1], x))
            R = R.drop(songs)
            songs.append(R.idxmax())
            rewards.append(rewards[-1] + R.max())
            print(time.time() - start_time)
        return songs, rewards
