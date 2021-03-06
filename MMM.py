import pandas as pd
import numpy as np

from prophet import Prophet
import theano
import theano.tensor as tt
import pymc3 as pm
import arviz as az

from scipy import optimize

from plotnine import *
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class mmm:

    """
    Marketing Mixture Model in PYMC3.

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self):
        self.model = None
        self.START_INDEX = None
        self.END_INDEX = None
        self.trace = None
        self.trace_summary = None
        self.prophet = None
        self.prophet_predict = None
        self.ppc_all = None
        self.y_true = None
        self.revenue_transformation = None

        

    def decomposition(self, data, encode_holidays = False, encode_events = False, holidays_file = "generated_holidays.csv", country_code = "DE"):
        if encode_holidays:
            holidays = pd.read_csv(holidays_file, parse_dates = ["ds"])
            holidays["begin_week"] = holidays["ds"].dt.to_period('W-SUN').dt.start_time

            #combine same week holidays into one holiday
            holidays_weekly = holidays.groupby(["begin_week", "country", "year"], as_index = False).agg({'holiday':'#'.join, 'country': 'first', 'year': 'first'}).rename(columns = {'begin_week': 'ds'})
            holidays_weekly_de = holidays_weekly.query("(country == '{}')".format(country_code)).copy()

        prophet_data = data.rename(columns = {'revenue': 'y', 'date': 'ds'})

        if encode_events:
            prophet_data = pd.concat([prophet_data, pd.get_dummies(prophet_data["events"], drop_first = True, prefix = "events")], axis = 1)

            self.prophet = Prophet(yearly_seasonality = True, holidays = holidays_weekly_de)
            self.prophet.add_regressor(name = "events_event2")
            self.prophet.add_regressor(name = "events_na")

            self.prophet.fit(prophet_data[["ds", "y", "events_event2", "events_na"]])
            self.prophet_predict = self.prophet.predict(prophet_data[["ds", "y", "events_event2", "events_na"]])

            prophet_columns = [col for col in self.prophet_predict.columns if (col.endswith("upper") == False) & (col.endswith("lower") == False)]
            events_numeric = self.prophet_predict[prophet_columns].filter(like = "events_").sum(axis = 1)

        else:
            if encode_holidays:
                self.prophet = Prophet(yearly_seasonality = True, holidays = holidays_weekly_de)
            else:
                self.prophet = Prophet(yearly_seasonality = True)
            self.prophet.fit(prophet_data[["ds", "y"]])
            self.prophet_predict = self.prophet.predict(prophet_data[["ds", "y"]])
            
        final_data = data.copy()
        final_data["trend"] = self.prophet_predict["trend"]
        final_data["season"] = self.prophet_predict["yearly"]

        if encode_holidays:
            final_data["holiday"] = self.prophet_predict["holidays"]

        if encode_events:
            final_data["events"] = (events_numeric - np.min(events_numeric)).values

        return final_data

    def plot_decomposition(self):
        self.prophet.plot_components(self.prophet_predict, figsize = (20, 10))

    def estimate_spend_exposure(self, data, media_exposures, media_spends):

        #We use the menten function as proposed by https://towardsdatascience.com/modeling-marketing-mix-using-pymc3-ba18dd9e6e68
        
        spend_to_exposure_menten_func = lambda spend, V_max, K_m: V_max * spend / (K_m + spend)
        media_spend_exposure_df = pd.DataFrame()

        for (media_exposure, media_spend) in zip(media_exposures, media_spends):
            V_max = data[media_exposure].values[self.START_INDEX : self.END_INDEX].max()
            K_m   = V_max / 2
            spend = data[media_spend].values[self.START_INDEX : self.END_INDEX]
            exposure = data[media_exposure].values[self.START_INDEX : self.END_INDEX]
            best_values, _ = optimize.curve_fit(f = spend_to_exposure_menten_func, xdata = spend, ydata = exposure, p0 = [V_max, K_m])
            media_spend_exposure_df = pd.concat([media_spend_exposure_df, pd.DataFrame({'spend': [media_spend], 'exposure': [media_exposure], 'V_max': [best_values[0]], 'K_m': [best_values[1]]})]).reset_index(drop = True)
            
        return media_spend_exposure_df

    def initialize(self, data, delay_channels, media_channels, control_variables, target_variable, revenue_transformation, START_INDEX, END_INDEX):

        self.delay_channels = delay_channels
        self.control_variables = control_variables
        self.target_variable = target_variable
        self.media_channels = media_channels
        self.START_INDEX = START_INDEX
        self.END_INDEX = END_INDEX
        self.revenue_transformation = revenue_transformation

        #The backbone of the model is inspired from: 
        #https://www.youtube.com/watch?time_continue=994&v=UznM_-_760Y&feature=emb_title
        #It's relatively standard but more simple then our first version. 

        response_mean = []
        with pm.Model() as self.model:
            for channel_name in self.delay_channels:
                print(f"Delay Channels: Adding {channel_name}")
                
                x = data[channel_name].values
                
                adstock_param = pm.Beta(f"{channel_name}_adstock", 3, 3)
                saturation_gamma = pm.Beta(f"{channel_name}_gamma", 2, 2)
                saturation_alpha = pm.Gamma(f"{channel_name}_alpha", 3, 1)
                
                x_new = self._geometric_adstock_pymc(x, adstock_param)
                x_new_sliced = x_new[self.START_INDEX:self.END_INDEX]
                saturation_tensor = self._hill_saturation(x_new_sliced, saturation_alpha, saturation_gamma)
                
                channel_b = pm.HalfNormal(f"{channel_name}_media_coef", sd = 0.1)
                response_mean.append(saturation_tensor * channel_b)
                
            for control_var in self.control_variables:
                print(f"Control Variables: Adding {control_var}")
                
                x = data[control_var].values[self.START_INDEX:self.END_INDEX]
                
                control_beta = pm.Normal(f"{control_var}_control_coef", 0.1, sd = 0.1)
                control_x = control_beta * x
                response_mean.append(control_x)
                
            intercept = pm.HalfNormal("intercept", 0.1)
                
            sigma = pm.HalfNormal("sigma", 0.15)
            
            likelihood = pm.Normal("outcome", mu = intercept + sum(response_mean), sd = sigma, observed = data[target_variable].values[self.START_INDEX:self.END_INDEX])

    def plot_prior_predictive(self, data):
        with self.model:
            prior_pred = pm.sample_prior_predictive()
        
        fig, ax = plt.subplots(figsize = (20, 8))
        _ = ax.plot(prior_pred["outcome"].T, color = "0.5", alpha = 0.1)
        _ = ax.plot(data[self.target_variable].values[self.START_INDEX:self.END_INDEX], color = "red")

    def fit(self, draws, tune, chains, cores, target_accept = 0.95):
        with self.model:
            self.trace = pm.sample(draws, tune = tune, chains = chains, step = None, target_accept = target_accept, cores = cores, return_inferencedata = True)
            self.trace_summary = az.summary(self.trace)
        return self.trace, self.trace_summary

    def predict(self, data, START_INDEX, END_INDEX, return_metrics = True):
        data["prediction"] = data[self.delay_channels + self.control_variables + ["intercept"]].sum(axis = 1)
        y_pred = self.revenue_transformation.inverse_transform(data["prediction"].values.reshape(-1,1))[START_INDEX:END_INDEX].reshape(-1)
        
        if return_metrics == True:
            y_true = self.y_true[START_INDEX:END_INDEX]

            print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred) ** 2))}")
            print(f"MAPE: {np.mean(np.abs((y_true - y_pred) / y_true))}")
            print(f"NRMSE: {self._nrmse(y_true, y_pred)}")
        
        return y_pred

    def plot_posterior_predictive(self):
        with self.model:
            self.ppc_all = pm.sample_posterior_predictive(self.trace, var_names = ["outcome"] + list(self.trace_summary.index))
            az.plot_ppc(az.from_pymc3(posterior_predictive = self.ppc_all, model=self.model), var_names = ["outcome"])

    def fit_metrics(self, data):
        self.trace_summary = self.trace_summary.rename(columns={'mean': 'old_mean'})
        self.trace_summary["mean"] = np.inf

        for variable in list(self.trace_summary.index):
            mean_variable = self.trace.posterior[variable].mean(axis = 0).mean().values
            self.trace_summary.loc[self.trace_summary.index == variable, "mean"] = mean_variable

        self.y_true = data[self.target_variable].values
        y_true = self.y_true[self.START_INDEX:self.END_INDEX]
        y_pred = self.revenue_transformation.inverse_transform(self.ppc_all["outcome"].mean(axis = 0).reshape(-1, 1)).reshape(-1)

        print(f"RMSE: {np.sqrt(np.mean((y_true - y_pred) ** 2))}")
        print(f"MAPE: {np.mean(np.abs((y_true - y_pred) / y_true))}")
        print(f"NRMSE: {self._nrmse(y_true, y_pred)}")

    def apply_transformations(self, data):
        adstock_params = self.trace_summary[self.trace_summary.index.str.contains("adstock")][["mean", "sd"]].reset_index().rename(columns = {'index': 'name'}).assign(name = lambda x: x["name"].str.replace("_adstock", ""))
        saturation_params = pd.merge(
            self.trace_summary[self.trace_summary.index.str.contains("gamma")][["mean"]].reset_index().rename(columns = {'mean': 'gamma'}).assign(name = lambda x: x["index"].str.replace("_gamma", "")), 
            self.trace_summary[self.trace_summary.index.str.contains("alpha")][["mean"]].reset_index().rename(columns = {'mean': 'alpha'}).assign(name = lambda x: x["index"].str.replace("_alpha", "")), on = "name" )
        control_coefficients = self.trace_summary[self.trace_summary.index.str.contains("_control_coef")][["mean", "sd"]].reset_index().assign(name = lambda x: x["index"].str.replace("_control_coef", ""))
        delay_coefficients = self.trace_summary[self.trace_summary.index.str.contains("_media_coef")][["mean", "sd"]].reset_index().assign(name = lambda x: x["index"].str.replace("_media_coef", ""))

        data_transformed_decomposed = data.copy()
        intercept = self.trace_summary[self.trace_summary.index == "intercept"]["mean"].iloc[0]
        data_transformed_decomposed["intercept"] = intercept

        for delay_channel in self.delay_channels:
            adstock = adstock_params[adstock_params.name == delay_channel]
            adstock_theta = adstock["mean"].iloc[0]
                
            data_transformed_decomposed[delay_channel] = self._geometric_adstock(data_transformed_decomposed[delay_channel].values, theta = adstock_theta)
            
            saturation = saturation_params[saturation_params.name == delay_channel]
            
            saturation_alpha = saturation["alpha"].iloc[0]
            saturation_gamma = saturation["gamma"].iloc[0]
                
            data_transformed_decomposed[delay_channel] = self._hill_saturation(data_transformed_decomposed[delay_channel].values, alpha = saturation_alpha, gamma = saturation_gamma)
            
            coefs = delay_coefficients[delay_coefficients.name == delay_channel]
            coef = coefs["mean"].iloc[0]
            
            data_transformed_decomposed[delay_channel] = data_transformed_decomposed[delay_channel] * coef
            
        for variable in self.control_variables:
            coefs = control_coefficients[control_coefficients.name == variable]
            coef = coefs["mean"].iloc[0]
            data_transformed_decomposed[variable] = data_transformed_decomposed[variable] * coef

        return data_transformed_decomposed

    def plot_model_fit(self, data):
        data["prediction"] = data[self.delay_channels + self.control_variables + ["intercept"]].sum(axis = 1)
        y_true = self.y_true[self.START_INDEX:self.END_INDEX]

        qs = mquantiles(self.revenue_transformation.inverse_transform(self.ppc_all["outcome"]), [0.025, 0.975], axis=0)
        fig, ax = plt.subplots(figsize = (20, 8))
        _ = ax.plot(self.revenue_transformation.inverse_transform(self.ppc_all["outcome"].mean(axis = 0).reshape(-1, 1)), color = "blue", label = "predicted posterior sampling")
        _ = ax.plot(y_true, 'ro', label = "true")
        _ = ax.plot(qs[0], '--', color = "grey", label = "2.5%", alpha = 0.5)
        _ = ax.plot(qs[1], '--', color = "grey", label = "97.5%", alpha = 0.5)
        _ = ax.legend()

    def compute_spend_effect_share(self, data, data_transformed_decomposed, media_spend_exposure_df):

        exposure_to_spend_menten_func = lambda exposure, V_max, K_m: exposure * K_m / (V_max - exposure)

        spend_df = pd.DataFrame()
        for media_channel in self.media_channels:
            temp_series = data[media_channel].iloc[self.START_INDEX:self.END_INDEX].values
            
            if len(media_spend_exposure_df[media_spend_exposure_df.exposure == media_channel]) > 0:
                vmax = media_spend_exposure_df[media_spend_exposure_df.exposure == media_channel]["V_max"].iloc[0]
                km = media_spend_exposure_df[media_spend_exposure_df.exposure == media_channel]["K_m"].iloc[0]
                spends = exposure_to_spend_menten_func(temp_series, V_max = vmax, K_m = km)
                spends_total = spends.sum()
                
            else:
                spends_total = temp_series.sum()
                
            spend_df = pd.concat([spend_df, pd.DataFrame({'media': [media_channel], 'total_spend': [spends_total]})]).reset_index(drop=True)

        spend_df["spend_share"] = spend_df["total_spend"] / spend_df["total_spend"].sum()

        response_df = pd.DataFrame()
        for media_channel in self.media_channels:
            response = data_transformed_decomposed[media_channel].iloc[self.START_INDEX:self.END_INDEX].values
            response_total = response.sum()
            response_df = pd.concat([response_df, pd.DataFrame({'media': [media_channel], 'total_effect': [response_total]})]).reset_index(drop=True)

        response_df["effect_share"] = response_df["total_effect"] / response_df["total_effect"].sum()

        spend_response_share_df = pd.concat([spend_df, response_df.drop(columns = ["media"])], axis = 1)
        
        return spend_response_share_df

    def plot_spend_vs_effect_share(self, spend_response_share_df):
            
            plot_spend_effect_share = spend_response_share_df.melt(id_vars = ["media"], value_vars = ["spend_share", "effect_share"])

            plt = ggplot(plot_spend_effect_share, aes("media", "value", fill = "variable")) \
            + geom_bar(stat = "identity", position = "dodge") \
            + geom_text(aes(label = "value * 100", group = "variable"), color = "darkblue", position=position_dodge(width = 0.5), format_string = "{:.2f}%") \
            + coord_flip() \
            + ggtitle("Share of Spend VS Share of Effect") + ylab("") + xlab("") \
            + theme(figure_size = (10, 6), 
                            legend_direction='vertical', 
                            legend_title=element_blank(),
                            legend_key_size=20, 
                            legend_entry_spacing_y=5) 
            return plt

    def _geometric_adstock_pymc(self, x, theta):
        #we use the function from https://towardsdatascience.com/modeling-marketing-mix-using-pymc3-ba18dd9e6e68
        #As the theano version of adstock is not straight forward. 
        x = tt.as_tensor_variable(x)
        
        def adstock_geometric_recurrence_theano(index, 
                                                input_x, 
                                                decay_x,   
                                                theta):
            return tt.set_subtensor(decay_x[index], 
                tt.sum(input_x + theta * decay_x[index - 1]))
        len_observed = x.shape[0]
        x_decayed = tt.zeros_like(x)
        x_decayed = tt.set_subtensor(x_decayed[0], x[0])
        output, _ = theano.scan(
            fn = adstock_geometric_recurrence_theano, 
            sequences = [tt.arange(1, len_observed), x[1:len_observed]], 
            outputs_info = x_decayed,
            non_sequences = theta, 
            n_steps = len_observed - 1
        )
        
        return output[-1]

    def _geometric_adstock(self, x: float, theta: float):
        x_decayed = np.zeros_like(x)
        x_decayed[0] = x[0]
                                
        for xi in range(1, len(x_decayed)):
            x_decayed[xi] = x[xi] + theta * x_decayed[xi - 1]

        return x_decayed

    def _hill_saturation(self, x, alpha, gamma): 
        x_s_hill = x ** alpha / (x ** alpha + gamma ** alpha)
        return x_s_hill

    def _nrmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))