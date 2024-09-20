import pandas as pd
import numpy as np
import optuna
import optuna.visualization as vis
import indicators as idct
import plotly.express as px

# Load and preprocess data
def load_data(file_path, start_time, end_time):
    data = pd.read_pickle(file_path)
    data = data[(data['open_time'] >= start_time) & (data['open_time'] <= end_time)].reset_index(drop=True)
    data['open_time'] = pd.to_datetime(data['open_time'])
    data[['open', 'high', 'low', 'close', 'fundingRate']] = data[['open', 'high', 'low', 'close', 'fundingRate']].astype(float)
    data['funding_signal'] = data['open_time'].dt.strftime('%H:%M').isin(['00:00', '08:00', '16:00'])
    return data

# Compute indicators using NumPy
def compute_indicators(data, ema_period, ub_multiplier, lb_multiplier):
    close_np = data['close'].values
    high_np = data['high'].values
    low_np = data['low'].values
    
    ema = idct.EMA(data['close'], ema_period)
    atr = idct.ATR(data['high'], data['low'], data['close'], ema_period)
    
    ub = ema + ub_multiplier * atr
    lb = ema - lb_multiplier * atr
    
    return ema, ub, lb

# Generate trading signals
def generate_signals(data, ub, lb):
    close_np = data['close'].values
    signals = np.zeros(len(close_np), dtype=int)
    signals[close_np > ub] = 1  # Buy signal
    signals[close_np < lb] = -1 # Sell signal
    return signals

# Vectorized backtest using NumPy
def run_backtest(data, signals, fee_rate, initial_capital):
    open_np = data['open'].values
    close_np = data['close'].values
    funding_rate_np = data['fundingRate'].values
    funding_signal_np = data['funding_signal'].values.astype(int)
    
    position = np.zeros(len(signals), dtype=float)
    cash = np.zeros(len(signals), dtype=float)
    cash[0] = initial_capital
    equity = np.zeros(len(signals), dtype=float)
    equity[0] = initial_capital

    # Vectorized processing for positions and equity
    for i in range(1, len(signals)):
        if signals[i-1] == 1 and position[i-1] == 0:
            # Enter long
            entry_price = open_np[i]
            position[i] = cash[i-1] / (entry_price * (1 + fee_rate))
            cash[i] = cash[i-1] - position[i] * entry_price * (1 + fee_rate)
        elif signals[i-1] == -1 and position[i-1] > 0:
            # Exit long
            exit_price = open_np[i]
            cash[i] = cash[i-1] + position[i-1] * exit_price * (1 - fee_rate)
            position[i] = 0
        else:
            # Carry forward previous values
            position[i] = position[i-1]
            cash[i] = cash[i-1]

        # Update equity
        equity[i] = cash[i] + position[i] * close_np[i]

        # Funding fee adjustment
        if funding_signal_np[i] == 1 and position[i] > 0:
            funding_fee = position[i] * open_np[i] * funding_rate_np[i]
            cash[i] -= funding_fee
            equity[i] -= funding_fee

    return equity

# Calculate Sharpe ratio
def calculate_sharpe_ratio(equity, annualization_factor):
    log_returns = np.diff(np.log(equity))
    sharpe_ratio = (np.mean(log_returns) / np.std(log_returns)) * np.sqrt(annualization_factor) if np.std(log_returns) != 0 else -1e6
    return np.round(sharpe_ratio, 4)

# Objective function for Optuna optimization
def objective(trial, data_slice, fee_rate, initial_capital, annualization_factor):
    ema_period = trial.suggest_int('EMA_p', 10, 600, step=10)
    ub_multiplier = trial.suggest_float('UB_m', 0.0, 9.0, step=0.01)
    lb_multiplier = trial.suggest_float('LB_m', -0.5, 1.5, step=0.01)

    # Compute indicators and signals
    ema, ub, lb = compute_indicators(data_slice, ema_period, ub_multiplier, lb_multiplier)
    signals = generate_signals(data_slice, ub, lb)

    # Run backtest
    equity = run_backtest(data_slice, signals, fee_rate, initial_capital)

    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(equity, annualization_factor)
    return sharpe_ratio

# Main execution
if __name__ == '__main__':
    # Parameters
    file_path = r"TMBA_data/BTCUSDT_30m.pkl"
    start_time = '2020/01/01 00:00:00'
    end_time = '2024/09/14 23:59:59'
    fee_rate = 0.0005
    initial_capital = 10000.0
    annualization_factor = 365 * 24 * 2  # Adjust based on your data frequency

    # Load data
    data = load_data(file_path, start_time, end_time)
    total_length = len(data)

    # Define walk-forward periods
    time_len = total_length // 10
    num_walks = 6
    best_params_list = []
    top_trials_list = []

    for j in range(num_walks):
        train_start = j * time_len
        train_end = (j + 4) * time_len
        data_slice = data.iloc[train_start:train_end].reset_index(drop=True)

        # Optimize using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, data_slice, fee_rate, initial_capital, annualization_factor), n_trials=800)

        # Record best parameters and top trials
        best_params = study.best_params
        best_value = study.best_value
        best_params['Sharpe_ratio'] = best_value
        best_params_list.append(best_params)

        study_df = study.trials_dataframe().drop_duplicates(subset=['params_EMA_p', 'params_UB_m', 'params_LB_m'])
        top_trials = study_df.sort_values(by='value', ascending=False).head(20)[['value', 'params_EMA_p', 'params_UB_m', 'params_LB_m']]
        top_trials_list.append(top_trials)
        
        # fig = px.parallel_coordinates(
        #     study_df,
        #     dimensions=[f'params_{params_name[0]}', f'params_{params_name[1]}', f'params_{params_name[2]}', 'value'],  # 使用正確的列名
        #     labels={
        #         f'params_{params_name[0]}': params_name[0],
        #         f'params_{params_name[1]}': params_name[1],
        #         f'params_{params_name[2]}': params_name[2],
        #         'value': params_name[-1]
        #     },
        #     color='value',
        #     color_continuous_scale=px.colors.sequential.Viridis
        # )
        # fig.show()
        # fig.write_image(f"parallel_coordinates_plot{j}.png", scale=4)
    
    for j in range(num_walks):
        # Print results
        print(f"Walk {j + 1} Best Parameters: {best_params_list[j]}")
        print(f"Top Trials for Walk {j + 1}:\n{top_trials_list[j]}\n")


# #%%
# data = pd.read_pickle(r"TMBA_data/BTCUSDT_30m.pkl")
# start_time, end_time = '2020/01/01 00:00:00', '2024/09/14 23:59:59'
# data = data[(data['open_time'] >= start_time) & (data['open_time'] <= end_time)]
# data.reset_index(drop=True, inplace=True)

# #%%
# columns_to_convert = ['open', 'high', 'low', 'close', 'fundingRate']
# open_np, high_np, low_np, close_np, funding_rate_np = [
#     np.array(data[col]).astype(float) for col in columns_to_convert
# ]
# open_pd, high_pd, low_pd, close_pd, funding_rate_pd = [
#     data[col] for col in columns_to_convert
# ]
# open_time_np = np.array(data['open_time']).astype('datetime64[m]')
# open_time_pd = data['open_time']
# funding_signal = np.isin(data['open_time'].dt.strftime('%H:%M'), ['00:00', '08:00', '16:00'])
# length = len(data)

# #%%
# T = 365 * 24 * 2
# fee_rate = 0.0005
# initial_cap = 10000.0
# params_name = ['EMA_p', 'UB_m', 'LB_m', 'Sharpe ratio']

# # Define the walk-forward periods
# time_len = int(length / 10)
# num_walks = 6
# idx_range = []
# for j in range(0, num_walks):
#     idx_range.append(range(j * time_len, (j + 4) * time_len))
# best_params_list = []
# top_trials_list = []

# #%%
# for j in range(0, num_walks):
#     def objective(trial):
#         ema_p = trial.suggest_int(params_name[0], 10, 300, step=10)
#         ub_m = trial.suggest_float(params_name[1], 0.0, 2.5, step=0.01)
#         lb_m = trial.suggest_float(params_name[2], -0.5, 1.5, step=0.01)
        
#         EMA = idct.EMA(close_pd.iloc[idx_range[j]], ema_p).to_numpy()
#         ATR = idct.ATR(high_pd.iloc[idx_range[j]], low_pd.iloc[idx_range[j]], close_pd.iloc[idx_range[j]], ema_p).to_numpy()
#         UB = EMA + ub_m * ATR
#         LB = EMA - lb_m * ATR
        
#         signal1 = close_np[idx_range[j]] > UB
#         signal2 = close_np[idx_range[j]] < LB
        
#         cap = np.zeros(len(idx_range[j]))
#         cap[0] = initial_cap
#         equity = np.zeros(len(idx_range[j]))
#         equity[0] = initial_cap
#         long_position = np.zeros(len(idx_range[j]))
        
#         for i in range(1, len(idx_range[j])):
#             cap[i] = cap[i-1]
#             long_position[i] = long_position[i-1]
#             if cap[i] >= 0 and long_position[i] <= 0 and signal1[i-1]:
#                 long_entry_price = open_np[idx_range[j][i]]
#                 long_entry_amount = cap[i-1]
#                 long_position[i] = long_entry_amount / (long_entry_price * (1 + fee_rate))
#                 cap[i] = cap[i-1] - long_entry_amount
            
#             elif long_position[i] > 0:
#                 if signal2[i-1]:
#                     cap[i] += long_position[i] * open_np[idx_range[j][i]] * (1 - fee_rate)
#                     long_position[i] = 0

#             equity[i] = cap[i] + long_position[i] * close_np[idx_range[j][i]]
            
#             if funding_signal[idx_range[j][i]]:
#                 if long_position[i] > 0:
#                     funding_fee = long_position[i] * open_np[idx_range[j][i]] * funding_rate_np[idx_range[j][i]]
#                     equity[i] -= funding_fee
#                     cap[i] -= funding_fee
        
#         log_rets = np.diff(np.log(equity))
#         sharpe_ratio = (np.mean(log_rets) / np.std(log_rets)) * np.sqrt(T) if np.std(log_rets) != 0 else -1e6
        
#         # #
#         # num_years = (length * 15) / (60 * 24 * 365)
#         # cagr = (equity[-1] / equity[0]) ** (1 / num_years) - 1
#         # equity_curve = pd.Series(equity)
#         # rolling_max = equity_curve.cummax()
#         # drawdown = equity_curve / rolling_max - 1
#         # max_drawdown = drawdown.min()
#         # max_drawdown = abs(max_drawdown)
#         # if max_drawdown == 0:
#         #     calmar_ratio = 0
#         # elif cagr > 0:
#         #     calmar_ratio = cagr / max_drawdown
#         # else:
#         #     calmar_ratio = cagr

#         return np.round(sharpe_ratio, 4)
    
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=200)
    
#     best_params = study.best_params
#     best_value = study.best_value
#     best_params['value'] = best_value
#     best_params_list.append(best_params)
    
#     study_df = study.trials_dataframe()
#     study_df = study_df.drop_duplicates(subset=[f'params_{params_name[0]}', f'params_{params_name[1]}', f'params_{params_name[2]}'])
#     top_trials_list.append(
#         study_df.sort_values(by="value", ascending=False).head(50)[
#             ['value', f'params_{params_name[0]}', f'params_{params_name[1]}', f'params_{params_name[2]}']
#         ]
#     )
#     print(top_trials_list)
#     # fig = px.parallel_coordinates(
#     #     study_df,
#     #     dimensions=[f'params_{params_name[0]}', f'params_{params_name[1]}', f'params_{params_name[2]}', 'value'],  # 使用正確的列名
#     #     labels={
#     #         f'params_{params_name[0]}': params_name[0],
#     #         f'params_{params_name[1]}': params_name[1],
#     #         f'params_{params_name[2]}': params_name[2],
#     #         'value': params_name[-1]
#     #     },
#     #     color='value',
#     #     color_continuous_scale=px.colors.sequential.Viridis
#     # )
#     # fig.show()
#     # fig.write_image(f"parallel_coordinates_plot{j}.png", scale=4)
    
# #%%
# # import pickle
# # pickle_file_name = 'wfo_results.pkl'
# # results = {
# #     'idx_range_OOS': idx_range_OOS,
# #     'best_params_list': best_params_list,
# #     'top_trials_list': top_trials_list
# # }

# # best_params_list = results['best_params_list']
# # values = [item['value'] for item in best_params_list]
# # mean_value = np.mean(values)
# # std_value = np.std(values)

# # print(f"mean of values: {mean_value:.4f}")
# # print(f"std of values: {std_value:.4f}")

# # with open(pickle_file_name, 'wb') as f:
# #     pickle.dump(results, f)

# # print(f"Results have been saved to {pickle_file_name}")
