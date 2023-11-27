import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

#Zdefiniowanie wskaźników wydajności
def ARC_test(balance, initial_balance, trading_periods_per_year):
    if initial_balance >= balance:
        return 0

    total_return = (balance / initial_balance) ** (1 / trading_periods_per_year) - 1
    return total_return


def ASD_test(returns, trading_periods_per_year):
    annualized_volatility = np.std(returns) * np.sqrt(trading_periods_per_year)
    return annualized_volatility


def MDD(returns):
    cum_returns = np.cumprod(1 + returns) - 1
    max_drawdown = np.min(cum_returns - np.maximum.accumulate(cum_returns))
    max_loss_duration = np.argmax(np.maximum.accumulate(cum_returns) - cum_returns)
    return max_drawdown, max_loss_duration


def IR_test(returns, benchmark_returns):
    excess_returns = returns - benchmark_returns
    information_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return information_ratio


def PC_test(signals):
    total_trading_days = len(signals)
    position_changes = np.sum(np.abs(signals['Buy_Signal'].diff()) + np.abs(signals['Sell_Signal'].diff()))

    position_changes = (position_changes / total_trading_days) * 100
    return position_changes

#pobranie danych
data = yf.download('^SPX', start='1988-12-31', end='2023-10-30')

#RSI
def rsi(data, period=14):
    data = data.copy()

    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    data['avg_gain'] = gain.rolling(window=period, min_periods=1).mean()
    data['avg_loss'] = loss.rolling(window=period, min_periods=1).mean()

    rs = data['avg_gain'] / data['avg_loss']
    data['RSI'] = 100 - (100 / (1 + rs))

    return data['RSI']

# wyliczenie RSI
data.loc[:, 'RSI'] = rsi(data)

# Generowanie sygnałów
def sygnaly(data, overbought_threshold=70, oversold_threshold=30, period=14):
    signals = pd.DataFrame(index=data.index)
    signals['Close'] = data['Close']

    # Buy signals
    signals['Buy_Signal'] = (data['RSI'] < oversold_threshold) & (data['RSI'].shift(1) >= oversold_threshold)

    # Sell signals
    signals['Sell_Signal'] = (data['RSI'] > overbought_threshold) & (data['RSI'].shift(1) <= overbought_threshold)

    return signals

# Symulacja strategii
def symulacja(signals, initial_balance=100000, transaction_cost=0.0005):
    balance = initial_balance
    position = 0

    for index, row in signals.iterrows():
        if row['Buy_Signal']:

            position = balance / row['Close']
            balance = 0
            balance -= transaction_cost * position * row['Close']

        elif row['Sell_Signal']:
            # Sell all stocks
            balance += position * row['Close']
            position = 0
            balance -= transaction_cost * balance


    balance += position * signals['Close'].iloc[-1]

    return balance

# Generowanie wykresów
def plot_results(data, signals):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Stock Price')
    plt.scatter(signals.index[signals['Buy_Signal']], signals['Close'][signals['Buy_Signal']], marker='^',
                color='g', label='Buy Signal')
    plt.scatter(signals.index[signals['Sell_Signal']], signals['Close'][signals['Sell_Signal']], marker='v',
                color='r', label='Sell Signal')
    plt.title('RSI-Based Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Parametry do symulacji
training_period = 2 * 252  # 2 lata (252 dni tradingowych w roku)
validation_period = 252
testing_period = 252
transaction_cost = 0.0005  # 0.05%
print(len(data))

# Walk-forward
for i in range(0, len(data) - training_period - validation_period - testing_period + 1, testing_period):
    train_data = data.iloc[i:i + training_period]
    val_data = data.iloc[i + training_period:i + training_period + validation_period]
    test_data = data.iloc[i + training_period + validation_period:i + training_period + validation_period + testing_period]

    train_data.loc[:, 'RSI'] = rsi(train_data)

    valid_signals = sygnaly(train_data)


    plot_results(train_data, valid_signals)


    final_balance = symulacja(valid_signals, initial_balance=100000, transaction_cost=transaction_cost)


    print(f"Testing Period {i // testing_period + 1} - Final Balance: ${final_balance}")


def optymalizacja(train_data, validation_data):
    best_balance = 0
    best_params = None

    for period in range(10, 31, 5):
        for overbought_threshold in range(70, 91, 5):
            for oversold_threshold in range(10, 31, 5):
                signals = sygnaly(train_data, overbought_threshold, oversold_threshold, period)
                balance = symulacja(signals)

                if balance > best_balance:
                    best_balance = balance
                    best_params = {'period': period, 'overbought_threshold': overbought_threshold,
                                   'oversold_threshold': oversold_threshold}

    return best_params

results_list = []

for i in range(0, len(data) - (training_period + validation_period + testing_period) + 1, testing_period):
    train_data = data.iloc[i:i + training_period]
    val_data = data.iloc[i + training_period:i + training_period + validation_period]
    test_data = data.iloc[
                i + training_period + validation_period:i + training_period + validation_period + testing_period]


    best_params = optymalizacja(train_data, val_data)


    train_data.loc[:, 'RSI'] = rsi(train_data, period=best_params['period'])


    valid_signals = sygnaly(train_data, overbought_threshold=best_params['overbought_threshold'],
                                   oversold_threshold=best_params['oversold_threshold'], period=best_params['period'])


    #plot_results(train_data, valid_signals)


    final_balance = symulacja(valid_signals, initial_balance=100000, transaction_cost=transaction_cost)


    print(f"Testing Period {i // testing_period + 1} - Final Balance: ${final_balance} (Optimized Parameters: {best_params})")

    returns = valid_signals['Close'].pct_change().dropna()
    trading_periods_per_year = 252

    arc = ARC_test(final_balance, initial_balance=100000, trading_periods_per_year=trading_periods_per_year)
    asd = ASD_test(returns, trading_periods_per_year=trading_periods_per_year)
    max_drawdown, max_loss_duration = MDD(returns)
    information_ratio = IR_test(returns, benchmark_returns=0)
    position_changes = PC_test(valid_signals)


    results_list.append({
        'Testing Period': i // testing_period + 1,
        'Final Balance': final_balance,
        'ARC': arc,
        'ASD': asd,
        'Max Drawdown': max_drawdown,
        'Max Loss Duration': max_loss_duration,
        'Information Ratio': information_ratio,
        'Position Changes': position_changes
    })


results_table = pd.DataFrame(results_list)


print(results_table)
results_table.to_excel('results.xlsx', index=False)

initial_balance=100000
transaction_cost=0.0005

# Ramka danych dla strategii Kup & Trzymaj
buy_hold_signals = pd.DataFrame(index=data.index)
buy_hold_signals['Close'] = data['Close']
buy_hold_signals['Buy_Signal'] = True
buy_hold_signals['Sell_Signal'] = False

# Kup akcje na starcie
buy_hold_signals['Position'] = initial_balance / buy_hold_signals['Close'].iloc[0]
buy_hold_signals['Balance'] = initial_balance - transaction_cost * buy_hold_signals['Position'] * buy_hold_signals['Close']

# Symulacja strategii Kup & Trzymaj - utrzymanie pozycji przez cały okres
buy_hold_signals['Position'] = buy_hold_signals['Position'].ffill()
buy_hold_signals['Balance'] = buy_hold_signals['Balance'].ffill()


# Wynik końcowy
final_balance = buy_hold_signals['Final_Balance'].iloc[-1]
print(f"Kup & Trzymaj - Final Balance: ${final_balance}")

# Wykres
plt.plot(data['Close'], label='Stock Price')
plt.title('Kup & Trzymaj - Buy & Hold Strategy')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
