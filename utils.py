import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import scipy.optimize as optim

def simulate_hsv_sample(
        n, tau, theta, ell, sigma,
        seed=None, weights=False
        ):
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Simulate market income (y)
    income = np.random.lognormal(mean=10, sigma=0.8, size=n)

    # Simulate household size 
    household_size = np.random.randint(1, 6, size=n)

    # Simulate number of dependents (a)
    dependents = np.random.binomial(household_size - 1, 0.5, size=n)
    dependents = np.clip(dependents, 0, household_size - 1)

    # Households with size 1 have 0 dependents
    dependents[household_size == 1] = 0

    # Calculate working adults (A)
    working = household_size - dependents

    # Simulate taxes (T)
    net_income = (
        pow(household_size, theta) / working * (
            ell * pow(income, 1 - tau) * (
                np.exp(np.random.normal(scale = sigma, size = n))
                )
            )
    )

    # Create DataFrame
    df = pd.DataFrame({
        'income': income,  # Market income
        'net_income': net_income,  # Net income after taxes
        'working': working,  # Number of working adults
        'household_size': household_size,  # Total household size
    })

    if weights:
        weights = np.random.uniform(0.5, 1.5, size=n)
        df['weights'] = weights

    # Filter the data
    df = df[(df['income'] > 0) & (df['net_income'] > 0)] 

    return df

def generate_ols_model(df):
    X = np.log(df[['income', 'household_size']])
    X = sm.add_constant(X)
    y = np.log(df['net_income']) + np.log(df['working'])

    model = sm.OLS(y, X).fit()

    return model

def get_sigma_hat(df, type=None):
    model = generate_ols_model(df)
    sigma_hat = pow(np.var(model.resid), 1/2)
    
    if (type is None) or (type == "sigma"):
        return sigma_hat
    elif type == "pearson":
        var_pred = np.var( model.predict().tolist() - np.log(df['working']) )
        var_actual = np.var( np.log( df['net_income'] ) )
        corr = ( var_actual + var_pred - pow(sigma_hat, 2) ) / (
            2 * pow(var_actual, 1/2) * pow(var_pred, 1/2)
        )
        return corr
    elif type == "spearman":
        df_corr = pd.DataFrame({
            "actual": np.log(df['net_income']), 
            "pred": model.predict()  - np.log(df['working'])
            })
        cov_matrix = df_corr.corr("spearman")
        return cov_matrix.actual.iloc[1]
    else:
        raise TypeError("Argument supplied for type is invalid.")
        


def generate_mle_model(df, initial_params):
    # Split dataframe
    # // is the floor division operator
    half_index = df.shape[0] // 2

    if df.shape[0] % 2 != 0:
        df = df.drop(df.index[-1])

    df_2 = pd.concat([
        # .iloc[:x] selects all rows from the beginning up to, but not including, the row at position x
            df.iloc[:half_index], 
            df.iloc[half_index:].reset_index(drop = True)
        ], axis=1)

    # Rename columns
    column_names = df.columns

    new_column_names = []
    for k in column_names:
        new_name = str(k) + "_i"
        new_column_names.append(new_name)
    for k in column_names:
        new_name = str(k) + "_j"
        new_column_names.append(new_name)

    df_2.columns = new_column_names

    # Add a rank indicator variable

    df_2 = df_2.assign(
        atr_i = df_2['net_income_i'] / df_2['income_i'],
        atr_j = df_2['net_income_j'] / df_2['income_j'],
    )

    df_2 = df_2.assign(
        rank_binary = np.where(df_2['atr_i'] < df_2['atr_j'], 1, 0)
    )

    # Specify a log-likelihood function
    def neg_log_likelihood(params, data):
        tau, theta, sigma = params
        
        # Specify random variable
        randvar = (
                tau * np.log(data['income_i'] / data['income_j'])
                + np.log(data['working_i'] / data['working_j'])
                - theta * (
                    np.log(data['household_size_i'] / data['household_size_j'])
                )
        )
        # Specify CDF
        cdf = norm.cdf(randvar, scale = np.pow(2, 1/2)*sigma)

        # Now specify log-likelihood function
        log_likelihood = np.sum(
            data['rank_binary'] * np.log(cdf) 
            + (1 - data['rank_binary']) * np.log(1 - cdf)
        )
        return -log_likelihood

    model = optim.minimize(
        neg_log_likelihood, initial_params, args = (df_2,),
        method='BFGS'
        )
    
    # print(f"Estimates are {model.x}.")
    
    return model