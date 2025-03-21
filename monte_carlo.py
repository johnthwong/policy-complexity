import numpy as np
import pandas as pd

def simulate_hsv_sample(
        n=int, tau=float, theta=float, ell=float, sigma=float,
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
        'working': working,  # Household size
        'household_size': household_size,  # Non-dependents in household
    })

    if weights:
        weights = np.random.uniform(0.5, 1.5, size=n)
        df['weights'] = weights

    # Filter the data
    df = df[(df['income'] > 0) & (df['net_income'] > 0)] 

    return df