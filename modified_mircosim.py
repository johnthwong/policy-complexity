from policyengine_us import Microsimulation

class ModifiedMicrosimulation(Microsimulation):
    def get_dataframe(self):
        df = self.calculate_dataframe(
            variable_names = ["household_market_income", "household_net_income", "household_size", "tax_unit_dependents", "household_weight"],
            period = 2024
        )
        self.dataframe = df
    
    def remove_zero_income(self):
        df = self.dataframe
        df = df.loc[df['household_market_income'] > 0]
        self.dataframe = df

    def get_household_mtrs(
            simulation, 
            step: float = 1000, 
            max_income: float = 1e6, 
            period: int = 2025
            ):
        # Get original income
        original_incomes = simulation.calculate(
            "market_income", 
            period=f"{period}"
            )
        sample_size = len(original_incomes)

        results = {}
        for household_id in range(sample_size):
            household_income = original_incomes[household_id]
            mtrs = []

