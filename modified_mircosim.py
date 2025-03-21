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