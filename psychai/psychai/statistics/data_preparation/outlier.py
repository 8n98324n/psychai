class OutlierProcess:

    def __init__(self):
        pass

    
def remove_outliers(input_data, column_name, taylor_multiplier):
        try:
            # Calculate Q1 and Q3
            Q1 = input_data[column_name].quantile(0.25)
            Q3 = input_data[column_name].quantile(0.75)
            
            # Calculate IQR and thresholds
            IQR = Q3 - Q1
            lower_threshold = Q1 - taylor_multiplier * IQR
            higher_threshold = Q3 + taylor_multiplier * IQR
            
            # Filter out outliers
            return_data = input_data[(input_data[column_name] > lower_threshold) & (input_data[column_name] < higher_threshold)]
            return return_data
        except Exception as e:
            print(f"ERROR: {column_name}: {e}")
            return input_data

