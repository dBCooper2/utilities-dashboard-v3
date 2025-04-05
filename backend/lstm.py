class LSTMForecast:
    def __init__(self, region_name=None, initial_df=None, season_length=None, log_level='INFO'):
        """
        Initialize the LSTM forecasting model.
        
        Parameters:
        -----------
        region_name : str, optional
            The name of the region being forecasted.
        initial_df : pandas.DataFrame, optional
            Initial dataframe to process during initialization.
        season_length : int, optional
            The seasonal period of the time series. If None, it will be inferred from the data.
        log_level : str, optional
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        import logging
        import sys
        import os
        
        # Setup logging
        self.logger = logging.getLogger(f"LSTMForecast_{region_name if region_name else 'default'}")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Create handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        self.model = None
        self.season_length = season_length
        self.data = None
        self.freq = None
        self.forecast_df = None
        self.fitted = False
        self.region_name = region_name
        self.scaler = None
        self.seq_length = 7  # Default lookback window (1 week)
        self.n_features = 1  # Default number of features (just the target variable)
        
        print(f"Initializing LSTM model for region: {region_name if region_name else 'default'}")
        self.logger.info(f"Initializing LSTM model for region: {region_name if region_name else 'default'}")
        
        # Process initial dataframe if provided
        if initial_df is not None:
            print(f"Processing initial dataframe with {len(initial_df)} rows")
            self.logger.info(f"Processing initial dataframe with {len(initial_df)} rows")
            self.fit(initial_df)
    
    def _detect_frequency(self, df):
        """
        Detect the frequency of the time series data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data with a 'ds' column.
            
        Returns:
        --------
        freq : str
            The detected frequency: 'D' (daily) for LSTM.
        """
        import pandas as pd
        
        print("Detecting time series frequency...")
        self.logger.info("Detecting time series frequency")
        
        # Convert 'ds' to datetime if it's not already
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Get the time differences
        diff = df['ds'].diff().dropna()
        
        # Calculate the most common time difference
        most_common_diff = diff.mode()[0]
        
        # Determine the frequency - for LSTM we use daily data
        if most_common_diff <= pd.Timedelta(minutes=15):
            freq = 'D'  # Override to daily
            print("Detected frequency: 15min - Overriding to daily (D) for LSTM")
            self.logger.info("Detected frequency: 15min - Overriding to daily (D) for LSTM")
        elif most_common_diff <= pd.Timedelta(hours=1):
            freq = 'D'  # Override to daily
            print("Detected frequency: hourly - Overriding to daily (D) for LSTM")
            self.logger.info("Detected frequency: hourly - Overriding to daily (D) for LSTM")
        else:
            freq = 'D'
            print(f"Detected frequency: {freq}")
            self.logger.info(f"Detected frequency: {freq}")
            
        return freq
    
    def _detect_season_length(self, freq):
        """
        Determine the appropriate season length based on frequency.
        
        Parameters:
        -----------
        freq : str
            The frequency of the time series.
            
        Returns:
        --------
        season_length : int
            The seasonal period for the given frequency.
        """
        print("Determining seasonal period...")
        self.logger.info("Determining seasonal period")
        
        # For LSTM we always use daily data, so season_length is 7 (weekly pattern)
        season_length = 7
            
        print(f"Determined season length: {season_length}")
        self.logger.info(f"Determined season length: {season_length}")
        return season_length
    
    def _create_sequences(self, data, seq_length):
        """
        Create sequences for LSTM training.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The normalized time series data.
        seq_length : int
            Length of sequences to create.
            
        Returns:
        --------
        X : numpy.ndarray
            Sequences for input.
        y : numpy.ndarray
            Target values for each sequence.
        """
        import numpy as np
        
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, seq_length, n_features):
        """
        Build the LSTM model.
        
        Parameters:
        -----------
        seq_length : int
            Length of input sequences.
        n_features : int
            Number of features in the data.
            
        Returns:
        --------
        model : tensorflow.keras.Model
            The compiled LSTM model.
        """
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def _resample_to_daily(self, df):
        """
        Resample data to daily frequency.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame resampled to daily frequency.
        """
        import pandas as pd
        
        print("Resampling data to daily frequency...")
        self.logger.info("Resampling data to daily frequency")
        
        # Ensure ds is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Group by unique_id and resample
        resampled_dfs = []
        for unique_id, group in df.groupby('unique_id'):
            # Set ds as index
            group_indexed = group.set_index('ds')
            
            # Resample to daily, taking the mean of each day
            daily = group_indexed.resample('D')['y'].mean().reset_index()
            
            # Add unique_id back
            daily['unique_id'] = unique_id
            
            # Add to list
            resampled_dfs.append(daily)
        
        # Combine all resampled data
        resampled_df = pd.concat(resampled_dfs)
        
        print(f"Resampled data: {len(df)} original rows -> {len(resampled_df)} daily rows")
        self.logger.info(f"Resampled data: {len(df)} original rows -> {len(resampled_df)} daily rows")
        
        return resampled_df
    
    def fit(self, df):
        """
        Fit the LSTM model to the provided dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data with columns 'ds', 'y', and 'unique_id'.
            
        Returns:
        --------
        self : LSTMForecast
            The fitted model instance.
        """
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Fitting LSTM model{region_suffix} with {len(df)} data points...")
        self.logger.info(f"Fitting LSTM model{region_suffix} with {len(df)} data points")
        
        # Store the original data
        self.data = df.copy()
        
        # Sort data by date to ensure proper time ordering
        self.data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Detect frequency
        self.freq = self._detect_frequency(df)
        
        # Set season length if not provided
        if self.season_length is None:
            self.season_length = self._detect_season_length(self.freq)
        
        # Resample to daily if not already daily
        if self.freq != 'D':
            self.data = self._resample_to_daily(self.data)
            self.freq = 'D'
        
        # LSTM requires numeric inputs, so we'll use sklearn's MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        unique_ids = self.data['unique_id'].unique()
        
        # If multiple time series, we'll train separate models for each
        # For now, we'll just use the first unique_id
        if len(unique_ids) > 1:
            print(f"Multiple unique_ids detected ({len(unique_ids)}). Using first id: {unique_ids[0]}")
            self.logger.warning(f"Multiple unique_ids detected ({len(unique_ids)}). Using first id: {unique_ids[0]}")
            
            # Filter data for the first unique_id
            self.data = self.data[self.data['unique_id'] == unique_ids[0]].copy()
        
        # Prepare data for LSTM
        scaled_data = self.scaler.fit_transform(self.data[['tavg']].values)
        
        # Create sequences for training
        X, y = self._create_sequences(scaled_data, self.seq_length)
        self.n_features = X.shape[2]
        
        # Build and train LSTM model
        print("Building and training LSTM model...")
        self.logger.info("Building and training LSTM model")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Build the model
        self.model = self._build_model(self.seq_length, self.n_features)
        
        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Use a 80/20 train-validation split
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Fit the model
        print(f"Training LSTM model with {len(X_train)} sequences...")
        self.logger.info(f"Training LSTM model with {len(X_train)} sequences")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.fitted = True
        
        print("Model fitting complete!")
        self.logger.info("Model fitting complete")
        
        return self
    
    def forecast(self, h=7, level=None):
        """
        Generate forecasts for the next h days.
        
        Parameters:
        -----------
        h : int, optional
            The forecast horizon in days. Default is 7 (one week).
        level : list, optional
            Confidence levels for prediction intervals. Implementation uses a simple approach
            based on the model's error distribution.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the forecasts.
        """
        if not self.fitted:
            error_msg = "Model has not been fitted yet. Call fit() first."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import pandas as pd
        import numpy as np
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Generating forecast{region_suffix} for next {h} days...")
        self.logger.info(f"Generating forecast{region_suffix} for next {h} days")
        
        # Get the last sequence from the data
        sorted_data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Get unique identifiers from the training data
        unique_id = sorted_data['unique_id'].iloc[0]
        
        # Get the last date in the training data
        last_date = pd.to_datetime(sorted_data['ds']).max()
        print(f"Last historical date: {last_date}")
        self.logger.info(f"Last historical date: {last_date}")
        
        # Prepare the last known sequence
        scaled_data = self.scaler.transform(sorted_data[['tavg']].values)
        last_sequence = scaled_data[-self.seq_length:].reshape(1, self.seq_length, self.n_features)
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=h,
            freq='D'
        )
        
        # Generate predictions recursively
        predictions = []
        prediction_intervals = []
        
        # Current sequence to predict from
        current_sequence = last_sequence.copy()
        
        # Calculate prediction intervals if requested
        # For LSTM, we'll calculate based on the model's historical error
        error_percentiles = {}
        if level is not None:
            # Get predictions for training data
            print("Calculating prediction intervals based on historical errors...")
            self.logger.info("Calculating prediction intervals based on historical errors")
            
            X, y_true = self._create_sequences(scaled_data, self.seq_length)
            y_pred = self.model.predict(X)
            
            # Calculate errors
            errors = (y_true - y_pred.flatten())
            
            # Store error percentiles for each confidence level
            for conf_level in level:
                lower_percentile = (100 - conf_level) / 2
                upper_percentile = 100 - lower_percentile
                
                error_percentiles[conf_level] = {
                    'lower': np.percentile(errors, lower_percentile),
                    'upper': np.percentile(errors, upper_percentile)
                }
        
        # Make predictions for h steps ahead
        for i in range(h):
            # Predict the next value
            next_pred = self.model.predict(current_sequence)[0][0]
            predictions.append(next_pred)
            
            # Store prediction intervals for this step
            if level is not None:
                step_intervals = {}
                for conf_level in level:
                    lower = next_pred + error_percentiles[conf_level]['lower']
                    upper = next_pred + error_percentiles[conf_level]['upper']
                    step_intervals[conf_level] = {'lower': lower, 'upper': upper}
                prediction_intervals.append(step_intervals)
            
            # Update the sequence for the next prediction
            current_sequence = np.append(
                current_sequence[:, 1:, :],
                [[[next_pred]]],
                axis=1
            )
        
        # Inverse transform predictions
        predictions_scaled_back = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'unique_id': [unique_id] * len(future_dates),
            'ds': future_dates,
            'LSTM': predictions_scaled_back.flatten()
        })
        
        # Add prediction intervals if requested
        if level is not None:
            for conf_level in level:
                lower_values = []
                upper_values = []
                
                for i, interval in enumerate(prediction_intervals):
                    lower = interval[conf_level]['lower']
                    upper = interval[conf_level]['upper']
                    
                    # Ensure values are within valid range for inverse scaling
                    lower = max(0, min(1, lower))
                    upper = max(0, min(1, upper))
                    
                    lower_values.append(lower)
                    upper_values.append(upper)
                
                # Inverse transform the interval bounds
                lower_scaled_back = self.scaler.inverse_transform(np.array(lower_values).reshape(-1, 1))
                upper_scaled_back = self.scaler.inverse_transform(np.array(upper_values).reshape(-1, 1))
                
                forecast_df[f'LSTM-lo-{conf_level}'] = lower_scaled_back.flatten()
                forecast_df[f'LSTM-hi-{conf_level}'] = upper_scaled_back.flatten()
        
        self.forecast_df = forecast_df
        
        print(f"Forecast generated with {len(self.forecast_df)} rows")
        self.logger.info(f"Forecast generated with {len(self.forecast_df)} rows")
        
        return self.forecast_df
    
    def evaluate(self, h=7, step_size=7, n_windows=5, metrics=None):
        """
        Evaluate the model using time series cross-validation.
        
        Parameters:
        -----------
        h : int, optional
            Forecast horizon (number of days to forecast). Default is 7.
        step_size : int, optional
            Step size between each window. Default is 7.
        n_windows : int, optional
            Number of windows for cross-validation. Default is 5.
        metrics : list, optional
            List of metrics to calculate. If None, uses MAE, MAPE, RMSE, and SMAPE.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing evaluation metrics for the model.
        """
        if not self.fitted:
            error_msg = "Model has not been fitted yet. Call fit() first."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data is None:
            error_msg = "No data available for evaluation."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Performing cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}...")
        self.logger.info(f"Performing cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}")
        
        # Sort data chronologically
        sorted_data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Initialize results dataframe
        result = pd.DataFrame(columns=['unique_id', 'metric', 'value'])
        
        # Get unique identifier
        unique_id = sorted_data['unique_id'].iloc[0]
        
        # Prepare data for evaluation
        # We need at least seq_length + h + step_size * (n_windows - 1) data points
        min_required_points = self.seq_length + h + step_size * (n_windows - 1)
        
        if len(sorted_data) < min_required_points:
            error_msg = f"Not enough data for cross-validation. Need at least {min_required_points} points, but have {len(sorted_data)}."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Define cutoff indices
        cutoffs = []
        for i in range(n_windows):
            cutoff_idx = len(sorted_data) - h - step_size * (n_windows - i - 1)
            cutoffs.append(cutoff_idx)
        
        # Collect actual and predicted values
        all_actuals = []
        all_predictions = []
        
        # Use scaled data for model training
        scaled_data = self.scaler.transform(sorted_data[['tavg']].values)
        
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        for i, cutoff_idx in enumerate(cutoffs):
            print(f"Cross-validation window {i+1}/{n_windows}")
            self.logger.info(f"Cross-validation window {i+1}/{n_windows}")
            
            # Split data at cutoff
            train_scaled = scaled_data[:cutoff_idx]
            test_scaled = scaled_data[cutoff_idx:cutoff_idx+h]
            
            # Prepare train sequences
            X_train, y_train = self._create_sequences(train_scaled, self.seq_length)
            
            # Create and train a new model for this window
            # Setting seeds for reproducibility
            np.random.seed(42 + i)
            tf.random.set_seed(42 + i)
            
            model = self._build_model(self.seq_length, self.n_features)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Fit the model
            model.fit(
                X_train, y_train,
                epochs=50,  # Reduced for speed
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make predictions
            # Get the last sequence from training data
            last_sequence = train_scaled[-self.seq_length:].reshape(1, self.seq_length, self.n_features)
            
            # Predict recursively
            predictions = []
            current_sequence = last_sequence.copy()
            
            for j in range(h):
                next_pred = model.predict(current_sequence, verbose=0)[0][0]
                predictions.append(next_pred)
                
                # Update the sequence for the next prediction
                current_sequence = np.append(
                    current_sequence[:, 1:, :],
                    [[[next_pred]]],
                    axis=1
                )
            
            # Compare with actual test values
            actuals = test_scaled.flatten()[:len(predictions)]
            
            all_actuals.extend(actuals)
            all_predictions.extend(predictions)
        
        # Convert back to original scale for metric calculation
        all_actuals_original = self.scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1)).flatten()
        all_predictions_original = self.scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(all_actuals_original, all_predictions_original)
        rmse = np.sqrt(mean_squared_error(all_actuals_original, all_predictions_original))
        
        # MAPE (handle zeros)
        non_zero = all_actuals_original != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((all_actuals_original[non_zero] - all_predictions_original[non_zero]) / all_actuals_original[non_zero])) * 100
        else:
            mape = np.nan
        
        # SMAPE
        smape = np.mean(200.0 * np.abs(all_predictions_original - all_actuals_original) / (np.abs(all_predictions_original) + np.abs(all_actuals_original) + 1e-8))
        
        # Add to results dataframe
        metrics_list = [
            {'metric': 'mae', 'value': mae},
            {'metric': 'rmse', 'value': rmse},
            {'metric': 'mape', 'value': mape},
            {'metric': 'smape', 'value': smape}
        ]
        
        for metric_item in metrics_list:
            result = pd.concat([
                result,
                pd.DataFrame({
                    'unique_id': [unique_id],
                    'metric': [metric_item['metric']],
                    'value': [metric_item['value']]
                })
            ])
        
        print("Evaluation complete")
        self.logger.info("Evaluation complete")
        
        return result
    
    def plot_forecast(self, test_df=None, figsize=(12, 6)):
        """
        Plot the forecast along with the historical data.
        
        Parameters:
        -----------
        test_df : pandas.DataFrame, optional
            Test data to overlay on the plot.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The forecast plot.
        """
        if not self.fitted or self.forecast_df is None:
            error_msg = "Model has not been fitted or forecast has not been generated."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from matplotlib.dates import DateFormatter
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Creating forecast plot{region_suffix}...")
        self.logger.info(f"Creating forecast plot{region_suffix}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure data is sorted by date
        sorted_data = self.data.sort_values('ds').reset_index(drop=True)
        sorted_forecast = self.forecast_df.sort_values('ds').reset_index(drop=True)
        
        # Get the last data point from historical data
        last_historical_date = sorted_data['ds'].max()
        last_historical_value = sorted_data[sorted_data['ds'] == last_historical_date]['tavg'].values[0]
        
        # Get the first forecast point
        first_forecast_date = sorted_forecast['ds'].min()
        first_forecast_value = sorted_forecast[sorted_forecast['ds'] == first_forecast_date]['LSTM'].values[0]
        
        print(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
        print(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
        print(f"Time difference: {first_forecast_date - last_historical_date}")
        self.logger.info(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
        self.logger.info(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
        self.logger.info(f"Time difference: {first_forecast_date - last_historical_date}")
        
        # Plot historical data
        ax.plot(sorted_data['ds'], sorted_data['tavg'], label='Historical Data')
        
        # Create continuous data for plotting by adding the last historical point
        bridge_point = pd.DataFrame({
            'unique_id': [sorted_forecast['unique_id'].iloc[0]],
            'ds': [last_historical_date],
            'LSTM': [last_historical_value]
        })
        
        # Add confidence interval columns if they exist
        for level in [80, 95]:
            lower_col = f'LSTM-lo-{level}'
            upper_col = f'LSTM-hi-{level}'
            
            if lower_col in sorted_forecast.columns and upper_col in sorted_forecast.columns:
                bridge_point[lower_col] = last_historical_value
                bridge_point[upper_col] = last_historical_value
        
        # Concatenate the bridge point with the forecast data
        continuous_forecast_df = pd.concat([bridge_point, sorted_forecast])
        continuous_forecast_df = continuous_forecast_df.sort_values('ds').reset_index(drop=True)
        
        # Plot forecast
        ax.plot(continuous_forecast_df['ds'], continuous_forecast_df['LSTM'], label='Forecast', color='red')
        
        # Plot confidence intervals if available
        for level in [80, 95]:
            lower_col = f'LSTM-lo-{level}'
            upper_col = f'LSTM-hi-{level}'
            
            if lower_col in continuous_forecast_df.columns and upper_col in continuous_forecast_df.columns:
                print(f"Adding {level}% confidence interval to plot...")
                self.logger.info(f"Adding {level}% confidence interval to plot")
                ax.fill_between(
                    continuous_forecast_df['ds'],
                    continuous_forecast_df[lower_col],
                    continuous_forecast_df[upper_col],
                    alpha=0.2,
                    label=f'{level}% Confidence Interval'
                )
        
        # Plot test data if provided
        if test_df is not None:
            print("Overlaying test data on plot...")
            self.logger.info("Overlaying test data on plot")
            ax.plot(test_df['ds'], test_df['tavg'], label='Actual Values', linestyle='--', color='green')
        
        # Format x-axis dates
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Add labels and legend
        title = f'LSTM Forecast{region_suffix}'
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.tight_layout()
        
        print("Forecast plot created successfully")
        self.logger.info("Forecast plot created successfully")
        
        return fig
    
    def get_results(self):
        """
        Get the forecast results.
        
        Returns:
        --------
        dict
            Dictionary containing the model results, including:
            - data: original data
            - forecast: forecast dataframe
            - freq: detected frequency
            - season_length: seasonal period
            - region_name: name of the region
        """
        if not self.fitted:
            error_msg = "Model has not been fitted yet. Call fit() first."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        results = {
            'data': self.data,
            'forecast': self.forecast_df,
            'freq': self.freq,
            'season_length': self.season_length,
            'region_name': self.region_name
        }
        
        print(f"Returning model results for {self.region_name if self.region_name else 'default'}")
        self.logger.info(f"Returning model results for {self.region_name if self.region_name else 'default'}")
        
        return results
    
    def get_model_summary(self):
        """
        Get a summary of the model configuration and status.
        
        Returns:
        --------
        dict
            Dictionary containing model configuration and status.
        """
        summary = {
            'region_name': self.region_name,
            'fitted': self.fitted,
            'frequency': self.freq,
            'season_length': self.season_length,
            'seq_length': self.seq_length,
            'data_points': len(self.data) if self.data is not None else 0,
            'forecast_generated': self.forecast_df is not None,
            'forecast_periods': len(self.forecast_df) if self.forecast_df is not None else 0
        }
        
        print(f"Model summary: {summary}")
        self.logger.info(f"Model summary: {summary}")
        
        return summary
