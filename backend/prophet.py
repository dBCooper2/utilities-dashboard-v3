zclass ProphetForecast:
    def __init__(self, region_name=None, data=None, season_length=None, log_level='INFO'):
        """
        Initialize the Prophet forecasting model.
        
        Parameters:
        -----------
        region_name : str, optional
            The name of the region being forecasted.
        data : pandas.DataFrame, optional
            DataFrame containing the time series data with columns 'ds', 'y', and 'unique_id'.
        season_length : int, optional
            The seasonal period of the time series. If None, it will be inferred from the data.
        log_level : str, optional
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        import logging
        import sys
        
        # Setup logging
        self.logger = logging.getLogger(f"ProphetForecast_{region_name if region_name else 'default'}")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Create handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.model = None
        self.season_length = season_length
        self.data = None
        self.freq = None
        self.forecast_df = None
        self.fitted = False
        self.region_name = region_name
        
        print(f"Initializing Prophet model for region: {region_name if region_name else 'default'}")
        self.logger.info(f"Initializing Prophet model for region: {region_name if region_name else 'default'}")
        
        # Process data if provided
        if data is not None:
            print(f"Processing initial dataframe with {len(data)} rows")
            self.logger.info(f"Processing initial dataframe with {len(data)} rows")
            self.fit(data)
    
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
            The detected frequency: '15min', 'H', or 'D'.
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
        
        # Determine the frequency based on the most common difference
        if most_common_diff <= pd.Timedelta(minutes=15):
            freq = '15min'
        elif most_common_diff <= pd.Timedelta(hours=1):
            freq = 'H'
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
        
        if freq == '15min':
            season_length = 96  # 24 hours * 4 (15-min intervals)
        elif freq == 'H':
            season_length = 24  # 24 hours
        else:  # 'D'
            season_length = 7   # 7 days
            
        print(f"Determined season length: {season_length}")
        self.logger.info(f"Determined season length: {season_length}")
        return season_length
    
    def fit(self, df=None):
        """
        Fit the Prophet model to the provided dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame containing the time series data with columns 'ds', 'y', and 'unique_id'.
            If None, uses the data stored in self.data.
            
        Returns:
        --------
        self : ProphetForecast
            The fitted model instance.
        """
        import pandas as pd
        from prophet import Prophet
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        
        # Use provided data or stored data
        if df is not None:
            self.data = df.copy()
        
        if self.data is None:
            error_msg = "No data available for fitting. Either provide data or initialize with data."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        print(f"Fitting Prophet model{region_suffix} with {len(self.data)} data points...")
        self.logger.info(f"Fitting Prophet model{region_suffix} with {len(self.data)} data points")
        
        # Sort data by date to ensure proper time ordering
        self.data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Detect frequency
        self.freq = self._detect_frequency(self.data)
        
        # Set season length if not provided
        if self.season_length is None:
            self.season_length = self._detect_season_length(self.freq)
        
        # Prophet requires only 'ds' and 'y' columns
        # For multiple time series (multiple unique_ids), we'll fit separate models
        # For now, we'll use the first unique_id if there are multiple
        unique_ids = self.data['unique_id'].unique()
        
        if len(unique_ids) > 1:
            print(f"Multiple unique_ids detected ({len(unique_ids)}). Using first id: {unique_ids[0]}")
            self.logger.warning(f"Multiple unique_ids detected ({len(unique_ids)}). Using first id: {unique_ids[0]}")
            
            # Filter data for the first unique_id
            self.data = self.data[self.data['unique_id'] == unique_ids[0]].copy()
        
        # Prepare data for Prophet
        prophet_df = self.data[['ds', 'y']].copy()
        
        # Configure the Prophet model
        print("Configuring Prophet model...")
        self.logger.info("Configuring Prophet model")
        
        # Initialize model with appropriate seasonality based on frequency
        if self.freq == '15min' or self.freq == 'H':
            # Sub-daily data: use hourly seasonality instead of daily
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            # Add hourly seasonality
            if self.freq == '15min':
                self.model.add_seasonality(
                    name='hourly',
                    period=24/24,  # 1 hour
                    fourier_order=5
                )
        else:
            # Daily data: use standard seasonalities
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
        
        # Fit the model
        print("Fitting model to data...")
        self.logger.info("Fitting model to data")
        self.model.fit(prophet_df)
        self.fitted = True
        
        print("Model fitting complete!")
        self.logger.info("Model fitting complete")
        
        return self
    
    def forecast(self, h=96, level=None):
        """
        Generate forecasts for the next h periods (default: 24 hours).
        
        Parameters:
        -----------
        h : int, optional
            The forecast horizon. Default is 96 (24 hours in 15-min intervals).
        level : list, optional
            Confidence levels for prediction intervals. In Prophet, uncertainty intervals are always provided.
            
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
        # Adjust h based on frequency to always forecast 24 hours
        original_h = h
        if self.freq == 'H':
            h = 24
        elif self.freq == 'D':
            h = 1
            
        print(f"Generating forecast{region_suffix} for next {h} periods (requested: {original_h}, adjusted based on freq: {self.freq})...")
        self.logger.info(f"Generating forecast{region_suffix} for next {h} periods (freq: {self.freq})")
        
        # Generate future dates dataframe
        print("Creating future dates dataframe...")
        self.logger.info("Creating future dates dataframe")
        future = self.model.make_future_dataframe(
            periods=h,
            freq=self.freq
        )
        
        # Generate forecast
        print("Generating forecast...")
        self.logger.info("Generating forecast")
        prophet_forecast = self.model.predict(future)
        
        # Get the unique_id from the original data
        unique_id = self.data['unique_id'].iloc[0]
        
        # Prepare forecast dataframe in the same format as StatsForecast
        forecast_df = pd.DataFrame({
            'unique_id': [unique_id] * len(prophet_forecast),
            'ds': prophet_forecast['ds'],
            'Prophet': prophet_forecast['yhat'],
            'Prophet-lo-80': prophet_forecast['yhat_lower'],
            'Prophet-hi-80': prophet_forecast['yhat_upper']
        })
        
        # Filter to only include future dates
        last_historical_date = self.data['ds'].max()
        forecast_only = forecast_df[forecast_df['ds'] > last_historical_date].copy()
        
        print(f"Forecast generated with {len(forecast_only)} future points")
        self.logger.info(f"Forecast generated with {len(forecast_only)} future points")
        
        # Store the forecast
        self.forecast_df = forecast_only
        
        # Also store the full forecast including historical fit
        self.full_forecast_df = forecast_df
        
        # Store the Prophet components for later plotting
        self.forecast_components = prophet_forecast
        
        return self.forecast_df
    
    def evaluate(self, h=12, step_size=12, n_windows=5, metrics=None):
        """
        Evaluate the model using time series cross-validation (following Prophet's cross-validation approach).
        
        Parameters:
        -----------
        h : int, optional
            Forecast horizon (number of periods to forecast). Default is 12.
        step_size : int, optional
            Step size between each window. How often to run the forecasting process. Default is 12.
        n_windows : int, optional
            Number of windows for cross-validation. How many forecasting processes in the past to evaluate. Default is 5.
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
        from prophet.diagnostics import cross_validation, performance_metrics
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Performing cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}...")
        self.logger.info(f"Performing cross-validation{region_suffix} with h={h}, step_size={step_size}, n_windows={n_windows}")
        
        # Determine the horizon in the appropriate unit based on frequency
        if self.freq == '15min':
            horizon = f"{h * 15} minutes"
        elif self.freq == 'H':
            horizon = f"{h} hours"
        else:  # 'D'
            horizon = f"{h} days"
        
        # Determine the period for cross-validation
        period = None
        if self.freq == '15min':
            period = f"{step_size * 15} minutes"
        elif self.freq == 'H':
            period = f"{step_size} hours"
        else:  # 'D'
            period = f"{step_size} days"
        
        # Get the number of data points
        n_data = len(self.data)
        
        # Determine initial train size to ensure we have n_windows
        initial = n_data - (n_windows * step_size) - h
        
        if initial <= 0:
            error_msg = f"Not enough data for cross-validation with given parameters. Need at least {(n_windows * step_size) + h + 1} points, but have {n_data}."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert initial to a string format
        if self.freq == '15min':
            initial = f"{initial * 15} minutes"
        elif self.freq == 'H':
            initial = f"{initial} hours"
        else:  # 'D'
            initial = f"{initial} days"
        
        try:
            # Perform cross-validation
            print("Executing Prophet cross-validation...")
            self.logger.info("Executing Prophet cross-validation")
            
            cv_results = cross_validation(
                model=self.model,
                horizon=horizon,
                period=period,
                initial=initial,
                parallel="processes"
            )
            
            # Calculate performance metrics
            print("Computing evaluation metrics...")
            self.logger.info("Computing evaluation metrics")
            
            # Default metrics for Prophet
            if metrics is None:
                metrics = ['mae', 'mape', 'mse', 'rmse']
            
            performance = performance_metrics(cv_results, metrics=metrics)
            
            # Convert to format consistent with AutoARIMA
            unique_id = self.data['unique_id'].iloc[0]
            result = pd.DataFrame()
            
            for metric in metrics:
                # Get the metric value from performance
                value = performance[metric].iloc[0]
                
                # Add to results
                metrics_row = pd.DataFrame({
                    'unique_id': [unique_id],
                    'metric': [metric],
                    'value': [value]
                })
                
                result = pd.concat([result, metrics_row])
            
            print("Evaluation complete")
            self.logger.info("Evaluation complete")
            
            return result
            
        except Exception as e:
            print(f"Error during cross-validation: {str(e)}")
            self.logger.error(f"Error during cross-validation: {str(e)}")
            
            # Fall back to manual approach
            print("Falling back to manual cross-validation...")
            self.logger.warning("Falling back to manual cross-validation")
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            # Initialize results dataframe
            result = pd.DataFrame(columns=['unique_id', 'metric', 'value'])
            
            # Sort data chronologically
            sorted_data = self.data.sort_values('ds').reset_index(drop=True)
            
            # Convert step_size and h to number of data points
            cutoffs = []
            for i in range(n_windows):
                cutoff_idx = len(sorted_data) - h - (n_windows - i - 1) * step_size
                if cutoff_idx <= 0:
                    continue
                cutoffs.append(sorted_data.iloc[cutoff_idx - 1]['ds'])
            
            if len(cutoffs) == 0:
                error_msg = "No valid cutoff points for cross-validation"
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # For each cutoff, fit model and make predictions
            all_actuals = []
            all_predictions = []
            
            from prophet import Prophet
            
            for cutoff in cutoffs:
                # Split data at cutoff
                train_df = sorted_data[sorted_data['ds'] <= cutoff][['ds', 'y']]
                test_df = sorted_data[sorted_data['ds'] > cutoff][['ds', 'y']]
                
                if len(test_df) == 0 or len(train_df) < 2:
                    continue
                
                # Fit Prophet model
                model = Prophet(
                    daily_seasonality=True if self.freq in ['15min', 'H'] else False,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )
                
                model.fit(train_df)
                
                # Make predictions
                future = model.make_future_dataframe(periods=len(test_df), freq=self.freq)
                forecast = model.predict(future)
                
                # Extract predictions for test period
                predictions = forecast[forecast['ds'].isin(test_df['ds'])]['yhat'].values
                actuals = test_df['y'].values
                
                all_actuals.extend(actuals)
                all_predictions.extend(predictions)
            
            # Calculate metrics
            if len(all_actuals) > 0 and len(all_predictions) > 0:
                # Convert to numpy arrays
                all_actuals = np.array(all_actuals)
                all_predictions = np.array(all_predictions)
                
                # MAE
                mae = mean_absolute_error(all_actuals, all_predictions)
                
                # RMSE
                rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
                
                # MAPE
                non_zero = all_actuals != 0
                if np.any(non_zero):
                    mape = np.mean(np.abs((all_actuals[non_zero] - all_predictions[non_zero]) / all_actuals[non_zero])) * 100
                else:
                    mape = np.nan
                
                # SMAPE
                smape = np.mean(200.0 * np.abs(all_predictions - all_actuals) / (np.abs(all_predictions) + np.abs(all_actuals) + 1e-8))
                
                # Add to results
                unique_id = sorted_data['unique_id'].iloc[0]
                metrics_list = [
                    {'metric': 'mae', 'value': mae},
                    {'metric': 'rmse', 'value': rmse},
                    {'metric': 'mape', 'value': mape},
                    {'metric': 'smape', 'value': smape}
                ]
                
                for metric in metrics_list:
                    result = pd.concat([
                        result,
                        pd.DataFrame({
                            'unique_id': [unique_id],
                            'metric': [metric['metric']],
                            'value': [metric['value']]
                        })
                    ])
                
                print("Manual cross-validation complete")
                self.logger.info("Manual cross-validation complete")
            else:
                print("Warning: No valid cross-validation results generated")
                self.logger.warning("No valid cross-validation results generated")
            
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
        
        # If we have the full forecast (including historical fit), use that
        if hasattr(self, 'full_forecast_df'):
            forecast_df = self.full_forecast_df.sort_values('ds').reset_index(drop=True)
        else:
            # Otherwise, use only future forecast and merge with historical
            forecast_df = self.forecast_df.sort_values('ds').reset_index(drop=True)
        
        # Get the last data point and first forecast point
        last_historical_date = sorted_data['ds'].max()
        last_historical_value = sorted_data[sorted_data['ds'] == last_historical_date]['y'].values[0]
        
        first_forecast_date = forecast_df[forecast_df['ds'] > last_historical_date]['ds'].min()
        if pd.notna(first_forecast_date):
            first_forecast_value = forecast_df[forecast_df['ds'] == first_forecast_date]['Prophet'].values[0]
            
            print(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
            print(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
            print(f"Time difference: {first_forecast_date - last_historical_date}")
            self.logger.info(f"Last historical date: {last_historical_date}, value: {last_historical_value}")
            self.logger.info(f"First forecast date: {first_forecast_date}, value: {first_forecast_value}")
            self.logger.info(f"Time difference: {first_forecast_date - last_historical_date}")
        
        # Plot historical data
        ax.plot(sorted_data['ds'], sorted_data['y'], label='Historical Data')
        
        # Get forecast data for the future only
        future_forecast = forecast_df[forecast_df['ds'] > last_historical_date]
        
        # Check if we have a gap between historical and forecast
        if len(future_forecast) > 0:
            # Check for a significant gap
            expected_interval = None
            if self.freq == '15min':
                expected_interval = pd.Timedelta(minutes=15)
            elif self.freq == 'H':
                expected_interval = pd.Timedelta(hours=1)
            else:  # 'D'
                expected_interval = pd.Timedelta(days=1)
            
            actual_interval = first_forecast_date - last_historical_date
            
            # Create continuous data for plotting
            continuous_forecast_df = future_forecast.copy()
            
            # If there's a large gap, create interpolation points
            if actual_interval > expected_interval * 2:
                print(f"Large gap detected ({actual_interval}), creating interpolation points for smoother transition")
                self.logger.info(f"Large gap detected ({actual_interval}), creating interpolation points for smoother transition")
                
                # Create interpolation points
                num_points = max(3, int(actual_interval / expected_interval) - 1)
                interp_dates = pd.date_range(
                    start=last_historical_date + expected_interval,
                    end=first_forecast_date - expected_interval,
                    periods=num_points
                )
                
                # Linear interpolation between last historical and first forecast values
                interp_values = np.linspace(last_historical_value, first_forecast_value, num_points + 2)[1:-1]
                
                # Create DataFrame with interpolated points
                interp_df = pd.DataFrame({
                    'unique_id': [continuous_forecast_df['unique_id'].iloc[0]] * len(interp_dates),
                    'ds': interp_dates,
                    'Prophet': interp_values
                })
                
                # Add confidence interval columns if they exist
                if 'Prophet-lo-80' in continuous_forecast_df.columns and 'Prophet-hi-80' in continuous_forecast_df.columns:
                    first_lower = continuous_forecast_df['Prophet-lo-80'].iloc[0]
                    first_upper = continuous_forecast_df['Prophet-hi-80'].iloc[0]
                    
                    # Create widening confidence intervals
                    lower_values = np.linspace(last_historical_value, first_lower, num_points + 2)[1:-1]
                    upper_values = np.linspace(last_historical_value, first_upper, num_points + 2)[1:-1]
                    
                    interp_df['Prophet-lo-80'] = lower_values
                    interp_df['Prophet-hi-80'] = upper_values
                
                # Add interpolation points to forecast dataframe
                continuous_forecast_df = pd.concat([interp_df, continuous_forecast_df])
                continuous_forecast_df = continuous_forecast_df.sort_values('ds').reset_index(drop=True)
                
                print(f"Added {len(interp_dates)} interpolation points to create smooth transition")
                self.logger.info(f"Added {len(interp_dates)} interpolation points to create smooth transition")
            
            # Add the last historical point to create a continuous line
            bridge_point = pd.DataFrame({
                'unique_id': [continuous_forecast_df['unique_id'].iloc[0]],
                'ds': [last_historical_date],
                'Prophet': [last_historical_value]
            })
            
            # Add confidence intervals if they exist
            if 'Prophet-lo-80' in continuous_forecast_df.columns and 'Prophet-hi-80' in continuous_forecast_df.columns:
                bridge_point['Prophet-lo-80'] = last_historical_value
                bridge_point['Prophet-hi-80'] = last_historical_value
            
            # Concatenate the bridge point with the forecast data
            continuous_forecast_df = pd.concat([bridge_point, continuous_forecast_df])
            continuous_forecast_df = continuous_forecast_df.sort_values('ds').reset_index(drop=True)
            
            print(f"Added bridge point to create continuous forecast line")
            self.logger.info(f"Added bridge point to create continuous forecast line")
            
            # Plot forecast with the bridge and interpolation points
            ax.plot(continuous_forecast_df['ds'], continuous_forecast_df['Prophet'], label='Forecast', color='red')
            
            # Plot confidence intervals if available
            if 'Prophet-lo-80' in continuous_forecast_df.columns and 'Prophet-hi-80' in continuous_forecast_df.columns:
                print(f"Adding 80% confidence interval to plot...")
                self.logger.info(f"Adding 80% confidence interval to plot")
                ax.fill_between(
                    continuous_forecast_df['ds'],
                    continuous_forecast_df['Prophet-lo-80'],
                    continuous_forecast_df['Prophet-hi-80'],
                    alpha=0.2,
                    label='80% Confidence Interval'
                )
        else:
            # If no future forecast available, plot the entire forecast
            historical_forecast = forecast_df[forecast_df['ds'] <= last_historical_date]
            ax.plot(historical_forecast['ds'], historical_forecast['Prophet'], label='Fitted Values', color='green', linestyle='--')
        
        # Plot test data if provided
        if test_df is not None:
            print("Overlaying test data on plot...")
            self.logger.info("Overlaying test data on plot")
            ax.plot(test_df['ds'], test_df['y'], label='Actual Values', linestyle='--', color='purple')
        
        # Format x-axis dates
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        # Add labels and legend
        title = f'Prophet Forecast{region_suffix}'
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        plt.tight_layout()
        
        print("Forecast plot created successfully")
        self.logger.info("Forecast plot created successfully")
        
        return fig
    
    def plot_components(self, figsize=(12, 10)):
        """
        Plot the forecast components (trend, seasonality, etc.) from Prophet.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The components plot.
        """
        if not self.fitted or not hasattr(self, 'forecast_components'):
            error_msg = "Model has not been fitted or forecast has not been generated."
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Creating forecast components plot{region_suffix}...")
        self.logger.info(f"Creating forecast components plot{region_suffix}")
        
        # Use Prophet's built-in component plotting
        fig = self.model.plot_components(self.forecast_components, figsize=figsize)
        
        print("Components plot created successfully")
        self.logger.info("Components plot created successfully")
        
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
            'data_points': len(self.data) if self.data is not None else 0,
            'forecast_generated': self.forecast_df is not None,
            'forecast_periods': len(self.forecast_df) if self.forecast_df is not None else 0
        }
        
        print(f"Model summary: {summary}")
        self.logger.info(f"Model summary: {summary}")
        
        return summary