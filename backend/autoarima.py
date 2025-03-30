class AutoARIMAForecast:
    def __init__(self, region_name=None, initial_df=None, season_length=None, log_level='INFO'):
        """
        Initialize the AutoARIMA forecasting model.
        
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
        
        # Setup logging
        self.logger = logging.getLogger(f"AutoARIMAForecast_{region_name if region_name else 'default'}")
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
        
        print(f"Initializing AutoARIMA model for region: {region_name if region_name else 'default'}")
        self.logger.info(f"Initializing AutoARIMA model for region: {region_name if region_name else 'default'}")
        
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
    
    def fit(self, df):
        """
        Fit the AutoARIMA model to the provided dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the time series data with columns 'ds', 'y', and 'unique_id'.
            
        Returns:
        --------
        self : AutoARIMAForecast
            The fitted model instance.
        """
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        print(f"Fitting AutoARIMA model{region_suffix} with {len(df)} data points...")
        self.logger.info(f"Fitting AutoARIMA model{region_suffix} with {len(df)} data points")
        
        # Store the original data
        self.data = df.copy()
        
        # Sort data by date to ensure proper time ordering
        self.data = self.data.sort_values('ds').reset_index(drop=True)
        
        # Detect frequency
        self.freq = self._detect_frequency(df)
        
        # Set season length if not provided
        if self.season_length is None:
            self.season_length = self._detect_season_length(self.freq)
        
        # Initialize the AutoARIMA model
        print(f"Creating AutoARIMA model with season_length={self.season_length}")
        self.logger.info(f"Creating AutoARIMA model with season_length={self.season_length}")
        models = [AutoARIMA(season_length=self.season_length)]
        
        # Create and fit the StatsForecast model
        print("Creating StatsForecast model...")
        self.logger.info("Creating StatsForecast model")
        self.model = StatsForecast(
            models=models,
            freq=self.freq,
            n_jobs=-1  # Use all available cores
        )
        
        # Fit the model
        print("Fitting model to data...")
        self.logger.info("Fitting model to data")
        self.model.fit(self.data)
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
            Confidence levels for prediction intervals.
            
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
        
        region_suffix = f" for region {self.region_name}" if self.region_name else ""
        # Adjust h based on frequency to always forecast 24 hours
        original_h = h
        if self.freq == 'H':
            h = 24
        elif self.freq == 'D':
            h = 1
            
        print(f"Generating forecast{region_suffix} for next {h} periods (requested: {original_h}, adjusted based on freq: {self.freq})...")
        self.logger.info(f"Generating forecast{region_suffix} for next {h} periods (freq: {self.freq})")
        
        # Create a forecast df with just the unique_id and ds columns
        import pandas as pd
        import numpy as np
        
        # Get unique identifiers from the training data
        unique_ids = self.data['unique_id'].unique()
        
        # Generate future dates based on the last date in the training data
        # Sort data to ensure we get the actual last date
        sorted_data = self.data.sort_values('ds')
        last_date = pd.to_datetime(sorted_data['ds']).max()
        print(f"Last historical date: {last_date}")
        self.logger.info(f"Last historical date: {last_date}")
        
        # Determine the next timestamp directly based on the frequency using explicit timedelta
        if self.freq == '15min':
            next_timestamp = last_date + pd.Timedelta(minutes=15)
        elif self.freq == 'H':
            next_timestamp = last_date + pd.Timedelta(hours=1)
        else:  # 'D'
            next_timestamp = last_date + pd.Timedelta(days=1)
        
        print(f"Next forecast timestamp: {next_timestamp}")
        self.logger.info(f"Next forecast timestamp: {next_timestamp}")
        
        # Create a dataframe with future dates for forecasting
        # Use explicit parameters to avoid any ambiguity
        future_dates = pd.date_range(
            start=next_timestamp,
            periods=h,
            freq=self.freq
        )
        
        # Verify the generated dates - this is important for debugging
        print(f"Generated future dates from {future_dates[0]} to {future_dates[-1]}")
        print(f"Time difference between last historical and first forecast: {future_dates[0] - last_date}")
        self.logger.info(f"Generated future dates from {future_dates[0]} to {future_dates[-1]}")
        self.logger.info(f"Time difference between last historical and first forecast: {future_dates[0] - last_date}")
        
        # Store expected dates for later verification and correction
        expected_first_date = future_dates[0]
        expected_dates = future_dates
        
        # Create forecast dataframe with all combinations of unique_ids and future dates
        forecast_input_df = pd.DataFrame()
        for unique_id in unique_ids:
            temp_df = pd.DataFrame({
                'unique_id': [unique_id] * len(future_dates),
                'ds': future_dates,
                'y': [np.nan] * len(future_dates)  # Add y column with NaN values
            })
            forecast_input_df = pd.concat([forecast_input_df, temp_df])
        
        # Ensure proper sorting of the dataframe
        forecast_input_df = forecast_input_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        print(f"Created forecast input dataframe with {len(forecast_input_df)} rows")
        self.logger.info(f"Created forecast input dataframe with {len(forecast_input_df)} rows")
        
        # Generate forecasts
        try:
            if level is not None:
                print(f"Including confidence intervals at levels: {level}")
                self.logger.info(f"Including confidence intervals at levels: {level}")
                self.forecast_df = self.model.forecast(df=forecast_input_df, h=h, level=level)
            else:
                self.forecast_df = self.model.forecast(df=forecast_input_df, h=h)
            
            print(f"Forecast generated with {len(self.forecast_df)} rows")
            self.logger.info(f"Forecast generated with {len(self.forecast_df)} rows")
            
            # Verify the forecast dates and correct them if needed
            forecast_dates = pd.to_datetime(self.forecast_df['ds']).sort_values().unique()
            actual_first_date = forecast_dates[0]
            
            print(f"Actual forecast dates: {actual_first_date} to {forecast_dates[-1]}")
            self.logger.info(f"Actual forecast dates: {actual_first_date} to {forecast_dates[-1]}")
            
            # Check if the actual first forecast date is significantly different from what we expected
            # This is likely due to StatsForecast possibly adding an extra day
            time_diff = actual_first_date - expected_first_date
            
            if abs(time_diff.total_seconds()) > 3600:  # More than an hour difference
                print(f"WARNING: First forecast date ({actual_first_date}) doesn't match expected date ({expected_first_date})")
                print(f"Time difference: {time_diff}")
                self.logger.warning(f"First forecast date ({actual_first_date}) doesn't match expected date ({expected_first_date})")
                self.logger.warning(f"Time difference: {time_diff}")
                
                # Correct the forecast dates to match what we expected
                print("Correcting forecast dates to match expected timeline...")
                self.logger.info("Correcting forecast dates to match expected timeline")
                
                # Create a mapping from the incorrect dates to the expected dates
                date_mapping = {}
                for i, incorrect_date in enumerate(forecast_dates):
                    if i < len(expected_dates):
                        date_mapping[incorrect_date] = expected_dates[i]
                    else:
                        # If we have more forecast dates than expected (unlikely), extend with the same interval
                        last_mapped_date = date_mapping[forecast_dates[i-1]]
                        if self.freq == '15min':
                            date_mapping[incorrect_date] = last_mapped_date + pd.Timedelta(minutes=15)
                        elif self.freq == 'H':
                            date_mapping[incorrect_date] = last_mapped_date + pd.Timedelta(hours=1)
                        else:  # 'D'
                            date_mapping[incorrect_date] = last_mapped_date + pd.Timedelta(days=1)
                
                # Apply the date correction
                corrected_forecast = self.forecast_df.copy()
                corrected_forecast['ds'] = corrected_forecast['ds'].apply(
                    lambda x: date_mapping.get(pd.to_datetime(x), pd.to_datetime(x))
                )
                
                self.forecast_df = corrected_forecast
                
                # Verify the correction
                corrected_dates = pd.to_datetime(self.forecast_df['ds']).sort_values().unique()
                print(f"Corrected forecast dates: {corrected_dates[0]} to {corrected_dates[-1]}")
                self.logger.info(f"Corrected forecast dates: {corrected_dates[0]} to {corrected_dates[-1]}")
                print(f"Time difference between last historical and first corrected forecast: {corrected_dates[0] - last_date}")
                self.logger.info(f"Time difference between last historical and first corrected forecast: {corrected_dates[0] - last_date}")
            
        except Exception as e:
            error_msg = f"Error generating forecast: {str(e)}"
            print(f"Error: {error_msg}")
            self.logger.error(error_msg)
            
            # Try a different approach - use predict instead of forecast
            print("Attempting alternative forecasting approach...")
            self.logger.info("Attempting alternative forecasting approach")
            
            try:
                # Some versions of statsforecast might use predict instead of forecast
                if level is not None:
                    self.forecast_df = self.model.predict(h=h, level=level)
                else:
                    self.forecast_df = self.model.predict(h=h)
                
                print(f"Forecast generated with {len(self.forecast_df)} rows")
                self.logger.info(f"Forecast generated with {len(self.forecast_df)} rows")
                
                # Apply the same date correction for the predict method
                forecast_dates = pd.to_datetime(self.forecast_df['ds']).sort_values().unique()
                actual_first_date = forecast_dates[0]
                
                # If there's a significant time difference, correct the dates
                if abs((actual_first_date - expected_first_date).total_seconds()) > 3600:
                    print(f"WARNING: First forecast date from predict ({actual_first_date}) doesn't match expected ({expected_first_date})")
                    self.logger.warning(f"First forecast date from predict ({actual_first_date}) doesn't match expected ({expected_first_date})")
                    
                    # Generate the correct series of dates
                    corrected_dates = pd.date_range(
                        start=expected_first_date,
                        periods=len(forecast_dates),
                        freq=self.freq
                    )
                    
                    # Map old dates to new dates
                    date_mapping = dict(zip(sorted(forecast_dates), corrected_dates))
                    
                    # Apply correction
                    corrected_forecast = self.forecast_df.copy()
                    corrected_forecast['ds'] = corrected_forecast['ds'].apply(
                        lambda x: date_mapping.get(pd.to_datetime(x), pd.to_datetime(x))
                    )
                    
                    self.forecast_df = corrected_forecast
                    print("Corrected predict dates to match expected timeline")
                    self.logger.info("Corrected predict dates to match expected timeline")
            except Exception as e2:
                error_msg = f"Error with alternative forecasting approach: {str(e2)}"
                print(f"Error: {error_msg}")
                self.logger.error(error_msg)
                raise ValueError(f"Unable to generate forecast: {str(e)}, then {str(e2)}")
        
        return self.forecast_df