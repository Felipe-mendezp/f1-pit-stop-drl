import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
Optimized script to build F1 race DataFrames for seasons 2021-2024.

This script processes Formula 1 race data using FastF1 and creates optimized
DataFrames with interval calculations, tire compound details, and pit information.

Key optimizations:
- Vectorized operations instead of iterrows()
- Parallel processing of race sessions
- Efficient interval calculations using sorted data and numpy
- Progress tracking and error handling
"""

import fastf1 as ff1
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('fastf1').setLevel(logging.ERROR)
logging.basicConfig(level=logging.CRITICAL)


# Compound mappings for each year and GP
COMPOUND_MAPS = {
    # C1 -> C0
    2021: {
        'Bahrain Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Portuguese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'French Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Styrian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Hungarian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Italian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Russian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Turkish Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'São Paulo Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Qatar Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    },
    # C1 -> C0
    2022: {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'French Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Hungarian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'Italian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C0'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'São Paulo Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    },
    2023: {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Qatar Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'São Paulo Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    },
    2024: {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Chinese Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'São Paulo Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Qatar Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    }
}


def add_intervals_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently add interval columns using vectorized operations.

    Calculates:
    - Interval_front: Time gap to car ahead in same lap
    - Interval_behind: Time gap to car behind in same lap

    Args:
        df: DataFrame with 'LapNumber' and 'Time' columns

    Returns:
        DataFrame with added interval columns
    """
    # Sort by lap number and time for efficient processing
    df = df.sort_values(['LapNumber', 'Time']).reset_index(drop=True)

    # Initialize interval columns with default value (100 seconds)
    df['Interval_front'] = 100.0
    df['Interval_behind'] = 100.0

    # Convert Time to numeric for calculations (total seconds)
    df['Time_seconds'] = pd.to_timedelta(df['Time']).dt.total_seconds()

    # Calculate intervals within same lap using shift
    df['prev_time'] = df.groupby('LapNumber')['Time_seconds'].shift(1)
    df['next_time'] = df.groupby('LapNumber')['Time_seconds'].shift(-1)

    # Interval front: difference with previous car (in same lap)
    mask_front = df['prev_time'].notna()
    df.loc[mask_front, 'Interval_front'] = df.loc[mask_front, 'Time_seconds'] - df.loc[mask_front, 'prev_time']

    # Interval behind: difference with next car (in same lap)
    mask_behind = df['next_time'].notna()
    df.loc[mask_behind, 'Interval_behind'] = df.loc[mask_behind, 'next_time'] - df.loc[mask_behind, 'Time_seconds']

    # Clean up temporary columns
    df = df.drop(columns=['Time_seconds', 'prev_time', 'next_time'])

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame with selected columns and type conversions.

    Args:
        df: Raw FastF1 laps DataFrame

    Returns:
        Preprocessed DataFrame with converted time columns
    """
    # Select columns of interest
    columns = [
        'Driver', 'LapTime', 'LapNumber', 'PitOutTime', 'PitInTime',
        'Compound', 'TyreLife', 'FreshTyre', 'Team', 'TrackStatus',
        'Position', 'Interval_front', 'Interval_behind', 'GP'
    ]

    laps_filter = df[columns].copy()

    # Convert time columns to seconds (intervals are already in seconds from add_intervals_vectorized)
    lap_time_columns = ['LapTime', 'PitOutTime', 'PitInTime']

    for col in lap_time_columns:
        laps_filter[col] = pd.to_timedelta(laps_filter[col]).dt.total_seconds()

    # Fill NaN in pit times with 0
    laps_filter[['PitOutTime', 'PitInTime']] = laps_filter[['PitOutTime', 'PitInTime']].fillna(0)

    return laps_filter


def map_compound_detail(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Map generic compound names (SOFT/MEDIUM/HARD) to specific compounds (C1-C5).

    Args:
        df: DataFrame with 'GP' and 'Compound' columns
        year: Season year

    Returns:
        DataFrame with added 'Compound_Detail' column
    """
    compound_map = COMPOUND_MAPS.get(year, {})

    def get_compound_detail(row):
        gp = row['GP']
        compound = row['Compound']
        return compound_map.get(gp, {}).get(compound, None)

    df['Compound_Detail'] = df.apply(get_compound_detail, axis=1)
    return df


def create_pit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary PitIn and PitOut columns.

    Args:
        df: DataFrame with 'PitInTime' and 'PitOutTime' columns

    Returns:
        DataFrame with added 'PitIn' and 'PitOut' binary columns
    """
    df['PitIn'] = (df['PitInTime'] != 0).astype(int)
    df['PitOut'] = (df['PitOutTime'] != 0).astype(int)
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the required final columns.

    Args:
        df: Processed DataFrame

    Returns:
        DataFrame with only the specified columns
    """
    final_columns = [
        'LapTime', 'LapNumber', 'Interval_front', 'Interval_behind',
        'PitIn', 'PitOut', 'Driver', 'Compound', 'Compound_Detail', 'Year',
        'TyreLife', 'GP', 'Position', 'TrackStatus'
    ]

    return df[final_columns].copy()


def process_single_race(year: int, race_num: int) -> pd.DataFrame:
    """
    Process a single race session.

    Args:
        year: Season year
        race_num: Race number in the season

    Returns:
        Processed DataFrame for the race, or empty DataFrame if error
    """
    try:
        # Load session
        session = ff1.get_session(year, race_num, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        # Get laps data
        laps = session.laps.copy()

        # Add GP name
        laps['GP'] = session.event.EventName

        # Add intervals (optimized)
        laps = add_intervals_vectorized(laps)

        # Preprocess
        laps = preprocess(laps)

        # Add year
        laps['Year'] = year

        # Map compound details
        laps = map_compound_detail(laps, year)

        # Create pit columns
        laps = create_pit_columns(laps)

        # Select final columns
        laps = select_final_columns(laps)

        return laps

    except Exception as e:
        print(f"Error processing {year} race {race_num}: {str(e)}")
        return pd.DataFrame()


def process_season(year: int, max_races: int, use_parallel: bool = True) -> pd.DataFrame:
    """
    Process all races for a given season.

    Args:
        year: Season year
        max_races: Maximum number of races in the season
        use_parallel: Whether to use parallel processing

    Returns:
        Combined DataFrame for all races in the season
    """
    print(f"\nProcessing {year} season...")

    if use_parallel:
        # Parallel processing
        results = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_single_race, year, i): i
                for i in range(1, max_races + 1)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{year}"):
                try:
                    result = future.result()
                    if not result.empty:
                        results.append(result)
                except Exception as e:
                    race_num = futures[future]
                    print(f"Error in race {race_num}: {str(e)}")

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        # Sequential processing with progress bar
        season_data = []
        for i in tqdm(range(1, max_races + 1), desc=f"{year}"):
            result = process_single_race(year, i)
            if not result.empty:
                season_data.append(result)

        if season_data:
            return pd.concat(season_data, ignore_index=True)
        else:
            return pd.DataFrame()


def main():
    """
    Main function to process all seasons and save results.
    """
    # Enable FastF1 cache for faster repeated runs
    cache_dir = config.DATA_DIR / 'fastf1_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(cache_dir))

    # Define seasons and race counts
    seasons = {
        2021: 22,
        2022: 22,
        2023: 22,
        2024: 24,
    }

    # Output directory
    output_dir = config.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each season
    all_data = {}

    for year, num_races in seasons.items():
        # Process season (set use_parallel=False if you encounter issues)
        season_df = process_season(year, num_races, use_parallel=False)

        if not season_df.empty:
            # Save individual season file
            output_file = output_dir / f'session_{year}_optimized.csv'
            season_df.to_csv(output_file, index=False)
            print(f"Saved {year} data: {len(season_df)} laps to {output_file}")

            all_data[year] = season_df
        else:
            print(f"No data collected for {year}")

    # Optionally combine all seasons
    if all_data:
        combined_df = pd.concat(all_data.values(), ignore_index=True)
        combined_file = output_dir / 'all_seasons_2021_2024.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"\nSaved combined data: {len(combined_df)} total laps to {combined_file}")

        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total laps: {len(combined_df)}")
        print(f"\nLaps per year:")
        print(combined_df.groupby('Year').size())
        print(f"\nMissing values:")
        print(combined_df.isna().sum())


if __name__ == "__main__":
    main()
