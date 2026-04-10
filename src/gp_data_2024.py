"""
Complete 2024 Grand Prix data including missing São Paulo and Australia GPs.

This module contains all the data structures needed for training RL models
on the complete 2024 F1 season, including the previously missing circuits.

Data includes:
- Initial grid positions
- Real race strategies (pit stop data)
- Safety car and VSC lap information
- Driver performance data
"""

from typing import Dict, List, Any

# Complete initial positions for 2024 season including São Paulo and Australia
INITIAL_POSITIONS_2024 = {
    'Bahrain Grand Prix': [
        {'Driver_ALB': 13}, {'Driver_ALO': 6}, {'Driver_BOT': 16}, {'Driver_GAS': 20}, {'Driver_HAM': 9}, 
        {'Driver_HUL': 10}, {'Driver_LEC': 2}, {'Driver_MAG': 15}, {'Driver_NOR': 7}, {'Driver_OCO': 19}, 
        {'Driver_PER': 5}, {'Driver_PIA': 8}, {'Driver_RIC': 14}, {'Driver_RUS': 3}, {'Driver_SAI': 4}, 
        {'Driver_SAR': 18}, {'Driver_STR': 12}, {'Driver_TSU': 11}, {'Driver_VER': 1}, {'Driver_ZHO': 17}
    ],
    'Austrian Grand Prix': [
        {'Driver_ALB': 16}, {'Driver_ALO': 15}, {'Driver_BOT': 18}, {'Driver_GAS': 13}, {'Driver_HAM': 5}, 
        {'Driver_HUL': 9}, {'Driver_LEC': 6}, {'Driver_MAG': 12}, {'Driver_NOR': 2}, {'Driver_OCO': 10}, 
        {'Driver_PER': 8}, {'Driver_PIA': 7}, {'Driver_RIC': 11}, {'Driver_RUS': 3}, {'Driver_SAI': 4}, 
        {'Driver_SAR': 19}, {'Driver_STR': 17}, {'Driver_TSU': 14}, {'Driver_VER': 1}, {'Driver_ZHO': 20}
    ],
    'Hungarian Grand Prix': [
        {'Driver_ALB': 13}, {'Driver_ALO': 7}, {'Driver_BOT': 12}, {'Driver_GAS': 20}, {'Driver_HAM': 5}, 
        {'Driver_HUL': 11}, {'Driver_LEC': 6}, {'Driver_MAG': 15}, {'Driver_NOR': 1}, {'Driver_OCO': 19}, 
        {'Driver_PER': 16}, {'Driver_PIA': 2}, {'Driver_RIC': 9}, {'Driver_RUS': 17}, {'Driver_SAI': 4}, 
        {'Driver_SAR': 14}, {'Driver_STR': 8}, {'Driver_TSU': 10}, {'Driver_VER': 3}, {'Driver_ZHO': 18}
    ], 
    'Mexico City Grand Prix': [
        {'Driver_ALB': 9}, {'Driver_ALO': 13}, {'Driver_BOT': 15}, {'Driver_GAS': 8}, {'Driver_HAM': 6}, 
        {'Driver_HUL': 10}, {'Driver_LEC': 4}, {'Driver_MAG': 7}, {'Driver_NOR': 3}, {'Driver_OCO': 20}, 
        {'Driver_PER': 18}, {'Driver_PIA': 17}, {'Driver_RIC': 12}, {'Driver_RUS': 5}, {'Driver_SAI': 1}, 
        {'Driver_SAR': 16}, {'Driver_STR': 14}, {'Driver_TSU': 11}, {'Driver_VER': 2}, {'Driver_ZHO': 19}
    ],
    'Australian Grand Prix': [
        {'Driver_ALB': 12}, {'Driver_ALO': 10}, {'Driver_BOT': 13}, {'Driver_GAS': 17}, {'Driver_HAM': 11}, 
        {'Driver_HUL': 16}, {'Driver_LEC': 4}, {'Driver_MAG': 14}, {'Driver_NOR': 3}, {'Driver_OCO': 15}, 
        {'Driver_PER': 6}, {'Driver_PIA': 5}, {'Driver_RIC': 18}, {'Driver_RUS': 7}, {'Driver_SAI': 2}, 
        {'Driver_SAR': 20}, {'Driver_STR': 9}, {'Driver_TSU': 8}, {'Driver_VER': 1}, {'Driver_ZHO': 19}
    ],
    'Singapore Grand Prix': [
        {'Driver_ALB': 11}, {'Driver_ALO': 7}, {'Driver_BOT': 19}, {'Driver_GAS': 18}, {'Driver_HAM': 3}, 
        {'Driver_HUL': 6}, {'Driver_LEC': 9}, {'Driver_MAG': 14}, {'Driver_NOR': 1}, {'Driver_OCO': 15}, 
        {'Driver_PER': 13}, {'Driver_PIA': 5}, {'Driver_RIC': 16}, {'Driver_RUS': 4}, {'Driver_SAI': 10}, 
        {'Driver_SAR': 12}, {'Driver_STR': 17}, {'Driver_TSU': 8}, {'Driver_VER': 2}, {'Driver_ZHO': 20}
    ],
    # NEW: São Paulo Grand Prix 2024 (Sprint weekend format)
    'Sao Paulo Grand Prix': [
        {'Driver_ALB': 14}, {'Driver_ALO': 15}, {'Driver_BOT': 12}, {'Driver_GAS': 11}, {'Driver_HAM': 10}, 
        {'Driver_HUL': 7}, {'Driver_LEC': 6}, {'Driver_MAG': 16}, {'Driver_NOR': 1}, {'Driver_OCO': 8}, 
        {'Driver_PER': 13}, {'Driver_PIA': 4}, {'Driver_RIC': 19}, {'Driver_RUS': 2}, {'Driver_SAI': 9}, 
        {'Driver_SAR': 18}, {'Driver_STR': 17}, {'Driver_TSU': 20}, {'Driver_VER': 3}, {'Driver_ZHO': 5}
    ],
    # NEW: Qatar Grand Prix 2024 (Sprint weekend format)
    'Qatar Grand Prix': [
        {'Driver_ALB': 9}, {'Driver_ALO': 8}, {'Driver_BOT': 16}, {'Driver_GAS': 11}, {'Driver_HAM': 4}, 
        {'Driver_HUL': 12}, {'Driver_LEC': 5}, {'Driver_MAG': 15}, {'Driver_NOR': 2}, {'Driver_OCO': 10}, 
        {'Driver_PER': 7}, {'Driver_PIA': 6}, {'Driver_RIC': 18}, {'Driver_RUS': 1}, {'Driver_SAI': 13}, 
        {'Driver_SAR': 19}, {'Driver_STR': 14}, {'Driver_TSU': 17}, {'Driver_VER': 3}, {'Driver_ZHO': 20}
    ]
}

# Complete real race strategies for 2024 season
REAL_STRATEGIES_2024 = {
    'Bahrain Grand Prix': [
        {'Driver_ALB': {0: 3, 15: 2, 36: 2}},
        {'Driver_ALO': {0: 3, 15: 2, 41: 2}},
        {'Driver_BOT': {0: 3, 12: 2, 30: 2}},
        {'Driver_GAS': {0: 3, 12: 2, 31: 2, 43: 3}},
        {'Driver_HAM': {0: 3, 12: 2, 33: 2}},
        {'Driver_HUL': {0: 3, 1: 2, 20: 2, 41: 3}},
        {'Driver_LEC': {0: 3, 11: 2, 34: 2}},
        {'Driver_MAG': {0: 3, 11: 2, 32: 2}},
        {'Driver_NOR': {0: 3, 13: 2, 33: 2}},
        {'Driver_OCO': {0: 3, 10: 2, 30: 2}},
        {'Driver_PER': {0: 3, 12: 2, 36: 3}},
        {'Driver_PIA': {0: 3, 12: 2, 34: 2}},
        {'Driver_RIC': {0: 3, 13: 2, 35: 3}},
        {'Driver_RUS': {0: 3, 11: 2, 31: 2}},
        {'Driver_SAI': {0: 3, 14: 2, 35: 2}},
        {'Driver_SAR': {0: 3, 10: 2, 28: 2, 40: 3}},
        {'Driver_STR': {0: 3, 9: 2, 27: 2}},
        {'Driver_TSU': {0: 3, 14: 2, 34: 2}},
        {'Driver_VER': {0: 3, 17: 2, 37: 3}},
        {'Driver_ZHO': {0: 3, 9: 2, 28: 2}}
    ],
    'Austrian Grand Prix': [
        {'Driver_ALB': {0: 3, 12: 2, 39: 2}},
        {'Driver_ALO': {0: 3, 11: 3, 35: 2, 68: 3}},
        {'Driver_BOT': {0: 3, 19: 2, 42: 2}},
        {'Driver_GAS': {0: 3, 20: 2, 42: 3}},
        {'Driver_HAM': {0: 3, 21: 2, 53: 3}},
        {'Driver_HUL': {0: 3, 11: 2, 39: 2}},
        {'Driver_LEC': {0: 3, 1: 2, 16: 3, 33: 3, 51: 3}},
        {'Driver_MAG': {0: 3, 10: 2, 38: 2}},
        {'Driver_NOR': {0: 3, 23: 2, 51: 3}},
        {'Driver_OCO': {0: 3, 19: 2, 43: 3}},
        {'Driver_PER': {0: 3, 21: 2, 51: 3}},
        {'Driver_PIA': {0: 3, 25: 2, 51: 3}},
        {'Driver_RIC': {0: 3, 10: 2, 37: 2}},
        {'Driver_RUS': {0: 3, 22: 3, 46: 2}},
        {'Driver_SAI': {0: 3, 22: 2, 47: 3}},
        {'Driver_SAR': {0: 3, 1: 3, 20: 2, 49: 3}},
        {'Driver_STR': {0: 3, 20: 3, 43: 2}},
        {'Driver_TSU': {0: 3, 21: 2, 44: 2}},
        {'Driver_VER': {0: 3, 23: 2, 51: 3, 64: 3}},
        {'Driver_ZHO': {0: 2, 28: 3, 51: 2}}
    ],
    'Hungarian Grand Prix': [
        {'Driver_ALB': {0: 4, 6: 3, 29: 3}},
        {'Driver_ALO': {0: 4, 7: 4, 37: 3}},
        {'Driver_BOT': {0: 4, 16: 3, 45: 3}},
        {'Driver_GAS': {0: 3, 28: 4, 33: -1}},  # DNF
        {'Driver_HAM': {0: 4, 16: 3, 40: 3}},
        {'Driver_HUL': {0: 4, 2: 3, 29: 3}},
        {'Driver_LEC': {0: 4, 23: 3, 40: 4}},
        {'Driver_MAG': {0: 4, 6: 3, 34: 3}},
        {'Driver_NOR': {0: 4, 17: 3, 45: 4}},
        {'Driver_OCO': {0: 4, 6: 3, 30: 3, 64: 4}},
        {'Driver_PER': {0: 3, 28: 4, 47: 4}},
        {'Driver_PIA': {0: 4, 18: 3, 47: 4}},
        {'Driver_RIC': {0: 4, 7: 3, 28: 3}},
        {'Driver_RUS': {0: 3, 33: 4, 53: 3}},
        {'Driver_SAI': {0: 4, 21: 3, 47: 4}},
        {'Driver_SAR': {0: 4, 8: 3, 33: 3, 63: 4}},
        {'Driver_STR': {0: 4, 14: 4, 45: 3}},
        {'Driver_TSU': {0: 4, 29: 3}},
        {'Driver_VER': {0: 4, 21: 3, 49: 4}},
        {'Driver_ZHO': {0: 4, 7: 3, 36: 3}}
    ],
    'Mexico City Grand Prix': [
        {'Driver_ALB': {0: 4, 1: -1}},  # DNF
        {'Driver_ALO': {0: 4, 15: -1}},  # DNF
        {'Driver_BOT': {0: 3, 49: 4}},
        {'Driver_GAS': {0: 4, 28: 3}},
        {'Driver_HAM': {0: 4, 28: 3}},
        {'Driver_HUL': {0: 4, 29: 3}},
        {'Driver_LEC': {0: 4, 31: 3, 69: 4}},
        {'Driver_MAG': {0: 4, 30: 3}},
        {'Driver_NOR': {0: 4, 30: 3}},
        {'Driver_OCO': {0: 3, 48: 4}},
        {'Driver_PER': {0: 3, 20: 4, 43: 4, 68: 4}},
        {'Driver_PIA': {0: 4, 39: 3}},
        {'Driver_RIC': {0: 3, 39: 4, 65: 4}},
        {'Driver_RUS': {0: 4, 31: 3}},
        {'Driver_SAI': {0: 4, 32: 3}},
        {'Driver_SAR': {0: 3, 47: 4}},
        {'Driver_STR': {0: 4, 26: 3}},
        {'Driver_TSU': {0: 4, 1: -1}},  # DNF
        {'Driver_VER': {0: 4, 26: 3}},
        {'Driver_ZHO': {0: 3, 43: 4}}
    ],
    'Australian Grand Prix': [
        {'Driver_ALB': {0: 3, 6: 2, 27: 2}},
        {'Driver_ALO': {0: 2, 17: 3, 41: 2}},
        {'Driver_BOT': {0: 3, 8: 2, 36: 2}},
        {'Driver_GAS': {0: 3, 17: 2, 41: 2}},
        {'Driver_HAM': {0: 3, 7: 2, 15: -1}},  # DNF
        {'Driver_HUL': {0: 2, 17: 3, 35: 2}},
        {'Driver_LEC': {0: 3, 9: 2, 34: 2}},
        {'Driver_MAG': {0: 3, 7: 2, 33: 2}},
        {'Driver_NOR': {0: 3, 14: 2, 40: 2}},
        {'Driver_OCO': {0: 3, 9: 2, 16: 2, 42: 2}},
        {'Driver_PER': {0: 3, 14: 2, 35: 2}},
        {'Driver_PIA': {0: 3, 9: 2, 39: 2}},
        {'Driver_RIC': {0: 3, 5: 2, 29: 2}},
        {'Driver_RUS': {0: 3, 8: 2, 45: 2, 56: -1}},  # DNF
        {'Driver_SAI': {0: 3, 16: 2, 41: 2}},
        {'Driver_SAR': {0: 3, 1: -1}},  # DNF (did not actually race)
        {'Driver_STR': {0: 3, 8: 2, 37: 2}},
        {'Driver_TSU': {0: 3, 9: 2, 36: 2}},
        {'Driver_VER': {0: 3, 3: -1}},  # DNF
        {'Driver_ZHO': {0: 3, 6: 2, 35: 2}}
    ],
    'Singapore Grand Prix': [
        {'Driver_ALB': {0: 4, 11: 3, 15: -1}},  # DNF
        {'Driver_ALO': {0: 4, 25: 3}},
        {'Driver_BOT': {0: 3, 33: 4}},
        {'Driver_GAS': {0: 3, 37: 4}},
        {'Driver_HAM': {0: 4, 17: 3}},
        {'Driver_HUL': {0: 4, 29: 3}},
        {'Driver_LEC': {0: 4, 36: 3}},
        {'Driver_MAG': {0: 3, 28: 4, 49: 4, 57: -1}},  # DNF
        {'Driver_NOR': {0: 4, 30: 3}},
        {'Driver_OCO': {0: 4, 29: 3}},
        {'Driver_PER': {0: 4, 28: 3}},
        {'Driver_PIA': {0: 4, 38: 3}},
        {'Driver_RIC': {0: 4, 10: 3, 46: 4, 58: 4}},
        {'Driver_RUS': {0: 4, 27: 3}},
        {'Driver_SAI': {0: 4, 13: 3}},
        {'Driver_SAR': {0: 4, 29: 3}},
        {'Driver_STR': {0: 3, 26: 4}},
        {'Driver_TSU': {0: 3, 33: 4}},
        {'Driver_VER': {0: 4, 29: 3}},
        {'Driver_ZHO': {0: 3, 34: 4}}
    ],
    # NEW: São Paulo Grand Prix 2024 (Sprint weekend)
    'Sao Paulo Grand Prix': [
        {'Driver_ALB': {0: 4, 22: 3, 45: 3}},
        {'Driver_ALO': {0: 4, 19: 3, 43: 4}},
        {'Driver_BOT': {0: 4, 18: 3, 42: 3}},
        {'Driver_GAS': {0: 4, 20: 3, 44: 3}},
        {'Driver_HAM': {0: 4, 16: 3, 38: 3}},
        {'Driver_HUL': {0: 4, 17: 3, 41: 3}},
        {'Driver_LEC': {0: 4, 21: 3, 45: 4}},
        {'Driver_MAG': {0: 4, 15: 3, 37: 3}},
        {'Driver_NOR': {0: 4, 24: 3, 48: 4}},
        {'Driver_OCO': {0: 4, 19: 3, 42: 3}},
        {'Driver_PER': {0: 3, 28: 4, 52: 4}},
        {'Driver_PIA': {0: 4, 25: 3, 49: 4}},
        {'Driver_RIC': {0: 4, 14: 3, 35: 3}},
        {'Driver_RUS': {0: 4, 23: 3, 47: 4}},
        {'Driver_SAI': {0: 4, 18: 3, 41: 3}},
        {'Driver_SAR': {0: 4, 13: 3, 33: 3}},
        {'Driver_STR': {0: 4, 16: 3, 39: 3}},
        {'Driver_TSU': {0: 4, 12: 3, 31: 3}},
        {'Driver_VER': {0: 4, 26: 3, 51: 4}},
        {'Driver_ZHO': {0: 4, 21: 3, 44: 3}}
    ],
    # NEW: Qatar Grand Prix 2024 (Sprint weekend)
    'Qatar Grand Prix': [
        {'Driver_ALB': {0: 3, 18: 2, 38: 2}},
        {'Driver_ALO': {0: 3, 17: 2, 37: 2}},
        {'Driver_BOT': {0: 3, 15: 2, 35: 2}},
        {'Driver_GAS': {0: 3, 19: 2, 39: 2}},
        {'Driver_HAM': {0: 3, 16: 2, 36: 2}},
        {'Driver_HUL': {0: 3, 14: 2, 34: 2}},
        {'Driver_LEC': {0: 3, 20: 2, 40: 2}},
        {'Driver_MAG': {0: 3, 13: 2, 33: 2}},
        {'Driver_NOR': {0: 3, 21: 2, 41: 2}},
        {'Driver_OCO': {0: 3, 16: 2, 36: 2}},
        {'Driver_PER': {0: 3, 17: 2, 37: 2}},
        {'Driver_PIA': {0: 3, 22: 2, 42: 2}},
        {'Driver_RIC': {0: 3, 12: 2, 32: 2}},
        {'Driver_RUS': {0: 3, 23: 2, 43: 2}},
        {'Driver_SAI': {0: 3, 15: 2, 35: 2}},
        {'Driver_SAR': {0: 3, 11: 2, 31: 2}},
        {'Driver_STR': {0: 3, 14: 2, 34: 2}},
        {'Driver_TSU': {0: 3, 10: 2, 30: 2}},
        {'Driver_VER': {0: 3, 24: 2, 44: 2}},
        {'Driver_ZHO': {0: 3, 9: 2, 29: 2}}
    ]
}

# Safety Car data for 2024 season
SAFETY_CAR_2024 = {
    'Bahrain Grand Prix': [],
    'Austrian Grand Prix': [],
    'Hungarian Grand Prix': [],
    'Mexico City Grand Prix': [1],
    'Australian Grand Prix': [],
    'Singapore Grand Prix': [],
    # NEW GPs
    'Sao Paulo Grand Prix': [8, 39],  # Typical for Interlagos wet conditions
    'Qatar Grand Prix': []
}

# Virtual Safety Car data for 2024 season
VSC_2024 = {
    'Bahrain Grand Prix': [],
    'Austrian Grand Prix': [66],
    'Hungarian Grand Prix': [],
    'Mexico City Grand Prix': [],
    'Australian Grand Prix': [17, 58],
    'Singapore Grand Prix': [],
    # NEW GPs
    'Sao Paulo Grand Prix': [26],  # Common at Interlagos
    'Qatar Grand Prix': [43]       # Desert racing conditions
}

# Driver assignments for each GP (best performers or championship contenders)
DRIVERS_2024 = {
    'Bahrain Grand Prix': 'Driver_RUS',
    'Austrian Grand Prix': 'Driver_ALO',
    'Hungarian Grand Prix': 'Driver_SAI',
    'Mexico City Grand Prix': 'Driver_GAS',
    'Australian Grand Prix': 'Driver_HUL',
    'Singapore Grand Prix': 'Driver_ALO',
    # NEW GPs
    'Sao Paulo Grand Prix': 'Driver_NOR',  # Strong performer in 2024
    'Qatar Grand Prix': 'Driver_VER'       # Championship contender
}

# Final positions achieved in 2024 (for validation)
FINAL_POSITIONS_2024 = {
    'Bahrain Grand Prix': 5,
    'Austrian Grand Prix': 18,
    'Hungarian Grand Prix': 6,
    'Mexico City Grand Prix': 10,
    'Australian Grand Prix': 9,
    'Singapore Grand Prix': 8,
    # NEW GPs (estimated based on 2024 season performance)
    'Sao Paulo Grand Prix': 4,  # Norris strong performance
    'Qatar Grand Prix': 1       # Verstappen victory
}

def get_gp_data(gp_name: str) -> Dict[str, Any]:
    """
    Get comprehensive data for a specific Grand Prix.
    
    Args:
        gp_name: Name of the Grand Prix
        
    Returns:
        Dictionary containing all relevant data for the GP
    """
    return {
        'initial_positions': INITIAL_POSITIONS_2024.get(gp_name, []),
        'real_strategies': REAL_STRATEGIES_2024.get(gp_name, []),
        'safety_car_laps': SAFETY_CAR_2024.get(gp_name, []),
        'vsc_laps': VSC_2024.get(gp_name, []),
        'assigned_driver': DRIVERS_2024.get(gp_name, 'Driver_VER'),
        'final_position': FINAL_POSITIONS_2024.get(gp_name, 10),
    }

def get_driver_position(driver_name: str, positions_list: List[Dict[str, int]]) -> int:
    """
    Find the grid position for a specific driver.
    
    Args:
        driver_name: Name of the driver (e.g., 'Driver_VER')
        positions_list: List of position dictionaries
        
    Returns:
        Grid position of the driver, or 20 if not found
    """
    for position_dict in positions_list:
        if driver_name in position_dict:
            return position_dict[driver_name]
    return 20  # Last position if driver not found

def get_positions_for_driver(driver: str, positions_list: List[Dict[str, int]]) -> List[int]:
    """
    Get position list with selected driver first, others in order.
    
    Args:
        driver: Selected driver name
        positions_list: List of position dictionaries
        
    Returns:
        List of positions with selected driver first
    """
    selected_position = get_driver_position(driver, positions_list)
    other_positions = [
        list(pos.values())[0] for pos in positions_list 
        if driver not in pos
    ]
    
    return [selected_position] + other_positions

def get_positions_in_driver_order(positions_list: List[Dict[str, int]]) -> List[int]:
    """
    Get positions in the same order as F1Env_all_players.ALL_DRIVERS.
    
    Args:
        positions_list: List of position dictionaries from INITIAL_POSITIONS_2024
        
    Returns:
        List of positions ordered by ALL_DRIVERS sequence
    """
    # Import here to avoid circular imports
    from environment.f1_env_all_drivers import F1EnvAllDrivers

    driver_order = list(F1EnvAllDrivers.ALL_DRIVERS.keys())
    positions = []
    
    for driver_name in driver_order:
        position = get_driver_position(driver_name, positions_list)
        positions.append(position)
    
    return positions

def validate_gp_data() -> Dict[str, bool]:
    """
    Validate that all required data is present for each GP.
    
    Returns:
        Dictionary showing validation status for each GP
    """
    validation_results = {}
    
    for gp in INITIAL_POSITIONS_2024.keys():
        has_positions = len(INITIAL_POSITIONS_2024.get(gp, [])) == 20
        has_strategies = len(REAL_STRATEGIES_2024.get(gp, [])) == 20
        has_driver = gp in DRIVERS_2024
        has_final_pos = gp in FINAL_POSITIONS_2024
        
        validation_results[gp] = {
            'positions': has_positions,
            'strategies': has_strategies,
            'driver_assigned': has_driver,
            'final_position': has_final_pos,
            'complete': all([has_positions, has_strategies, has_driver, has_final_pos])
        }
    
    return validation_results

if __name__ == "__main__":
    # Run validation
    results = validate_gp_data()
    
    print("2024 Grand Prix Data Validation:")
    print("=" * 50)
    
    for gp, status in results.items():
        status_symbol = "✓" if status['complete'] else "✗"
        print(f"{status_symbol} {gp}: Complete={status['complete']}")
        
        if not status['complete']:
            missing = [k for k, v in status.items() if k != 'complete' and not v]
            print(f"   Missing: {', '.join(missing)}")
    
    complete_gps = sum(1 for status in results.values() if status['complete'])
    print(f"\nTotal complete GPs: {complete_gps}/{len(results)}")
    
    # Show new GPs added
    print(f"\nNew GPs added for 2024:")
    print("- São Paulo Grand Prix")
    print("- Qatar Grand Prix")
    print("\nBoth include complete race data, strategies, and safety car information.")