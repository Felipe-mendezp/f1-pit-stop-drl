import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pickle

from config import SIMULATION_DIR

# ============================================================================
# EXPONENTIAL CACHE FOR INTERVAL CALCULATIONS (OPTIMIZATION)
# ============================================================================
# Pre-compute exp(-x) for common gap values to avoid repeated np.exp() calls.
# This provides 3-5x speedup for interval calculations.
# Values outside cache range fall back to np.exp().

# Cache exp(-x) for x in [0, 100] with 0.1 precision
_EXP_CACHE = {round(v, 1): np.exp(-v) for v in np.arange(0, 100.1, 0.1)}
_EXP_CACHE_MAX = 100.0


def _get_exp_neg(value: float) -> float:
    """
    Fast lookup for exp(-value) using pre-computed cache.

    Args:
        value: The gap value (will compute exp(-value))

    Returns:
        exp(-value), from cache if available, computed otherwise
    """
    # Clamp to cache range
    if value < 0:
        value = 0.0
    elif value > _EXP_CACHE_MAX:
        return _EXP_CACHE[_EXP_CACHE_MAX]

    # Round to nearest 0.1 for cache lookup
    key = round(value, 1)
    cached = _EXP_CACHE.get(key)
    if cached is not None:
        return cached

    # Fallback for edge cases
    return np.exp(-value)


def initial_lap(initial_compound: int, GP: str, driver: str, reg_dd: dict) -> pd.DataFrame:
    """
    Actualiza el estado de una vuelta en una simulacion de carreras de Formula 1.

    Args:
        initial_compound (int): Componente inicial.
        GP (str): Grand Prix.
        driver: Piloto.
        reg_dd: Diccionario con los modelos de regresion de los circuitos.

    Returns:
        pd.DataFrame: Nuevo estado de la vuelta despues de aplicar las actualizaciones.
    """
    initial_lap = pd.DataFrame(0.0, index=np.arange(1), columns=reg_dd[GP].params.keys().tolist())
    if f'Compound_Detail_C{initial_compound}' in initial_lap.columns:
          initial_lap[f'Compound_Detail_C{initial_compound}'] = 1.0

    initial_lap['Intercept'] = 1.0
    initial_lap[driver] = 1.0
    initial_lap['Interval_front'] = 1 / (1 + 50.0)
    initial_lap['Interval_front_diff_lap'] = 1 / (1 + 100.0)
    initial_lap['Interval_behind'] = 1 / (1 + 50.0)
    initial_lap['LapNumber'] = 1.0
    initial_lap['LapNumber_square'] = 1.0
    try:
      year = [s for s in reg_dd[GP].params.keys().tolist() if s.startswith('Year')][-1]
      initial_lap[year] = 1.0
    except Exception as e:
      pass
    return initial_lap


def initial_lap2(initial_compound: int, driver: str, model_laptime) -> dict:
    """
    Actualiza el estado de una vuelta en una simulacion de carreras de Formula 1.

    Args:
        initial_compound (int): Componente inicial.
        GP (str): Grand Prix.
        driver: Piloto.
        reg_dd: Diccionario con los modelos de regresion de los circuitos.

    Returns:
        pd.DataFrame: Nuevo estado de la vuelta despues de aplicar las actualizaciones.
    """
    initial_lap = dict.fromkeys(model_laptime.params.keys(), 0.0)
    if f'Compound_Detail_C{initial_compound}' in initial_lap.keys():
          initial_lap[f'Compound_Detail_C{initial_compound}'] = 1.0

    initial_lap.update({'Intercept': 1.0,
                        driver: 1.0,
                        'Interval_front': 1 / (1 + 50.0),
                        'Interval_front_diff_lap': 1 / (1 + 100.0),
                        'Interval_behind': 1 / (1 + 50.0),
                        'LapNumber': 1.0,
                        'LapNumber_square': 1.0})

    try:
        year = [s for s in list(initial_lap.keys()) if s.startswith('Year')][-1]
        initial_lap[year] = 1.0
    except Exception as e:
      pass
    return initial_lap


def initial_lap3(initial_compound: int, driver: str, model_laptime) -> dict:
    """
    Creates initial lap state for a single driver on lap 1.

    Args:
        initial_compound (int): Starting compound (0 if not chosen yet, 1-5 for C1-C5)
        driver (str): Driver name (e.g., 'Driver_VER')
        model_laptime: Regression model with .params.keys() containing all required variables

    Returns:
        dict: Initial lap state with all model variables set to appropriate values
    """
    # Initialize all model parameters to 0.0
    initial_lap = dict.fromkeys(model_laptime.params.keys(), 0.0)

    # Set compound only if a valid compound is provided (not 0)
    if initial_compound > 0 and f'Compound_Detail_C{initial_compound}' in initial_lap.keys():
        initial_lap[f'Compound_Detail_C{initial_compound}'] = 1.0

    # Set core lap state variables
    initial_lap.update({
        'Intercept': 1.0,
        'Interval_front': np.exp(-100.0),  # Large gap initially (leader)
        'Interval_front_diff_lap': np.exp(-100.0),  # Large initial difference
        'Interval_behind': np.exp(-100.0),  # Large gap behind initially
        'LapNumber': 1.0,
        'LapNumber_square': 1.0,
        'PitIn': 0.0,
        'PitOut': 0.0,
    })

    # Set driver indicator if present in model
    if driver in initial_lap.keys():
        initial_lap[driver] = 1.0

    # Set year indicator if present in model
    try:
        year_keys = [s for s in list(initial_lap.keys()) if s.startswith('Year')]
        if year_keys:
            initial_lap[year_keys[-1]] = 1.0  # Use most recent year
    except Exception:
        pass

    # Set additional variables that might be present
    if 'DRS' in initial_lap.keys():
        initial_lap['DRS'] = 0.0  # No DRS on first lap
    if 'FirstLap_pos' in initial_lap.keys():
        initial_lap['FirstLap_pos'] = 0.0  # Will be set by environment

    return initial_lap


def initial_lap4(initial_compound: list[int], drivers: list[str], model_laptime) -> list[dict]:
    """
    Creates initial lap states for multiple drivers on lap 1.

    Args:
        initial_compound (list[int]): Starting compounds for each driver (1-5 for C1-C5)
        drivers (list[str]): Driver names (e.g., ['Driver_VER', 'Driver_HAM'])
        model_laptime: Regression model with .params.keys() containing all required variables

    Returns:
        list[dict]: List of initial lap states, one for each driver
    """
    initial_lap_ls = []

    for i in range(len(initial_compound)):
        # Use initial_lap3 function for consistency
        initial_lap = initial_lap3(initial_compound[i], drivers[i], model_laptime)
        initial_lap_ls.append(initial_lap)

    return initial_lap_ls


def update_lap_state(lap_state: pd.DataFrame, action: int, pit_out: int, yellow_flag: bool) -> pd.DataFrame:
    """
    Actualiza el estado de una vuelta en una simulacion de carreras de Formula 1.

    Args:
        lap_state (pd.DataFrame): Estado actual de la vuelta, representado como un diccionario.
        env: Ambiente de Gym de formula 1.

    Returns:
        pd.DataFrame: Nuevo estado de la vuelta despues de aplicar las actualizaciones.
    """
    compounds = [col for col in lap_state.columns if 'Compound' in col]
    tyreslifes = [col for col in lap_state.columns if 'TyreLife' in col]

    # Caso si entra a pits
    if action != 0:
       # Reiniciamos los compounds y TyreLife * Compound, y actualizamos PitIn
        lap_state[compounds] = 0.0
        lap_state[tyreslifes] = 0.0
        lap_state['PitIn'] = 1.0

        # Actualizamos el compound segun la accion
        if f'Compound_Detail_C{action}' in lap_state.columns:
          lap_state[f'Compound_Detail_C{action}'] = 1.0

    # Si no entra a pits
    else:
        lap_state['PitIn'] = 0.0
        compounds.reverse()
        tyre_life_col = None
        for compound in compounds:
            if lap_state[compound].eq(1.0).any():
                tyre_life_col = tyreslifes[compounds.index(compound)]
                break
        if tyre_life_col is None:
            tyre_life_col = tyreslifes[-3]

        if not yellow_flag:
            lap_state[tyre_life_col] += 1.0

    # Actualizamos el PitOut
    lap_state['PitOut'] = pit_out

    # Actualizamos el lapnumber y lapnumber square
    lap_state['LapNumber'] += 1.0
    lap_state['LapNumber_square'] = np.square(lap_state['LapNumber'])

    return lap_state


def update_lap_state2(lap_state: dict, action: int, prev_action: int, yf: bool) -> dict:
    """
    Actualiza el estado de una vuelta en una simulacion de carreras de Formula 1.

    Args:
        lap_state (dict): Estado actual de la vuelta, representado como un diccionario.
        action (int): Accion que toma el agente

    Returns:
        dict: Nuevo estado de la vuelta despues de aplicar las actualizaciones.
    """
    compounds = [col for col in lap_state.keys() if 'Compound' in col]
    tyreslifes = [col for col in lap_state.keys() if 'TyreLife' in col]

    lap_state['PitIn'] = 1.0 if action != 0 else 0.0

    # Caso si entra a pits
    if prev_action != 0:
       # Reiniciamos los compounds y TyreLife * Compound, y actualizamos PitIn
        compound_tyers = {**{col: 0.0 for col in compounds}, **{col: 0.0 for col in tyreslifes}}
        lap_state.update(compound_tyers)
        lap_state['PitOut'] = 1.0

        # Actualizamos el compound segun la accion
        if f'Compound_Detail_C{prev_action}' in lap_state.keys():
          lap_state[f'Compound_Detail_C{prev_action}'] = 1.0

    # Si no entra a pits
    else:
        lap_state['PitOut'] = 0.0
        # Verificar si todos los valores de Compound_Detail son 0
        if all(lap_state[compound] == 0.0 for compound in compounds):
            # Si todos los Compound_Detail son 0, usamos el menor Compound disponible (primer TyreLife)
            tyre_life_col = tyreslifes[0]  # El primer TyreLife corresponde al menor Compound
        else:
            # Encontrar el Compound_Detail activo y asignar el correspondiente TyreLife
            for i, compound in enumerate(compounds):
                if lap_state[compound] == 1.0:
                    tyre_life_col = tyreslifes[i+1]
                    break

        if not yf:
            lap_state[tyre_life_col] += 1.0

    # Actualizamos el lapnumber y lapnumber square
    lap_state['LapNumber'] += 1.0
    lap_state['LapNumber_square'] = lap_state['LapNumber'] * lap_state['LapNumber']

    return lap_state


def update_lap_state3(lap_state: dict, action: int, prev_action: int, position: int, interval: float, yf: bool) -> dict:
    """
    Actualiza el estado de una vuelta en una simulacion de carreras de Formula 1.

    Args:
        lap_state (dict): Estado actual de la vuelta, representado como un diccionario.
        action (int): Accion que toma el agente

    Returns:
        dict: Nuevo estado de la vuelta despues de aplicar las actualizaciones.
    """
    compounds = [col for col in lap_state.keys() if 'Compound' in col]
    tyreslifes = [col for col in lap_state.keys() if 'TyreLife' in col]

    lap_state['PitIn'] = 1.0 if action != 0 else 0.0

    # Caso si entra a pits
    if prev_action != 0:
       # Reiniciamos los compounds y TyreLife * Compound, y actualizamos PitIn
        compound_tyers = {**{col: 0.0 for col in compounds}, **{col: 0.0 for col in tyreslifes}}
        lap_state.update(compound_tyers)
        # lap_state['PitIn'] = 1.0
        lap_state['PitOut'] = 1.0

        # Actualizamos el compound segun la accion
        if f'Compound_Detail_C{prev_action}' in lap_state.keys():
          lap_state[f'Compound_Detail_C{prev_action}'] = 1.0

    # Si no entra a pits
    else:
        lap_state['PitOut'] = 0.0
        compounds.reverse()
        tyre_life_col = None
        for compound in compounds:
            if lap_state[compound] == 1.0:
                tyre_life_col = tyreslifes[compounds.index(compound)]
                break
        if tyre_life_col is None:
            tyre_life_col = tyreslifes[-3]

        if not yf:
            lap_state[tyre_life_col] += 1.0

    # Actualizamos el lapnumber y lapnumber square
    lap_state['LapNumber'] += 1.0
    lap_state['LapNumber_square'] = lap_state['LapNumber'] * lap_state['LapNumber']
    if position == 1:
        lap_state.update({'Interval_front': 1 / (1 + 50.0),
                          'Interval_behind': 1 / (1 + interval)})
    else:
        lap_state.update({'Interval_front': 1 / (1 + interval),
                          'Interval_behind': 1 / (1 + 50.0)})

    return lap_state


def update_lap_state4(
    lap_state: dict[str, float],
    action: int,
    prev_action: int,
    position: int,
    intervals: list[float],
    yf: bool = False
    ) -> dict[str, float]:
    """
    Updates lap state for subsequent laps, handling pit stops and tyre degradation.

    Args:
        lap_state: Current lap state dictionary with all model variables
        action: Current action (0 = no pit, 1-5 = pit to compound C1-C5)
        prev_action: Previous action (used for PitOut logic)
        position: Current position (1-20)
        intervals: [gap_to_car_ahead, gap_to_car_behind] in seconds
        yf: Whether yellow flag conditions are active

    Returns:
        dict: Updated lap state for next model prediction
    """
    # Input validation
    if not (1 <= position <= 20):
        raise ValueError(f"Position must be between 1 and 20, got {position}")

    # Get relevant columns based on actual model parameters
    compound_cols = sorted([col for col in lap_state.keys() if col.startswith('Compound_Detail_C')])
    tyre_life_cols = sorted([col for col in lap_state.keys() if col.startswith('TyreLife_C')])

    # Update pit status for current action
    if 'PitIn' in lap_state:
        lap_state['PitIn'] = 1.0 if action != 0 else 0.0

    lap_state['FirstLap_pos'] = position if lap_state['LapNumber'] == 1 else 0.0

    if lap_state['LapNumber'] > 2:
        lap_state['DRS'] = 1.0 if intervals[0] <= 1.0 else 0.0

    # Handle pit stop from previous lap (PitOut logic)
    if prev_action != 0:
        # Reset all compounds and tyre life
        for col in compound_cols + tyre_life_cols:
            if col in lap_state:
                lap_state[col] = 0.0

        if 'PitOut' in lap_state:
            lap_state['PitOut'] = 1.0

        # Set new compound from previous action
        new_compound = f'Compound_Detail_C{prev_action}'
        if new_compound in lap_state:
            lap_state[new_compound] = 1.0
    else:
        # No pit stop: update tyre life for current compound
        if 'PitOut' in lap_state:
            lap_state['PitOut'] = 0.0

        # Find active compound and update corresponding tyre life
        active_compound = None
        for compound_col in compound_cols:
            if lap_state.get(compound_col, 0.0) == 1.0:
                # Extract compound number from name (e.g., 'Compound_Detail_C1' -> '1')
                compound_num = compound_col.split('_C')[-1]
                active_compound = compound_num
                break

        # If no active compound found, default to first available
        if active_compound is None and compound_cols:
            first_compound = compound_cols[0].split('_C')[-1]
            lap_state[f'Compound_Detail_C{first_compound}'] = 1.0
            active_compound = first_compound

        # Update tyre life only if not under yellow flag
        if active_compound and not yf:
            tyre_life_col = f'TyreLife_C{active_compound}'
            if tyre_life_col in lap_state:
                lap_state[tyre_life_col] += 1.0

    # Update lap information
    current_lap = lap_state.get('LapNumber', 0.0)
    lap_state['LapNumber'] = current_lap + 1.0
    lap_state['LapNumber_square'] = lap_state['LapNumber'] ** 2

    # Update intervals with exponential transformation
    if len(intervals) >= 2:
        front_gap = 100.0 if position == 1 else max(intervals[0], 0.0)
        behind_gap = 100.0 if position == 20 else max(intervals[1], 0.0)
    else:
        front_gap = behind_gap = 100.0

    if 'Interval_front' in lap_state:
        lap_state['Interval_front'] = np.exp(-front_gap)
    if 'Interval_behind' in lap_state:
        lap_state['Interval_behind'] = np.exp(-behind_gap)

    return lap_state


def mellowmax(a: torch.Tensor, w: int):
    m = torch.max(a)
    N = torch.Tensor([len(a),])
    lse = torch.exp((a - m)*w).sum().log_()
    return m + (lse - N.log_())/w


def compute_td_loss(agent, target_network, states, actions, rewards, next_states, done_flags, env,
                    gamma=0.99, w=1.0, deepmellow=False, device='cpu'):
    """
    Calcula la perdida de error cuadratico medio (MSE) para la actualizacion de la red neuronal.

    Args:
        agent: Red neuronal que representa el agente.
        target_network: Red neuronal objetivo utilizada para calcular los valores Q futuros.
        states: Estados actuales.
        actions: Acciones tomadas en los estados actuales.
        rewards: Recompensas recibidas despues de tomar las acciones.
        next_states: Estados siguientes despues de tomar las acciones.
        done_flags: Indicadores de finalizacion para cada transicion.
        gamma (float): Factor de descuento para las recompensas futuras (predeterminado: 0.99).
        device: Dispositivo (CPU o GPU) en el que se deben realizar los calculos.

    Returns:
        torch.Tensor: Perdida de error cuadratico medio (MSE) para la actualizacion de la red neuronal.
    """
    # Convertir arrays de numpy a tensores de torch
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype('float32'), device=device, dtype=torch.float)
    # valid_actions = env.action_space
    # valid_indices = [valid_actions.index(action) for action in actions]

    # Obtener los valores Q para todas las acciones en los estados actuales usando la red del agente
    predicted_qvalues = agent(states)

    # Seleccionar los valores Q para las acciones elegidas
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    next_q_values = agent(next_states)

    if deepmellow:
        # Calcular mellowmax(next_states, acciones) utilizando los valores Q futuros predichos
        next_state_values = mellowmax(next_q_values, w)

        # Calcular los "valores Q objetivo"
        target_qvalues_for_actions = rewards + gamma * next_state_values * (1 - done_flags)

        # Calcular la perdida de error cuadratico medio (MSE) para minimizar
        loss = F.mse_loss(predicted_qvalues_for_actions, target_qvalues_for_actions)

    else:
        max_next_acts = torch.max(next_q_values, dim=1)[1].detach()

        target_next_q_values = target_network(next_states)
        max_next_q_values = target_next_q_values.gather(index=max_next_acts.view(-1, 1), dim=1)
        max_next_q_values = max_next_q_values.view(-1).detach()

        actual_qs = rewards + gamma * max_next_q_values * (1 - done_flags)

        loss = F.mse_loss(actual_qs, predicted_qvalues_for_actions)

    return loss


def epsilon_schedule(start_eps: float, end_eps: float, step: int, final_step: int) -> float:
    """
    Programa una interpolacion lineal para la programacion de la tasa de exploracion (epsilon).

    Args:
        start_eps (float): Tasa de exploracion al principio.
        end_eps (float): Tasa de exploracion al final.
        step (int): Paso actual en la programacion.
        final_step (int): Paso final en la programacion.

    Returns:
        float: Tasa de exploracion en el paso actual segun la programacion.
    """
    return start_eps + (end_eps - start_eps) * min(step, final_step) / final_step


def evaluate_strategy(gp: str, driver: str, strategy: list[tuple]) -> tuple[list, float]:
    """_summary_

    Args:
        model_laptime (_type_): modelo de la regresion para predecir el tiempo de cada vuelta.
        driver (str): Piloto a evaluar.
        strategy (list[tuple]): Estrategia a evaluar. Debe ser una lista de tuplas con el tipo de nuematico y la cantidad de vueltas.

    Returns:
        tuple[list, float]: Tupla con la lista con los tiempos de vuelta y el tiempo total de la carrera.
    """
    path = SIMULATION_DIR
    with open(f"{path}/{gp}_reg_Pickle.pkl", "rb") as f:
        model_laptime = pickle.load(f)
    params = model_laptime.params
    driver = params[driver]
    intercept = params['Intercept']
    lap_number = params['LapNumber']
    lap_number_square = params['LapNumber_square']
    year = params['Year_2023']
    interval_front = params['Interval_front'] * (1 / (1 + 50.0))
    interval_front_diff_lap = params['Interval_front_diff_lap'] * (1 / (1 + 100.0))
    interval_behind = params['Interval_behind'] * (1 / (1 + 50.0))
    intervals = interval_front + interval_behind + interval_front_diff_lap
    pit_in = params['PitIn']
    pit_out = params['PitOut']
    laptimes_ls = []
    total_time = 0.0
    lap_count = 1
    total_laps = sum(element[1] for element in strategy)

    for stint in strategy:
        compound = f'Compound_Detail_{stint[0]}'
        tyre_life = f'TyreLife_{stint[0]}'
        compound_coef = params.get(compound, 0.0)
        tyre_life_coef = params.get(tyre_life, 0.0)
        for i in range(1, stint[1]+1):
            laptime = intercept + lap_count * lap_number + (lap_count**2) * lap_number_square + year + intervals + compound_coef + tyre_life_coef * i + driver
            # Agregar pit_in en la vuelta especificada
            if i == (stint[1]-1) and lap_count != total_laps:
                laptime += pit_in
            # Agregar pit_out en la siguiente vuelta a la especificada
            if lap_count != 1 and i == 0 and i != len(strategy)-1:
                laptime += pit_out
            print(f'Laptime {lap_count}: {laptime:,.3f}')
            laptimes_ls.append(laptime)
            total_time += laptime
            lap_count += 1

    print(f'Total laptime: {total_time:,.3f}')
    return laptimes_ls, np.round(total_time, 3)


def evaluate_strategy2(gp: str, driver: str, strategy: list[tuple]) -> tuple[list, float]:
    """_summary_

    Args:
        model_laptime (_type_): modelo de la regresion para predecir el tiempo de cada vuelta.
        driver (str): Piloto a evaluar.
        strategy (list[tuple]): Estrategia a evaluar. Debe ser una lista de tuplas con el tipo de nuematico y la cantidad de vueltas.

    Returns:
        tuple[list, float]: Tupla con la lista con los tiempos de vuelta y el tiempo total de la carrera.
    """
    path = SIMULATION_DIR
    with open(f"{path}/{gp}_reg_Pickle.pkl", "rb") as f:
        model_laptime = pickle.load(f)

    laptimes_ls = []
    total_time = 0.0
    lap_count = 1
    total_laps = sum(element[1] for element in strategy)
    laps_list = [element[1] for element in strategy]
    laps_pits = [sum(laps_list[:i+1]) for i in range(len(laps_list))]
    compounds_list = [element[0] for element in strategy]
    initial_compound = compounds_list.pop(0)
    lap_state = initial_lap2(int(initial_compound[-1]), driver, model_laptime)
    prev_action = 0

    for i in range(1, total_laps + 1):
        laptime = model_laptime.predict(np.fromiter(lap_state.values(), dtype=float))[0]
        print(f'Laptime {lap_count}: {laptime:,.3f}')
        laptimes_ls.append(laptime)
        total_time += laptime
        if i+1 in laps_pits[:-1]:
            action = compounds_list.pop(0)
            lap_state = update_lap_state2(lap_state, int(action[-1]), prev_action)
            prev_action = int(action[-1])
        else:
            lap_state = update_lap_state2(lap_state, 0, prev_action)
            prev_action = 0
        lap_count += 1


    print(f'Total laptime: {total_time:,.3f}')
    return laptimes_ls, np.round(total_time, 3)


# ============================================================================
# V3 UTILITY FUNCTIONS FOR F1EnvV3
# ============================================================================

def initial_lap_v3(initial_compound: int, driver: str, fast_predictor) -> dict:
    """
    Creates initial lap state compatible with FastLinearPredictor.
    Uses fast_predictor.feature_names_ to build correct dictionary.

    Args:
        initial_compound (int): Starting compound (0 if not chosen yet, 1-5 for C1-C5)
        driver (str): Driver name (e.g., 'Driver_VER')
        fast_predictor: FastLinearPredictor instance with feature_names_ attribute

    Returns:
        dict: Initial lap state with all model variables set to appropriate values
    """
    # Initialize all features to 0.0 based on fast_predictor's feature names
    initial_lap = dict.fromkeys(fast_predictor.feature_names_, 0.0)

    # Set compound if valid compound is provided (not 0)
    if initial_compound > 0:
        # Match exact Treatment indicator [T.C{n}] to avoid matching the reference compound
        target = f'[T.C{initial_compound}]'
        for key in initial_lap.keys():
            if 'Compound_Detail' in key and target in key and 'TyreLife' not in key:
                initial_lap[key] = 1.0
                break

    # Set core lap state variables
    initial_lap['Intercept'] = 1.0
    initial_lap['LapNumber'] = 1.0

    # Set squared lap number if present
    if 'LapNumber_square' in initial_lap:
        initial_lap['LapNumber_square'] = 1.0
    if 'LapNumber_2' in initial_lap:
        initial_lap['LapNumber_2'] = 1.0

    # Set intervals to large gap initially (leader position) - OPTIMIZED with cache
    if 'Interval_front' in initial_lap:
        initial_lap['Interval_front'] = _get_exp_neg(100.0)
    if 'Interval_behind' in initial_lap:
        initial_lap['Interval_behind'] = _get_exp_neg(100.0)

    # Set pit flags
    if 'PitIn' in initial_lap:
        initial_lap['PitIn'] = 0.0
    if 'PitOut' in initial_lap:
        initial_lap['PitOut'] = 0.0

    # Set DRS (no DRS on first lap)
    if 'DRS' in initial_lap:
        initial_lap['DRS'] = 0.0

    # Set FirstLap_pos (will be updated by environment)
    if 'FirstLap_pos' in initial_lap:
        initial_lap['FirstLap_pos'] = 0.0

    # Set driver indicator - match exact Treatment indicator [T.{code}]
    driver_code = driver.replace('Driver_', '')
    driver_target = f'[T.{driver_code}]'
    for key in initial_lap.keys():
        if 'Driver' in key and driver_target in key:
            initial_lap[key] = 1.0
            break

    # Set year indicator if present (use most recent year)
    # Filter to only Treatment-encoded keys (contain '[T.') to exclude the semantic 'Year' key
    year_keys = [k for k in initial_lap.keys() if 'Year' in k and '[T.' in k]
    if year_keys:
        # Set the most recent year
        initial_lap[year_keys[-1]] = 1.0

    # Add semantic keys for _build_design_vector (categorical lookups)
    initial_lap['Driver'] = driver_code
    initial_lap['Compound_Detail'] = f'C{initial_compound}' if initial_compound > 0 else ''
    initial_lap['TyreLife'] = 0.0
    # Extract year from the Treatment-encoded key name
    if year_keys:
        initial_lap['Year'] = year_keys[-1].split('[T.')[-1].rstrip(']')
    else:
        initial_lap['Year'] = ''

    return initial_lap


def initial_lap_v3_batch(initial_compounds: list[int], drivers: list[str], fast_predictor) -> list[dict]:
    """
    Creates initial lap states for multiple drivers on lap 1.

    Args:
        initial_compounds (list[int]): Starting compounds for each driver (1-5 for C1-C5)
        drivers (list[str]): Driver names (e.g., ['Driver_VER', 'Driver_HAM'])
        fast_predictor: FastLinearPredictor instance

    Returns:
        list[dict]: List of initial lap states, one for each driver
    """
    return [
        initial_lap_v3(compound, driver, fast_predictor)
        for compound, driver in zip(initial_compounds, drivers)
    ]


def update_lap_state_v3(
    lap_state: dict,
    action: int,
    prev_action: int,
    position: int,
    intervals: tuple,
    yf: bool,
    compound_cols: list[str] = None,
    tyre_life_cols: list[str] = None,
    compound_col_map: dict = None,
    tyre_life_col_map: dict = None
) -> dict:
    """
    Optimized lap state update for V3 environment.
    Pre-receives column names and O(1) lookup maps for maximum performance.

    Args:
        lap_state: Current lap state dictionary with all model variables
        action: Current action (0 = no pit, 1-5 = pit to compound C1-C5)
        prev_action: Previous action (used for PitOut logic)
        position: Current position (1-20)
        intervals: (gap_to_car_ahead, gap_to_car_behind) in seconds
        yf: Whether yellow flag conditions are active
        compound_cols: Pre-computed list of compound column names
        tyre_life_cols: Pre-computed list of tyre life column names
        compound_col_map: O(1) lookup: {compound_num: column_name} (OPTIMIZATION)
        tyre_life_col_map: O(1) lookup: {compound_num: column_name} (OPTIMIZATION)

    Returns:
        dict: Updated lap state for next model prediction
    """
    # Get compound and tyre life columns if not provided
    # Filter with '[' to exclude the semantic 'Compound_Detail' key
    if compound_cols is None:
        compound_cols = sorted([col for col in lap_state.keys() if 'Compound_Detail' in col and 'TyreLife' not in col and '[' in col])
    if tyre_life_cols is None:
        # Filter with ':' to exclude the semantic 'TyreLife' key (interaction cols have ':')
        tyre_life_cols = sorted([col for col in lap_state.keys() if 'TyreLife' in col and ':' in col])

    # Build O(1) lookup maps if not provided
    if compound_col_map is None:
        compound_col_map = {}
        for col in compound_cols:
            for i in range(1, 6):
                if f'C{i}' in col:
                    compound_col_map[i] = col
                    break

    if tyre_life_col_map is None:
        tyre_life_col_map = {}
        for col in tyre_life_cols:
            for i in range(1, 6):
                if f'C{i}' in col:
                    tyre_life_col_map[i] = col
                    break

    # Update PitIn for current action
    if 'PitIn' in lap_state:
        lap_state['PitIn'] = 1.0 if action != 0 else 0.0

    # Update FirstLap_pos
    current_lap = lap_state.get('LapNumber', 1.0)
    if 'FirstLap_pos' in lap_state:
        lap_state['FirstLap_pos'] = float(position) if current_lap == 1 else 0.0

    # Update DRS (available after lap 2 if within 1 second)
    if 'DRS' in lap_state and current_lap > 2:
        gap_ahead = intervals[0] if len(intervals) > 0 else 100.0
        lap_state['DRS'] = 1.0 if gap_ahead <= 1.0 else 0.0

    # Handle pit stop from previous lap (PitOut logic)
    if prev_action != 0:
        # Reset all compounds and tyre life (use lists for clearing)
        for col in compound_cols:
            lap_state[col] = 0.0
        for col in tyre_life_cols:
            lap_state[col] = 0.0

        if 'PitOut' in lap_state:
            lap_state['PitOut'] = 1.0

        # Set new compound from previous action - O(1) lookup!
        compound_col = compound_col_map.get(prev_action)
        if compound_col:
            lap_state[compound_col] = 1.0
        else:
            # Fallback: linear search
            for col in compound_cols:
                if f'C{prev_action}' in col:
                    lap_state[col] = 1.0
                    break

        # Update semantic keys for _build_design_vector
        lap_state['Compound_Detail'] = f'C{prev_action}'
        lap_state['TyreLife'] = 0.0
    else:
        # No pit stop: update tyre life for current compound
        if 'PitOut' in lap_state:
            lap_state['PitOut'] = 0.0

        # Find active compound using O(1) reverse lookup
        active_compound = None
        for compound_num, col in compound_col_map.items():
            if lap_state.get(col, 0.0) == 1.0:
                active_compound = compound_num
                break

        # If no active compound found, default to first available
        if active_compound is None and compound_col_map:
            active_compound = min(compound_col_map.keys())
            lap_state[compound_col_map[active_compound]] = 1.0

        # Update tyre life only if not under yellow flag - O(1) lookup!
        if active_compound and not yf:
            tyre_col = tyre_life_col_map.get(active_compound)
            if tyre_col:
                lap_state[tyre_col] = lap_state.get(tyre_col, 0.0) + 1.0
            else:
                # Fallback: linear search
                for col in tyre_life_cols:
                    if f'C{active_compound}' in col:
                        lap_state[col] = lap_state.get(col, 0.0) + 1.0
                        break

            # Update semantic TyreLife for _build_design_vector
            lap_state['TyreLife'] = lap_state.get('TyreLife', 0.0) + 1.0

    # Update lap information
    lap_state['LapNumber'] = current_lap + 1.0
    if 'LapNumber_square' in lap_state:
        lap_state['LapNumber_square'] = lap_state['LapNumber'] ** 2
    if 'LapNumber_2' in lap_state:
        lap_state['LapNumber_2'] = lap_state['LapNumber'] ** 2

    # Update intervals with exponential transformation (OPTIMIZED with cache)
    front_gap = 100.0 if position == 1 else max(intervals[0], 0.0) if len(intervals) > 0 else 100.0
    behind_gap = 100.0 if position == 20 else max(intervals[1], 0.0) if len(intervals) > 1 else 100.0

    if 'Interval_front' in lap_state:
        lap_state['Interval_front'] = _get_exp_neg(front_gap)
    if 'Interval_behind' in lap_state:
        lap_state['Interval_behind'] = _get_exp_neg(behind_gap)

    return lap_state


def build_yf_features(
    lap_state: dict,
    is_sc: bool,
    is_vsc: bool,
    is_first_yf_lap: bool
) -> dict:
    """
    Build feature dictionary for yellow flag lap time prediction.
    Matches Equation (4) from paper: YF lap time model.

    The YF model uses: [const, PitIn, PitOut, SC, VSC, SC_firstlap, VSC_firstlap]

    Args:
        lap_state: Current lap state dictionary
        is_sc: Whether Safety Car is active
        is_vsc: Whether Virtual Safety Car is active
        is_first_yf_lap: Whether this is the first lap of the YF period

    Returns:
        dict: Feature dictionary for YF lap time prediction
    """
    features = {
        'const': 1.0,
        'Intercept': 1.0,
        'PitIn': lap_state.get('PitIn', 0.0),
        'PitOut': lap_state.get('PitOut', 0.0),
        'SC': 1.0 if is_sc else 0.0,
        'VSC': 1.0 if is_vsc else 0.0,
        'SC_firstlap': 1.0 if (is_sc and is_first_yf_lap) else 0.0,
        'VSC_firstlap': 1.0 if (is_vsc and is_first_yf_lap) else 0.0,
    }

    return features


def predict_pit_stop_v3(
    driver_state: dict,
    betas_stop: np.ndarray,
    gp_compounds: list[str],
    n_laps: int,
    threshold: float = None
) -> tuple:
    """
    Fast pit stop prediction using pre-computed coefficient array.

    Assumes logit formula: PitIn ~ SC + VSC + used_two_compounds + LapsLeft +
                           TyreLife:C(Compound_Detail) + Position + GP

    Args:
        driver_state: Driver state dictionary containing:
            - SC, VSC: Yellow flag indicators
            - used_two_compounds: Whether driver has changed compound
            - LapNumber: Current lap
            - TyreLife: Current tire age
            - Compound_Detail: Current compound (e.g., 'C3')
            - Position: Current position
        betas_stop: Pre-computed coefficient array for the GP
        gp_compounds: List of compounds available (e.g., ['C2', 'C3', 'C4'])
        n_laps: Total race laps
        threshold: Decision threshold (if None, random threshold between 0-1)

    Returns:
        tuple: (pit_decision: int 0/1, probability: float)
    """
    # Build feature vector
    x_vector_1 = np.array([
        1.0,  # Intercept
        driver_state.get('SC', 0.0),
        driver_state.get('VSC', 0.0),
        driver_state.get('used_two_compounds', 0.0),
        n_laps - driver_state.get('LapNumber', 1.0),  # LapsLeft
    ], dtype=np.float32)

    # TyreLife interaction (only for current compound)
    current_compound = driver_state.get('Compound_Detail', 'C3')
    tyre_life = driver_state.get('TyreLife', 0.0)
    x_vector_2 = np.array([
        tyre_life if current_compound == compound else 0.0
        for compound in gp_compounds
    ], dtype=np.float32)

    x_vector_3 = np.array([
        driver_state.get('Position', 10.0),
        1.0  # GP dummy (always 1 since we use GP-specific betas)
    ], dtype=np.float32)

    x_vector = np.concatenate([x_vector_1, x_vector_2, x_vector_3])

    # Compute linear predictor and probability
    lin_pred = np.dot(betas_stop, x_vector)
    prob_stop = 1.0 / (1.0 + np.exp(-lin_pred))

    # Decision
    if threshold is None:
        threshold = np.random.uniform(0, 1)
    pit_in = 1 if prob_stop >= threshold else 0

    return pit_in, float(prob_stop)


def predict_compound_choice_v3(
    driver_state: dict,
    betas_compound: np.ndarray,
    gp_compounds: list[str],
    n_laps: int
) -> str:
    """
    Fast compound choice prediction using pre-computed 2D coefficient array.

    Assumes conditional logit: chosen ~ 0 + C(alt) + C(alt):LapsLeft + not_change_compound + GP

    Args:
        driver_state: Driver state dictionary
        betas_compound: Pre-computed 2D coefficient array (n_compounds, 4)
                       Columns: [Intercept(alt), GP, LapsLeft(alt), not_change_compound]
        gp_compounds: List of compounds available (e.g., ['C2', 'C3', 'C4'])
        n_laps: Total race laps

    Returns:
        str: Chosen compound (e.g., 'C3')
    """
    current_compound = driver_state.get('Compound_Detail', 'C3')
    used_two = driver_state.get('used_two_compounds', 0.0)
    laps_left = n_laps - driver_state.get('LapNumber', 1.0)

    # Build utility for each compound
    utilities = []
    for c, compound in enumerate(gp_compounds):
        # not_change_compound: 1 if putting same compound and haven't changed yet
        not_change = 1.0 if (compound == current_compound and used_two == 0) else 0.0

        x_vector = np.array([
            1.0,           # Intercept (alt)
            1.0,           # GP dummy
            laps_left,     # LapsLeft (alt)
            not_change     # not_change_compound
        ], dtype=np.float32)

        utility = np.dot(betas_compound[c], x_vector)
        utilities.append(utility)

    # Softmax for choice probabilities
    utilities = np.array(utilities)
    exp_u = np.exp(utilities - np.max(utilities))  # Subtract max for numerical stability
    probs = exp_u / exp_u.sum()

    # Random choice based on probabilities
    chosen_idx = np.random.choice(len(gp_compounds), p=probs)

    return gp_compounds[chosen_idx]


def get_tyre_life_from_lap_state(lap_state: dict, compound: int) -> float:
    """
    Extract tyre life for a specific compound from lap state.

    Args:
        lap_state: Current lap state dictionary
        compound: Compound number (1-5)

    Returns:
        float: Tyre life for the compound
    """
    # Look for TyreLife column containing the compound
    for key, value in lap_state.items():
        if 'TyreLife' in key and f'C{compound}' in key:
            return value
    return 0.0


def get_active_compound_from_lap_state(lap_state: dict) -> int:
    """
    Extract active compound number from lap state.

    Args:
        lap_state: Current lap state dictionary

    Returns:
        int: Active compound number (1-5), or 3 as default
    """
    for key, value in lap_state.items():
        if 'Compound_Detail' in key and value == 1.0:
            for i in range(1, 6):
                if f'C{i}' in key:
                    return i
    return 3  # Default to C3
