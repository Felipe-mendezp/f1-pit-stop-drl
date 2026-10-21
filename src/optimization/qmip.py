import os
import sys
import gurobipy as gp
from gurobipy import GRB

Tires = ["Soft", "Medium", "Hard"]

def slot_tires(p, set_ages=None):
    """Compound (1 = Soft, 2 = Medium, 3 = Hard) of each candidate future stint, in the order of the y variables.
    Without set_ages: p copies of (Soft, Medium, Hard) (published model). With set_ages (tyre sets, 2026-10-21):
    one slot per set left of each compound, at most p per compound, Soft slots first, then Medium, then Hard."""
    if set_ages is None:
        return [1, 2, 3] * p
    return [ti + 1 for ti, t in enumerate(Tires) for _ in range(min(p, len(set_ages.get(t, []))))]


def slot_ages(p, set_ages):
    """Age of the set of each slot of slot_tires(p, set_ages) (ascending within a compound)."""
    return [a for t in Tires for a in sorted(set_ages.get(t, []))[:p]]


def solve_MIP_GUROBI(betas_dict, time_stop, n, t_current, w_current, t_previous, p, gamma, N, verbose=True,
                     time_stop_yf=None, set_ages=None):
    # betas_dict : dict of regression betas; key = tire compound name, value = list of length 2 [beta0, beta1]
    # time_stop  : pit-stop time loss (seconds)
    # n          : number of remaining laps
    # t_current  : current tire compound in use
    # w_current  : current tire wear (laps on this compound)
    # t_previous : list of previously used tire compounds (subset of Soft / Medium / Hard)
    # p          : number of stints to consider
    # gamma      : 1 if a yellow flag is active now, 0 otherwise
    # N          : total race length in laps (used to detect the start-of-race case)
    # time_stop_yf : (2026-10-09, opcional) pérdida de parada bajo SC/VSC. None = descuento
    #                legacy time_stop/2 (publicado); si no, descuento time_stop - time_stop_yf
    # set_ages   : (2026-10-21, opcional, juegos de neumaticos F1_TYRE_SETS) {"Soft": [edades], ...} de los juegos
    #              que quedan (sin el montado). None = modelo publicado (toda tanda futura con neumatico nuevo y
    #              oferta ilimitada). Con set_ages cada tanda futura es un juego concreto: a lo mas un uso por
    #              juego, desgaste inicial = su edad, y los juegos de un compuesto se usan de menor a mayor edad
    #              (y no creciente dentro del compuesto: sin perdida de optimalidad, el orden temporal de las
    #              tandas no cambia el objetivo y un juego menos gastado nunca cuesta mas). Total de tandas <= p.
    if set_ages is not None:
        return _solve_sets(betas_dict, time_stop, n, t_current, w_current, t_previous, p, gamma, N, verbose,
                           time_stop_yf, set_ages)

    # Create a new optimization model
    model = gp.Model("MIQP_Problem")
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Encode the candidate stint sequence: p copies of (Soft, Medium, Hard).
    t_s = [1, 2, 3] * p
    k = len(t_s)

    # Define the decision variables
    x = [model.addVar(lb=0, vtype=GRB.INTEGER, name=f'x_{i}') for i in range(k + 1)]
    y = [model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f'y_{i}') for i in range(k)]
    z = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z')

    # Ahorro de parar bajo amarilla (z = 1)
    yf_discount = time_stop / 2 if time_stop_yf is None else time_stop - time_stop_yf

    # Define the objective function
    dx = []
    dy = []
    Dxx = []
    for j in range(k + 1):
        if j == 0:
            w = w_current
            t = t_current
        else:
            w = 0
            t = Tires[t_s[j - 1] - 1]
        beta_0 = betas_dict[t][0]
        beta_1 = betas_dict[t][1]
        dx.append(beta_0 + (w - 0.5) * beta_1)
        Dxx.append(beta_1)
    for j in range(k):
        dy.append(time_stop)

    objective = gp.quicksum(dx[i] * x[i] for i in range(len(dx))) + \
                gp.quicksum(dy[i] * y[i] for i in range(len(dy))) + \
                gp.quicksum(Dxx[i] * x[i] * x[i] / 2 for i in range(len(Dxx))) - \
                yf_discount * z
    model.setObjective(objective, GRB.MINIMIZE)

    # Add constraints
    A1x = gp.quicksum(x[i] for i in range(k + 1)) == n
    model.addConstr(A1x)

    for j in range(1, k + 1):
        A2x = x[j] >= 0
        model.addConstr(A2x)

    for j in range(k):
        A3x = x[j + 1] - n * y[j] <= 0
        model.addConstr(A3x)

    for j in range(k):
        A4x = x[j + 1] - y[j] >= 0
        model.addConstr(A4x)

    # Force different tire if only one tire compound has been used
    if len(t_previous) == 1:
        A5y = gp.quicksum(y[i] for i in range(k) if Tires[t_s[i]-1] != t_previous[0]) >= 1
        model.addConstr(A5y)
     
    #eliminate symmetry    
    
    model.addConstr(gp.quicksum(y[i] for i in range(0, 3)) <= 1)

    for i in range(0, 3 * (p - 1), 3):
        model.addConstr(y[i] + y[i + 1] + y[i + 2] >= y[i + 3] + y[i + 4] + y[i + 5])

    # 2026-10-21, opt-in with F1_QMIP_SYMMETRY=full: complete symmetry breaking among the future stints
    # (all start on new tyres and cost the same, so their order does not change the objective). Stints are
    # sorted Soft -> Medium -> Hard (cumulative sums of y non-increasing; the Hard case is the constraint
    # above) and, within a compound, by non-increasing length. Without the variable: only the Soft
    # ordering below (published behaviour); Medium/Hard ties are left to the solver.
    full_symmetry = os.environ.get('F1_QMIP_SYMMETRY') == 'full'
    if full_symmetry:
        for i in range(0, 3 * (p - 1), 3):
            model.addConstr(y[i] >= y[i + 3])
            model.addConstr(y[i] + y[i + 1] >= y[i + 3] + y[i + 4])
            for c in range(3):
                model.addConstr(x[i + c + 1] >= x[i + c + 4] - n * (1 - y[i + c]))

    # Constraint to enforce pitting under a yellow flag if one is active
    model.addConstr(z <= gamma  - x[0] * gamma / n)
    model.addConstr(gamma  - x[0] * gamma <= z)

    # Constraints used by the strategy-plot post-processing
    model.addConstr(x[0] >= int(n == N))
    
    if not full_symmetry:  # with full_symmetry the Soft ordering is part of the block above
        for i in range(1, 3 * (p - 1) + 1, 3):
            model.addConstr(x[i] >= x[i + 3])

    model.addConstr(n * gamma + x[0] + w_current >= x[1]*int(t_current== "Soft") + x[2]*int(t_current== "Medium") + x[3]* int(t_current== "Hard"))

    # Optimize the MIQP problem using Gurobi solver
    model.optimize()

    # Return the optimal variable values
    x_values = [float(v.X) for v in x]
    y_values = [float(v.X) for v in y]
    # get the optimal value
    opt_val = model.objVal
    if verbose:
        print(opt_val)

    # return x_values + y_values + [opt_val]
    return x_values, y_values, opt_val

def _solve_sets(betas_dict, time_stop, n, t_current, w_current, t_previous, p, gamma, N, verbose, time_stop_yf,
                set_ages):
    """QMIP with the tyre sets left (see solve_MIP_GUROBI, set_ages). Same objective and constraints as the
    published model, with the candidate stints indexed by set instead of p blocks of (Soft, Medium, Hard)."""
    model = gp.Model("MIQP_Problem_sets")
    if not verbose:
        model.setParam('OutputFlag', 0)
    t_s = slot_tires(p, set_ages)
    w_s = slot_ages(p, set_ages)
    k = len(t_s)
    x = [model.addVar(lb=0, vtype=GRB.INTEGER, name=f'x_{i}') for i in range(k + 1)]
    y = [model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f'y_{i}') for i in range(k)]
    z = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z')
    yf_discount = time_stop / 2 if time_stop_yf is None else time_stop - time_stop_yf
    dx, Dxx = [], []
    for j in range(k + 1):
        w, t = (w_current, t_current) if j == 0 else (w_s[j - 1], Tires[t_s[j - 1] - 1])
        beta_0, beta_1 = betas_dict[t][0], betas_dict[t][1]
        dx.append(beta_0 + (w - 0.5) * beta_1)
        Dxx.append(beta_1)
    model.setObjective(gp.quicksum(dx[i] * x[i] for i in range(k + 1)) +
                       gp.quicksum(time_stop * y[i] for i in range(k)) +
                       gp.quicksum(Dxx[i] * x[i] * x[i] / 2 for i in range(k + 1)) - yf_discount * z, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(x) == n)
    for j in range(k):
        model.addConstr(x[j + 1] - n * y[j] <= 0)
        model.addConstr(x[j + 1] - y[j] >= 0)
    if len(t_previous) == 1:
        model.addConstr(gp.quicksum(y[i] for i in range(k) if Tires[t_s[i] - 1] != t_previous[0]) >= 1)
    model.addConstr(gp.quicksum(y) <= p)
    for i in range(k - 1):  # sets of a compound used from the least worn
        if t_s[i] == t_s[i + 1]:
            model.addConstr(y[i] >= y[i + 1])
    model.addConstr(z <= gamma - x[0] * gamma / n)
    model.addConstr(gamma - x[0] * gamma <= z)
    model.addConstr(x[0] >= int(n == N))
    # same-compound rule of the published model, on the first (least-worn) slot of the current compound
    first = [i for i in range(k) if Tires[t_s[i] - 1] == t_current][:1]
    if first:
        model.addConstr(n * gamma + x[0] + w_current >= x[first[0] + 1])
    model.optimize()
    x_values = [float(v.X) for v in x]
    y_values = [float(v.X) for v in y]
    if verbose:
        print(model.objVal)
    return x_values, y_values, model.objVal


def solve_MIP_GUROBI_start(betas_dict, time_stop, n, w_current, t_previous, p, gamma, N, verbose=True,
                           set_ages=None):
    # Start-of-race variant: also chooses the starting compound by sweeping all Tires.
    # betas_dict : dict of regression betas; key = tire compound name, value = list of length 2 [beta0, beta1]
    # time_stop  : pit-stop time loss (seconds)
    # n          : number of remaining laps
    # w_current  : current tire wear
    # t_previous : list of previously used tire compounds
    # p          : maximum number of pit stops to consider
    # gamma      : 1 if a yellow flag is active now, 0 otherwise
    # N          : total race length in laps

    x_opt = None
    y_opt = None
    obj_opt = 10000000.0
    t0_opt = None
    for t_current in Tires:
        t_previous = [t_current]
        if set_ages is None:
            x, y, obj = solve_MIP_GUROBI(betas_dict, time_stop, n, t_current, w_current, t_previous, p, gamma, N,
                                         verbose=verbose)
        else:  # tyre sets: start on the least-worn set of t_current (none left: compound not available)
            left = {t: sorted(a) for t, a in set_ages.items()}
            if not left.get(t_current):
                continue
            w0 = left[t_current].pop(0)
            x, y, obj = solve_MIP_GUROBI(betas_dict, time_stop, n, t_current, w_current + w0, t_previous, p,
                                         gamma, N, verbose=verbose, set_ages=left)
        if obj < obj_opt:
            x_opt = x.copy()
            y_opt = y.copy()
            obj_opt = obj
            t0_opt = t_current
    # Return the OPTIMAL objective (obj would be the last compound tried)
    return x_opt, y_opt, obj_opt, t0_opt

if __name__ == "__main__":
    beta_dict = {"Soft" : [float(sys.argv[1]), float(sys.argv[2])],
                "Medium" : [float(sys.argv[3]), float(sys.argv[4])],
                "Hard" : [float(sys.argv[5]), float(sys.argv[6])]}
    time_stop = float(sys.argv[7])
    n = int(sys.argv[8])
    t_current = sys.argv[9]
    w_current = float(sys.argv[10])
    t_previous = list(set(sys.argv[11].split('-')+[t_current]))
    p = int(sys.argv[12])
    gamma = int(sys.argv[13])
    N = int(sys.argv[14])
    # print(p)
    # print(gamma)
    # remove '_' from t_previous
    t_previous = [t for t in t_previous if t != '_']
    # sort t_previous according to Soft Medium Hard
    t_previous = sorted(t_previous, key=lambda x: Tires.index(x))
    x = None; y = None; obj = None; t0 = ""
    if t_current != "_":
        x, y, obj = solve_MIP_GUROBI(beta_dict, time_stop, n, t_current, w_current, t_previous, p, gamma, N)
        print(x + y + [obj])
    else:
        x, y, obj, t0 = solve_MIP_GUROBI_start(beta_dict, time_stop, n, w_current, t_previous, p, gamma, N)
        print(x + y + [obj], t0)
