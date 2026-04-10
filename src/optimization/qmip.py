import sys
import gurobipy as gp
from gurobipy import GRB

Tires = ["Soft", "Medium", "Hard"]

def solve_MIP_GUROBI(betas_dict, time_stop, n, t_current, w_current, t_previous, p, gamma, N, verbose=True):
    # betas_dicts : diccionario de los beta, la llave el el tipo de neumatico, y el valor es una lista de tamaño 2 (beta0 y beta1)
    # time_stop : tiempo de parada en piuts
    # n : numero de vueltas faltantes
    # t_current : tipo de neumatico actual
    # w_current : desgaste actual del neumatico
    # t_previous : tipos de neumaticos anteriores usados // Soft, Medium, Hard
    # p : number of stints to consider
    # gamma : 1 if we are at a yellow flag right now, 0 if not

    # Create a new optimization model
    model = gp.Model("MIQP_Problem")
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Define your parameters
    # n = 20
    # w_current = 5
    t_s = [1, 2, 3] * p
    k = len(t_s)

    # Define the decision variables
    x = [model.addVar(lb=0, vtype=GRB.INTEGER, name=f'x_{i}') for i in range(k + 1)]
    y = [model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f'y_{i}') for i in range(k)]
    z = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z')

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
                (time_stop / 2) * z
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

    # restricción de parar durante la bandera amarilla
    model.addConstr(z <= gamma  - x[0] * gamma / n)
    model.addConstr(gamma  - x[0] * gamma <= z)
    
    # restricciones para arreglar grafico estrategias 
    model.addConstr(x[0] >= int(n == N))
    
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

def solve_MIP_GUROBI_start(betas_dict, time_stop, n, w_current, t_previous, p, gamma, N, verbose=True):
    # betas_dicts : diccionario de los beta, la llave el el tipo de neumatico, y el valor es una lista de tamaño 2 (beta0 y beta1)
    # time_stop : tiempo de parada en piuts
    # n : numero de vueltas faltantes
    # t_current : tipo de neumatico actual
    # w_current : desgaste actual del neumatico
    # t_previous : tipos de neumaticos anteriores usados
    # p : cantidad máxima de paradas en pits
    # gamma : 1 if we are at a yellow flag right now, 0 if not

    x_opt = None
    y_opt = None
    obj_opt = 10000000.0
    t0_opt = None
    for t_current in Tires:
        t_previous = [t_current]
        x, y, obj = solve_MIP_GUROBI(betas_dict, time_stop, n, t_current, w_current, t_previous, p, gamma, N, verbose=verbose)
        if obj < obj_opt:
            x_opt = x.copy()
            y_opt = y.copy()
            obj_opt = obj
            t0_opt = t_current
    return x_opt, y_opt, obj, t0_opt

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
