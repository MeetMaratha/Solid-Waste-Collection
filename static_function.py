from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

np.random.seed(42)

# Constants
B_TO_T = 10 # Bin to Truck
B_TO_B = 100 # Bin to Bin

def optimize(df, distances, NTaken, n_trucks = 1, w1 = 0.5, w2 = 0.5, folder_Path = '', ward_name = '', t_name = '', t_no = None):
    mdl = Model('CVRP')

    # Initializations
    
    startNode = [0]
    objs = [0]
    fills = []
    active_arcs = []
    Ns = []
    Vs = []
    As = []
    Cs = []
    Xs = []
    Ys = []
    Us = []
    
    Ns.append(None)
    Vs.append(None)
    As.append(None)
    Cs.append(None)
    Xs.append(None)
    Ys.append(None)
    Us.append(None)
    active_arcs.append(None)
    fills.append(None)
    fillNewName = 'fill_ratio_' + str(t_no)
    dist = distances.iloc[startNode[0], df.index.tolist()].tolist()
    distName = 'distance_from_' + str(startNode[0]) + '_' + str(t_no)
    if distName not in df.columns.tolist() : 
        temp = df.copy()
        temp[distName] = dist
        df = temp.copy()
    fillpmNewName1 = 'fill_per_m' + '_' + str(t_no)
    temp = df.copy()
    temp[fillpmNewName1] = B_TO_B * temp.loc[:, fillNewName] / temp.loc[:, distName]
    df = temp.copy()
    df = df.sort_values(by = fillpmNewName1, ascending = False)
    fills[0] = pd.DataFrame(
            {'fill' : df.loc[:, fillNewName].tolist() + [0.0]}, index = df.index.tolist() + [0]
            )
    N = []
    for i in df.index.tolist():
        if (i not in NTaken) and ( df.loc[i, fillNewName] + sum(df.loc[N, fillNewName]) ) * B_TO_T <= 100:
            N.append(i)
            NTaken.append(i)
    Ns[0] = N
    Vs[0] = N + [0]
    As[0] = [(p, q) for p in Vs[0] for q in Vs[0] if p != q]
    Cs[0] = {(p, q) : distances.iloc[p, q] for p,q in As[0]}
    Xs[0] = mdl.addVars(As[0], vtype = GRB.BINARY)
    Ys[0] = mdl.addVars(Vs[0], vtype = GRB.BINARY)
    Us[0] = mdl.addVars(Ns[0], vtype = GRB.CONTINUOUS)
    objs[0] = quicksum( 
            (w1 * Xs[0][p, q] * Cs[0][(p, q)]) - (w2 * Ys[0][p] * fills[0].loc[p, 'fill'] * B_TO_T) for p,q in As[0]
            )
        
# Optimization Function defination

    
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(sum(objs))

    # Constraints

    mdl.addConstrs(
        quicksum( Xs[0][i, j] for j in Vs[0] if j != i ) == 1 for i in Ns[0]
    )
    mdl.addConstrs(
        quicksum( Xs[0][i, j] for i in Vs[0] if i != j ) == 1 for j in Ns[0]
    )
    mdl.addConstr(
        quicksum( Ys[0][i] * fills[0].loc[i, 'fill'] * B_TO_T for i in Ns[0] ) <= ( 100)
    )
    mdl.addConstr(
        quicksum( Xs[0][startNode[0], j] for j in Ns[0]) == 1
    )
    mdl.addConstr(
        quicksum( Xs[0][j, startNode[0]] for j in Ns[0] ) == 1
    )
    mdl.addConstrs(
        (Xs[0][i, j] == 1) >> (Us[0][i] + fills[0].loc[j, 'fill'] * B_TO_T == Us[0][j]) for i,j in As[0] if i != 0 and j != 0
    )

    mdl.addConstrs(
        Us[0][i] >= (fills[0].loc[i, 'fill'] * B_TO_T) for i in Ns[0]
    )
    mdl.addConstrs(Us[0][i] <= (100) for i in Ns[0])

    # Model Restrictions

    mdl.Params.MIPGap = 0.1
    mdl.Params.TIMELimit = 900

    # Optimization
    mdl.optimize()
    objValue = mdl.getObjective().getValue()
        
    active_arcs = [a for a in As[0] if Xs[0][a].x > 0.99]
    
    return objValue, df, active_arcs, NTaken
    
    

def update_fill(data, t_no):
    fillNewName = 'fill_ratio_' + str(t_no)
    fillRatio = [np.random.rand() for _ in range(data.shape[0])]
    data.insert(data.shape[1], fillNewName, fillRatio)
    return data


def stat_multi_opt(df, visit, distances, t_name, n_trucks = 1, folder_Path = '', ward_name = '', w1 = 0.5, w2 = 0.5, obj_value = 0, t_no = None):

    SPEED = 13.88


    # Initializations
        
    arcs = []
    startNode = [0]
    counts = []
    NTaken = []
    for K in range(n_trucks):
        fillNewName = 'fill_ratio_' + str(K)
        fillpmNewName = 'fill_per_m_' + str(K)
        df = update_fill(df, K)
        objValue, df, active_arcs, NTaken = optimize(df = df, distances = distances, n_trucks = K, w1 = w1, w2 = w2, NTaken = NTaken, folder_Path = folder_Path, ward_name = ward_name, t_name = t_name, t_no = K)

        next_element = next(
                y for x, y in active_arcs if x == 0
            )
        while next_element != 0:
            visit[K].loc[len(visit[K])] = [next_element, df.loc[next_element, fillNewName]]
            df.loc[next_element, [fillNewName, fillpmNewName]] = [0.0, 0.0]
            next_element = next(
                y for x, y in active_arcs if x == int(visit[K].iloc[-1, 0])
            )

        visit[K].loc[len(visit[K])] = [next_element, sum(visit[K].iloc[:, 1])]
        # ----------------------------------------
        print(f'Optimization done for truck {K}')
        # ----------------------------------------
        fileName = folder_Path + 'Visited ' + ward_name + '/visited_' + t_name + '_' + str(K + 1) + '_' + str(w1) + '_' + str(w2) + '.csv'
        visit[K].to_csv(fileName, index = False)
        fileName = folder_Path + ward_name + ' Data/' + t_name + '_multi_' + str(w1) + '_' + str(w2) + '.csv'
        df.to_csv(fileName, index = False)
        
        
    return obj_value
