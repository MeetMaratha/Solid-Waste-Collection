from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

# Constants
B_TO_T = 10 # Bin to Truck
B_TO_B = 100 # Bin to Bin

def update_fill(data, m):
    fillPrevName = 'fill_ratio_' + str(m - 1)
    fillpmNewName = 'fill_per_m_' + str(m)
    fillNewName = 'fill_ratio_' + str(m)
    if m == 0:
        fillRatio = [0.0] + [np.random.rand() for _ in range(data.shape[0] - 1)]
    else:
        fillRatio = data.loc[:, fillPrevName].tolist()
        for k in range(len(fillRatio)):
            if fillRatio[k] != 0.0 and np.random.rand() < 0.80:
                fillRatio[k] = fillRatio[k] + np.random.uniform(0, 1 - fillRatio[k]) / 10
    data.insert(data.shape[1], fillNewName, fillRatio)
    # data[fillNewName] = fillRatio
    return data

def calc_V(N, m, st):
    if m == 0:
        V = N + [0]
    else:
        V = [st] + N + [0]
    return V


def dyn_multi_opt(df, visit, distances, t_name, n_done, n_trucks = 1, folder_Path = '', ward_name = '', w1 = 0.5, w2 = 0.5, m = 0, obj_value = []):
    SPEED = 13.88
    TIME = 900
    # Model
    mdl = Model('CVRP')

    # Initializations

    startNode = []
    objs = []
    Ns = []
    Vs = []
    As = []
    Us = []
    Xs = []
    Ys = []
    Cs = []
    (visit1, visit2, visit3, visit4, visit5) = (None, None, None, None, None)
    visits = [visit1, visit2, visit3, visit4, visit5]
    visitedNodes = []
    NTaken = []

    for i in range(n_trucks):
        visits[i] = visit[i]
        visits[i].Node = visits[i].Node.astype('int')
        startNode.append(visits[i].iloc[-1, 0])
        objs.append(0)
        for j in visits[i].Node.tolist():
            visitedNodes.append(j)
    
    fillPrevName = 'fill_ratio_' + str(m - 1)
    fillpmNewName = 'fill_per_m_' + str(m)
    fillNewName = 'fill_ratio_' + str(m)
    fills = []
    

    #  ---------------------------------
    print(f"Start Nodes : {startNode}\nFill Ratios : {[sum(visits[z].iloc[:, 1]) for z in range(n_trucks)]}")
    # -------------------------------------

    df = update_fill(df, m)
    for k in range(len(n_done)):
        Ns.append(None)
        Vs.append(None)
        As.append(None)
        Cs.append(None)
        Xs.append(None)
        Ys.append(None)
        Us.append(None)
        fills.append(None)
        if n_done[k] == 0:
            dist = distances.iloc[startNode[k], df.index.tolist()].tolist()
            distName = 'distance_from_' + str(startNode[k]) + '_' + str(k)
            df.insert(df.shape[1], distName, dist)
            # df[distName] = dist
            fillpmNewName1 = fillpmNewName + '_' + str(startNode[k]) + '_' + str(k)
            df.insert(df.shape[1], fillpmNewName1, B_TO_B * df.loc[:, fillNewName] / df.loc[:, distName])
            # df[fillpmNewName1] = B_TO_B * df.loc[:, fillNewName] / df.loc[:, distName]
            df.sort_values(by = fillpmNewName1, ascending = False)
            fills[k] = pd.DataFrame(
                    {'fill' : df.loc[:, fillNewName].tolist() + [0.0]}, index = df.index.tolist() + [0]
                    )
            N = []
            for i in df.index.tolist():
                if i not in visitedNodes and i not in NTaken and i != 0 and ( df.loc[i, fillNewName] + sum(df.loc[N, fillNewName]) ) * B_TO_T <= 100 - sum(visits[k].iloc[:, 1]) * B_TO_T:
                    N.append(i)
                    NTaken.append(i)
            Ns[k] = N
            Vs[k] = calc_V(N, m, startNode[k])
            As[k] = [(p, q) for p in Vs[k] for q in Vs[k] if p != q]
            Cs[k] = {(p, q) : distances.iloc[p, q] for p,q in As[k]}
            Xs[k] = mdl.addVars(As[k], vtype = GRB.BINARY)
            Ys[k] = mdl.addVars(Vs[k], vtype = GRB.BINARY)
            Us[k] = mdl.addVars(Ns[k], vtype = GRB.CONTINUOUS)
            # print(Ns[k])
            objs[k] = quicksum( 
                    (w1 * Xs[k][p, q] * Cs[k][(p, q)]) - (w2 * Ys[k][p] * fills[k].loc[p, 'fill'] * B_TO_T) for p,q in As[k]
                    )

    # Optimization Function defination

    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(sum(objs))

    # Constraints

    for k in range(len(n_done)):
        if n_done[k] == 0:
            mdl.addConstrs(
                quicksum( Xs[k][i, j] for j in Vs[k] if j != i ) == 1 for i in Ns[k]
            )
            mdl.addConstrs(
                quicksum( Xs[k][i, j] for i in Vs[k] if i != j ) == 1 for j in Ns[k]
            )
            mdl.addConstr(
                quicksum( Ys[k][i] * fills[k].loc[i, 'fill'] * B_TO_T for i in Ns[k] ) <= ( 100 - sum(visits[k].iloc[:, 1]) * B_TO_T )
            )
            mdl.addConstr(
                quicksum( Xs[k][startNode[k], j] for j in Ns[k]) == 1
            )
            mdl.addConstr(
                quicksum( Xs[k][j, 0] for j in Ns[k] ) == 1
            )
            if startNode[k] != 0:
                mdl.addConstr(
                    quicksum( Xs[k][0, j] for j in Ns[k]) == 0
                    )
            if startNode[k] != 0:
                    mdl.addConstr(
                        quicksum( Xs[k][j, startNode[k]] for j in Ns[k]) == 0
                    )
            mdl.addConstrs(
                (Xs[k][i, j] == 1) >> (Us[k][i] + fills[k].loc[i, 'fill'] * B_TO_T == Us[k][j]) for i,j in As[k] if i not in [0, visits[k].iloc[-1, 0]] and j not in [0, visits[k].iloc[-1, 0]]
            )
            a = mdl.addVars(Ns[k], vtype = GRB.BINARY)
            b = mdl.addVars(Ns[k], vtype =  GRB.BINARY)
            y = mdl.addVars(Ns[k], vtype =  GRB.BINARY)

            mdl.addConstrs(
                Us[k][i] >= (fills[k].loc[i, 'fill'] * B_TO_T) for i in Ns[k]
            )
            mdl.addConstrs(Us[k][i] <= (100 - np.sum(visits[k].iloc[:, 1] * B_TO_T)) for i in Ns[k])
    
    # Model Restrictions

    mdl.Params.MIPGap = 0.1
    mdl.Params.TIMELimit = 900

    # Optimization

    mdl.optimize()
    obj_value.append(mdl.getObjective().getValue())

    # Time simulation

    for k in range(len(n_done)):
        if n_done[k] == 0:
            TIME = 900
            arcs = [a for a in As[k] if Xs[k][a].x > 0.99]
            imp = 1
            next_element = next(
                y for x, y in arcs if x == visits[k].iloc[-1 ,0]
            )
            unnormalize = np.max(pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1).iloc[startNode[k], :])
            while TIME - (unnormalize * Cs[k][visits[k].iloc[-1, 0], next_element]) / SPEED >= 0 and next_element != 0:
                TIME -= (unnormalize * Cs[k][visits[k].iloc[-1, 0], next_element]) / SPEED
                visits[k].loc[len(visits[k])] = [next_element, df.loc[next_element, fillNewName]]
                df.loc[next_element, [fillNewName, fillpmNewName + '_' + str(startNode[k])]] = [0.0, 0.0]
                next_element = next(
                    y for x, y in arcs if x == visits[k].iloc[-1 ,0]
                )
                imp = 0
            if imp == 1 and next_element != 0:

                # ------------------------------------
                print('Forcefully entering value.')
                # ------------------------------------
                TIME = 0
                visits[k].loc[len(visits[k])] = [next_element, df.loc[next_element, fillNewName]]
                df.loc[next_element, [fillNewName, fillpmNewName + '_' + str(startNode[k])]] = [0.0, 0.0]
                next_element = next(
                    y for x, y in arcs if x == visits[k].iloc[-1 ,0]
                )
            
            if next_element == 0:
                visits[k].loc[len(visits[k])] = [next_element, sum(visits[k].iloc[:, 1])]

                # ----------------------------------------
                print(f'Optimization done for truck {k}')
                # ----------------------------------------

                fileName = folder_Path + 'Visited ' + ward_name + '/visited_' + t_name + '_' + str(k + 1) + '_' + str(w1) + '_' + str(w2) + '.csv'
                visits[k].to_csv(fileName, index = False)
                n_done[k] = 1
            # ----------------------------------------
            print(f"Start Node : {startNode[k]}")
            print(f"Active Arcs : {arcs}")
            # ----------------------------------------

    print(f"Done Status : {n_done}")
    m += 1
    if n_done == [1] * len(n_done):
        
        # -----------------------------------------------
        print('Done Computation')
        # -----------------------------------------------

        fileName = folder_Path + ward_name + ' Data/' + t_name + '_multi_' + str(w1) + '_' + str(w2) + '.csv'
        df.to_csv(fileName, index = False)
        return obj_value
    
    # Recursive call
    
    dyn_multi_opt(df =df, visit = visits, distances = distances, t_name = t_name, n_done = n_done, w1 = w1, w2 = w2, n_trucks = n_trucks, folder_Path = folder_Path, ward_name = ward_name, obj_value = obj_value, m = m)
    return obj_value
        
