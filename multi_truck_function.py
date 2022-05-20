from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

# Constants
B_TO_T = 10 # Bin to Truck
B_TO_B = 100 # Bin to Bin



def dyn_multi_opt(df, visit1, visit2, folder_Path, t_name, ward_name, distances, w1 = 1.0, w2 = 0.0, m = 0, n1_done = 0, n2_done = 0, obj_value = []):
    SPEED = 13.88
    # Model
    mdl = Model('CVRP')

    visit1.Node = visit1.Node.astype('int')
    visit2.Node = visit2.Node.astype('int')
    

    # Initalization
    obj1, obj2 = 0, 0
    f_prev = 'fill_ratio_' + str(m-1)
    fpm = 'fill_per_m_' + str(m)
    f_new = 'fill_ratio_' + str(m)
    start_node = [
        visit1.iloc[-1,0],
        visit2.iloc[-1,0]
        ]
    print(f'\nStart nodes : {start_node}\nFill Ratios : {sum(visit1.iloc[:,1])} | {sum(visit2.iloc[:,1])}')
    if m == 0:
        fill1 = df.fill_ratio
    else:
        fill1 = df.loc[:, f_prev]
    df.insert(df.shape[1], f_new, fill1)
    if m != 0:
        for i in df.index.to_list():
            if i not in visit1.Node.to_list() and i not in visit2.Node.to_list() and np.random.rand() < 0.80:
                df.loc[i, f_new] = df.loc[i, f_new] + np.random.uniform(0, 1 - df.loc[i, f_new])/10
            else:
                df.loc[i, f_new] = df.loc[i, f_new]
    if n1_done != 1:
        # df.insert(df.shape[1], f_new, fill1)
        # if m != 0:
        #     for i in df.index.tolist():
        #         if i not in visit1.Node.to_list() and np.random.rand() < 0.80:
        #             df.loc[i, f_new] = df.loc[i, f_new] + np.random.uniform(0, 1 - df.loc[i, f_new])/10
        dist = distances.iloc[start_node[0], df.index.to_list()].tolist()
        dist_name = 'distance_from_' + str(start_node[0])
        if dist_name in df.columns:
            dist_name = dist_name + '_' + str(np.random.rand())
        fill_1 = pd.DataFrame({'Node' : df.index.tolist() + [0], 'fill' : df.loc[:, f_new].to_list() + [0]})
        df.insert(df.shape[1], dist_name, dist)
        df.insert(df.shape[1], fpm, B_TO_B*df.loc[:,f_new]/df.loc[:,dist_name])
        df.sort_values(by = f_new, ascending = False)
        N1 = []
        for i in df.index.tolist():
            if i not in visit1.Node.to_list() and i not in visit2.Node.tolist() and (df.loc[i, f_new] + sum(df.loc[N1, f_new]))*B_TO_T <= 100 - sum(visit1.iloc[:,1])*B_TO_T:
                N1.append(i)
        if m == 0:
            V1 = N1 + [0]
        else:
            V1 = [start_node[0]] + N1 + [0]
        A1 = [(i,j) for i in V1 for j in V1 if i != j]
        c1 = {(i,j) : distances.iloc[i, j] for i,j in A1}
        x1 = mdl.addVars(A1, vtype = GRB.BINARY)
        y1 = mdl.addVars(V1, vtype = GRB.BINARY)
        u1 = mdl.addVars(N1, vtype = GRB.CONTINUOUS)
        obj1 = quicksum( (w1*x1[i, j]*c1[(i, j)]) - w2*y1[i]*fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T for i,j in A1)

    if n2_done != 1 and n1_done != 1:
        N2 = []
        for i in df.index.tolist():
            if i not in visit2.Node.to_list() and i not in visit1.Node.tolist() and i not in N1 and (df.loc[i, f_new] + sum(df.loc[N2, f_new]))*B_TO_T <= 100 - sum(visit2.iloc[:,1])*B_TO_T:
                N2.append(i)
        if m == 0:
            V2 = N2 + [0]
        else:
            V2 = [start_node[1]] + N2 + [0]
        if V2 != []:
            A2 = [(i,j) for i in V2 for j in V2 if i != j]
            c2 = {(i,j) : distances.iloc[i, j] for i,j in A2}
            x2 = mdl.addVars(A2, vtype = GRB.BINARY)
            y2 = mdl.addVars(V2, vtype = GRB.BINARY)
            u2 = mdl.addVars(N2, vtype = GRB.CONTINUOUS)
            obj2 = quicksum( (w1*x2[i, j]*c2[(i, j)]) - w2*y2[i]*fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T for i,j in A2)
    elif n2_done != 1 and n1_done == 1:
        # df.insert(df.shape[1], f_new, fill1)
        # if m != 0:
        #     for i in df.index.tolist():
        #         if i not in visit1.Node.to_list() and np.random.rand() < 0.80:
        #             df.loc[i, f_new] = df.loc[i, f_new] + np.random.uniform(0, 1 - df.loc[i, f_new])/10
        dist = distances.iloc[start_node[1], df.index.to_list()].tolist()
        dist_name = 'distance_from_' + str(start_node[1])
        if dist_name in df.columns:
            dist_name = dist_name + '_' + str(np.random.rand())
        fill_1 = pd.DataFrame({'Node' : df.index.tolist() + [0], 'fill' : df.loc[:, f_new].to_list() + [0]})
        df.insert(df.shape[1], dist_name, dist)
        df.insert(df.shape[1], fpm, B_TO_B*df.loc[:,f_new]/df.loc[:,dist_name])
        df.sort_values(by = f_new, ascending = False)
        N2 = []
        for i in df.index.tolist():
            if i not in visit2.Node.to_list() and i not in visit1.Node.tolist() and i not in N1 and (df.loc[i, f_new] + sum(df.loc[N2, f_new]))*B_TO_T <= 100 - sum(visit2.iloc[:,1])*B_TO_T:
                N2.append(i)
        if m == 0:
            V2 = N2 + [0]
        else:
            V2 = [start_node[1]] + N2 + [0]
        if V2 != []:
            A2 = [(i,j) for i in V2 for j in V2 if i != j]
            c2 = {(i,j) : distances.iloc[i, j] for i,j in A2}
            x2 = mdl.addVars(A2, vtype = GRB.BINARY)
            y2 = mdl.addVars(V2, vtype = GRB.BINARY)
            u2 = mdl.addVars(N2, vtype = GRB.CONTINUOUS)
            obj2 = quicksum( (w1*x2[i, j]*c2[(i, j)]) - w2*y2[i]*fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T for i,j in A2)
        
    # Model
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(obj1 + obj2)

    # Constraints
    if n1_done == 0:
        mdl.addConstrs( quicksum( x1[i,j] for j in V1 if j != i) == 1 for i in N1 )
        mdl.addConstrs( quicksum( x1[i,j] for i in V1 if i != j) == 1 for j in N1 )
        mdl.addConstr( quicksum( y1[i]*fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T for i in N1 ) <= (100 - sum(visit1.iloc[:, 1])*B_TO_T) )
        mdl.addConstr( quicksum( x1[start_node[0], j] for j in N1) == 1)
        mdl.addConstr( quicksum( x1[j, 0] for j in N1) == 1)
        if start_node[0] != 0:
            mdl.addConstr( quicksum( x1[0, j] for j in N1) == 0)
        if start_node[0] != 0:
                    mdl.addConstr( quicksum( x1[j, start_node[0]] for j in N1) == 0)
        mdl.addConstrs(
        (x1[i,j] == 1) >> (u1[i] + fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T == u1[j]) for i,j in A1 if i != 0 and j != 0 and i != int(visit1.iloc[-1,0]) and j != int(visit1.iloc[-1,0]) 
        )
        mdl.addConstrs( u1[i] >= fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T for i in N1 )
        mdl.addConstrs( u1[i] <= 100 - sum(visit1.iloc[:, 1])*B_TO_T for i in N1 )
    
    if n2_done == 0:
        mdl.addConstrs( quicksum( x2[i,j] for j in V2 if j != i) == 1 for i in N2 )
        mdl.addConstrs( quicksum( x2[i,j] for i in V2 if i != j) == 1 for j in N2 )
        mdl.addConstr( quicksum( y2[i]*fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T for i in N2 ) <= (100 - sum(visit2.iloc[:, 1])*B_TO_T) )
        mdl.addConstr( quicksum( x2[start_node[1], j] for j in N2) == 1)
        mdl.addConstr( quicksum( x2[j, 0] for j in N2) == 1)
        if start_node[1] != 0:
            mdl.addConstr( quicksum( x2[0, j] for j in N2) == 0)
        if start_node[1] != 0:
                    mdl.addConstr( quicksum( x2[j, start_node[1]] for j in N2) == 0)
        mdl.addConstrs(
        (x2[i,j] == 1) >> (u2[i] + fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T == u2[j]) for i,j in A2 if i != 0 and j != 0 and i != int(visit2.iloc[-1,0]) and j != int(visit2.iloc[-1,0]) 
        )
        mdl.addConstrs( u2[i] >= fill_1[fill_1.Node == i].iloc[0, 1]*B_TO_T for i in N2 )
        mdl.addConstrs( u2[i] <= 100 - sum(visit2.iloc[:, 1])*B_TO_T for i in N2 )
    
    # Model TIME Restrictions

    mdl.Params.MIPGap = 0.1
    mdl.Params.TIMELimit = 900

    # Optimize model
    mdl.optimize()
    q = mdl.getObjective()
    obj_value.append(q.getValue())

    # TODO : TIME simulation
    if n1_done == 0:
        active_arcs1 = [a for a in A1 if x1[a].x > 0.99]
        TIME = 900 # 15 minutes
        visited1 = 0
        next_element = next( y for x, y in active_arcs1 if x == visit1.iloc[-1, 0] )
        temp = np.max(pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1).iloc[start_node[0], :])
        while (TIME - temp * c1[visit1.iloc[-1, 0], next_element]/(SPEED) >= 0) and next_element != 0:
            TIME = TIME - temp * c1[visit1.iloc[-1, 0], next_element]/(SPEED)
            visit1.loc[len(visit1)] = [next_element, df.loc[next_element, f_new]]
            df.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs1 if x == visit1.iloc[-1, 0] )
            visited1 = visited1 + 1
        if visited1 == 0:
            print('Forecully entered the value 2.')
            visit1.loc[len(visit1)] = [next_element, df.loc[next_element, f_new]]
            df.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs1 if x == visit1.iloc[-1, 0] )
        if next_element == 0:
            visit1.loc[len(visit1)] = [next_element, sum(visit1.iloc[:, 1])]
            print(f'\nOptimization done for truck 1 - 1')
            file_name = folder_Path + 'Visited ' + ward_name + '/visited_'+ t_name + '_1_' + str(w1) + '_' + str(w2) + '.csv'
            visit1.to_csv(file_name, index = False)
            n1_done = 1
        print(f'Active arcs | Truck 1 | Start Node : {start_node[0]} :\n{active_arcs1}')

    if n2_done == 0:
        active_arcs2 = [a for a in A2 if x2[a].x > 0.99]
        TIME = 900 # 15 minutes
        visited2 = 0
        next_element = next( y for x, y in active_arcs2 if x == visit2.iloc[-1, 0] )
        temp = np.max(pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1).iloc[start_node[1], :])
        while (TIME - temp * c2[visit2.iloc[-1, 0], next_element]/(SPEED) >= 0) and next_element != 0:
            TIME = TIME - temp * c2[visit2.iloc[-1, 0], next_element]/(SPEED)
            visit2.loc[len(visit2)] = [next_element, df.loc[next_element, f_new]]
            df.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs2 if x == visit2.iloc[-1, 0] )
            visited2 = visited2 + 1
        if visited2 == 0:
            print('Forecully entered the value 2.')
            visit2.loc[len(visit2.index)] = [next_element, df.loc[next_element, f_new]]
            df.loc[next_element, [f_new, fpm]] = [0.0, 0.0]
            next_element = next( y for x, y in active_arcs2 if x == visit2.iloc[-1, 0] )
        if next_element == 0:
            visit2.loc[len(visit2)] = [next_element, sum(visit2.iloc[:, 1])]
            print(f'\nOptimization done for truck 1 - 2')
            file_name = folder_Path + 'Visited ' + ward_name + '/visited_'+ t_name + '_2_' + str(w1) + '_' + str(w2) + '.csv'
            visit2.to_csv(file_name, index = False)
            n2_done = 1
        print(f'Active arcs | Truck 2 | Start Node : {start_node[1]} :\n{active_arcs2}')

    print(f'\n{n1_done, n2_done}')
    m = m + 1
    if n1_done == 1 and n2_done == 1:
        print('\nDone computation')
        file_name = folder_Path + ward_name + ' Data/' + t_name + '_multi_' + str(w1) + '_' + str(w2) + '.csv'
        df.to_csv(file_name, index = False)
        return obj_value
    dyn_multi_opt(df = df, distances = distances, visit1 = visit1, visit2 = visit2, folder_Path = folder_Path, t_name = t_name, ward_name = ward_name, m = m, w1 = w1, w2 = w2, n1_done = n1_done, n2_done = n2_done, obj_value = obj_value)
    return obj_value