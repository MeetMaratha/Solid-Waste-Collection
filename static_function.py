from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

# Constants
B_TO_T = 10 # Bin to Truck
B_TO_B = 100 # Bin to Bin

# Static Optimization function
def opt(df, distances, w1 = 0.2, w2 = 0.8):
  df1 = df[df.Ward == 0].sort_values(by = ['fill_p_m'], ascending = False) 
  df2 = df[df.Ward == 1].sort_values(by = ['fill_p_m'], ascending = False) 
  df3 = df[df.Ward == 2].sort_values(by = ['fill_p_m'], ascending = False)
  fill_ratio = df.fill_ratio
  mdl = Model('CVRP')
  N = []
  for i in df1.index.to_list():
    if sum(df1.loc[N, 'fill_ratio']*B_TO_T) + df1.loc[i, 'fill_ratio']*B_TO_T <= 100:
      N.append(i)

  N1 = []
  for i in df2.index.to_list():
    if sum(df2.loc[N1, 'fill_ratio']*B_TO_T) + df2.loc[i, 'fill_ratio']*B_TO_T <= 100:
      N1.append(i)
  
  N2 = []
  for i in df3.index.to_list():
    if sum(df3.loc[N2, 'fill_ratio']*B_TO_T) + df3.loc[i, 'fill_ratio']*B_TO_T <= 100:
      N2.append(i)

  V = [0] + N                                                                                   # All the vertices
  V1 = [0] + N1                                                                                   # All the vertices
  V2 = [0] + N2                                                                                   # All the vertices
  A = [(i, j) for i in V for j in V if i != j]                                                  # Arcs
  A1 = [(i, j) for i in V1 for j in V1 if i != j]                                                  # Arcs
  A2 = [(i, j) for i in V2 for j in V2 if i != j]                                                  # Arcs
  c = {(i,j) : distances.iloc[i,j] for i,j in A}                                                # cost
  c1 = {(i,j) : distances.iloc[i,j] for i,j in A1}                                                # cost
  c2 = {(i,j) : distances.iloc[i,j] for i,j in A2}                                                # cost
  # Starting Gurobi Model

  x = mdl.addVars(A, vtype = GRB.BINARY)                                                        # X(ij)
  x1 = mdl.addVars(A1, vtype = GRB.BINARY)                                                        # X(ij)
  x2 = mdl.addVars(A2, vtype = GRB.BINARY)                                                        # X(ij)
  y = mdl.addVars(V, vtype = GRB.BINARY)                                                        # Y(i)
  y1 = mdl.addVars(V1, vtype = GRB.BINARY)                                                        # Y(i)
  y2 = mdl.addVars(V2, vtype = GRB.BINARY)                                                        # Y(i)
  u = mdl.addVars(N, vtype = GRB.CONTINUOUS)                                                    # u(i)
  u1 = mdl.addVars(N1, vtype = GRB.CONTINUOUS)                                                    # u(i)
  u2 = mdl.addVars(N2, vtype = GRB.CONTINUOUS)                                                    # u(i)
  mdl.modelSense = GRB.MINIMIZE                                                                 # Minimization model


  mdl.setObjective(quicksum(w1*x[i,j]*c[i,j] - w2*y[i]*fill_ratio.loc[i]*B_TO_T for i,j in A) + quicksum(w1*x1[i,j]*c1[i,j] - w2*y1[i]*fill_ratio.loc[i]*B_TO_T for i,j in A1) + quicksum(w1*x2[i,j]*c2[i,j] - w2*y2[i]*fill_ratio.loc[i]*B_TO_T for i,j in A2))


  # Constraints

  #start at 0
  mdl.addConstr( quicksum(x[0,j] for j in N) == 1 )
  mdl.addConstr( quicksum(x1[0,j] for j in N1) == 1 )
  mdl.addConstr( quicksum(x2[0,j] for j in N2) == 1 )

  #End at 0
  mdl.addConstr( quicksum(x[j,0] for j in N) == 1 )
  mdl.addConstr( quicksum(x1[j,0] for j in N1) == 1 )
  mdl.addConstr( quicksum(x2[j,0] for j in N2) == 1 )

  # Updation of route
  mdl.addConstrs(quicksum(x[i,j] for j in V if j != i) == 1 for i in N)                       
  mdl.addConstrs(quicksum(x1[i,j] for j in V1 if j != i) == 1 for i in N1)                    
  mdl.addConstrs(quicksum(x2[i,j] for j in V2 if j != i) == 1 for i in N2)                    
  
  mdl.addConstrs(quicksum(x[i,j] for i in V if i != j) == 1 for j in N)                       
  mdl.addConstrs(quicksum(x1[i,j] for i in V1 if i != j) == 1 for j in N1)                    
  mdl.addConstrs(quicksum(x2[i,j] for i in V2 if i != j) == 1 for j in N2)                    
  
  # Capacity constraint
  mdl.addConstr(quicksum(y[i]*fill_ratio.loc[i]*B_TO_T for i in N) <= 100)                          
  mdl.addConstr(quicksum(y1[i]*fill_ratio.loc[i]*B_TO_T for i in N1) <= 100)                       
  mdl.addConstr(quicksum(y2[i]*fill_ratio.loc[i]*B_TO_T for i in N2) <= 100)                       
 
  # Tracking fill ratio
  mdl.addConstrs((x[i,j] == 1) >> (u[i] + fill_ratio.loc[i]*B_TO_T == u[j])
                  for i,j in A if i != 0 and j!= 0)                                            
  
  mdl.addConstrs((x1[i,j] == 1) >> (u1[i] + fill_ratio.loc[i]*B_TO_T == u1[j])
                  for i,j in A1 if i != 0 and j!= 0)                                           
  
  mdl.addConstrs((x2[i,j] == 1) >> (u2[i] + fill_ratio.loc[i]*B_TO_T == u2[j])
                  for i,j in A2 if i != 0 and j!= 0)                                           
  
  
  mdl.addConstrs(u[i] >= fill_ratio.loc[i]*B_TO_T for i in N)                                  
  mdl.addConstrs(u1[i] >= fill_ratio.loc[i]*B_TO_T for i in N1)                                
  mdl.addConstrs(u2[i] >= fill_ratio.loc[i]*B_TO_T for i in N2)                                
  
  
  mdl.addConstrs(u[i] <= 100 for i in N)                                                   
  mdl.addConstrs(u1[i] <= 100 for i in N1)                                                 
  mdl.addConstrs(u2[i] <= 100 for i in N2)                                                 
  
  # Time constraint to make model feasible to use
  mdl.Params.MIGap = 0.1                                                                        # Stop at < 10% gap
  mdl.Params.TimeLimit = 900                                                                    # Stop after 900s (15 minutes)
  
  # Optimize the model

  mdl.optimize()

#   q = mdl.getObjective()
  obj_value = mdl.getObjective().getValue()
  active_arcs = [a for a in A if x[a].x > 0.99]                                                 # Store the arcs in the list active_arcs
  active_arcs1 = [a for a in A1 if x1[a].x > 0.99]                                              # Store the arcs in the list active_arcs
  active_arcs2 = [a for a in A2 if x2[a].x > 0.99]                                              # Store the arcs in the list active_arcs
  return [active_arcs, active_arcs1, active_arcs2], obj_value                                              # Return those arcs                                           # Return those arcs 

