from show_routes import CreateMap
B_TO_B = 100
B_TO_T = 10
N_WARDS = 3
N_TRUCKS = 3
W1 = 0.9
W2 = 0.1
map = CreateMap()
map.createRoutes('Data/Static Data/3 Truck/', N_WARDS, N_TRUCKS, W1, W2, Multiple_truck = True)
map.createLatLong('Data/Bin Locations.csv', N_WARDS)
map.createRoutesDict(N_WARDS)
map.addRoutesToMap(N_WARDS, N_TRUCKS)
map.addDepot()
map.addNodes('Data/Bin Locations.csv')
map.saveMap('Data/Static Data/3 Truck/')
map.displayMap('Data/Static Data/3 Truck/')