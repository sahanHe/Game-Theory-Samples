# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions
import random

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
ctrl_vehicle_ids = ["t_y", "t_x"]
depart_delays = {vid: random.uniform(0, 5) for vid in ctrl_vehicle_ids}
print(ctrl_vehicle_ids)
print(depart_delays)


Sumo_config = [
    'sumo-gui',
    '-c', 'GT_crossing.sumocfg',
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define Variables
vehicle_speed = 0
total_speed = 0

# Step 7: Define Functions

def within_ctzone(vid, edge, position_on_edge):

    if edge in ["-E4", "-E0"]:
        return position_on_edge >120
    else:
        return position_on_edge <30




# Step 8: Take simulation steps until there are no more vehicles in the network
# while traci.simulation.getMinExpectedNumber() > 0:
#     traci.simulationStep() # Move simulation forward 1 step
#     # Here you can decide what to do with simulation data at each step
#     vehicle_ids = traci.vehicle.getIDList()
#     print(traci.vehicle.getIDList())
#     for vid in vehicle_ids:
#         x, y = traci.vehicle.getPosition(vid)
#         print()
#         print(f"Vehicle {vid} at position x={x}, y={y}")

delay_set = {}

for veh_id in ctrl_vehicle_ids:
    delay_set[veh_id] = True


while traci.simulation.getMinExpectedNumber() > 0:
    sim_time = traci.simulation.getTime()
    traci.simulationStep()

    

    # vehicle_ids = traci.vehicle.getIDList()
    
    for vid in list(traci.vehicle.getIDList()):

        if delay_set[vid]:
            print(delay_set[vid])
            if vid in depart_delays and sim_time < depart_delays[vid]:
                # Freeze the vehicle until its time
                traci.vehicle.setSpeed(vid, 0.0)
            else:
                # Allow it to move
                traci.vehicle.setSpeed(vid, 20)  # default behavior
                traci.vehicle.setSpeedMode(vid, 0b00000)
                delay_set[vid] = False
        else:
            print(delay_set[vid])
            route = traci.vehicle.getRoute(vid)                     # List of edge IDs in the route
            current_edge = traci.vehicle.getRoadID(vid)             # Current edge ID
            position_on_edge = traci.vehicle.getLanePosition(vid)   # Position along the lane (in meters)
            route_index = route.index(current_edge) if current_edge in route else -1
            ct_plus = within_ctzone(vid, current_edge, position_on_edge)
            # print(f"Vehicle {vid}: on edge {current_edge}, is in ctrl zone : {ct_plus}")




# Step 9: Close connection between SUMO and Traci
traci.close()