import glob
import os
import sys
import time
import random
import numpy as np
import math as math
import matplotlib.pyplot as plt
import pandas as pd


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

N = 6                                                                           # Number of vehicles in the Platoon
Vel_max = 55                                                                    # Maximum Velocity of of leader vehicle in KMPH
Spacing = 8                                                                    # Desired spacing between vehicles
AttVeh = 0
K_p = 70
k_d = 20
K_dAtt = -9
world = None
Iter = 1

rel_vel_list = []
vehicle_vel = []
vehicle_data = []
pos_gap_list = []
vehicle2_list = []
acc_list = [17]
class CarEnv:

    def __init__(self):        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)        
        
        # Get the CARLA world and pick the town map
        self.world = self.client.get_world()
        global world
        world = self.world        
        self.world = self.client.load_world('Town06')
        self.spectator = self.world.get_spectator()

        # Pick if you want rendering enabled or not. Uncomment if you wish for a no rendering mode
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

        # Get the blueprint Library
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]       
        
        self.N = N
        self.Vel_max = Vel_max

        

    def reset(self):        
        self.actor_list = []
        self.sensor_list = []
        self.collision_hist = []  
        self.vehicle_data = vehicle_data
        self.vehicle_data = [] 
        self.vehicle_vel = vehicle_vel
        self.vehicle_vel = []   
        self.pos_gap_list = pos_gap_list
        self.pos_gap_list = []
        step = 0

        # Adding randomness to the attack
        self.ranVar  = random.randint(0,2)
        print(self.ranVar) 

        for i in range(self.N):
            # self.transform = carla.Transform(carla.Location(x=380+step, y= -20.0, z=  1.5), carla.Rotation(yaw = -180))
            self.transform = carla.Transform(carla.Location(x=-50-step, y= -20.0, z=  1.5))

            self.actor_role_name = "hero"+str(i)
            self.model_3.set_attribute('role_name', self.actor_role_name)
            
            # Spawn the desired number of vehicles with desired spacing
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)            
            self.actor_list.append(self.vehicle)
            # self.spectator.set_transform(self.transform)
            step += int(Spacing)
            self.vehicle.apply_control(carla.VehicleControl(throttle= 0, steer=0))

            # attach collision sensors to each of the vehicles
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, self.transform, attach_to = self.vehicle)
            self.sensor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event))           
             
             
  
    def collision_data(self, event):
        # self.collision_hist.append(event)
        impulse = event.normal_impulse        
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)                
        # print("Collision when leader vehicle at %d" % self.P)

    def speed(self, actor):
        v = actor.get_velocity()
        speed =   3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2)
        return speed

    def NormalizeData(self, vel, data):
        acc = (vel - np.min(data)) / (np.max(data) - np.min(data))
        return acc

    def action(self, attacker):
        count = 0                   
        
        for idx,vehicle in enumerate(self.actor_list):
            # print(idx,vehicle, "to check iterator",self.actor_list[idx])
            if vehicle.attributes.get('role_name') == "hero0":
                if count == 0:                    
                    vehicle.apply_control(carla.VehicleControl(throttle = 0.65, steer = 0))                    
                count = count+1
                # vehicle.set_velocity(carla.Vector3D(x = -15, y = 0, z = 0))
                self.P  = vehicle.get_location().x
                # print(self.P)
                self.vehicle_data.append(self.P)               
                self.v1 = self.speed(vehicle)
                self.vehicle_vel.append(self.v1)
                # print(vehicle.get_acceleration().x                )
            else:
                
                if attacker == 0:
                    self.P_i = self.actor_list[idx].get_location().x
                    self.vehicle_data.append(self.P_i)
                                        
                    self.v_i = self.speed(self.actor_list[idx])
                    
                    self.vehicle_vel.append(self.v_i)

                    self.P_n = self.actor_list[idx-1].get_location().x                    
                    self.v_n = self.speed(self.actor_list[idx-1])

                    self.pos_gap_list.append(abs(self.P_i - self.P_n)-Spacing)
                    # Condition to check if the vehicle is one at the end
                    if idx != np.amax(idx):                            
                        self.P_p = self.actor_list[idx+1].get_location().x                    
                        self.v_p = self.speed(self.actor_list[idx+1])
                        
                        # Apply control
                        acc_com = K_p*(self.P_n - self.P_i - Spacing) + k_d*(self.v_n - self.v_i) + K_p*(self.P_p - self.P_i + Spacing) + k_d*(self.v_p - self.v_i)
                                           
                    else:
                        acc_com = K_p*(self.P_n - self.P_i - Spacing) + k_d*(self.v_n - self.v_i)
                        # self.spectator.set_transform(carla.Transform(carla.Location(x = self.P_i-10, y = self.actor_list[idx].get_location().y, z = self.actor_list[idx].get_location().z+10)))
                    acc_list.append(acc_com)
                    acc_com_norm = self.NormalizeData(acc_com, acc_list)
                    if math.isnan(acc_com_norm):
                        acc_com_norm = 0.3
                    # print(acc_com_norm)
                    self.actor_list[idx].apply_control(carla.VehicleControl(throttle = acc_com_norm, steer = 0))

                else:
                    
                    if self.P < (100 + self.ranVar):
                        self.P_i = self.actor_list[idx].get_location().x
                        self.vehicle_data.append(self.P_i)
                                            
                        self.v_i = self.speed(self.actor_list[idx])
                        
                        self.vehicle_vel.append(self.v_i)

                        self.P_n = self.actor_list[idx-1].get_location().x                    
                        self.v_n = self.speed(self.actor_list[idx-1])

                        self.pos_gap_list.append(abs(self.P_i - self.P_n)-Spacing)
                        # Condition to check if the vehicle is one at the end
                        if idx != np.amax(idx):                            
                            self.P_p = self.actor_list[idx+1].get_location().x                    
                            self.v_p = self.speed(self.actor_list[idx+1])
                            
                            # Apply control
                            acc_com = K_p*(self.P_n - self.P_i - Spacing) + k_d*(self.v_n - self.v_i) + K_p*(self.P_p - self.P_i + Spacing) + k_d*(self.v_p - self.v_i)
                                            
                        else:
                            acc_com = K_p*(self.P_n - self.P_i - Spacing) + k_d*(self.v_n - self.v_i)
                            # self.spectator.set_transform(carla.Transform(carla.Location(x = self.P_i-10, y = self.actor_list[idx].get_location().y, z = self.actor_list[idx].get_location().z+10)))
                        acc_list.append(acc_com)
                        acc_com_norm = self.NormalizeData(acc_com, acc_list)
                        if math.isnan(acc_com_norm):
                            acc_com_norm = 0.3
                        # print(acc_com_norm)
                        self.actor_list[idx].apply_control(carla.VehicleControl(throttle = acc_com_norm, steer = 0))
                    
                    # Inducing Attack    
                    else:
                        if idx != attacker:
                            self.P_i = self.actor_list[idx].get_location().x
                            self.vehicle_data.append(self.P_i)
                                                
                            self.v_i = self.speed(self.actor_list[idx])
                            
                            self.vehicle_vel.append(self.v_i)

                            if idx == 1:
                                vehicle2_list.append(self.P_i)

                            self.P_n = self.actor_list[idx-1].get_location().x                    
                            self.v_n = self.speed(self.actor_list[idx-1])

                            self.pos_gap_list.append(abs(self.P_i - self.P_n)-Spacing)
                            # Condition to check if the vehicle is one at the end
                            if idx != np.amax(idx):                            
                                self.P_p = self.actor_list[idx+1].get_location().x                    
                                self.v_p = self.speed(self.actor_list[idx+1])
                                
                                # Apply control
                                acc_com = K_p*(self.P_n - self.P_i - Spacing) + k_d*(self.v_n - self.v_i) + K_p*(self.P_p - self.P_i + Spacing) + k_d*(self.v_p - self.v_i)
                                                
                            else:
                                acc_com = K_p*(self.P_n - self.P_i - Spacing) + k_d*(self.v_n - self.v_i)
                                # self.spectator.set_transform(carla.Transform(carla.Location(x = self.P_i-10, y = self.actor_list[idx].get_location().y, z = self.actor_list[idx].get_location().z+10)))
                            acc_list.append(acc_com)
                            acc_com_norm = self.NormalizeData(acc_com, acc_list)
                            if math.isnan(acc_com_norm):
                                acc_com_norm = 0.3
                            # print(acc_com_norm)
                            self.actor_list[idx].apply_control(carla.VehicleControl(throttle = acc_com_norm, steer = 0))
                        elif idx == attacker:
                            # print(idx,attacker)
                            self.P_i = self.actor_list[attacker].get_location().x
                            self.vehicle_data.append(self.P_i)
                                                
                            self.v_i = self.speed(self.actor_list[attacker])
                            
                            self.vehicle_vel.append(self.v_i)

                            self.P_n = self.actor_list[attacker-1].get_location().x                    
                            self.v_n = self.speed(self.actor_list[attacker-1])

                            self.pos_gap_list.append(abs(self.P_i - self.P_n)-Spacing)
                            # Condition to check if the vehicle is one at the end
                            if attacker != np.amax(idx):                            
                                self.P_p = self.actor_list[attacker+1].get_location().x                    
                                self.v_p = self.speed(self.actor_list[attacker+1])
                                
                                # Apply control
                                acc_com = K_p*(self.P_n - self.P_i - Spacing) + K_dAtt*(self.v_n - self.v_i) + K_p*(self.P_p - self.P_i + Spacing) + K_dAtt*(self.v_p - self.v_i)
                                                
                            else:
                                acc_com = K_p*(self.P_n - self.P_i - Spacing) + K_dAtt*(self.v_n - self.v_i)
                                # self.spectator.set_transform(carla.Transform(carla.Location(x = self.P_i-10, y = self.actor_list[idx].get_location().y, z = self.actor_list[idx].get_location().z+10)))
                            acc_list.append(acc_com)
                            acc_com_norm = self.NormalizeData(acc_com, acc_list)
                            if math.isnan(acc_com_norm):
                                acc_com_norm = 0.3
                            # print(acc_com_norm)
                            self.actor_list[attacker].apply_control(carla.VehicleControl(throttle = acc_com_norm, steer = 0))
                            
                        
                    
              

def main():
    data1 = []
    env = CarEnv()
    # collide = False
    try:
        for i in range(Iter):            
            data2 = []            
            env.reset()
            time.sleep(5)

            while True:
                env.action(AttVeh)
                print(len(env.vehicle_vel))
                if len(env.vehicle_vel) > 8000:
                    break
            for actor in env.actor_list:
                actor.destroy()
            for sensor in env.sensor_list:
                sensor.destroy()                    
            print("All actors destroyed!")

            # if len(env.collision_hist) == 0:
            #     print("No collisions! Hurray!")
            #     #     collide = False
            rem = len(env.vehicle_data) % N
            env.vehicle_data = env.vehicle_data[: len(env.vehicle_data) - rem]
            env.vehicle_data = np.array(env.vehicle_data)
            vehicle_data_new = env.vehicle_data.reshape(-1,N)

            env.vehicle_vel = env.vehicle_vel[: len(env.vehicle_vel) - rem]
            env.vehicle_vel = np.array(env.vehicle_vel)
            vehicle_vel_new = env.vehicle_vel.reshape(-1,N)            

            for j in range(np.size(vehicle_data_new,1)):
                data2.append(vehicle_vel_new[3200:3700,j])            
            # print(np.size(data2,1))            
            data1.append(data2)           
                

            rem2 = len(env.pos_gap_list) % (N-1)
            env.pos_gap_list = env.pos_gap_list[: len(env.pos_gap_list) - rem2]
            env.pos_gap_list = np.array(env.pos_gap_list)
            pos_gap_new = env.pos_gap_list.reshape(-1,(N-1))

            
            # plt.figure(1)
            # plt.subplot(311)             # the first subplot in the first figure
            # plt.plot(vehicle_data_new, label = "Vehicle position")        
            # plt.title('Distance')
            # plt.ylabel('Distance')
            
            # plt.subplot(312)             # the first subplot in the first figure
            # plt.plot(vehicle_vel_new, label = "Vehicle velocities")        
            # # plt.title('Speed')
            # plt.ylabel('Vehicle Speed')
            
            # plt.subplot(313)             # the first subplot in the first figure
            # plt.plot(pos_gap_new)        
            # # plt.title('Error')
            # plt.ylabel('Error')
            # plt.show()

            # plt.figure(2)
            # plt.plot(env.collision_hist)
            # plt.title("Collision History")
            # plt.show() 
           
        # data1 = np.array(data1)        
        # str1 = "AbsoluteVelDataAtt" + str(AttVeh)
        # np.save(str1, data1)


    except KeyboardInterrupt:
        pass        

    finally:
        
        plt.figure(1)
        plt.subplot(311)             # the first subplot in the first figure
        plt.plot(vehicle_data_new, label = "Vehicle position")        
        plt.title('Distance')
        plt.ylabel('Distance')
        
        plt.subplot(312)             # the first subplot in the first figure
        plt.plot(vehicle_vel_new, label = "Vehicle velocities")        
        # plt.title('Speed')
        plt.ylabel('Vehicle Speed')
        
        plt.subplot(313)             # the first subplot in the first figure
        plt.plot(pos_gap_new)        
        # plt.title('Error')
        plt.ylabel('Error')
        plt.show()

        # plt.figure(2)
        # plt.plot(env.collision_hist)
        # plt.title("Collision History")
        # plt.show() 



if __name__ == "__main__":
    main()
