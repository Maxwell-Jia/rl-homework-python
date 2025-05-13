# Assuming one of your packages in src is named 'my_actual_package_name_1'
import stable_baselines3
print(stable_baselines3.__file__)

# Assuming the other is 'my_actual_package_name_2'
import rl_zoo3
print(rl_zoo3.__file__)