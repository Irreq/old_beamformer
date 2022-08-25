# Configuration
fs = 44100               # Samplingrate
window = fs // 8         # Chunks of data to be processed
propagation_speed = 340  # How fast the signals travel
distance = 0.2           # Distance between the elements in meters
columns = 8              # Number of columns
rows = 8                 # Number of rows
position = [0, 0, 0]     # Coordinate position of origin of array
yaw = 0                  # Listen up/down
pitch = 0                # Listen right/left
coefficients_path = "data/coefficients.txt"