256 0.01 0.02 0.01 120.0 200 1e-9
g++ -O3 -std=c++17 cpp_serial.cpp -o sim_serial

./sim_serial 256 1.0 0.02 0.005 100

# Run on Core 0 with maximum priority (-20)
sudo nice -n -20 taskset -c 0 ./sim_serial 256

# Run on Core 0 only
taskset -c 0 ./sim_serial 256

g++ -O3 -std=c++17 -ffast-math simulation_serial.cpp -o sim_serial