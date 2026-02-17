# Linux / macOS (GCC/Clang)
g++ -O3 -std=c++17 -fopenmp simulation_cpu.cpp -o sim_cpu

# Run
./sim_cpu 256

Argument List (in order)Only the first argument (N) is mandatory. The rest are optional and will default to the values defined in the code if omitted.PositionVariableTypeDescriptionDefault (if omitted)1NintGrid size (Required)None2ISFdoubleInteraction Scale Factor1.0003PSIdoubleAvg RBC volume fraction0.024ITdoubleTime increment (dt)0.0055TdoubleTotal simulation time1200.06NOdoubleOutput interval (steps)10007DdoubleDiffusion coefficient0.0Usage Examples1. Minimal (Just Grid Size):Runs with $N=256$ and all other parameters at default.Bash./sim_cpu 256
2. Custom Physics (Grid + Physics):Runs with $N=512$, $ISF=1.0$, $PSI=0.05$, $dt=0.001$, TotalTime=500.0.Bash./sim_cpu 512 1.0 0.05 0.001 500.0
3. Full Configuration:Explicitly sets every parameter, including diffusion.Bash./sim_cpu 256 1.0 0.02 0.005 1200.0 500 1e-9