# A Simple RealTime PathFinding Robot Based on Implementation of DQN Algoriththm on Xilinx Zynq ARM Cortex-A Hard Processor (My B.Sc. Thesis, Phase 2/3)
+ Adjust settings in config.hpp header file
+ Implementation is done by heavy use of pointers and is not appropriate for processors that support SIMD (which can run much faster using cpp vectors)
+ The code can be tested via UART interface connected to a simulator, or it can be used on a development board (Zybo is used in this project)
+ Do not use ARM Cortex-M Series or things will be messed up due to heavy use of floating-point operations (which are not directly supported on cortex-M on default)
  * ARM Cortex M4F with floating-point mode enabled may be used
+ Pointers are used efficiently so there's no need for garbage collection
  * Design consumes a very small portion of RAM, sth between 10KB to 1MB (depending on neural network hyperparameters)
