Experiment with io_uring and compute shaders.</br>
Generally, the program generates vectors for subsequent dot product calculation on the GPU.</br>
Vector generation imitates receiving data from somewhere. It was made intentionally in a slow way.</br>
Processes run in a separate threads synchronizing by io_uring.</br>
For interaction with GPU the vulkano lib was used. It allows you to use low-level constructs </br>
while remaining quite high-level. The timeline semafore was used for waiting completion signal</br>
from GPU.</br>
General scheme of application:</br>

|vector generation| -> |waiting two vector| -> |running compute shader| -> |waiting semafore| -></br>

|receiving message in main ring|

