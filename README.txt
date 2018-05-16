Realistic Flame Demo

Uses CUDA to update an approximation of Navier-Stokes each physics step
Renders using a particle engine

Controls:
z/x        - zoom in/out
arrow keys - rotate camera angles
m          - toggle per-frame/constant-timestep updates
g          - toggle cpu/gpu computation


Steps in fluid simulation:
Advect (follow velocities backward in time to update fluid properties)
Bouyancy (update vertical velocities based on temperature)
Jacobi Iteration (iterative solver to find pressure values)
Update Velocities (subtract pressure gradient)
Update partcle positions

I was able to write the important cuda functions, but I couldn't overcome the hurdles of initializing the cuda texture objects and target surfaces. The math is all there, but the objects they're supposed to work on aren't. The way I implemented it made it more difficult to write a cpu version alongside it, but not doing that was also a mistake, because now I don't have any kind of working implementation.
If nothing else, I will probably have time at the end of this week or after finals are over to figure out what I'm doing wrong. Graded or not, I want to make this work.
