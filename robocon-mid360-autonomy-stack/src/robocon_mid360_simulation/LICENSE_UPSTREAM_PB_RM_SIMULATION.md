# Upstream notice: livox_laser_simulation_RO2

The files under `src/` and the MID-360 scan pattern were adapted from
`LihanChen2004/PB_RMSimulation`, commit `457cae56c33e2eb407ad412608b02e2c2c716286`.
The upstream `livox_laser_simulation_RO2` files carry the MIT license by Ricardo
Casimiro. This notice is retained with the adapted source. The adapter changes
the package namespace, sets `CustomMsg.timebase`, and derives deterministic
per-point offsets from the Gazebo scan pattern.

MIT License

Copyright (c) 2023 Ricardo Casimiro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
