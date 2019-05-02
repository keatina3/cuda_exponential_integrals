  <h1 class="title page-title"><span class="field field--name-title field--type-string field--label-hidden">5615-03 - Cuda Exponential Integral calculation</span>
</h1>

<p>The goal of this assignment is to develop a fast cuda implementation of the provided exponential integral source code. This code integrates the integrals of the exponential functions, from E_0 to E_n. You can find more information about the algorithm in:<br />
http://mathworld.wolfram.com/En-Function.html<br />
http://mathworld.wolfram.com/ExponentialIntegral.html<br />
Or in any of the Numerical Recipes books (page 266 in the Numerical Recipes Third Edition C++).</p>

<p><br /><strong>Task 1 - cuda implementation</strong></p>

<p>Starting from the provided source code (in exponentialIntegral.tar.gz), modify the main.cpp and add .h and .cu files that contain your cuda code to calculates both the floating point and double precision versions of what has been implemented in the CPU and that will be executed unless the program is executed with the "-g" flag (see the usage of the provided code, which skips the cpu test when passing a "-c" flag as an argument).</p>

<p>The cuda implementation must time *all* the cuda part of the code (including memory transfers and allocations). Add as well separate time measures for both the single and double precision versions (so we can see the difference in performance between either precision in cuda). Calculate the speedup for the total cuda timing (that is, including the memory allocations, transfers and execution).</p>

<p>Add a comparison between the results obtained from the gpu and the cpu. If any values diverge by more than 1.E-5 (which shouldn't happen), print them.</p>

<p>There are no restrictions on the cuda techniques that can be used for this assignment, other than not using libraries (other than cuda itself).</p>

<p>Most of the marks will be given for good performance of the implemented code, and additional marks will be given for:<br />
* Using the constant and shared memories to save register memory, so it doesn't get demoted to local memory.<br />
* Using streams to let the compute overlap.<br />
* Using multiple cards to split up the computation.<br />
* Any other advanced cuda (dynamic parallelism, etc) that is tested for performance.</p>

<p><strong>Task 2 - performance</strong></p>

<p>Run the final version of the program with the following sizes:<br />
-n 5000 -m 5000<br />
-n 8192 -m 8192<br />
-n 16384 -m 16384<br />
-n 20000 -m 20000<br />
and find the best grid sizes for each problem size (do not forget to test cases in which the n and m values are different, as they have to work as well!).</p>

<p>===================================================================================================================</p>

<p>Submit a tar ball with your source code files (including a working Makefile for cuda01), speedup graphs and a writeup of what you did and any observations you have made on the behaviour and performance of your code, as well as problems that you came across while writing the assignmentas well as the bottlenecks that you found while implementing the cuda version.</p>

<p><strong>Note:</strong> Marks will be deducted for tarbombing. http://en.wikipedia.org/wiki/Tar_%28computing%29#Tarbomb</p>

<p><strong>Note:</strong> Extra marks will be given for separating the C/C++ and cuda code. You can find examples on how to do that on the "makefileCpp" and "makefileExternC" sample code.</p>

<p><strong>Note:</strong> Remember that the code must work for non-square systems too, even when, for this assignment, we are using square ones for benchmarking. You can run use cuda-memcheck to test that, as in: cuda-memcheck ./my_exec</p>

<p><strong>Note:</strong> When you are benchmarking the performance of your code, you can check the current load on cuda01 with the <em>nvidia-smi</em> command.</p>

<p><strong>Note:</strong> When benchmarking the gpu code, remember to use the -c flag so the cpu part doesn't have to be run each time (the -n 20000 -m 20000 takes about 120 seconds)</p></div>
