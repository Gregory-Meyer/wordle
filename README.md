# wordle

CUDA C++ program to score Wordle guesses using expected information gain, then prune a wordlist of answers based on the result of the guess.

## Building

The minimum [CMake](https://cmake.org/) version required to build this project is 3.17. I tested with CMake 3.23. If you are running Ubuntu 16.04, 18.04, or 20.04, you may use the [Kitware APT repositories](https://apt.kitware.com/) to get the most recent versions of CMake. I prefer to use the [`Ninja`](https://ninja-build.org/) generator, but `Unix Makefiles` will work as well.

```console
$ git clone https://github.com/Gregory-Meyer/wordle.git
$ cmake -S wordle -B wordle/build -DCMAKE_BUILD_TYPE=Release -G Ninja 
$ cmake --build wordle/build --parallel "$(nproc)"
```

I tested with GCC 9.4.0, Clang 11.1.0, and NVCC 11.3.109. Other compiler versions may work, but they must support C++17.

<details>

```console
$ gcc --version
gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
$ clang --version
Ubuntu clang version 11.1.0-++20211011094159+1fdec59bffc1-1~exp1~20211011214622.5
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/local/bin
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
```

</details>

## Usage

```console
$ # score each allowed input based on the expected size of the resulting potential answer list
$ ./wordle score <allowed inputs> <potential answers> <output scores.csv>
$ # remove answers from the word list that are no longer possible given this guess
$ ./wordle prune <word guessed> <result of the guess> <potential answers> <output potential answers>
```

The result of the guess is five letters, one of:
* A: Absent / Grey
* E: Exact Match / Green
* C: Contained / Yellow

For example, the guess "reais" for the target word "sheen" would give:

* grey, yellow, grey, grey, yellow
* ⬛🟨⬛⬛🟨
* ACAAC

## Example Game

The target word for this game will be "sheen."

### reais

The expected information gain metric will always suggest your starting guess is "reais":

```console
$ nvidia-smi --list-gpus
GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-<snipped>)
$ time build/wordle score wordle-allowed-inputs.txt wordle-allowed-inputs.txt scores.csv
loaded 12972 allowed inputs
loaded 12972 potential answers
65536 registers per block
65536 registers per multiprocessor
49152 bytes of shared memory per block
101376 bytes of shared memory per block, opt-in
102400 bytes of shared memory per multiprocessor
1024 threads per block
1536 threads per multiprocessor
0 bytes of constant memory requested
40 bytes of local memory requested
48640 bytes of shared memory requested
1024 threads per block requested
38 registers per thread requested
evaluated wordlist sizes of 12972 allowed inputs
________________________________________________________
Executed in   30.71 secs    fish           external
   usr time   30.43 secs  390.00 micros   30.43 secs
   sys time    0.25 secs   69.00 micros    0.25 secs
$ head --lines 2 scores.csv
guess,is_potential_answer,average_num_allowed,num_allowed_variance,max_wordlist_len
reais,true,447.809,89011.2,965
```

### poult

Given the target word "sheen," we will get the result ⬛🟨⬛⬛🟨 -- Absent, Contained, Absent, Absent, Contained -- for the guess "reais." The next word we'll guess is "poult":

```console
$ time build/wordle prune reais ACAAC wordle-answers-alphabetical.txt reais-answers.txt
loaded 2315 potential answers
pruned wordlist of 2315 potential answers

________________________________________________________
Executed in  112.87 millis    fish           external
   usr time   45.26 millis   33.06 millis   12.21 millis
   sys time   74.09 millis    0.85 millis   73.24 millis
$ wc --lines reais-answers.txt
87 reais-answers.txt
$ time build/wordle score wordle-allowed-inputs.txt reais-answers.txt reais-scores.csv
loaded 12972 allowed inputs
loaded 87 potential answers
65536 registers per block
65536 registers per multiprocessor
49152 bytes of shared memory per block
101376 bytes of shared memory per block, opt-in
102400 bytes of shared memory per multiprocessor
1024 threads per block
1536 threads per multiprocessor
0 bytes of constant memory requested
40 bytes of local memory requested
48640 bytes of shared memory requested
1024 threads per block requested
38 registers per thread requested
evaluated wordlist sizes of 12972 allowed inputs

________________________________________________________
Executed in  116.11 millis    fish           external
   usr time   20.51 millis  344.00 micros   20.16 millis
   sys time   72.65 millis   53.00 micros   72.59 millis
$ head --lines 2 reais-scores.csv
guess,is_potential_answer,average_num_allowed,num_allowed_variance,max_wordlist_len
poult,false,5.26437,10.5921,11
```

### deeve

This gives us result ⬛⬛⬛⬛⬛ -- Absent, Absent, Absent, Absent, Absent. The next guess will be "deeve":

```console
$ time build/wordle prune poult AAAAA reais-answers.txt poult-answers.txt
loaded 87 potential answers
pruned wordlist of 87 potential answers

________________________________________________________
Executed in  111.00 millis    fish           external
   usr time    7.67 millis  244.00 micros    7.43 millis
   sys time   78.04 millis   23.00 micros   78.02 millis
$ wc --lines poult-answers.txt
9 poult-answers.txt
$ time build/wordle score wordle-allowed-inputs.txt poult-answers.txt poult-scores.csv
loaded 12972 allowed inputs
loaded 9 potential answers
65536 registers per block
65536 registers per multiprocessor
49152 bytes of shared memory per block
101376 bytes of shared memory per block, opt-in
102400 bytes of shared memory per multiprocessor
1024 threads per block
1536 threads per multiprocessor
0 bytes of constant memory requested
40 bytes of local memory requested
48640 bytes of shared memory requested
1024 threads per block requested
38 registers per thread requested
evaluated wordlist sizes of 12972 allowed inputs

________________________________________________________
Executed in  116.11 millis    fish           external
   usr time   27.67 millis  247.00 micros   27.42 millis
   sys time   66.61 millis   21.00 micros   66.59 millis
$ head --lines 2 poult-scores.csv
guess,is_potential_answer,average_num_allowed,num_allowed_variance,max_wordlist_len
deeve,false,1.22222,0.194444,2
```

### scene

This gives us result ⬛🟨🟩⬛⬛ -- Absent, Contained, Exact Match, Absent, Absent. The next guess will be "scene," because it lexicographically sorts before "sheen."

```console
$ time build/wordle prune deeve ACEAA poult-answers.txt deeve-answers.txt
loaded 9 potential answers
pruned wordlist of 9 potential answers

________________________________________________________
Executed in  120.70 millis    fish           external
   usr time   20.07 millis  413.00 micros   19.66 millis
   sys time   70.83 millis   46.00 micros   70.78 millis
$ wc --lines deeve-answers.txt
2 deeve-answers.txt
$ time build/wordle score wordle-allowed-inputs.txt deeve-answers.txt deeve-scores.csv
loaded 12972 allowed inputs
loaded 2 potential answers
65536 registers per block
65536 registers per multiprocessor
49152 bytes of shared memory per block
101376 bytes of shared memory per block, opt-in
102400 bytes of shared memory per multiprocessor
1024 threads per block
1536 threads per multiprocessor
0 bytes of constant memory requested
40 bytes of local memory requested
48640 bytes of shared memory requested
1024 threads per block requested
38 registers per thread requested
evaluated wordlist sizes of 12972 allowed inputs

________________________________________________________
Executed in  112.62 millis    fish           external
   usr time   13.27 millis  318.00 micros   12.95 millis
   sys time   77.75 millis   34.00 micros   77.72 millis
$ head --lines 3 deeve-scores.csv
scene,true,1,0,1
sheen,true,1,0,1
```

### sheen

This gives us result 🟩⬛🟩🟨🟨 -- Exact Match, Absent, Exact Match, Contained, Contained. The only remaining answer left is "sheen," which is the target word! 🎉

```console
$ time build/wordle prune scene EAECC deeve-answers.txt scene-answers.txt
loaded 2 potential answers
pruned wordlist of 2 potential answers

________________________________________________________
Executed in   99.22 millis    fish           external
   usr time   17.50 millis  289.00 micros   17.21 millis
   sys time   60.24 millis   22.00 micros   60.22 millis
$ cat scene-answers.txt
sheen
```

### Summary

⬛🟨⬛⬛🟨  
⬛⬛⬛⬛⬛  
⬛🟨🟩⬛⬛  
🟩⬛🟩🟨🟨  
🟩🟩🟩🟩🟩

```
reais
poult
deeve
scene
sheen
```

## Algorithm and Optimizations

I use a brute force algorithm to evaluate the number of remaining allowed guesses for each unique (guess, answer) pair, then compute the average number of remaining allowed guesses over all answers. Computing the average number of guesses is done using the [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) at 64-bit precision -- this is much less efficient, but is done to ensure that the result has as much precision as possible.

Blocks are (32 x 32), for a total of 1024 threads per block.

### Global Memory Access Coalescing

Words are stored with 8-byte alignment. One array is of allowed guesses, and one array is the possible answers. The first 32 threads in each block load 32 adjacent guesses, then 32 adjacent answers into shared memory. The x-dimension of each thread+block combination is the index of the guess to be evaluted, and the y-dimension is the index of the answer to evaluate against.

Once the per-guess/answer allowed guess counts have been computed, the rows of the global memory allowed guess count matrix are naturally aligned so that each thread's 8-byte write is coalesced into a warp-level 256-byte write.

### Warp-Level Primitives for Synchronized Data Exchange

Inside of the inner loop for evaluating which answers are allowed by a given (guess, result) pair, I loop over the entire set of answers. To reduce shared memory pressure, each thread in a warp loads one word out of a block of 32, then uses the `__shfl_sync` intrinsic to broadcast its result to the other threads in the warp. Each thread in the warp evaluates the same possible answer.

### SIMD Word Comparison

This is where things get hairy. I increased register pressure to significantly increase throughput by using CUDA SIMD instructions on 4-byte unsigned integers.

1. Given a guess and its corresponding result, fill some lookup tables:
    * Create a mask for each byte index; this is an 8-byte integer that is `0x00` at all byte indices other than `i`, and `0xFF` at byte index `i`
    * Iterate over the "exact match" results. Binary-and the byte index mask for this these indices to determine a mask of bytes that have to match the guess word's letter exactly
    * Iterate over the "contained" and "absent" results. The number of "contained" results for a given letter is the lower bound for the number of times that letter may appear in the answer, and not in an exact match position. If there is at least one "absent" result for that letter, then the number of times that the letter appears in the answer must be **exactly** the number of times that that letter had a "contained" result.
    * Shift the lower and upper bound letter counts into their index in a pair of 4-byte integers. We'll do bytewise vector comparisons on these later.
    * Zero out the last three bytes of the guess. Uninitialized memory can cause trouble for us.
2. Then, given a potential answer:
   * Bytewise compare the guess to the potential answer, then binary-and this result with the mask of guess letters that were exact matches.
   * Bytewise compare the potential answer to broadcasted vectors of each letter in the guess, then take the population count to get the number of letters in the potential answer that compare equal to that guess. Place the count of equal letters that are **not** in an exact match position in the corresponding index of a vector.
   * If each exact match in the guess is also an exact match to the potential answer, then we have fulfilled the exact match requirement
   * If each non-exact match in the guess has a number of equal letters in the potential answer between the lower and upper bounds, then we have fulfilled the absent / contained requirements
