#include "io.hpp"
#include "result.hpp"
#include "util.hpp"
#include "word.hpp"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>

#ifdef __clang__
#include "__clang_cuda_intrinsics.h"
#endif

enum class TestResult {
  ExactMatch,
  Contained,
  Absent,
};

struct GuessStats {
  double average_wordlist_len;
  double wordlist_len_variance;
  std::uint32_t max_wordlist_len;
};

__global__ static void compute_stats(std::uint32_t num_inputs,
                                     std::uint32_t num_answers,
                                     const std::uint32_t *num_allowed_words,
                                     std::uint32_t num_allowed_words_pitch,
                                     GuessStats *stats) {
  const std::uint32_t guess_word_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  GuessStats this_guess_stats = {0.0, 0.0, 0};

  if (guess_word_idx < num_inputs) {
    double sum = 0.0;
    double c = 0.0;

    for (std::uint32_t i = 0; i < num_answers; ++i) {
      const std::uint32_t this_actual_num_allowed = ptr_offset(
          num_allowed_words, num_allowed_words_pitch, i)[guess_word_idx];

      if (this_actual_num_allowed > this_guess_stats.max_wordlist_len) {
        this_guess_stats.max_wordlist_len = this_actual_num_allowed;
      }

      const double y = double(this_actual_num_allowed) - c;
      const double t = sum + y;
      c = t - sum - y;
      sum = t;
    }

    this_guess_stats.average_wordlist_len = sum / double(num_answers);

    sum = 0.0;
    c = 0.0;

    for (std::uint32_t i = 0; i < num_answers; ++i) {
      const auto this_actual_num_allowed = double(ptr_offset(
          num_allowed_words, num_allowed_words_pitch, i)[guess_word_idx]);
      const auto this_diff =
          (this_actual_num_allowed - this_guess_stats.average_wordlist_len);
      const auto this_diff_sq = this_diff * this_diff;

      const double y = this_diff_sq - c;
      const double t = sum + y;
      c = t - sum - y;
      sum = t;
    }

    this_guess_stats.wordlist_len_variance = sum / double(num_answers - 1);

    stats[guess_word_idx] = this_guess_stats;
  }
}

__global__ static void filter_answers(Result result, const Word *answers,
                                      std::uint32_t num_answers,
                                      bool *answer_is_allowed) {
  const std::uint32_t answer_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (answer_idx < num_answers) {
    const Word this_answer = answers[answer_idx];
    answer_is_allowed[answer_idx] = result.allows_word(this_answer);
  }
}

inline constexpr std::uint32_t GUESSES_PER_BLOCK = 32;
inline constexpr std::uint32_t ANSWERS_PER_BLOCK = 32;
inline constexpr std::uint32_t BLOCK_SIZE =
    GUESSES_PER_BLOCK * ANSWERS_PER_BLOCK;
inline constexpr std::uint32_t WARP_SIZE = 32;
inline constexpr std::uint32_t WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
static_assert(BLOCK_SIZE % WARP_SIZE == 0);

__global__ static void get_num_allowed_words(
    const Word *guesses, std::uint32_t num_guesses, const Word *answers,
    std::uint32_t num_answers,
    std::uint32_t *num_allowed_words, /* num_answer rows x num_guesses cols */
    std::uint32_t num_allowed_words_pitch) {
  assert(WARP_SIZE == warpSize);

  const std::uint32_t this_block_first_guess_idx = blockIdx.x * blockDim.x;
  const std::uint32_t this_block_first_answer_idx = blockIdx.y * blockDim.y;

  // guess is the x dimension -- adjacent threads in the same block are loading
  // from different guesses
  const std::uint32_t this_thread_guess_idx =
      this_block_first_guess_idx + threadIdx.x;

  // actual word is the y dimension -- each block loads a different real word,
  // so we can load it into shared memory. only the first thread in each block
  // needs to load this word
  const std::uint32_t this_thread_answer_idx =
      this_block_first_answer_idx + threadIdx.y;

  const std::uint32_t thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
  const std::uint32_t warp_idx = thread_idx / WARP_SIZE;
  const std::uint32_t warp_thread_idx = thread_idx % WARP_SIZE;

  __shared__ Word block_guesses[GUESSES_PER_BLOCK];
  if (thread_idx < GUESSES_PER_BLOCK) {
    if (this_block_first_guess_idx + thread_idx < num_guesses) {
      block_guesses[thread_idx] =
          guesses[this_block_first_guess_idx + thread_idx];
    }
  }

  __shared__ Word block_answers[ANSWERS_PER_BLOCK];
  if (thread_idx < ANSWERS_PER_BLOCK) {
    if (this_block_first_answer_idx + thread_idx < num_answers) {
      block_answers[thread_idx] =
          answers[this_block_first_answer_idx + thread_idx];
    }
  }

  __syncthreads();

  const Word this_thread_guess = block_guesses[threadIdx.x];
  const Word this_thread_answer = block_answers[threadIdx.y];
  const auto this_thread_result =
      Result::from_answer(this_thread_guess, this_thread_answer);

  std::uint32_t this_thread_num_allowed = 0;

  for (std::uint32_t i = 0; i < num_answers; i += WARP_SIZE) {
    const std::uint32_t last_test_idx = min(i + WARP_SIZE, num_answers);
    const std::uint32_t num_to_test = last_test_idx - i;

    const unsigned active_mask = ~unsigned(0);
    const bool this_thread_will_collect_data = warp_thread_idx < num_to_test;
    const unsigned has_data_mask =
        __ballot_sync(active_mask, this_thread_will_collect_data);

    Word this_thread_answer_to_share;
    if (this_thread_will_collect_data) {
      this_thread_answer_to_share = answers[i + warp_thread_idx];
    }

    for (std::uint32_t i = 0; i < num_to_test; ++i) {
      Word this_answer_to_test;
      this_answer_to_test.storage =
          __shfl_sync(active_mask, this_thread_answer_to_share.storage, i);

      if (this_thread_result.allows_word(this_answer_to_test)) {
        ++this_thread_num_allowed;
      }
    }
  }

  if ((this_thread_answer_idx < num_answers) &&
      (this_thread_guess_idx < num_guesses)) {
    ptr_offset(num_allowed_words, num_allowed_words_pitch,
               this_thread_answer_idx)[this_thread_guess_idx] =
        this_thread_num_allowed;
  }
}

struct EvaluatedWord {
  Word word;
  GuessStats stats;
  bool is_not_answer;
};

auto as_tuple(const EvaluatedWord &evaluated) noexcept {
  return std::make_tuple(
      evaluated.stats.average_wordlist_len, evaluated.stats.max_wordlist_len,
      evaluated.stats.wordlist_len_variance, evaluated.is_not_answer ? 1 : 0,
      std::cref(evaluated.word));
}

static constexpr std::size_t div_round_up(std::size_t num,
                                          std::size_t den) noexcept {
  std::size_t quotient = num / den;

  if (num % den != 0) {
    ++quotient;
  }

  return quotient;
}

static int compute_scores(const std::string &inputs_pathname,
                          const std::string &answers_pathname,
                          const std::string &output_pathname) {
  const std::vector<Word> inputs = read_line_delimited_words(inputs_pathname);
  const std::vector<Word> answers = read_line_delimited_words(answers_pathname);

  std::ofstream output(output_pathname);

  if (!output.is_open()) {
    std::cerr << "error: couldn't create '" << output_pathname
              << "' for writing\n";
    return EXIT_FAILURE;
  }

  const std::uint32_t num_inputs = inputs.size();
  const std::uint32_t num_answers = answers.size();
  std::cout << "loaded " << num_inputs << " allowed inputs\n";
  std::cout << "loaded " << num_answers << " potential answers\n";

  const auto guesses_device = to_device(inputs);
  const auto answers_device = to_device(answers);

  const auto [num_allowed_words_device, num_allowed_words_device_pitch] =
      make_unique_device_pitched<std::uint32_t>(num_inputs, num_answers);

  const auto stats_device = make_unique_device<GuessStats>(num_inputs);

  const std::uint32_t num_guess_blocks =
      div_round_up(num_inputs, GUESSES_PER_BLOCK);
  const std::uint32_t num_answer_blocks =
      div_round_up(num_answers, ANSWERS_PER_BLOCK);

  cudaDeviceProp properties;
  check_cuda(cudaGetDeviceProperties(&properties, 0));
  std::cout << properties.regsPerBlock << " registers per block\n";
  std::cout << properties.regsPerMultiprocessor
            << " registers per multiprocessor\n";
  std::cout << properties.sharedMemPerBlock
            << " bytes of shared memory per block\n";
  std::cout << properties.sharedMemPerBlockOptin
            << " bytes of shared memory per block, opt-in\n";
  std::cout << properties.sharedMemPerMultiprocessor
            << " bytes of shared memory per multiprocessor\n";
  std::cout << properties.maxThreadsPerBlock << " threads per block\n";
  std::cout << properties.maxThreadsPerMultiProcessor
            << " threads per multiprocessor\n";

  cudaFuncAttributes attributes;
  check_cuda(cudaFuncGetAttributes(&attributes, get_num_allowed_words));
  std::cout << attributes.constSizeBytes
            << " bytes of constant memory requested\n";
  std::cout << attributes.localSizeBytes
            << " bytes of local memory requested\n";
  std::cout << attributes.maxDynamicSharedSizeBytes
            << " bytes of shared memory requested\n";
  std::cout << attributes.maxThreadsPerBlock
            << " threads per block requested\n";
  std::cout << attributes.numRegs << " registers per thread requested\n";

  get_num_allowed_words<<<dim3(num_guess_blocks, num_answer_blocks, 1),
                          dim3(GUESSES_PER_BLOCK, ANSWERS_PER_BLOCK, 1)>>>(
      guesses_device.get(), num_inputs, answers_device.get(), num_answers,
      num_allowed_words_device.get(), num_allowed_words_device_pitch);
  check_cuda(cudaGetLastError());

  compute_stats<<<dim3(num_inputs, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(
      num_inputs, num_answers, num_allowed_words_device.get(),
      num_allowed_words_device_pitch, stats_device.get());
  check_cuda(cudaGetLastError());

  std::vector<std::uint32_t> num_allowed_words(num_inputs * num_answers, 0);

  copy_to_host_pitched(num_allowed_words_device.get(),
                       num_allowed_words_device_pitch, num_allowed_words.data(),
                       num_inputs * sizeof(num_allowed_words[0]), num_inputs,
                       num_answers);

  std::vector<GuessStats> stats(num_inputs);

  copy_to_host(stats_device.get(), stats.data(), num_inputs);

  std::set<Word> answers_set(answers.cbegin(), answers.cend());

  std::vector<EvaluatedWord> evaluated;
  evaluated.reserve(num_inputs);

  for (std::size_t i = 0; i < num_inputs; ++i) {
    evaluated.push_back(
        EvaluatedWord{inputs[i], stats[i], answers_set.count(inputs[i]) == 0});
  }

  std::sort(evaluated.begin(), evaluated.end(),
            [](const EvaluatedWord &lhs, const EvaluatedWord &rhs) {
              return as_tuple(lhs) < as_tuple(rhs);
            });

  std::cout << "evaluated wordlist sizes of " << num_inputs
            << " allowed inputs\n";

  if (!(output << "guess,is_potential_answer,average_num_allowed,num_allowed_"
                  "variance,max_wordlist_len\n")) {
    std::cout << "error: couldn't write to '" << output_pathname << "'\n";
    return EXIT_FAILURE;
  }

  for (const EvaluatedWord &this_evaluated : evaluated) {
    if (!(output << this_evaluated.word << ','
                 << (this_evaluated.is_not_answer ? "false" : "true") << ','
                 << this_evaluated.stats.average_wordlist_len << ','
                 << this_evaluated.stats.wordlist_len_variance << ','
                 << this_evaluated.stats.max_wordlist_len << '\n')) {
      std::cout << "error: couldn't write to '" << output_pathname << "'\n";
      return EXIT_FAILURE;
    }
  }

  output.close();
  if (!output) {
    std::cout << "error: couldn't write to '" << output_pathname << "'\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static int prune_answers(const std::string &guess_str,
                         const std::string &result_str,
                         const std::string &answers_pathname,
                         const std::string &output_pathname) {
  const std::optional<Word> maybe_guess = make_word(guess_str);
  if (!maybe_guess) {
    std::cerr << "error: guess '" << guess_str << "' is not a valid word\n";
    return EXIT_FAILURE;
  }

  const Word guess = *maybe_guess;

  if (result_str.size() != WORD_LEN) {
    std::cerr << "error: result '" << guess_str << "' is not a valid result\n";
    return EXIT_FAILURE;
  }

  Comparison comparisons[WORD_LEN];

  for (std::size_t i = 0; i < WORD_LEN; ++i) {
    const char ch = result_str[i];

    if ((ch == 'E') || (ch == 'e')) {
      comparisons[i] = Comparison::ExactMatch;
    } else if ((ch == 'C') || (ch == 'c')) {
      comparisons[i] = Comparison::Contained;
    } else if ((ch == 'A') || (ch == 'a')) {
      comparisons[i] = Comparison::Absent;
    } else {
      std::cerr << "error: result '" << guess_str
                << "' is not a valid result\n";
      return EXIT_FAILURE;
    }
  }

  const auto result = Result::from_comparisons(guess, comparisons);

  const std::vector<Word> answers = read_line_delimited_words(answers_pathname);

  const std::size_t num_answers = answers.size();
  std::cout << "loaded " << num_answers << " potential answers\n";

  const auto answers_device = to_device(answers);

  const auto is_allowed_device = make_unique_device<bool>(num_answers);

  constexpr std::size_t THREADS_PER_BLOCK = 256;

  const std::size_t num_x_blocks = div_round_up(num_answers, THREADS_PER_BLOCK);

  filter_answers<<<dim3(num_x_blocks, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1)>>>(
      result, answers_device.get(), num_answers, is_allowed_device.get());
  check_cuda(cudaGetLastError());

  std::vector<std::uint8_t> is_allowed(num_answers, 0);

  // vector<bool> grr
  copy_to_host(is_allowed_device.get(),
               reinterpret_cast<bool *>(is_allowed.data()), num_answers);

  std::vector<Word> new_potential_answers;
  new_potential_answers.reserve(answers.size());

  for (std::size_t i = 0; i < num_answers; ++i) {
    if (is_allowed[i] != 0) {
      new_potential_answers.push_back(answers[i]);
    }
  }

  std::cout << "pruned wordlist of " << num_answers << " potential answers\n";

  write_line_delimited_words(new_potential_answers, output_pathname);

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "error: missing required positional argument SUBCOMMAND\n";
    return EXIT_FAILURE;
  }

  const std::string subcommand = argv[1];

  if (subcommand == "score") {
    if (argc < 3) {
      std::cerr
          << "error: missing required positional argument ALLOWED_INPUTS\n";
      return EXIT_FAILURE;
    }

    if (argc < 4) {
      std::cerr
          << "error: missing required positional argument POTENTIAL_ANSWERS\n";
      return EXIT_FAILURE;
    }

    if (argc < 5) {
      std::cerr << "error: missing required positional argument OUTPUT\n";
      return EXIT_FAILURE;
    }

    return compute_scores(argv[2], argv[3], argv[4]);
  }

  if (subcommand == "prune") {
    if (argc < 3) {
      std::cerr << "error: missing required positional argument INPUT\n";
      return EXIT_FAILURE;
    }

    if (argc < 4) {
      std::cerr << "error: missing required positional argument RESULT\n";
      return EXIT_FAILURE;
    }

    if (argc < 5) {
      std::cerr
          << "error: missing required positional argument POTENTIAL_ANSWERS\n";
      return EXIT_FAILURE;
    }

    if (argc < 6) {
      std::cerr << "error: missing required positional argument OUTPUT\n";
      return EXIT_FAILURE;
    }

    return prune_answers(argv[2], argv[3], argv[4], argv[5]);
  }

  std::cerr << "error: unrecognized subcommand '" << subcommand << "'\n";
  return EXIT_FAILURE;
}
