#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>

constexpr std::size_t WORD_LEN = 5;
constexpr std::size_t NUM_CHARS =
    std::size_t(std::numeric_limits<unsigned char>::max()) + 1;

#define check_cuda(...)                                                        \
  _do_check_cuda((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__, __func__)
#define check_cuda_safe(...)                                                   \
  _do_check_cuda_safe((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__, __func__)

static void _do_check_cuda(cudaError err, std::string_view what,
                           std::string_view file, std::int32_t line,
                           std::string_view func);
static std::optional<std::string>
_do_check_cuda_safe(cudaError err, std::string_view what, std::string_view file,
                    std::int32_t line, std::string_view func);

struct CudaDeleter {
  void operator()(void *ptr) const {
    if (std::optional<std::string> err_str = check_cuda_safe(cudaFree(ptr))) {
      std::cerr << "warning: " << *err_str << '\n';
    }
  }
};

template <typename T>
static std::pair<std::unique_ptr<T[], CudaDeleter>, std::size_t>
make_unique_device_pitched(std::size_t width, std::size_t height) {
  T *ptr = nullptr;
  std::size_t pitch = 0;
  check_cuda(cudaMallocPitch(&ptr, &pitch, width * sizeof(T), height));

  return std::make_pair(std::unique_ptr<T[], CudaDeleter>(ptr), pitch);
}

template <typename T>
static std::unique_ptr<T[], CudaDeleter> make_unique_device(std::size_t len) {
  T *ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, len * sizeof(T)));

  return std::unique_ptr<T[], CudaDeleter>(ptr);
}

template <typename T>
static void copy_to_device_pitched(const T *src, std::size_t src_pitch, T *dst,
                                   std::size_t dst_pitch, std::size_t width,
                                   std::size_t height) {
  check_cuda(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width * sizeof(T),
                          height, cudaMemcpyHostToDevice));
}

template <typename T>
static void copy_to_host_pitched(const T *src, std::size_t src_pitch, T *dst,
                                 std::size_t dst_pitch, std::size_t width,
                                 std::size_t height) {
  check_cuda(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width * sizeof(T),
                          height, cudaMemcpyDeviceToHost));
}

template <typename T>
static void copy_to_host(const T *src, T *dst, std::size_t len) {
  check_cuda(cudaMemcpy(dst, src, len * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
__device__ static T *ptr_offset(T *ptr, std::uint32_t pitch,
                                std::uint32_t row) {
  return reinterpret_cast<T *>(reinterpret_cast<char *>(ptr) + pitch * row);
}

template <typename T>
__device__ static const T *ptr_offset(const T *ptr, std::uint32_t pitch,
                                      std::uint32_t row) {
  return reinterpret_cast<const T *>(reinterpret_cast<const char *>(ptr) +
                                     pitch * row);
}

enum class TestResult {
  ExactMatch,
  Contained,
  Absent,
};

struct GuessStats {
  double average_wordlist_len;
  double wordlist_len_variance;
};

__global__ static void compute_stats(std::uint32_t num_inputs,
                                     std::uint32_t num_answers,
                                     const std::uint32_t *num_allowed_words,
                                     std::uint32_t num_allowed_words_pitch,
                                     GuessStats *stats) {
  const std::uint32_t guess_word_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  GuessStats this_guess_stats = {0.0, 0.0};

  if (guess_word_idx < num_inputs) {
    double sum = 0.0;
    double c = 0.0;

    for (std::uint32_t i = 0; i < num_answers; ++i) {
      const auto this_actual_num_allowed = double(ptr_offset(
          num_allowed_words, num_allowed_words_pitch, i)[guess_word_idx]);

      const double y = this_actual_num_allowed - c;
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

struct Word {
  unsigned char word[WORD_LEN];
};

struct Result {
  TestResult result[WORD_LEN];
};

__device__ static bool
is_answer_allowed(const unsigned char guess[WORD_LEN],
                  const TestResult result[WORD_LEN],
                  const unsigned char answer[WORD_LEN],
                  const bool answer_contains[NUM_CHARS]) {
  bool is_allowed = true;
  for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
    switch (result[i]) {
    case TestResult::ExactMatch: {
      is_allowed = is_allowed && guess[i] == answer[i];
      break;
    }

    case TestResult::Contained: {
      is_allowed = is_allowed && answer_contains[guess[i]];
      break;
    }

    case TestResult::Absent: {
      is_allowed = is_allowed && !answer_contains[guess[i]];
      break;
    }
    }
  }

  return is_allowed;
}

__global__ static void
filter_answers(Word input, Result result, const unsigned char *answers,
               std::uint32_t answers_pitch, std::uint32_t num_answers,
               const unsigned char *answers_tr, std::uint32_t answers_tr_pitch,
               bool *answer_is_allowed) {
  const std::uint32_t answer_word_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  unsigned char this_answer[WORD_LEN + 1] = {0};
  bool this_answer_contains[NUM_CHARS];

  if (answer_word_idx < num_answers) {
    for (std::uint32_t i = 0; i < NUM_CHARS; ++i) {
      this_answer_contains[i] = false;
    }

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      this_answer[i] =
          ptr_offset(answers_tr, answers_tr_pitch, answer_word_idx)[i];
      this_answer_contains[this_answer[i]] = true;
    }
  }

  if (answer_word_idx < num_answers) {
    answer_is_allowed[answer_word_idx] = is_answer_allowed(
        input.word, result.result, this_answer, this_answer_contains);
  }
}

__global__ static void set_filter_table_entries(
    const unsigned char *inputs, std::uint32_t inputs_pitch,
    std::uint32_t num_inputs, const unsigned char *answers,
    std::uint32_t answers_pitch, std::uint32_t num_answers,
    const unsigned char *inputs_tr, std::uint32_t inputs_tr_pitch,
    const unsigned char *answers_tr, std::uint32_t answers_tr_pitch,
    std::uint32_t *num_allowed_words, std::uint32_t num_allowed_words_pitch) {
  // guess is the x dimension -- adjacent threads in the same block are loading
  // from different guesses
  const std::uint32_t guess_word_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // actual word is the y dimension -- each block loads a different real word,
  // so we can load it into shared memory. only the first thread in each block
  // needs to load this word
  const std::uint32_t actual_word_idx = (blockIdx.y * blockDim.y) + threadIdx.y;

  unsigned char guess_word[WORD_LEN + 1] = {0};

  if (guess_word_idx < num_inputs) {
    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      guess_word[i] = ptr_offset(inputs, inputs_pitch, i)[guess_word_idx];
    }
  }

  // all threads in a block are assigned the same actual word, so we can use the
  // first couple of threads in a block
  __shared__ unsigned char actual_word[WORD_LEN + 1];

  if (actual_word_idx < num_answers) {
    if (threadIdx.x < WORD_LEN) {
      actual_word[threadIdx.x] = ptr_offset(answers_tr, answers_tr_pitch,
                                            actual_word_idx)[threadIdx.x];
    } else if (threadIdx.x == WORD_LEN) {
      actual_word[WORD_LEN] = '\0';
    }
  }

  __syncthreads();

  TestResult results[WORD_LEN];

  for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
    if (guess_word[i] == actual_word[i]) {
      results[i] = TestResult::ExactMatch;
    } else {
      results[i] = TestResult::Absent;

      for (std::uint32_t j = 0; j < WORD_LEN; ++j) {
        if (guess_word[i] == actual_word[j]) {
          results[i] = TestResult::Contained;
          break;
        }
      }
    }
  }

  std::uint32_t this_guess_and_actual_num_allowed_words = 0;

  for (std::uint32_t i = 0; i < num_answers; ++i) {
    __shared__ unsigned char this_possible_word[WORD_LEN + 1];
    __shared__ bool this_possible_word_contains[NUM_CHARS];

    if (threadIdx.x == 0) {
      for (std::uint32_t i = 0; i < NUM_CHARS; ++i) {
        this_possible_word_contains[i] = false;
      }
    }

    if (threadIdx.x < WORD_LEN) {
      this_possible_word[threadIdx.x] =
          ptr_offset(answers_tr, answers_tr_pitch, i)[threadIdx.x];

      // this isn't really a data race if a word contains a given character more
      // than once; both writing threads will write true
      this_possible_word_contains[this_possible_word[threadIdx.x]] = true;
    } else if (threadIdx.x == WORD_LEN) {
      this_possible_word[WORD_LEN] = '\0';
    }

    __syncthreads();

    bool is_allowed = false;
    if ((guess_word_idx < num_inputs) && (actual_word_idx < num_answers)) {
      is_allowed = is_answer_allowed(guess_word, results, this_possible_word,
                                     this_possible_word_contains);
    }

    // we need the syncthreads here to ensure that the shared memory isn't
    // clobbered by another thread before we are finished with it
    __syncthreads();

    if (is_allowed) {
      ++this_guess_and_actual_num_allowed_words;
    }
  }

  if ((guess_word_idx < num_inputs) && (actual_word_idx < num_answers)) {
    ptr_offset(num_allowed_words, num_allowed_words_pitch,
               actual_word_idx)[guess_word_idx] =
        this_guess_and_actual_num_allowed_words;
  }
}

struct EvaluatedWord {
  std::string word;
  GuessStats stats;
};

#define format(...)                                                            \
  (std::move(static_cast<std::ostringstream &>(std::ostringstream().flush()    \
                                               << __VA_ARGS__))                \
       .str())

static bool is_valid_word(const std::string &maybe_word) noexcept {
  return maybe_word.size() == WORD_LEN;
}

static std::vector<std::string>
read_line_delimited_words(const std::string &pathname) {
  std::ifstream file(pathname);

  if (!file.is_open()) {
    throw std::runtime_error(format("open '" << pathname << "' for reading"));
  }

  std::uint32_t line = 0;
  std::string this_word;

  std::vector<std::string> words;

  while (std::getline(file, this_word)) {
    ++line;

    if (!is_valid_word(this_word)) {
      std::cerr << "warning: skipping word '" << this_word << "' on line "
                << line << " of file '" << pathname << "'\n";
      continue;
    }

    words.push_back(this_word);
  }

  std::sort(words.begin(), words.end());

  return words;
}

static std::pair<std::vector<unsigned char>, std::vector<unsigned char>>
to_contiguous(const std::vector<std::string> &words) {
  const std::size_t num_words = words.size();

  std::vector<unsigned char> words_contiguous(num_words * WORD_LEN);
  std::vector<unsigned char> words_contiguous_tr(WORD_LEN * num_words);

  for (std::size_t i = 0; i < num_words; ++i) {
    for (std::size_t j = 0; j < WORD_LEN; ++j) {
      words_contiguous[j * num_words + i] = words[i][j];
      words_contiguous_tr[i * WORD_LEN + j] = words[i][j];
    }
  }

  return {std::move(words_contiguous), std::move(words_contiguous_tr)};
}

template <typename T>
static std::pair<std::unique_ptr<T[], CudaDeleter>, std::size_t>
to_device_pitched(const std::vector<T> &host, std::size_t width,
                  std::size_t height) {
  assert(host.size() == width * height);

  auto ret = make_unique_device_pitched<T>(width, height);
  const auto &[device, device_pitch] = ret;

  copy_to_device_pitched(host.data(), width * sizeof(T), device.get(),
                         device_pitch, width, height);

  return std::move(ret);
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
  const std::vector<std::string> inputs =
      read_line_delimited_words(inputs_pathname);
  const std::vector<std::string> answers =
      read_line_delimited_words(answers_pathname);

  std::ofstream output(output_pathname);

  if (!output.is_open()) {
    std::cerr << "error: couldn't create '" << output_pathname
              << "' for writing\n";
    return EXIT_FAILURE;
  }

  const std::size_t num_inputs = inputs.size();
  const std::size_t num_answers = answers.size();
  std::cout << "loaded " << num_inputs << " allowed inputs\n";
  std::cout << "loaded " << num_answers << " potential answers\n";

  const auto [inputs_contiguous, inputs_tr_contiguous] = to_contiguous(inputs);
  const auto [answers_contiguous, answers_tr_contiguous] =
      to_contiguous(answers);

  const auto [inputs_device, inputs_device_pitch] =
      to_device_pitched(inputs_contiguous, num_inputs, WORD_LEN);
  const auto [inputs_tr_device, inputs_tr_device_pitch] =
      to_device_pitched(inputs_tr_contiguous, WORD_LEN, num_inputs);

  const auto [answers_device, answers_device_pitch] =
      to_device_pitched(answers_contiguous, num_answers, WORD_LEN);
  const auto [answers_tr_device, answers_tr_device_pitch] =
      to_device_pitched(answers_tr_contiguous, WORD_LEN, num_answers);

  const auto [num_allowed_words_device, num_allowed_words_device_pitch] =
      make_unique_device_pitched<std::uint32_t>(num_inputs, num_answers);

  const auto stats_device = make_unique_device<GuessStats>(num_inputs);

  constexpr std::size_t THREADS_PER_BLOCK = 256;

  const std::size_t num_x_blocks = div_round_up(num_inputs, THREADS_PER_BLOCK);

  set_filter_table_entries<<<dim3(num_x_blocks, num_answers, 1),
                             dim3(THREADS_PER_BLOCK, 1, 1)>>>(
      inputs_device.get(), inputs_device_pitch, num_inputs,
      answers_device.get(), answers_device_pitch, num_answers,
      inputs_tr_device.get(), inputs_tr_device_pitch, answers_tr_device.get(),
      answers_tr_device_pitch, num_allowed_words_device.get(),
      num_allowed_words_device_pitch);
  check_cuda(cudaGetLastError());

  compute_stats<<<dim3(num_inputs, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1)>>>(
      num_inputs, num_answers, num_allowed_words_device.get(),
      num_allowed_words_device_pitch, stats_device.get());
  check_cuda(cudaGetLastError());

  std::vector<std::uint32_t> num_allowed_words(num_inputs * num_answers, 0);

  copy_to_host_pitched(num_allowed_words_device.get(),
                       num_allowed_words_device_pitch, num_allowed_words.data(),
                       num_inputs * sizeof(std::uint32_t), num_inputs,
                       num_answers);

  std::vector<GuessStats> stats(num_inputs);

  copy_to_host(stats_device.get(), stats.data(), num_inputs);

  std::vector<EvaluatedWord> evaluated;
  evaluated.reserve(num_inputs);

  for (std::size_t i = 0; i < num_inputs; ++i) {
    evaluated.push_back(EvaluatedWord{inputs[i], stats[i]});
  }

  std::sort(evaluated.begin(), evaluated.end(),
            [](const EvaluatedWord &lhs, const EvaluatedWord &rhs) {
              return lhs.stats.average_wordlist_len <
                     rhs.stats.average_wordlist_len;
            });

  std::cout << "evaluated wordlist sizes of " << num_inputs
            << " allowed inputs\n";

  if (!(output << "guess,average_num_allowed,num_allowed_variance\n")) {
    std::cout << "error: couldn't write to '" << output_pathname << "'\n";
    return EXIT_FAILURE;
  }

  for (const EvaluatedWord &this_evaluated : evaluated) {
    if (!(output << this_evaluated.word << ','
                 << this_evaluated.stats.average_wordlist_len << ','
                 << this_evaluated.stats.wordlist_len_variance << '\n')) {
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
  if (!is_valid_word(guess_str)) {
    std::cerr << "error: guess '" << guess_str << "' is not a valid word\n";
    return EXIT_FAILURE;
  }

  Word guess;
  std::copy(guess_str.cbegin(), guess_str.cend(), guess.word);

  if (result_str.size() != WORD_LEN) {
    std::cerr << "error: result '" << guess_str << "' is not a valid result\n";
    return EXIT_FAILURE;
  }

  Result result;

  for (std::size_t i = 0; i < WORD_LEN; ++i) {
    const char ch = result_str[i];

    if ((ch == 'E') || (ch == 'e')) {
      result.result[i] = TestResult::ExactMatch;
    } else if ((ch == 'C') || (ch == 'c')) {
      result.result[i] = TestResult::Contained;
    } else if ((ch == 'A') || (ch == 'a')) {
      result.result[i] = TestResult::Absent;
    } else {
      std::cerr << "error: result '" << guess_str
                << "' is not a valid result\n";
      return EXIT_FAILURE;
    }
  }

  const std::vector<std::string> answers =
      read_line_delimited_words(answers_pathname);

  std::ofstream output(output_pathname);

  if (!output.is_open()) {
    std::cerr << "error: couldn't create '" << output_pathname
              << "' for writing\n";
    return EXIT_FAILURE;
  }

  const std::size_t num_answers = answers.size();
  std::cout << "loaded " << num_answers << " potential answers\n";

  const auto [answers_contiguous, answers_tr_contiguous] =
      to_contiguous(answers);

  const auto [answers_device, answers_device_pitch] =
      to_device_pitched(answers_contiguous, num_answers, WORD_LEN);
  const auto [answers_tr_device, answers_tr_device_pitch] =
      to_device_pitched(answers_tr_contiguous, WORD_LEN, num_answers);

  const auto is_allowed_device = make_unique_device<bool>(num_answers);

  constexpr std::size_t THREADS_PER_BLOCK = 256;

  const std::size_t num_x_blocks = div_round_up(num_answers, THREADS_PER_BLOCK);

  filter_answers<<<dim3(num_x_blocks, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1)>>>(
      guess, result, answers_device.get(), answers_device_pitch, num_answers,
      answers_tr_device.get(), answers_tr_device_pitch,
      is_allowed_device.get());
  check_cuda(cudaGetLastError());

  std::vector<std::uint8_t> is_allowed(num_answers, 0);

  // vector<bool> grr
  copy_to_host(is_allowed_device.get(),
               reinterpret_cast<bool *>(is_allowed.data()), num_answers);

  std::vector<std::string> new_potential_answers;
  new_potential_answers.reserve(answers.size());

  for (std::size_t i = 0; i < num_answers; ++i) {
    if (is_allowed[i] != 0) {
      new_potential_answers.push_back(answers[i]);
    }
  }

  std::cout << "pruned wordlist of " << num_answers << " potential answers\n";

  for (const std::string &this_answer : new_potential_answers) {
    if (!(output << this_answer << '\n')) {
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

static void _do_check_cuda(cudaError err, std::string_view what,
                           std::string_view file, std::int32_t line,
                           std::string_view func) {
  if (std::optional<std::string> err_str =
          _do_check_cuda_safe(err, what, file, line, func)) {
    throw std::runtime_error(std::move(*err_str));
  }
}

static std::optional<std::string>
_do_check_cuda_safe(cudaError err, std::string_view what, std::string_view file,
                    std::int32_t line, std::string_view func) {
  if (err == cudaSuccess) {
    return std::nullopt;
  }

  const std::string_view description = cudaGetErrorString(err);
  const std::string_view name = cudaGetErrorName(err);

  std::ostringstream oss;
  oss << file << ':' << line << ": " << func << ": " << what << ": "
      << description << " (" << name << ')';

  return std::move(oss).str();
}
