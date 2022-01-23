#ifndef RESULT_HPP
#define RESULT_HPP

#include "word.hpp"

enum class Comparison {
  Absent,
  ExactMatch,
  Contained,
};

struct Result {
  Word guess;
  Comparison comparisons[WORD_LEN];

  __host__ __device__ constexpr bool
  allows_word(const Word &word) const noexcept {
    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      switch (comparisons[i]) {
      case Comparison::Absent: {
        if (word.contains_char(guess.char_at(i))) {
          return false;
        }

        break;
      }

      case Comparison::ExactMatch: {
        if (!word.has_char_at(guess.char_at(i), i)) {
          return false;
        }

        break;
      }

      case Comparison::Contained: {
        if (!word.contains_char(guess.char_at(i))) {
          return false;
        }

        break;
      }
      }
    }

    return true;
  }
};

__host__ __device__ constexpr Result make_result(const Word &guess,
                                                 const Word &answer) noexcept {
  Result result = {guess};

  for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
    const unsigned char ch = guess.char_at(i);

    if (answer.has_char_at(ch, i)) {
      result.comparisons[i] = Comparison::ExactMatch;
    } else if (answer.contains_char(ch)) {
      result.comparisons[i] = Comparison::Contained;
    } else {
      result.comparisons[i] = Comparison::Absent;
    }
  }

  return result;
}

#endif
