#ifndef RESULT_HPP
#define RESULT_HPP

#include "word.hpp"

#include <cstdint>

enum class Comparison {
  Absent,
  ExactMatch,
  Contained,
};

template <typename T, std::uint32_t N>
__host__ __device__ constexpr void zero_fill(T (&arr)[N]) noexcept {
  for (std::uint32_t i = 0; i < N; ++i) {
    arr[i] = T(0);
  }
}

class Result {
public:
  __host__ __device__ static Result from_answer(const Word &guess,
                                                const Word &answer) noexcept {
    struct char_map_entry {
      unsigned char ch;
      std::uint32_t num_occurences;
    };

    Result result = {guess};

    std::uint32_t num_unique_chars = 0;
    struct char_map_entry char_map[WORD_LEN] = {0};

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      const unsigned char ch = answer.char_at(i);

      bool have_char_already = false;
      for (std::uint32_t j = 0; j < num_unique_chars; ++j) {
        if (char_map[j].ch == ch) {
          have_char_already = true;
          ++char_map[j].num_occurences;

          break;
        }
      }

      if (!have_char_already) {
        char_map[num_unique_chars].ch = ch;
        char_map[num_unique_chars].num_occurences = 1;
        ++num_unique_chars;
      }
    }

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      const unsigned char guess_ch = guess.char_at(i);

      if (answer.has_char_at(guess_ch, i)) {
        result.comparisons[i] = Comparison::ExactMatch;

        for (std::uint32_t j = 0; j < num_unique_chars; ++j) {
          if (char_map[j].ch == guess_ch) {
            --char_map[j].num_occurences;

            break;
          }
        }
      }
    }

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      const unsigned char guess_ch = guess.char_at(i);

      if (!answer.has_char_at(guess_ch, i)) {
        result.comparisons[i] = Comparison::Absent;

        for (std::uint32_t j = 0; j < num_unique_chars; ++j) {
          if ((char_map[j].ch == guess_ch) &&
              (char_map[j].num_occurences > 0)) {
            result.comparisons[i] = Comparison::Contained;
            --char_map[j].num_occurences;

            break;
          }
        }
      }
    }

    result.fill_lanes();

    return result;
  }

  __host__ __device__ static Result
  from_comparisons(const Word &guess,
                   const Comparison (&comparisons)[WORD_LEN]) noexcept {
    Result result = {guess};

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      result.comparisons[i] = comparisons[i];
    }

    result.fill_lanes();

    return result;
  }

  __device__ bool allows_word(const Word &word) const noexcept {
    const WordVec guess_matches_word = guess.vec == word.vec;

    WordVec occurences_not_in_exact_match_positions = WordVec::zeros();

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      const WordVec equals_this_char = word.vec == char_at_broadcast[i];
      const WordVec equals_this_char_and_not_in_exact_match_position =
          equals_this_char & is_occurences_bounds_check;
      const int num_occurences_not_in_exact_match_positions =
          count_lanes_set(equals_this_char_and_not_in_exact_match_position);

      occurences_not_in_exact_match_positions |= WordVec::with_byte_at_index(
          num_occurences_not_in_exact_match_positions, i);
    }

    const WordVec has_required_match = guess_matches_word & is_exact_match;

    const WordVec meets_lower_bound =
        occurences_not_in_exact_match_positions >=
        lower_bound_equal_to_this_char_but_not_exact_match;
    const WordVec meets_upper_bound =
        occurences_not_in_exact_match_positions <=
        upper_bound_equal_to_this_char_but_not_exact_match;
    const WordVec meets_bounds =
        (meets_lower_bound & meets_upper_bound) & is_occurences_bounds_check;

    const WordVec conditions_met = has_required_match | meets_bounds;

    return are_all_set(conditions_met | ~WordVec::relevant_mask());
  }

private:
  __host__ __device__ Result(const Word &word) noexcept : guess(word) {}

  __host__ __device__ constexpr void fill_lanes() noexcept {
    is_exact_match = WordVec::zeros();
    is_occurences_bounds_check = WordVec::zeros();
    lower_bound_equal_to_this_char_but_not_exact_match = WordVec::zeros();
    upper_bound_equal_to_this_char_but_not_exact_match = WordVec::zeros();

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      const unsigned char ch = guess.char_at(i);
      const WordVec this_mask = WordVec::mask_for_index(i);

      switch (comparisons[i]) {
      case Comparison::ExactMatch: {
        is_exact_match |= this_mask;
        break;
      }

      case Comparison::Absent:
      case Comparison::Contained: {
        is_occurences_bounds_check |= this_mask;

        int lower_bound = 0;
        int upper_bound = WORD_LEN;

        bool has_absent = false;
        for (std::uint32_t j = 0; j < WORD_LEN; ++j) {
          if (guess.has_char_at(ch, j)) {
            if (comparisons[j] == Comparison::Contained) {
              ++lower_bound;
            } else if (comparisons[j] == Comparison::Absent) {
              has_absent = true;
            }
          }
        }

        if (has_absent) {
          upper_bound = lower_bound;
        }

        lower_bound_equal_to_this_char_but_not_exact_match |=
            WordVec::with_byte_at_index(lower_bound, i);
        upper_bound_equal_to_this_char_but_not_exact_match |=
            WordVec::with_byte_at_index(upper_bound, i);

        break;
      }
      }

      char_at_broadcast[i] = WordVec::broadcast_scalar_to_elem(ch);
    }

    guess.vec &= WordVec::relevant_mask();
  }

  Word guess;
  WordVecElem char_at_broadcast[WORD_LEN];
  WordVec is_exact_match;
  WordVec is_occurences_bounds_check;

  WordVec lower_bound_equal_to_this_char_but_not_exact_match;
  WordVec upper_bound_equal_to_this_char_but_not_exact_match;

  Comparison comparisons[WORD_LEN];
};
#endif
