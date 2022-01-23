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
  std::uint32_t guess_wide[WORD_LEN];
  std::uint32_t is_absent[2];
  std::uint32_t is_exact_match[2];
  std::uint32_t is_contained[2];
  std::uint32_t is_index[WORD_LEN][2];

  Comparison comparisons[WORD_LEN];

  __device__ bool allows_word(const Word &word) const noexcept {
    const std::uint32_t guess_matches_word[2] = {
        __vcmpeq4(guess.storage_u32[0], word.storage_u32[0]),
        __vcmpeq4(guess.storage_u32[1], word.storage_u32[1]),
    };

    std::uint32_t contains_char_at[2] = {0, 0};
    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      const std::uint32_t low = __vcmpeq4(guess_wide[i], word.storage_u32[0]);
      const std::uint32_t high = __vcmpeq4(guess_wide[i], word.storage_u32[1]);

      const std::uint32_t contains_this_char =
          (low | high) != 0 ? ~std::uint32_t(0) : 0;

      contains_char_at[0] |= contains_this_char & is_index[i][0];
      contains_char_at[1] |= contains_this_char & is_index[i][1];
    }

    const std::uint32_t has_required_match[2] = {
        guess_matches_word[0] & is_exact_match[0],
        guess_matches_word[1] & is_exact_match[1]};
    const std::uint32_t contains_required_char[2] = {
        contains_char_at[0] & is_contained[0],
        contains_char_at[1] & is_contained[1]};
    const std::uint32_t doesnt_contain_required_absent_char[2] = {
        ~contains_char_at[0] & is_absent[0],
        ~contains_char_at[1] & is_absent[1]};

    const std::uint32_t conditions_met[2] = {
        has_required_match[0] | contains_required_char[0] |
            doesnt_contain_required_absent_char[0],
        has_required_match[1] | contains_required_char[1] |
            doesnt_contain_required_absent_char[1],
    };

    return (conditions_met[0] == ~std::uint32_t(0)) &&
           (conditions_met[1] == std::uint32_t(0xff));
  }

  __host__ __device__ constexpr void fill_lanes() noexcept {
    is_exact_match[0] = 0;
    is_exact_match[1] = 0;

    is_contained[0] = 0;
    is_contained[1] = 0;

    is_absent[0] = 0;
    is_absent[1] = 0;

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      const unsigned char ch = guess.char_at(i);

      const std::uint32_t this_ch_pos_mask =
          (std::uint32_t(0xff) << (8 * (i % 4)));

      switch (comparisons[i]) {
      case Comparison::Absent: {
        is_absent[i / 4] |= this_ch_pos_mask;
        break;
      }

      case Comparison::ExactMatch: {
        is_exact_match[i / 4] |= this_ch_pos_mask;
        break;
      }

      case Comparison::Contained: {
        is_contained[i / 4] |= this_ch_pos_mask;
        break;
      }
      }

      guess_wide[i] = (std::uint32_t(ch) << 24) | (std::uint32_t(ch) << 16) |
                      (std::uint32_t(ch) << 8) | (std::uint32_t(ch) << 0);
    }

    is_index[0][0] = std::uint32_t(0xff) << 0;
    is_index[0][1] = std::uint32_t(0);
    is_index[1][0] = std::uint32_t(0xff) << 8;
    is_index[1][1] = std::uint32_t(0);
    is_index[2][0] = std::uint32_t(0xff) << 16;
    is_index[2][1] = std::uint32_t(0);
    is_index[3][0] = std::uint32_t(0xff) << 24;
    is_index[3][1] = std::uint32_t(0);
    is_index[4][0] = std::uint32_t(0);
    is_index[4][1] = std::uint32_t(0xff) << 0;
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

  result.fill_lanes();

  return result;
}

#endif
