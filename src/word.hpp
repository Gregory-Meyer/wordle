#ifndef WORD_H
#define WORD_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

inline constexpr std::size_t WORD_LEN = 5;
static_assert(WORD_LEN < sizeof(std::uint64_t));

inline constexpr char WORD_MAPPING[] = "abcdefghijklmnopqrstuvwxyz0123456789";
inline constexpr std::size_t NUM_CHARS = sizeof(WORD_MAPPING) - 1;

struct Word {
  union {
    unsigned char data[WORD_LEN];
    std::uint64_t storage;
  };

  __host__ __device__ bool contains_char(unsigned char ch) const noexcept {
    for (std::uint32_t i = 0; i < std::uint32_t(WORD_LEN); ++i) {
      if (data[i] == ch) {
        return true;
      }
    }

    return false;
  }

  __host__ __device__ bool has_char_at(unsigned char ch,
                                       std::uint32_t idx) const noexcept {
    return data[idx] == ch;
  }

  __host__ __device__ unsigned char char_at(std::uint32_t idx) const noexcept {
    return data[idx];
  }
};

constexpr std::optional<Word> make_word(std::string_view sv) noexcept {
  if (sv.size() != WORD_LEN) {
    return std::nullopt;
  }

  Word w = {0};

  for (std::size_t i = 0; i < sv.size(); ++i) {
    const unsigned char ch = sv[i];

    if ((ch >= 'a') && (ch <= 'z')) {
      w.data[i] = ch - 'a';
    } else if ((ch >= 'A') && (ch <= 'Z')) {
      w.data[i] = 'a' + (ch - 'A');
    } else if ((ch >= '0') && (ch <= '9')) {
      w.data[i] = ('z' - 'a' + 1) + (ch - '0');
    } else {
      return std::nullopt;
    }
  }

  return w;
}

inline std::ostream &operator<<(std::ostream &os, const Word &word) {
  char as_chars[WORD_LEN + 1] = {0};
  for (std::size_t i = 0; i < WORD_LEN; ++i) {
    as_chars[i] = WORD_MAPPING[word.data[i]];
  }

  return os.write(as_chars, std::streamsize(WORD_LEN));
}

#endif
