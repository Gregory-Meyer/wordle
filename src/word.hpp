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

struct Word {
  union {
    unsigned char data[WORD_LEN + 1];
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
  std::copy(sv.begin(), sv.end(), w.data);
  w.data[WORD_LEN] = '\0';

  return w;
}

inline std::ostream &operator<<(std::ostream &os, const Word &word) {
  return os.write(reinterpret_cast<const char *>(word.data),
                  std::streamsize(WORD_LEN));
}

#endif
