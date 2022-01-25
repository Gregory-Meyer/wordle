#ifndef WORD_H
#define WORD_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

__host__ __device__ constexpr std::uint32_t
div_round_up(std::uint32_t numerator, std::uint32_t denominator) noexcept {
  const std::uint32_t quotient = numerator / denominator;
  if (numerator % denominator != 0) {
    return quotient + 1;
  }

  return quotient;
}

inline constexpr std::uint32_t WORD_LEN = 5;
inline constexpr std::uint32_t WORD_DATA_LEN = WORD_LEN + 1;
using WordData = unsigned char[WORD_DATA_LEN];

using WordStorageElem = std::uint64_t;
inline constexpr std::uint32_t WORD_STORAGE_LEN =
    div_round_up(sizeof(WordData), sizeof(WordStorageElem));

class WordStorage {
public:
  WordStorage() noexcept = default;

  [[gnu::always_inline]] __host__ __device__ static constexpr WordStorage
  zeroed() noexcept {
    return WordStorage{0};
  }

  [[gnu::always_inline]] __device__ constexpr friend WordStorage
  __shfl_sync(unsigned int mask, const WordStorage &var, int offset) noexcept {
    WordStorage result = WordStorage::zeroed();

    for (std::uint32_t i = 0; i < WORD_STORAGE_LEN; ++i) {
      result.elems[i] = __shfl_sync(mask, var.elems[i], offset);
    }

    return result;
  }

private:
  [[gnu::always_inline]] __host__
      __device__ constexpr WordStorage(WordStorageElem elem) noexcept
      : elems{elem} {}

  WordStorageElem elems[WORD_STORAGE_LEN];
};

using WordVecElem = std::uint32_t;
inline constexpr std::uint32_t WORD_VEC_LEN =
    div_round_up(sizeof(WordData), sizeof(WordVecElem));
inline constexpr std::uint32_t CHARS_PER_VEC_ELEM =
    sizeof(WordVecElem) / sizeof(unsigned char);
inline constexpr std::uint32_t CHARS_PER_VEC =
    CHARS_PER_VEC_ELEM * WORD_VEC_LEN;

class WordVec {
public:
  WordVec() noexcept = default;

  constexpr WordVec(const WordVec &other) noexcept = default;

  constexpr WordVec(WordVec &&other) noexcept = default;

  constexpr WordVec &operator=(const WordVec &other) noexcept = default;

  constexpr WordVec &operator=(WordVec &&other) noexcept = default;

  [[gnu::always_inline]] __host__ __device__ static constexpr WordVec
  zeros() noexcept {
    return WordVec::from_scalar(0);
  }

  [[gnu::always_inline]] __host__ __device__ static constexpr WordVec
  mask_for_index(std::uint32_t i) noexcept {
    return WordVec::with_byte_at_index(UCHAR_MAX, i);
  }

  [[gnu::always_inline]] __host__ __device__ static constexpr WordVec
  relevant_mask() noexcept {
    WordVec vec = WordVec::zeros();

    for (std::uint32_t i = 0; i < WORD_LEN; ++i) {
      vec |= WordVec::mask_for_index(i);
    }

    return vec;
  }

  [[gnu::always_inline]] __host__ __device__ static constexpr WordVec
  with_byte_at_index(unsigned char ch, std::uint32_t i) noexcept {
    WordVec vec = WordVec::zeros();

    vec.elems[i / CHARS_PER_VEC_ELEM] =
        WordVecElem(ch) << (CHAR_BIT * (i % CHARS_PER_VEC_ELEM));

    return vec;
  }

  [[gnu::always_inline]] __host__ __device__ static constexpr WordVecElem
  broadcast_scalar_to_elem(unsigned char ch) noexcept {
    return (WordVecElem(ch) << 24) | (WordVecElem(ch) << 16) |
           (WordVecElem(ch) << 8) | (WordVecElem(ch) << 0);
  }

  [[gnu::always_inline]] __host__ __device__ static constexpr WordVec
  from_scalar(unsigned char ch) noexcept {
    const WordVecElem elem = WordVec::broadcast_scalar_to_elem(ch);

    return WordVec{elem, elem};
  }

  [[gnu::always_inline]] __host__ __device__ constexpr WordVec &
  operator&=(const WordVec &other) noexcept {
    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      elems[i] &= other.elems[i];
    }

    return *this;
  }

  [[gnu::always_inline]] __host__ __device__ constexpr WordVec &
  operator&=(WordVecElem other) noexcept {
    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      elems[i] &= other;
    }

    return *this;
  }

  [[gnu::always_inline]] __host__ __device__ constexpr WordVec &
  operator|=(const WordVec &other) noexcept {
    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      elems[i] |= other.elems[i];
    }

    return *this;
  }

  [[gnu::always_inline]] __host__ __device__ constexpr WordVec &
  operator|=(WordVecElem other) noexcept {
    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      elems[i] |= other;
    }

    return *this;
  }

  [[gnu::always_inline]] __device__ constexpr friend WordVec
  __shfl_sync(unsigned int mask, const WordVec &var, int offset) noexcept {
    WordVec result = WordVec::zeros();

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __shfl_sync(mask, var.elems[i], offset);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend bool
  are_any_set(const WordVec &vec) noexcept {
    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      if (vec.elems[i] != 0) {
        return true;
      }
    }

    return false;
  }

  [[gnu::always_inline]] __host__ __device__ constexpr friend bool
  are_all_set(const WordVec &vec) noexcept {
    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      if (vec.elems[i] != 0xffffffff) {
        return false;
      }
    }

    return true;
  }

  [[gnu::always_inline]] __device__ friend int
  count_lanes_set(const WordVec &vec) noexcept {
    int num_set = 0;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      num_set += __popc(vec.elems[i]);
    }

    return num_set / 8;
  }

  [[gnu::always_inline]] __host__ __device__ constexpr friend WordVec
  operator&(const WordVec &lhs, const WordVec &rhs) noexcept {
    WordVec result = WordVec::zeros();

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = lhs.elems[i] & rhs.elems[i];
    }

    return result;
  }

  [[gnu::always_inline]] __host__ __device__ constexpr friend WordVec
  operator|(const WordVec &lhs, const WordVec &rhs) noexcept {
    WordVec result = WordVec::zeros();

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = lhs.elems[i] | rhs.elems[i];
    }

    return result;
  }

  [[gnu::always_inline]] __host__ __device__ constexpr friend WordVec
  operator~(const WordVec &vec) noexcept {
    WordVec result = WordVec::zeros();

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = ~vec.elems[i];
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator==(const WordVec &lhs, const WordVec &rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpeq4(lhs.elems[i], rhs.elems[i]);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator==(WordVecElem lhs, const WordVec &rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpeq4(lhs, rhs.elems[i]);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator==(const WordVec &lhs, WordVecElem rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpeq4(lhs.elems[i], rhs);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator!=(const WordVec &lhs, const WordVec &rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpne4(lhs.elems[i], rhs.elems[i]);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator!=(WordVecElem lhs, const WordVec &rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpne4(lhs, rhs.elems[i]);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator!=(const WordVec &lhs, WordVecElem rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpne4(lhs.elems[i], rhs);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator>=(const WordVec &lhs, const WordVec &rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpgeu4(lhs.elems[i], rhs.elems[i]);
    }

    return result;
  }

  [[gnu::always_inline]] __device__ friend WordVec
  operator<=(const WordVec &lhs, const WordVec &rhs) noexcept {
    WordVec result;

    for (std::uint32_t i = 0; i < WORD_VEC_LEN; ++i) {
      result.elems[i] = __vcmpleu4(lhs.elems[i], rhs.elems[i]);
    }

    return result;
  }

private:
  [[gnu::always_inline]] __host__
      __device__ constexpr WordVec(WordVecElem lower,
                                   WordVecElem upper) noexcept
      : elems{lower, upper} {}

  WordVecElem elems[WORD_VEC_LEN];
};

struct Word {
  union {
    WordData data;
    WordStorage storage;
    WordVec vec;
  };

  __host__ __device__ std::uint32_t has_char(unsigned char ch) const noexcept {
    std::uint32_t num_occurences = 0;

    for (std::uint32_t i = 0; i < std::uint32_t(WORD_LEN); ++i) {
      if (data[i] == ch) {
        ++num_occurences;
      }
    }

    return num_occurences;
  }

  __host__ __device__ bool has_char_at(unsigned char ch,
                                       std::uint32_t idx) const noexcept {
    return data[idx] == ch;
  }

  __host__ __device__ unsigned char char_at(std::uint32_t idx) const noexcept {
    return data[idx];
  }

  friend bool operator==(const Word &lhs, const Word &rhs) noexcept {
    return std::string_view(lhs) == std::string_view(rhs);
  }

  friend bool operator!=(const Word &lhs, const Word &rhs) noexcept {
    return std::string_view(lhs) != std::string_view(rhs);
  }

  friend bool operator<(const Word &lhs, const Word &rhs) noexcept {
    return std::string_view(lhs) < std::string_view(rhs);
  }

  friend bool operator<=(const Word &lhs, const Word &rhs) noexcept {
    return std::string_view(lhs) <= std::string_view(rhs);
  }

  friend bool operator>(const Word &lhs, const Word &rhs) noexcept {
    return std::string_view(lhs) > std::string_view(rhs);
  }

  friend bool operator>=(const Word &lhs, const Word &rhs) noexcept {
    return std::string_view(lhs) >= std::string_view(rhs);
  }

  explicit operator std::string_view() const noexcept {
    return std::string_view(reinterpret_cast<const char *>(&data[0]), WORD_LEN);
  }
};

constexpr std::optional<Word> make_word(std::string_view sv) noexcept {
  if (sv.size() != WORD_LEN) {
    return std::nullopt;
  }

  Word w = {0};

  std::copy(sv.cbegin(), sv.cend(), w.data);
  w.data[WORD_LEN] = '\0';

  return w;
}

inline std::ostream &operator<<(std::ostream &os, const Word &word) {
  return os.write(reinterpret_cast<const char *>(word.data),
                  std::streamsize(WORD_LEN));
}

#endif
