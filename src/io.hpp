#ifndef IO_HPP
#define IO_HPP

#include "util.hpp"
#include "word.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<Word>
read_line_delimited_words(const std::string &pathname) {
  std::ifstream input(pathname);
  if (!input.is_open()) {
    throw std::runtime_error(
        format("open word list '" << pathname << "' for reading"));
  }

  std::vector<Word> words;

  std::uint64_t line = 0;
  std::string this_word_buf;
  while (std::getline(input, this_word_buf)) {
    ++line;

    if (const std::optional<Word> this_word = make_word(this_word_buf)) {
      words.push_back(*this_word);
    } else {
      std::cerr << "warning: skipping line " << line << " of word list '"
                << pathname << "'\n";
    }
  }

  return words;
}

inline void write_line_delimited_words(const std::vector<Word> &words,
                                       const std::string &pathname) {
  std::ofstream output(pathname);
  if (!output.is_open()) {
    throw std::runtime_error(
        format("create word list '" << pathname << "' for writing"));
  }

  std::copy(words.cbegin(), words.cend(),
            std::ostream_iterator<Word>(output, "\n"));

  output.close();

  if (!output) {
    throw std::runtime_error(
        format("write word to word list '" << pathname << '\''));
  }
}

#endif
