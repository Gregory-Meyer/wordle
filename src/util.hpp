#ifndef UTIL_HPP
#define UTIL_HPP

#include <cuda.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define format(...)                                                            \
  (std::move(static_cast<std::ostringstream &>(std::ostringstream().flush()    \
                                               << __VA_ARGS__))                \
       .str())

#define check_cuda(...)                                                        \
  _do_check_cuda((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__, __func__)
#define check_cuda_safe(...)                                                   \
  _do_check_cuda_safe((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__, __func__)

inline void _do_check_cuda(cudaError err, std::string_view what,
                           std::string_view file, std::int32_t line,
                           std::string_view func);
inline std::optional<std::string>
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
__host__ __device__ T *ptr_offset(T *ptr, std::uint32_t pitch,
                                  std::uint32_t row) noexcept {
  return reinterpret_cast<T *>(reinterpret_cast<char *>(ptr) + pitch * row);
}

template <typename T>
__host__ __device__ const T *ptr_offset(const T *ptr, std::uint32_t pitch,
                                        std::uint32_t row) noexcept {
  return reinterpret_cast<const T *>(reinterpret_cast<const char *>(ptr) +
                                     pitch * row);
}

template <typename T>
inline std::pair<std::unique_ptr<T[], CudaDeleter>, std::size_t>
make_unique_device_pitched(std::size_t width, std::size_t height) {
  T *ptr = nullptr;
  std::size_t pitch = 0;
  check_cuda(cudaMallocPitch(&ptr, &pitch, width * sizeof(T), height));

  return std::make_pair(std::unique_ptr<T[], CudaDeleter>(ptr), pitch);
}

template <typename T>
inline std::unique_ptr<T[], CudaDeleter> make_unique_device(std::size_t len) {
  T *ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, len * sizeof(T)));

  return std::unique_ptr<T[], CudaDeleter>(ptr);
}

template <typename T>
inline void copy_to_device_pitched(const T *src, std::size_t src_pitch, T *dst,
                                   std::size_t dst_pitch, std::size_t width,
                                   std::size_t height) {
  check_cuda(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width * sizeof(T),
                          height, cudaMemcpyHostToDevice));
}

template <typename T>
inline void copy_to_host_pitched(const T *src, std::size_t src_pitch, T *dst,
                                 std::size_t dst_pitch, std::size_t width,
                                 std::size_t height) {
  check_cuda(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width * sizeof(T),
                          height, cudaMemcpyDeviceToHost));
}

template <typename T>
inline void copy_to_host(const T *src, T *dst, std::size_t len) {
  check_cuda(cudaMemcpy(dst, src, len * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
inline void copy_to_device(const T *src, T *dst, std::size_t len) {
  check_cuda(cudaMemcpy(dst, src, len * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
inline std::unique_ptr<T[], CudaDeleter> to_device(const std::vector<T> &host) {
  auto device = make_unique_device<T>(host.size());

  copy_to_device(host.data(), device.get(), host.size());

  return device;
}

template <typename T>
inline std::vector<T> to_host(const T *device, std::size_t len) {
  std::vector<T> host(len);

  copy_to_host(device, host.data(), len);

  return host;
}

template <typename T> constexpr T min(T lhs, T rhs) noexcept {
  return (rhs < lhs) ? rhs : lhs;
}

inline void _do_check_cuda(cudaError err, std::string_view what,
                           std::string_view file, std::int32_t line,
                           std::string_view func) {
  if (std::optional<std::string> err_str =
          _do_check_cuda_safe(err, what, file, line, func)) {
    throw std::runtime_error(std::move(*err_str));
  }
}

inline std::optional<std::string>
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

#endif
