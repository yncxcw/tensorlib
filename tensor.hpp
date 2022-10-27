#pragma once

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <type_traits>

template <typename T> T sum_array(const T *t, size_t length) {
  T sum = 0;
  for (size_t i = 0; i < length; i++) {
    sum += t[i];
  }
  return sum;
}

template <typename T> void add_array(T *src, const T *dst, size_t length) {
  for (size_t i = 0; i < length; i++) {
    src[i] += dst[i];
  }
}

template <typename T> void add_array(T *src, const T v, size_t length) {
  for (size_t i = 0; i < length; i++) {
    src[i] += v;
  }
}

template <typename T, size_t... Ns> class TensorDelegate;

template <typename T, size_t N> class TensorDelegate<T, N> {
public:
  TensorDelegate(T *data) : data(data) {}
  T &operator[](const size_t idx) {
    assert(idx < N);
    return data[idx];
  }

private:
  T *data;
};

template <typename T, size_t N, size_t... Ns>
class TensorDelegate<T, N, Ns...> {
public:
  TensorDelegate(T *data) : data(data) {}
  TensorDelegate<T, Ns...> operator[](const size_t idx) {
    assert(idx < N);
    return TensorDelegate<T, Ns...>(data + idx * stride);
  }

private:
  static constexpr size_t stride = (... * Ns);
  T *data;
};

template <class T, size_t... N> class Tensor;

template <class T, size_t N, size_t... Ns> class Tensor<T, N, Ns...> {
public:
  Tensor() {}
  Tensor(const T v) {
    for (size_t i = 0; i < size; i++) {
      data[i] = std::move(v);
    }
  }
  ~Tensor() {}
  Tensor(const Tensor<T, N, Ns...> &tensor) {
    static_assert(tensor.size == size);
    std::memcpy(data, tensor.data, size * sizeof(T));
  }

  Tensor(const Tensor<T, N, Ns...> &&tensor) = delete;

  constexpr size_t get_size() const { return size; }
  T *get_data() { return data; }
  TensorDelegate<T, Ns...> operator[](const size_t idx) {
    assert(idx < N);
    return TensorDelegate<T, Ns...>(data + idx * stride);
  }
  T sum() { return sum_array<T>(data, size); }
  void add(const T v) { add_array(data, v, size); }
  Tensor<T, N, Ns...> &operator+=(const Tensor<T, N, Ns...> &other) {
    static_assert(size == other.get_size());
    add_array<T>(data, other.data, size);
    return *this;
  }

private:
  constexpr static size_t size = N * (... * Ns);
  constexpr static size_t stride = (... * Ns);
  T data[size];
};

template <class T, size_t N> class Tensor<T, N> {
public:
  Tensor() {}
  Tensor(const T v) {
    for (size_t i = 0; i < size; i++) {
      data[i] = std::move(v);
    }
  }
  ~Tensor() {}
  Tensor(const Tensor<T, N> &tensor) {
    static_assert(tensor.size == size);
    std::memcpy(data, tensor.data, size * sizeof(T));
  }
  Tensor(const Tensor<T, N> &&tensor) = delete;

  constexpr size_t get_size() const { return size; }
  T *get_data() { return data; }
  T &operator[](const size_t idx) {
    assert(idx < N);
    return data[idx];
  }

  T sum() { return sum_array<T>(data, size); }

  void add(const T v) { add_array(data, v, size); }

  Tensor<T, N> &operator+=(const Tensor<T, N> &other) {
    static_assert(size == other.get_size());
    add_array<T>(data, other.get_data(), size);
    return *this;
  }

private:
  constexpr static size_t size = N;
  T data[size];
};