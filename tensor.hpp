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
  constexpr size_t get_size() const { return size; }
  T *get_data() const { return data; }
  TensorDelegate<T, Ns...> operator[](const size_t idx) {
    assert(idx < N);
    return TensorDelegate<T, Ns...>(data + idx * stride);
  }
  T sum() { return sum_array<T>(data, size); }

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
  constexpr size_t get_size() const { return size; }
  T *get_data() const { return data; }
  T &operator[](const size_t idx) {
    assert(idx < N);
    return data[idx];
  }

  T sum() { return sum_array<T>(data, size); }

private:
  constexpr static size_t size = N;
  T data[size];
};
