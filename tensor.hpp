#include<cassert>
#include<cstdio>
#include<cstring>
#include<type_traits>

template<typename T, size_t... Ns>
class TensorDelegate;

template<typename T, size_t N>
class TensorDelegate<T, N> {
    public:
        TensorDelegate(T* data): data(data) {}
        T& operator[](const size_t idx) {
            assert(idx < N);
            return data[idx];
        }
    private:
        T *data;
};

template<typename T, size_t N, size_t... Ns>
class TensorDelegate<T, N, Ns...> {
    public:
        TensorDelegate(T* data): data(data) {}
        TensorDelegate<T, Ns...> operator[](const size_t idx) {
            assert(idx < N);
            return TensorDelegate<T, Ns...>(data + idx * stride);
        }
    private:
        static constexpr size_t stride = (... * Ns);
        T* data;
};

template<class T, size_t... N>
class Tensor;

template<class T, size_t N, size_t...Ns>
class Tensor<T, N, Ns...> {
    public:
        Tensor(){
            data = new T[size];     
        }
        ~Tensor() {
            delete data;
        }
        Tensor(const Tensor& tensor) {
            data = new T[size];
            std::memcpy(data, tensor.data, size*sizeof(T));
        }
        static size_t get_size() {
            return size;
        }
        T* get_data() const {
            return data;
        }
        TensorDelegate<T, Ns...> operator[] (const size_t idx) {
            assert(idx < N);
            return TensorDelegate<T, Ns...>(data + idx * stride);
        } 
    private:
        T* data;
        constexpr static size_t size = N * (... * Ns);
        constexpr static size_t stride = (... * Ns);
};

template<class T, size_t N>
class Tensor<T, N> {
    public:
        Tensor(){
            data = new T[size];     
        }
        ~Tensor() {
            delete data;
        }
        Tensor(const Tensor& tensor) {
            data = new T[size];
            std::memcpy(data, tensor.data, size*sizeof(T));
        }
        static size_t get_size() {
            return size;
        }
        T* get_data() const {
            return data;
        }
        T& operator[] (const size_t idx) {
            assert(idx < N);
            return data[idx];
        } 
    private:
        T* data;
        constexpr static size_t size = N;
};
