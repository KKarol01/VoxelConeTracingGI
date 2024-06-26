#pragma once

#include <cstdint>
#include <atomic>
#include <compare>

typedef int8_t i8;
typedef uint8_t u8;
typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;
typedef float f32;
typedef double f64;
typedef const char* cstring;

struct HandleGenerate_T {};
inline constexpr HandleGenerate_T HandleGenerate;

template<typename T> struct Handle;

template<typename T> struct HandleGenerator {
    inline static u64 generate() { return ++counter; }
    inline static std::atomic_uint64_t counter{0ull};
};

template<typename T> struct Handle { 
    constexpr Handle() = default;
    constexpr explicit Handle(u64 handle): handle(handle) {}
    constexpr explicit Handle(HandleGenerate_T): Handle(HandleGenerator<T>::generate()) {}
    constexpr Handle(const Handle& other) noexcept = default;
    constexpr Handle& operator=(const Handle& other) noexcept = default;
    constexpr Handle(Handle&& other) noexcept { *this = std::move(other); }
    constexpr Handle& operator=(Handle&& other) noexcept {
        handle = std::exchange(other.handle, 0u);
        return *this;
    }
    constexpr operator bool() const noexcept { return handle != 0ull; }
    constexpr auto operator<=>(const Handle<T>& other) const noexcept = default;
    u64 handle{0ull}; 
};

namespace std {
    template<typename T> struct hash<Handle<T>> {
        size_t operator()(const Handle<T>& h) const { return h.handle; }
    };
}

template<class... Ts> struct visitor : Ts...  { using Ts::operator()...; };
template<class... Ts> visitor(Ts...) -> visitor<Ts...>;