/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Containers used to store and manage other objects.
 *************************************************************************/
#ifndef CONTAINER_H_
#define CONTAINER_H_

#include <mutex>
#include "common/macros.h"
/*
 * An implement of lazy-static singleton class.
 */
template <typename T>
class Singleton {
 public:
  static T &Global() {
    static T _instance;
    return _instance;
  }

 private:
  Singleton() {}
  ~Singleton() = default;

  //! disable copy construction
  Singleton(const Singleton &);
  //! disable assignment
  Singleton &operator=(const Singleton &);
};
/*
 * An implement to wrap Destroy() function as operator().
 * It will be used for smart pointers to automatically release
 * some objects.
 */
struct Destroyer {
  template <class T>
  void operator()(T *t) {
    if (t) {
      USED_VAR(t->Destroy());
    }
  }
};

template <typename T>
using SUniquePtr = std::unique_ptr<T, Destroyer>;

template <typename T>
class RingQueue {
 public:
  explicit RingQueue(uint32_t size) : size_(size) { container_.resize(size); }
  void push(const T &data) {
    container_[(front_ + current_) % size_] = data;
    ++current_;
    CHECK_LE(current_, size_);
  };
  T &pop() {
    if (current_) {
      T &ret = container_[front_];
      front_ = (front_ + 1) % size_;
      --current_;
      return ret;
    }
    abort();
  };
  size_t size() const { return current_; }

 private:
  std::vector<T> container_;
  uint32_t front_   = 0;
  uint32_t current_ = 0;
  uint32_t size_    = 0;
};

#endif  // CONTAINER_H_
