/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: An implement for threadpool.
 *************************************************************************/
#ifndef THREADPOOL_H_
#define THREADPOOL_H_
#include <vector>
#include <queue>
#include <atomic>
#include <future>
#include <mutex>
#include <functional>
#include <thread>
#include <utility>
#include <memory>
#include <condition_variable>
#include "common/logger.h"
#include "common/container.h"

const size_t kThreadPoolMaxNum     = std::thread::hardware_concurrency();
const size_t kThreadPoolDefaultNum = kThreadPoolMaxNum / 4;

class ThreadPool {
 public:
  ThreadPool();
  explicit ThreadPool(uint32_t thread_size, bool dynamic_grow = false);
  int GetThreadPoolSize();
  int GetCurrentThreadSize();
  int GetIdleThreadSize();
  void ActivateThreadPool();
  void DeactivateThreadPool();

  template <class F, class... Args>
  auto AddTask(F &&f, Args &&... args) -> std::future<decltype(f(args...))> {
    if (!running_) {
      SLOG(ERROR) << "ThreadPool running failed.";
      abort();
    }
    using RetType = decltype(f(args...));
    auto task     = std::make_shared<std::packaged_task<RetType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<RetType> future = task->get_future();
    {
      std::lock_guard<std::mutex> lk(mtx_);
      jobs_.emplace([task]() { (*task)(); });
    }
    if (idle_num_.load(std::memory_order_acquire) < 1 && thread_num_ < kThreadPoolMaxNum &&
        dynamic_grow_ == true) {
      CreateThread(1);
    }
    cv_.notify_one();

    return future;
  }
  ~ThreadPool();

 private:
  void CreateThread(int thread_size = kThreadPoolDefaultNum);
  void DestroyThreadPool();

 private:
  using Job = std::function<void()>;
  std::mutex mtx_;
  bool dynamic_grow_ = false;
  std::condition_variable cv_;
  std::vector<std::thread> pool_;
  std::queue<Job> jobs_;
  std::atomic<bool> running_{true};
  std::atomic<int> idle_num_{0};
  std::atomic<size_t> thread_num_{0};
};

typedef Singleton<ThreadPool> ThreadPoolSingleton;
#endif  // THREADPOOL_H_
