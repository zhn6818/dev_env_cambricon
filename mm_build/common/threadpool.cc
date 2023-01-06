/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "common/threadpool.h"

ThreadPool::ThreadPool() : dynamic_grow_(true) {
  CreateThread();
}

ThreadPool::ThreadPool(uint32_t thread_size, bool dynamic_grow) : dynamic_grow_(dynamic_grow) {
  CreateThread(thread_size);
}

int ThreadPool::GetThreadPoolSize() {
  return thread_num_;
}

int ThreadPool::GetCurrentThreadSize() {
  std::unique_lock<std::mutex> lk(mtx_);
  return thread_num_ - idle_num_;
}

int ThreadPool::GetIdleThreadSize() {
  return idle_num_;
}

void ThreadPool::ActivateThreadPool() {
  running_.store(true, std::memory_order_relaxed);
  CreateThread(kThreadPoolMaxNum);
}

void ThreadPool::DeactivateThreadPool() {
  DestroyThreadPool();
}

void ThreadPool::DestroyThreadPool() {
  {
    std::unique_lock<std::mutex> lk(mtx_);
    running_.store(false, std::memory_order_relaxed);
  }
  cv_.notify_all();
  for (auto &thd : pool_) {
    if (thd.joinable()) {
      thd.join();
    }
  }
}

ThreadPool::~ThreadPool() {
  DestroyThreadPool();
}

void ThreadPool::CreateThread(int thread_size) {
  std::unique_lock<std::mutex> lk(mtx_);
  for (; thread_num_ < kThreadPoolMaxNum && thread_size > 0; --thread_size) {
    pool_.emplace_back([this] {
      while (running_.load(std::memory_order_acquire)) {
        Job job;
        {
          std::unique_lock<std::mutex> lk(mtx_);
          cv_.wait(lk,
                   [this] { return !running_.load(std::memory_order_acquire) || !jobs_.empty(); });
          if (!running_.load(std::memory_order_acquire) && jobs_.empty())
            return;
          job = std::move(jobs_.front());
          jobs_.pop();
        }
        idle_num_--;
        job();
        idle_num_++;
      }
    });
    thread_num_++;
    idle_num_++;
  }
}
