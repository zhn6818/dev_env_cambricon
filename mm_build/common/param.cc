#include "common/param.h"

std::string binary_name = "UserBinary";

Args ArrangeArgs(const int argc, char *argv[], int offset) {
  if (argc == 0) {
    SLOG(ERROR) << "Empty command line";
    abort();
  }
  ArgValue all_args{};
  all_args.assign(argv, argv + argc);
  binary_name = all_args[0];
  Args ret{};
  // Rearrangment
  std::string head = "";
  for (auto iter = all_args.begin() + offset; iter != all_args.end(); ++iter) {
    if (iter->find("--") == 0) {
      head = iter->substr(2);
      if (ret.find(head) != ret.end()) {
        SLOG(ERROR) << "Duplicated command line arg: " << head << " for " << binary_name;
        abort();
      }
      ret[head] = std::vector<std::string>();
    } else if (!head.empty()) {
      ret[head].push_back(*iter);
    } else {
      SLOG(ERROR) << "Unknown command line arg format: " << *iter << " for " << binary_name;
      SLOG(ERROR) << "Please use the following format: --key value";
      abort();
    }
  }
  return ret;
}

std::string Trim(std::string s) {
  if (!s.empty()) {
    s.erase(0, s.find_first_not_of(" "));
  }
  if (!s.empty()) {
    s.erase(s.find_last_not_of(" ") + 1);
  }
  return s;
}

template <>
bool ParseValue<int>(const std::string &s, int *value) {
  const char *head = s.c_str();
  char *end_pointer;
  int64_t v = std::strtoll(head, &end_pointer, 10);
  if (end_pointer != head + s.size()) {
    return false;
  }
  *value = Clamp<int>(v);
  if ((*value) != v) {
    SLOG(WARNING) << "An int string is overflow.";
    return false;
  }
  return true;
}

template <>
bool ParseValue<float>(const std::string &s, float *value) {
  const char *head = s.c_str();
  char *end_pointer;
  *value = std::strtof(head, &end_pointer);
  if (end_pointer != head + s.size()) {
    return false;
  }
  return true;
}

template <>
bool ParseValue<int64_t>(const std::string &s, int64_t *value) {
  const char *head = s.c_str();
  char *end_pointer;
  *value = std::strtoll(head, &end_pointer, 10);
  if (end_pointer != head + s.size()) {
    return false;
  }
  return true;
}

ArgBase::~ArgBase() {
  if (next_) {
    delete next_;
  }
}

void ArgBase::SetNext(ArgBase *next) {
  if (next_) {
    next_->SetNext(next);
  } else {
    next_ = next;
  }
}

bool ArgBase::ReadIn(Args *obj) {
  auto ret = UpdateArg(obj);
  if (next_) {
    ret = next_->ReadIn(obj) && ret;
  }
  return ret;
}

std::string ArgBase::DebugString() const {
  std::string s{};
  s += WriteOut();
  if (next_) {
    s += next_->DebugString();
  }
  return s;
}

std::string ArgBase::Synopsis() const {
  std::stringstream ss;
  ss << "\nSynopsis:\n";
  ss << "          " << binary_name << " ";
  ss << SynopCollect(0, false);
  ss << "\nOptions:\n";
  ss << SynopCollect(10, true);
  return ss.str();
}

std::string ArgBase::SynopCollect(int align, bool detailed) const {
  std::string s{};
  s += Doc(align, detailed);
  if (next_) {
    s += next_->SynopCollect(align, detailed);
  }
  return s;
}

std::ostream &operator<<(std::ostream &os, const ArgBase &arg) {
  return os << arg.DebugString();
}

void ArgListBase::ReadIn(Args args) {
  for (auto e_ : args) {
    if (e_.first == "help") {
      SLOG(INFO) << Synopsis();
      exit(0);
    }
  }
  auto res = front->ReadIn(&args);
  if (args.size() > 0) {
    SLOG(ERROR) << "Unrecognized args:";
    for (auto e_ : args) {
      SLOG(ERROR) << "--" << e_.first << e_.second;
    }
    res = false;
  }
  if (!res) {
    SLOG(ERROR) << Synopsis();
    abort();
  }
}

std::string ArgListBase::DebugString() const {
  return front->DebugString();
}

std::string ArgListBase::Synopsis() const {
  return front->Synopsis();
}

ArgListBase::~ArgListBase() {
  if (front) {
    delete front;
    front = nullptr;
  }
}

std::ostream &operator<<(std::ostream &os, const ArgListBase &arg) {
  return os << arg.DebugString();
}
