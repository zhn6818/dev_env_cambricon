/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Command-lind arg parser.
 *************************************************************************/
#ifndef PARAM_H_
#define PARAM_H_
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <unordered_map>
#include "common/logger.h"
#include "common/macros.h"
#include "common/data.h"

using ArgValue = std::vector<std::string>;
using Args = std::unordered_map<std::string, ArgValue>;
/*
 * Binary name for command line.
 */
extern std::string binary_name;
/*
 * A function to rearrange input args from char *[] list
 * to map<key, value>. Key matches args start with prefix
 * "--".
 */
Args ArrangeArgs(const int argc, char *argv[], int offset = 1);
/*
 * A helper function to remove spaces in head and tail for string.
 * e.g. "  function  " to “"function"
 */
std::string Trim(std::string s);
/*
 * A wrapper for strtof/strtoll, false will return if string cannot
 * be fully convert to target type.
 */
template <typename T>
bool ParseValue(const std::string &s, T *value);

template <>
bool ParseValue<int>(const std::string &s, int *value);

template <>
bool ParseValue<float>(const std::string &s, float *value);

template <>
bool ParseValue<int64_t>(const std::string &s, int64_t *value);
/*
 * A struct for arg conversion from ArgValue to derived types.
 * Support types:
 * 1.Bool (from 1/0, true/false, ignore capsules) to vector<string> with size 1.
 * 2.Float to vector<string> with size 1.
 * 3.Int to vector<string> with size 1.
 * 4.String to vector<string> (ignore capsules) with size 1.
 * 4.Vector<T>, T can be bool, float, int, or another vector<T>. Given that only
 * two separators, space and comma, can be used to define a vector in bash command,
 * nesting vectors here can not be more than 2.
 * Only string type can be set on alternative value manually.
 * ArgResolver::success shows parsing success or not.
 */
template <typename T>
struct ArgResolver
{
};
/*
 * Partial template specialization for string type.
 * String type args can be set alternative values.
 */
template <>
struct ArgResolver<std::string>
{
  std::string Dispatch(const ArgValue &v)
  {
    success = true;
    if (v.size() != 1)
    {
      success = false;
      return "";
    }
    auto ret = Trim(v[0]);
    if (alters_.size() && !ret.empty())
    {
      if (std::find(alters_.begin(), alters_.end(), ret) == alters_.end())
      {
        success = false;
        SLOG(ERROR) << ret << " fails to match arg alternative values. " << AlterString();
        return "";
      }
    }
    return ret;
  }
  bool SetAlternative(const ArgValue &v)
  {
    for (auto e_ : v)
    {
      alters_.push_back(Trim(e_));
    }
    return true;
  }
  std::string AlterString() const
  {
    std::string ret = "";
    if (alters_.size())
    {
      ret += "Allowed: ";
      for (size_t i = 0; i < alters_.size() - 1; ++i)
      {
        ret += alters_[i];
        ret += "/";
      }
      ret += alters_[alters_.size() - 1];
    }
    return ret;
  }
  ArgValue alters_ = {};
  bool success = true;
  std::string name = "<Str>";
};
/*
 * Partial template specialization for bool type.
 * Bool type args receives 0/1/true/false, and will be casted to lowercase
 * for argument matching to ignore capsules.
 * Bool type has no custom alternative values.
 */
template <>
struct ArgResolver<bool>
{
  bool Dispatch(const ArgValue &v)
  {
    success = true;
    if (v.size() != 1)
    {
      success = false;
      return false;
    }
    auto ret = Trim(v[0]);
    std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
    if (ret == "true" || ret == "1")
    {
      return true;
    }
    else if (ret == "false" || ret == "0")
    {
      return false;
    }
    success = false;
    return false;
  }
  bool SetAlternative(const ArgValue &v) { return false; }
  std::string AlterString() const { return "Allowed: 0/1/true/false(ignore capsules)"; }
  bool success = true;
  std::string name = "<Bool>";
};
/*
 * Partial template specialization for float type.
 * Float type has no custom alternative values.
 */
template <>
struct ArgResolver<float>
{
  float Dispatch(const ArgValue &v)
  {
    success = true;
    if (v.size() != 1)
    {
      success = false;
      return 0;
    }
    auto value = Trim(v[0]);
    float ret = 0;
    success = ParseValue<float>(value, &ret);
    return ret;
  }
  bool SetAlternative(const ArgValue &v) { return false; }
  std::string AlterString() const { return ""; }
  bool success = true;
  std::string name = "<Float>";
};
/*
 * Partial template specialization for int type.
 * Int type has no custom alternative values.
 */
template <>
struct ArgResolver<int>
{
  int Dispatch(const ArgValue &v)
  {
    success = true;
    if (v.size() != 1)
    {
      success = false;
      return 0;
    }
    auto value = Trim(v[0]);
    int ret = 0;
    success = ParseValue<int>(value, &ret);
    if (alters_.size())
    {
      if (std::find(alters_.begin(), alters_.end(), std::to_string(ret)) == alters_.end())
      {
        SLOG(ERROR) << "Fail to match arg alternative.";
        success = false;
        return 0;
      }
    }
    return ret;
  }
  bool SetAlternative(const ArgValue &v)
  {
    for (auto e_ : v)
    {
      auto value = Trim(e_);
      int r = 0;
      if (!ParseValue(value, &r))
      {
        return false;
      }
      alters_.push_back(Trim(e_));
    }
    return true;
  }
  std::string AlterString() const
  {
    std::string ret = "";
    if (alters_.size())
    {
      ret += "Allowed: ";
      for (size_t i = 0; i < alters_.size() - 1; ++i)
      {
        ret += alters_[i];
        ret += "/";
      }
      ret += alters_[alters_.size() - 1];
    }
    return ret;
  }
  ArgValue alters_ = {};
  bool success = true;
  std::string name = "<Int>";
};
/*
 * Partial template specialization for int64 type.
 * Int64 type has no custom alternative values.
 */
template <>
struct ArgResolver<int64_t>
{
  int64_t Dispatch(const ArgValue &v)
  {
    success = true;
    if (v.size() != 1)
    {
      success = false;
      return 0;
    }
    auto value = Trim(v[0]);
    int64_t ret = 0;
    success = ParseValue<int64_t>(value, &ret);
    if (alters_.size())
    {
      if (std::find(alters_.begin(), alters_.end(), std::to_string(ret)) == alters_.end())
      {
        SLOG(ERROR) << "Fail to match arg alternative.";
        success = false;
        return 0;
      }
    }
    return ret;
  }
  bool SetAlternative(const ArgValue &v)
  {
    for (auto e_ : v)
    {
      auto value = Trim(e_);
      int64_t r = 0;
      if (!ParseValue(value, &r))
      {
        return false;
      }
      alters_.push_back(Trim(e_));
    }
    return true;
  }
  std::string AlterString() const
  {
    std::string ret = "";
    if (alters_.size())
    {
      ret += "Allowed: ";
      for (size_t i = 0; i < alters_.size() - 1; ++i)
      {
        ret += alters_[i];
        ret += "/";
      }
      ret += alters_[alters_.size() - 1];
    }
    return ret;
  }
  ArgValue alters_ = {};
  bool success = true;
  std::string name = "<Int64>";
};
/*
 * Partial template specialization for more than two layers of nested vectors.
 * This type is not allowed for arg resolver.
 */
template <typename T>
struct ArgResolver<std::vector<std::vector<std::vector<T>>>>
{
};
/*
 * Partial template specialization for vector type.
 * Vector type's alternative values and limitations depends on its inner type.
 * Vector type with two layers of nested vectors uses comma as its first spacer,
 * and uses space as its second spacer.
 */
template <typename T>
struct ArgResolver<std::vector<T>>
{
  std::vector<T> Dispatch(const ArgValue &v)
  {
    success = true;
    std::vector<T> ret;
    ArgValue dispatch_v;
    if ((v.size() == 1) && inner_getter.name.find("Vec") == std::string::npos)
    {
      std::stringstream in(v[0]);
      std::string tmp;
      while (std::getline(in, tmp, ','))
      {
        dispatch_v.push_back(tmp);
      }
    }
    else
    {
      dispatch_v = v;
    }
    for (auto e_ : dispatch_v)
    {
      if (e_ == "_")
      {
        ret.push_back(inner_getter.Dispatch({}));
      }
      else
      {
        ret.push_back(inner_getter.Dispatch({e_}));
      }
      success = success && inner_getter.success;
    }
    return ret;
  }
  bool SetAlternative(const ArgValue &v) { return inner_getter.SetAlternative(v); }
  std::string AlterString() const
  {
    if (inner_getter.name.find("Vec") == std::string::npos)
    {
      return inner_getter.AlterString();
    }
    else
    {
      return "Use \'_\' as empty vector;" + inner_getter.AlterString();
    }
  }
  ArgResolver<T> inner_getter;
  bool success = true;
  std::string name = "<Vec" + inner_getter.name + ">";
};
/*
 * A base class for arg parsing with responsibility chain.
 * One arg will firstly try to consume args by its own method, then pass the
 * rest to the next.
 */
class ArgBase
{
public:
  ArgBase(const std::string &name) : name_(name) {}
  virtual ~ArgBase();
  void SetNext(ArgBase *next);
  bool ReadIn(Args *args);
  std::string DebugString() const;
  std::string Synopsis() const;

protected:
  std::string SynopCollect(int align, bool detailed) const;
  virtual std::string WriteOut() const = 0;
  virtual std::string Doc(int align, bool detailed) const = 0;
  virtual bool UpdateArg(Args *args) = 0;

protected:
  ArgBase *next_ = nullptr;
  ArgValue arg_;
  std::string name_;
  std::string description_ = "";
};

std::ostream &operator<<(std::ostream &os, const ArgBase &arg);
/*
 * An arg object is one single arg with its value. e.g.
 * --option_name <option_type> [optional/required]
 *           Allowed: alternative_inputs
 *           Default: default
 *           Description
 */
template <typename T>
class Arg : public ArgBase
{
public:
  Arg(const std::string &name) : ArgBase(name), required_(true) {}

  Arg<T> *SetAlternative(const ArgValue &value)
  {
    if (!r_.SetAlternative(value))
    {
      SLOG(ERROR) << "Internal error, " << Name() << " does not support alternative value setting";
      abort();
    }
    if (default_.size() > 0)
    {
      if (!Check(default_))
      {
        SLOG(ERROR) << "Internal error, invalid default value for arg: " << Name();
        abort();
      }
    }
    return this;
  }

  Arg<T> *SetDefault(const ArgValue &default_value)
  {
    required_ = false;
    if (default_value.size() > 0)
    {
      if (Check(default_value))
      {
        default_ = default_value;
        v_ = default_value;
      }
      else
      {
        SLOG(ERROR) << "Internal error, invalid default value for arg: " << Name();
        abort();
      }
    }
    return this;
  }

  Arg<T> *SetDescription(const std::string &desc)
  {
    description_ = desc;
    return this;
  }

  Arg<T> *SetOnList(ArgBase **head)
  {
    if (*head)
    {
      (*head)->SetNext(this);
    }
    else
    {
      (*head) = this;
    }
    return this;
  }

  std::tuple<bool, T> ValueOrNull()
  {
    if (v_.size() > 0)
    {
      return std::make_tuple<bool, T>(true, r_.Dispatch(v_));
    }
    else
    {
      return std::make_tuple<bool, T>(false, {});
    }
  }

private:
  std::string Name() const { return "--" + name_ + r_.name; }

  bool Check(const ArgValue &value)
  {
    USED_VAR(r_.Dispatch(value));
    return r_.success;
  }

  bool UpdateArg(Args *args) override final
  {
    auto arg = args->find(name_);
    if (name_ == "onnx")
    {
      return true;
    }
    if (arg == args->end())
    {
      if (required_)
      {
        SLOG(ERROR) << "Command line arg: " << Name() << " is required but not given.";
        return false;
      }
      return true;
    }
    else
    {
      if (arg->second.size() == 0)
      {
        if (required_)
        {
          SLOG(ERROR) << "Command line arg: " << Name() << " is required but not given.";
          return false;
        }
      }
      if (Check(arg->second))
      {
        v_ = arg->second;
        args->erase(arg);
        return true;
      }
      else
      {
        SLOG(ERROR) << "Command line arg: " << Name() << " miss its type restriction.";
        return false;
      }
    }
  }

  std::string WriteOut() const override final
  {
    std::stringstream ss;
    ss << std::setw(30) << std::left << Name() << ": ";
    for (auto e_ : v_)
    {
      ss << e_ << " ";
    }
    ss << "\n";
    return ss.str();
    ;
  }

  std::string Doc(int align, bool detailed) const override final
  {
    std::stringstream ss;
    if (detailed)
    {
      ss << std::string(align, ' ');
      ss << Name();
      if (required_)
      {
        ss << "[required]: \n";
      }
      else
      {
        ss << "[optional]: \n";
      }
      auto alter_ = r_.AlterString();
      if (!alter_.empty())
      {
        ss << std::string(align + 10, ' ');
        ss << alter_ << "\n";
      }
      if (default_.size() > 0)
      {
        ss << std::string(align + 10, ' ');
        ss << "Default: ";
        for (size_t i = 0; i < default_.size() - 1; ++i)
        {
          ss << default_[i] << " ";
        }
        ss << default_[default_.size() - 1] << "\n";
      }
      ss << std::string(align + 10, ' ');
      ss << description_ << "\n";
    }
    else
    {
      if (!required_)
      {
        ss << "[";
      }
      ss << Name();
      if (!required_)
      {
        ss << "]";
      }
      ss << " ";
    }
    return ss.str();
  }

  bool required_ = false;
  ArgResolver<T> r_;
  ArgValue v_;
  ArgValue default_;
};

template <class T>
bool HasValue(const std::tuple<bool, T> &t)
{
  return std::get<0>(t);
}

template <class T>
T Value(const std::tuple<bool, T> &t)
{
  return std::get<1>(t);
}
/*
 * An arglist object is a list of args with methods can reach each of them to
 * get values.
 * ArgListBase class are designed to be used with macro "DECLARE_ARG" below.
 */
class ArgListBase
{
public:
  void ReadIn(Args args);
  std::string DebugString() const;
  std::string Synopsis() const;
  virtual ~ArgListBase();

protected:
  ArgBase *front = nullptr;
};

std::ostream &operator<<(std::ostream &os, const ArgListBase &arg);
/*
 * Macro for arglist defination.
 * An arglist can be define like this:
 * class UserArgList : public ArgListBase {
 *   DECLARE_ARG(ARG_NAME, (ARG_TYPE))->SetDescription()
 *                                    ->SetDefault()
 *                                    ->SetAlternative();
 * };
 * If there is no default value set for an argument, it will be treated as a required argument.
 */
#define DECLARE_ARG(name, type)        \
public:                                \
  std::tuple<bool, UNPACK type> name() \
  {                                    \
    return __##name->ValueOrNull();    \
  }                                    \
                                       \
private:                               \
  Arg<UNPACK type> *__##name = (new Arg<UNPACK type>(#name))->SetOnList(&front)

#endif // PARAM_H_
