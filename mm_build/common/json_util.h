/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef JSON_UTIL_H_
#define JSON_UTIL_H_
#include <vector>
#include <utility>
#include <map>
#include <string>
#include "third_party/json11/json11.h"
#include "common/logger.h"

bool ReadJsonFromFile(const std::string &json_name, json11::Json *obj);

bool ReadJsonFromString(const std::string &json_string, json11::Json *obj);

bool WriteJsonToFile(const std::string &json_name, json11::Json obj);

std::string WriteJsonToString(json11::Json obj);

std::string JsonTypeToString(json11::Json::Type e);

template <typename T>
struct JsonTypeResolver {};

#define MATCH_TYPE_AND_JSON_AND_GETTER(TYPE, ENUM, GETTER)           \
  template <>                                                        \
  struct JsonTypeResolver<TYPE> {                                    \
    static constexpr json11::Json::Type value = ENUM;                \
    TYPE Dispatch(json11::Json x) { return x.GETTER(); };            \
    json11::Json Patch(const TYPE &x) { return json11::Json(x); };   \
    bool Check(json11::Json x) {                                     \
      if (x.type() != value) {                                       \
        SLOG(ERROR) << "Json field type:" << JsonTypeToString(value) \
                    << ", but got:" << JsonTypeToString(x.type());   \
        return false;                                                \
      }                                                              \
      return true;                                                   \
    }                                                                \
  }

MATCH_TYPE_AND_JSON_AND_GETTER(int32_t, json11::Json::Type::NUMBER, int_value);
MATCH_TYPE_AND_JSON_AND_GETTER(float, json11::Json::Type::NUMBER, number_value);
MATCH_TYPE_AND_JSON_AND_GETTER(double, json11::Json::Type::NUMBER, number_value);
MATCH_TYPE_AND_JSON_AND_GETTER(bool, json11::Json::Type::BOOL, bool_value);
MATCH_TYPE_AND_JSON_AND_GETTER(std::string, json11::Json::Type::STRING, string_value);

#undef MATCH_TYPE_AND_JSON_AND_GETTER

template <typename T>
struct JsonTypeResolver<std::vector<T>> {
  std::vector<T> Dispatch(json11::Json x) {
    std::vector<T> ret;
    auto data = x.array_items();
    for (auto e_ : data) {
      ret.push_back(inner_getter.Dispatch(e_));
    }
    return ret;
  };
  json11::Json Patch(const std::vector<T> &x) {
    std::vector<json11::Json> ret;
    for (auto e_ : x) {
      ret.push_back(inner_getter.Patch(e_));
    }
    return json11::Json(ret);
  };
  bool Check(json11::Json x) {
    if (x.type() != value) {
      SLOG(ERROR) << "Json field type:" << JsonTypeToString(value) << ", but got:",
          JsonTypeToString(x.type());
      return false;
    } else {
      auto data = x.array_items();
      for (auto &e_ : data) {
        bool ret = inner_getter.Check(e_);
        if (!ret) {
          return ret;
        }
      }
      return true;
    }
  }
  JsonTypeResolver<T> inner_getter;
  static constexpr json11::Json::Type value = json11::Json::Type::ARRAY;
};

template <typename T>
struct JsonTypeResolver<std::map<std::string, T>> {
  std::map<std::string, T> Dispatch(json11::Json x) {
    std::map<std::string, T> ret;
    auto data = x.object_items();
    for (auto e_ : data) {
      ret.insert(std::make_pair(e_.first, inner_getter.Dispatch(e_.second)));
    }
    return ret;
  };
  json11::Json Patch(const std::map<std::string, T> &x) {
    std::map<std::string, json11::Json> ret;
    for (auto e_ : x) {
      ret.insert(std::make_pair(e_.first, inner_getter.Patch(e_.second)));
    }
    return json11::Json(ret);
  };
  bool Check(json11::Json x) {
    if (x.type() != value) {
      SLOG(ERROR) << "Json field type:" << JsonTypeToString(value) << ", but got:",
          JsonTypeToString(x.type());
      return false;
    } else {
      auto data = x.object_items();
      for (auto it = data.begin(); it != data.end(); ++it) {
        bool ret = inner_getter.Check(it->second);
        if (!ret) {
          return ret;
        }
      }
      return true;
    }
  }
  JsonTypeResolver<T> inner_getter;
  static constexpr json11::Json::Type value = json11::Json::Type::OBJECT;
};

template <typename T>
struct JsonTypeResolver<std::pair<std::string, T>> {
  std::pair<std::string, T> Dispatch(json11::Json x) {
    std::pair<std::string, T> ret;
    if (x.type() == value1) {
      auto data = x.object_items().begin();
      ret       = std::make_pair(data->first, inner_getter.Dispatch(data->second));
    } else {
      ret = std::make_pair(x.string_value(), T({}));
    }
    return ret;
  };
  json11::Json Patch(const std::pair<std::string, T> &x) {
    std::map<std::string, json11::Json> ret = {{x.first, inner_getter.Patch(x.second)}};
    return json11::Json(ret);
  };
  bool Check(json11::Json x) {
    if ((x.type() != value1) && x.type() != value2) {
      SLOG(ERROR) << "Json field type:" << JsonTypeToString(x.type()) << " or "
                  << JsonTypeToString(value2) << ", but got:",
          JsonTypeToString(value1);
      return false;
    } else if (x.type() == value1) {
      auto data = x.object_items();
      if (data.size() != 1) {
        SLOG(ERROR) << "Json field type:" << JsonTypeToString(value1)
                    << " with size 1, but got: " << data.size();
        return false;
      } else {
        return inner_getter.Check(data.begin()->second);
      }
    }
    return true;
  }
  JsonTypeResolver<T> inner_getter;
  static constexpr json11::Json::Type value1 = json11::Json::Type::OBJECT;
  static constexpr json11::Json::Type value2 = json11::Json::Type::STRING;
};

template <typename T>
bool CheckJsonTypeForKey(json11::Json root_obj, const std::string &key) {
  json11::Json json_val;
  if (!key.empty()) {
    json_val = root_obj[key];
  } else {
    json_val = root_obj;
  }
  if (json_val.is_null()) {
    SLOG(ERROR) << key << " doesnt existed in json obj";
    return false;
  }
  JsonTypeResolver<T> r;
  auto ret = r.Check(json_val);
  if (!ret) {
    SLOG(ERROR) << "Duraing query " << key;
  }
  return ret;
}

template <typename T>
bool GetJsonValueFromObj(json11::Json root_obj, const std::string &key, T *value) {
  auto ret = CheckJsonTypeForKey<T>(root_obj, key);
  if (ret) {
    json11::Json json_val;
    if (!key.empty()) {
      json_val = root_obj[key];
    } else {
      json_val = root_obj;
    }
    *value = JsonTypeResolver<T>().Dispatch(json_val);
  }
  return ret;
}

template <typename T>
json11::Json GetJsonObjFromValue(const std::string &key, const T &value) {
  auto obj = JsonTypeResolver<T>().Patch(value);
  if (!key.empty()) {
    std::map<std::string, json11::Json> ret = {{key, obj}};
    return json11::Json(ret);
  } else {
    return obj;
  }
}

json11::Json RemoveLineFromObj(json11::Json obj, const std::string &key);
json11::Json ReplaceLineFromObj(json11::Json obj1, json11::Json obj2, const std::string &key);
json11::Json GetLineFromObj(json11::Json obj, const std::string &key);

// map: {key : value,}
// array: [key, key, key]
std::string PrettyJson(const std::string &json_string);
bool IsEmpty(json11::Json obj);

#endif  // JSON_UTIL_H_
