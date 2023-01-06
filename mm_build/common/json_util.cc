/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include <fstream>
#include <string>
#include <map>
#include "common/json_util.h"

static std::map<json11::Json::Type, std::string> kJsonTypeMap{
    {json11::Json::Type::NUL, "Null"},    {json11::Json::Type::NUMBER, "Number"},
    {json11::Json::Type::BOOL, "Bool"},   {json11::Json::Type::STRING, "String"},
    {json11::Json::Type::ARRAY, "Array"}, {json11::Json::Type::OBJECT, "Object"}};

bool ReadJsonFromFile(const std::string &json_name, json11::Json *obj) {
  if (json_name.empty()) {
    SLOG(ERROR) << "Parse json failed due to empty file name.";
    return false;
  }
  std::ifstream json(json_name.c_str());
  if (!json.is_open()) {
    SLOG(ERROR) << "Parse json failed due to open file failure. File name is " << json_name << ".";
    return false;
  }
  std::stringstream buffer;
  buffer << json.rdbuf();
  json.close();
  return ReadJsonFromString(buffer.str(), obj);
}

bool ReadJsonFromString(const std::string &json_string, json11::Json *obj) {
  std::string err;
  *obj = json11::Json::parse(json_string, err);
  if (obj->is_null()) {
    SLOG(ERROR) << "Parse json failed for: ";
    SLOG(ERROR) << json_string;
    SLOG(ERROR) << "Error msg: " << err;
    return false;
  }
  return true;
}

bool WriteJsonToFile(const std::string &json_name, json11::Json obj) {
  if (json_name.empty()) {
    SLOG(ERROR) << "Write json failed due to empty file name.";
    return false;
  }
  std::ofstream json(json_name.c_str());
  if (!json.is_open()) {
    SLOG(ERROR) << "Write json failed due to open file failure. File name is " << json_name << ".";
    return false;
  }
  std::string out = PrettyJson(WriteJsonToString(obj));
  json << out;
  json.close();
  return true;
}

std::string WriteJsonToString(json11::Json obj) {
  return obj.dump();
}

std::string JsonTypeToString(json11::Json::Type e) {
  auto iter = kJsonTypeMap.find(e);
  if (iter != kJsonTypeMap.end()) {
    return iter->second;
  } else {
    return "Invalid";
  }
}

json11::Json RemoveLineFromObj(json11::Json obj, const std::string &key) {
  if (obj.type() != json11::Json::Type::OBJECT) {
    SLOG(ERROR) << "Fail to remove line: " << key << " from a non-object json: " << obj.dump();
    SLOG(ERROR) << "Check your json input.";
    return obj;
  }
  std::map<std::string, json11::Json> obj_item = obj.object_items();
  obj_item.erase(key);
  return json11::Json(obj_item);
}

json11::Json ReplaceLineFromObj(json11::Json obj1, json11::Json obj2, const std::string &key) {
  if ((obj1.type() != json11::Json::Type::OBJECT) || (obj2.type() != json11::Json::Type::OBJECT)) {
    SLOG(ERROR) << "Fail to replace line: " << key << " from a non-object json:" << obj1.dump()
                << " to " << obj2.dump();
    SLOG(ERROR) << "Check your json input.";
    return obj1;
  }
  std::map<std::string, json11::Json> obj_item = obj1.object_items();
  obj_item[key] = obj2;
  return json11::Json(obj_item);
}

json11::Json GetLineFromObj(json11::Json obj, const std::string &key) {
  std::map<std::string, json11::Json> ret = {{key, obj[key]}};
  return json11::Json(ret);
}

std::string PrettyJson(const std::string &json_string) {
  std::string ret;
  int brackletstack = 0;
  int last_pos = 0;
  size_t i = 0;
  for (; i < json_string.size(); ++i) {
    if ((json_string[i] == '{') || (json_string[i] == '[')) {
      brackletstack += 1;
      ret = ret + json_string.substr(last_pos, i - last_pos + 1);
      ret = ret + "\n" + std::string(brackletstack * 2, ' ');
      last_pos = i + 1;
    }
    if ((json_string[i] == '}') || (json_string[i] == ']')) {
      brackletstack -= 1;
      ret = ret + json_string.substr(last_pos, i - last_pos);
      ret = ret + "\n" + std::string(brackletstack * 2, ' ');
      last_pos = i;
    }
    if (json_string[i] == ',') {
      ret = ret + json_string.substr(last_pos, i - last_pos + 1);
      ret = ret + "\n" + std::string(brackletstack * 2, ' ');
      last_pos = i + 2;
    }
  }
  ret = ret + json_string.substr(last_pos, i - last_pos + 1);
  return ret;
}

bool IsEmpty(json11::Json obj) {
  if (obj.type() == json11::Json::Type::OBJECT) {
    return (obj.object_items().size() == 0);
  } else if (obj.type() == json11::Json::Type::ARRAY) {
    return (obj.array_items().size() == 0);
  } else {
    return obj.is_null();
  }
}
