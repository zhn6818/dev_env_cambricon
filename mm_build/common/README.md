# MagicMind C++ Common Samples

## Contents
| 模块名称     | 内容描述                                                                                |
|---|---|
| buffer       | 对一组推理输入、输出地址及Tensor描述符的对象封装，提供简单的可变复用机制|
| calib_data   | 对一组量化校准数据集的对象封装，提供随机初始化和文件读入机制            |
| container    | 对单例和自销毁智能指针的封装，可以自行管理销毁函数名称为Destroy的类对象 |
| data         | 对数据处理和读写的函数封装，包括读写数据，初始化与精度计算，上下溢转换等|
| device       | 对设备相关的宏/函数/对象封装，包括异常处理，设备状态，驱动队列抽象等    |
| timer        | 基本计时器封装                                                          |
| logger       | 基本日志系统                                                            |
| macros       | 常用检查宏封装，包括Status和bool的处理与返回                            |
| param        | 命令行读入参数的类封装，支持以--key value的形式注册命令行参数           |
| threadpool   | 线程池封装，支持动态扩张和静态初始化                                    |
| type         | 对MagicMind基础数据类型的进一步函数封装                                 |

## Notes

  - Common下模块不直接进行编译，具体使用见文件内注释和各个具体实例调用方法。
