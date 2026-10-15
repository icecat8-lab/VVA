cmake_minimum_required(VERSION 3.22.1)

add_library(vva_core STATIC 
    src/vva.cpp 
    src/scheduler.cpp
)

target_include_directories(vva_core PUBLIC include)
