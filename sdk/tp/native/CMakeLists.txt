cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("argeo-jjml-tp-native" C CXX)

add_subdirectory(llama.cpp)

#set_target_properties(llama PROPERTIES LIBRARY_OUTPUT_DIRECTORY "../a2/lib/x86_64-linux-gnu")
