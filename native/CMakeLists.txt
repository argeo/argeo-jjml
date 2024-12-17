cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.

# Java
find_package(JNI)


include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(org_argeo_jjml_llama)
