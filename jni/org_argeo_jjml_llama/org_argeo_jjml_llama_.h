#include <string>

#include <argeo/argeo_jni.h>

/*
 * package org.argeo.jjml.llama
 */
#ifndef org_argeo_jjml_llama_h
#define org_argeo_jjml_llama_h

/** To be concatenated with Java class names.*/
const std::string JNI_PKG = "org/argeo/jjml/llama/";

const std::string JCLASS_MODEL_PARAMS = JNI_PKG + "params/ModelParams";
const std::string JCLASS_CONTEXT_PARAMS = JNI_PKG + "params/ContextParams";

// NOTE: Only standard Java or this package's classes should be cached,
// as the class loader may change in a dynamic environment (such as OSGi).
// NOTE: Java methods and fields can be cached, but not classes.
/*
 * Standard Java
 */
// METHODS
extern jmethodID Integer$valueOf;

extern jmethodID DoublePredicate$test;

//extern jmethodID IntBuffer$limit;
//extern jmethodID IntBuffer$position;

extern jmethodID CompletionHandler$completed;
extern jmethodID CompletionHandler$failed;

/*
 * org.argeo.jjml.llama package
 */
extern jmethodID ModelParams$init;
extern jmethodID ContextParams$init;

#endif
