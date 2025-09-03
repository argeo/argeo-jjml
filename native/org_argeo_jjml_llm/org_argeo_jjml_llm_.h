#include <string>

#include <jni.h>
/*
 * package org.argeo.jjml.llama
 */
#ifndef org_argeo_jjml_llama_h
#define org_argeo_jjml_llama_h

/** To be concatenated with Java class names.*/
const std::string JNI_PKG = "org/argeo/jjml/llm/";

const std::string JCLASS_MODEL_PARAMS = JNI_PKG + "params/ModelParams";
const std::string JCLASS_CONTEXT_PARAMS = JNI_PKG + "params/ContextParams";
const std::string JCLASS_JAVA_SAMPLER = JNI_PKG + "LlamaCppJavaSampler";

// NOTE: Only standard Java or this package's classes should be cached,
// as the class loader may change in a dynamic environment (such as OSGi).
// NOTE: Java methods and fields can be cached, but not classes.
/*
 * Standard Java
 */
// METHODS
extern jmethodID Integer__valueOf;
extern jmethodID DoublePredicate__test;
extern jmethodID CompletionHandler__completed;
extern jmethodID CompletionHandler__failed;

/*
 * org.argeo.jjml.llama package
 */
extern jmethodID LlamaCppJavaSampler__apply;
extern jmethodID LlamaCppJavaSampler__accept;
extern jmethodID LlamaCppJavaSampler__reset;

/*
 * org.argeo.jjml.llama.params package
 */
extern jmethodID ModelParams__init;
extern jmethodID ContextParams__init;

#endif
