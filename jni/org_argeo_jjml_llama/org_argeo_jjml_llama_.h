#include <string>

#include <jni.h>
#include <jni_md.h>

#include <argeo/argeo_jni.h>

/*
 * package org.argeo.jjml.llama
 */
#ifndef org_argeo_jjml_llama_h
#define org_argeo_jjml_llama_h

/** To be concatenated with Java class names.*/
const std::string JNI_PKG = "org/argeo/jjml/llama/";

// NOTE: Only standard Java or this package's classes should be cached,
// as the class loader may change in a dynamic environment (such as OSGi)
/*
 * Standard Java
 */
// METHODS
extern jclass Integer;
extern jmethodID Integer$valueOf;

extern jmethodID DoublePredicate$test;

extern jmethodID IntBuffer$limit;
extern jmethodID IntBuffer$limitI;
extern jmethodID IntBuffer$position;
extern jmethodID IntBuffer$positionI;

extern jmethodID CompletionHandler$completed;
extern jmethodID CompletionHandler$failed;

// EXCEPTIONS
extern jclass IllegalStateException;
extern jclass IllegalArgumentException;

/*
 * LlamaCppModelParams
 */
//extern jclass LlamaCppModelParams;
//// FIELDS
//extern jfieldID LlamaCppModelParams$gpuLayerCount;
//extern jfieldID LlamaCppModelParams$vocabOnly;
//extern jfieldID LlamaCppModelParams$useMlock;
//// CONSTRUCTORS
//extern jmethodID LlamaCppModelParams$LlamaCppModelParams;

/*
 * LlamaCppContextParams
 */
//extern jclass LlamaCppContextParams;
//// FIELDS
//// integers
//extern jfieldID LlamaCppContextParams$contextSize;
//extern jfieldID LlamaCppContextParams$maxBatchSize;
//extern jfieldID LlamaCppContextParams$physicalMaxBatchSize;
//extern jfieldID LlamaCppContextParams$maxSequencesCount;
//extern jfieldID LlamaCppContextParams$generationThreadCount;
//extern jfieldID LlamaCppContextParams$batchThreadCount;
//// enums
//extern jfieldID LlamaCppContextParams$poolingTypeCode;
//// booleans
//extern jfieldID LlamaCppContextParams$embeddings;
//// CONSTRUCTORS
//extern jmethodID LlamaCppContextParams$LlamaCppContextParams;

/*
 * NativeReference
 */
// METHODS
//extern jmethodID NativeReference$getPointer;

/*
 * FUNCTIONS
 */
/**
 * Get a native pointer from a Java object implementing a method returning a (Java) long.
 *
 * @param env the JNI environment
 * @param obj the Java object
 * @return the native pointer
 */
//template<typename T>
//inline T getPointer(JNIEnv *env, jobject obj) {
//	jlong pointer = env->CallLongMethod(obj, NativeReference$getPointer);
//	return argeo::jni::getPointer<T>(pointer);
//}
//
#endif
