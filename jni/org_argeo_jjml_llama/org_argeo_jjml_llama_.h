#include <jni.h>
#include <type_traits>

/*
 * package org.argeo.jjml.llama
 */
#ifndef org_argeo_jjml_llama_h
#define org_argeo_jjml_llama_h

// NOTE: Only standard Java or this package's classes should be cached,
// as the class loader may change in a dynamic environment (such as OSGi)
/*
 * Standard Java
 */
// METHODS
extern jmethodID DoublePredicate$test;
// EXCEPTIONS
extern jclass IllegalStateException;
extern jclass IllegalArgumentException;

/*
 * LlamaCppModelParams
 */
extern jclass LlamaCppModelParams;
// FIELDS
extern jfieldID LlamaCppModelParams$gpuLayerCount;
extern jfieldID LlamaCppModelParams$vocabOnly;
extern jfieldID LlamaCppModelParams$useMlock;
// CONSTRUCTORS
extern jmethodID LlamaCppModelParams$LlamaCppModelParams;

/*
 * LlamaCppContextParams
 */
extern jclass LlamaCppContextParams;
// FIELDS
// integers
extern jfieldID LlamaCppContextParams$contextSize;
extern jfieldID LlamaCppContextParams$maxBatchSize;
extern jfieldID LlamaCppContextParams$physicalMaxBatchSize;
extern jfieldID LlamaCppContextParams$maxSequencesCount;
extern jfieldID LlamaCppContextParams$generationThreadCount;
extern jfieldID LlamaCppContextParams$batchThreadCount;
// enums
extern jfieldID LlamaCppContextParams$poolingTypeCode;
// booleans
extern jfieldID LlamaCppContextParams$embeddings;
// CONSTRUCTORS
extern jmethodID LlamaCppContextParams$LlamaCppContextParams;

/*
 * NativeReference
 */
// METHODS
extern jmethodID NativeReference$getPointer;

/*
 * FUNCTIONS
 */
/** Cast a jlong to a pointer. */
template<typename T>
inline T getPointer(jlong pointer) {
	static_assert(std::is_pointer<T>::value);
	// Check the (unlikely) case where casting pointers as jlong would fail,
	// since it is relied upon in order to map Java and native structures
	// TODO provide alternative mechanism, such as a registry?
	static_assert(sizeof(T) <= sizeof(jlong));
	return reinterpret_cast<T>(pointer);
}

/**
 * Get a native pointer from a Java object implementing a method returning a (Java) long.
 *
 * @param env the JNI environment
 * @param obj the Java object
 * @return the native pointer
 */
template<typename T>
inline T getPointer(JNIEnv *env, jobject obj) {
	jlong pointer = env->CallLongMethod(obj, NativeReference$getPointer);
	return getPointer<T>(pointer);
}

#endif
