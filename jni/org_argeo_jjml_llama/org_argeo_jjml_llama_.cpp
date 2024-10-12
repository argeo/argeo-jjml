/*
 * package org.argeo.jjml.llama
 */
#include <string>

#include <jni.h>

#include "argeo_jni_utils.h"

/*
 * Standard Java
 */
// METHODS
jmethodID DoublePredicate$test;
// EXCEPTIONS
jclass IllegalStateException;
jclass IllegalArgumentException;

/*
 * LlamaCppModelParams
 */
jclass LlamaCppModelParams;
// FIELDS
jfieldID LlamaCppModelParams$gpuLayerCount;
jfieldID LlamaCppModelParams$vocabOnly;
jfieldID LlamaCppModelParams$useMlock;
// CONSTRUCTORS
jmethodID LlamaCppModelParams$LlamaCppModelParams;

/*
 * LlamaCppContextParams
 */
jclass LlamaCppContextParams;
// FIELDS
// integers
jfieldID LlamaCppContextParams$contextSize;
jfieldID LlamaCppContextParams$maxBatchSize;
jfieldID LlamaCppContextParams$physicalMaxBatchSize;
jfieldID LlamaCppContextParams$maxSequencesCount;
jfieldID LlamaCppContextParams$generationThreadCount;
jfieldID LlamaCppContextParams$batchThreadCount;
// enums
jfieldID LlamaCppContextParams$poolingTypeCode;
// booleans
jfieldID LlamaCppContextParams$embeddings;
// CONSTRUCTORS
jmethodID LlamaCppContextParams$LlamaCppContextParams;

static void org_argeo_jjml_llama_(JNIEnv *env) {
	/*
	 * Standard Java
	 */
	// METHODS
	DoublePredicate$test = env->GetMethodID(
			env->FindClass("java/util/function/DoublePredicate"), "test",
			"(D)Z");
	// EXCEPTIONS
	IllegalStateException = env->FindClass("java/lang/IllegalStateException");
	IllegalArgumentException = env->FindClass(
			"java/lang/IllegalArgumentException");

	/*
	 * LlamaCppModelParams
	 */
	LlamaCppModelParams = env->FindClass(
			"org/argeo/jjml/llama/LlamaCppModelParams");

	// FIELDS
	LlamaCppModelParams$gpuLayerCount = env->GetFieldID(LlamaCppModelParams,
			"gpuLayersCount", "I");
	LlamaCppModelParams$vocabOnly = env->GetFieldID(LlamaCppModelParams,
			"vocabOnly", "Z");
	LlamaCppModelParams$useMlock = env->GetFieldID(LlamaCppModelParams,
			"useMlock", "Z");
	// CONSTRUCTORS
	LlamaCppModelParams$LlamaCppModelParams = env->GetMethodID(
			LlamaCppModelParams, "<init>", "()V");

	/*
	 * LlamaCppContextParams
	 */
	LlamaCppContextParams = env->FindClass(
			"org/argeo/jjml/llama/LlamaCppContextParams");
	// FIELDS
	// integers
	LlamaCppContextParams$contextSize = env->GetFieldID(LlamaCppContextParams,
			"contextSize", "I");
	LlamaCppContextParams$maxBatchSize = env->GetFieldID(LlamaCppContextParams,
			"maxBatchSize", "I");
	LlamaCppContextParams$physicalMaxBatchSize = env->GetFieldID(
			LlamaCppContextParams, "physicalMaxBatchSize", "I");
	LlamaCppContextParams$maxSequencesCount = env->GetFieldID(
			LlamaCppContextParams, "maxSequencesCount", "I");
	LlamaCppContextParams$generationThreadCount = env->GetFieldID(
			LlamaCppContextParams, "generationThreadCount", "I");
	LlamaCppContextParams$batchThreadCount = env->GetFieldID(
			LlamaCppContextParams, "batchThreadCount", "I");
	// enums
	LlamaCppContextParams$poolingTypeCode = env->GetFieldID(
			LlamaCppContextParams, "poolingTypeCode", "I");
	// booleans
	LlamaCppContextParams$embeddings = env->GetFieldID(LlamaCppContextParams,
			"embeddings", "Z");

	// CONSTRUCTORS
	LlamaCppContextParams$LlamaCppContextParams = env->GetMethodID(
			LlamaCppContextParams, "<init>", "()V");
}

/**
 *  Check the (unlikely) case where casting pointers as jlong would fail,
 *  since it is relied upon in order to map Java and native structures
 */
static void checkPlatformPointerSize(JNIEnv *env) {
	int pointerSize = sizeof(void*);
	int jlongSize = sizeof(jlong);
	if (pointerSize > jlongSize) {
		env->ThrowNew(IllegalStateException,
				"Platform pointer size is not supported");
		return;
	}
}

/*
 * JNI
 */
JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
	//std::cerr << "JNI_OnLoad";

	// load a new JNIEnv
	JNIEnv *env;
	vm->AttachCurrentThreadAsDaemon((void**) &env, nullptr);

	// make sure pointer casting will work
	// TODO provide alternative mechanism, such as a registry
	checkPlatformPointerSize(env);

	// cache Java references
	org_argeo_jjml_llama_(env);

	vm->DetachCurrentThread();
	return JNI_VERSION_10;
}
