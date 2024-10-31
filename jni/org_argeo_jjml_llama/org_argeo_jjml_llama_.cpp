/*
 * package org.argeo.jjml.llama
 */

#include "org_argeo_jjml_llama_.h"

/*
 * DEBUG & PERF
 */

/*
 * Standard Java
 */
// METHODS
jclass Integer;
jmethodID Integer$valueOf;

jmethodID DoublePredicate$test;
jmethodID LongSupplier$getAsLong;

jmethodID IntBuffer$limit;
jmethodID IntBuffer$limitI;
jmethodID IntBuffer$position;
jmethodID IntBuffer$positionI;

jmethodID CompletionHandler$completed;
jmethodID CompletionHandler$failed;

// EXCEPTIONS
jclass IllegalStateException;
jclass IllegalArgumentException;

/*
 * LlamaCppModelParams
 */
//jclass LlamaCppModelParams;
//// FIELDS
//jfieldID LlamaCppModelParams$gpuLayerCount;
//jfieldID LlamaCppModelParams$vocabOnly;
//jfieldID LlamaCppModelParams$useMlock;
//// CONSTRUCTORS
//jmethodID LlamaCppModelParams$LlamaCppModelParams;

/*
 * LlamaCppContextParams
 */
//jclass LlamaCppContextParams;
//// FIELDS
//// integers
//jfieldID LlamaCppContextParams$contextSize;
//jfieldID LlamaCppContextParams$maxBatchSize;
//jfieldID LlamaCppContextParams$physicalMaxBatchSize;
//jfieldID LlamaCppContextParams$maxSequencesCount;
//jfieldID LlamaCppContextParams$generationThreadCount;
//jfieldID LlamaCppContextParams$batchThreadCount;
//// enums
//jfieldID LlamaCppContextParams$poolingTypeCode;
//// booleans
//jfieldID LlamaCppContextParams$embeddings;
//// CONSTRUCTORS
//jmethodID LlamaCppContextParams$LlamaCppContextParams;

/*
 * NativeReference
 */
// METHODS
//jmethodID NativeReference$getPointer;

static void org_argeo_jjml_llama_(JNIEnv *env) {
	/*
	 * Standard Java
	 */
	Integer = env->FindClass("java/lang/Integer");
	Integer$valueOf = env->GetStaticMethodID(Integer, "valueOf",
			"(I)Ljava/lang/Integer;");

	// METHODS
	DoublePredicate$test = env->GetMethodID(
			env->FindClass("java/util/function/DoublePredicate"), "test",
			"(D)Z");
	LongSupplier$getAsLong = LongSupplier$getAsLong(env);

	jclass IntBuffer = env->FindClass("java/nio/IntBuffer");
	IntBuffer$limit = env->GetMethodID(IntBuffer, "limit", "()I");
	IntBuffer$limitI = env->GetMethodID(IntBuffer, "limit",
			"(I)Ljava/nio/IntBuffer;");
	IntBuffer$position = env->GetMethodID(IntBuffer, "position", "()I");
	IntBuffer$positionI = env->GetMethodID(IntBuffer, "position",
			"(I)Ljava/nio/IntBuffer;");

	// EXCEPTIONS
	IllegalStateException = env->FindClass("java/lang/IllegalStateException");
	IllegalArgumentException = env->FindClass(
			"java/lang/IllegalArgumentException");

	jclass CompletionHandler = env->FindClass(
			"java/nio/channels/CompletionHandler");
	CompletionHandler$completed = env->GetMethodID(CompletionHandler,
			"completed", "(Ljava/lang/Object;Ljava/lang/Object;)V");
	CompletionHandler$failed = env->GetMethodID(CompletionHandler, "failed",
			"(Ljava/lang/Throwable;Ljava/lang/Object;)V");
	/*
	 * LlamaCppModelParams
	 */
	// Note: jclass are apparently local references (TODO check it in details)
//	LlamaCppModelParams = static_cast<jclass>(env->NewWeakGlobalRef(
//			env->FindClass("org/argeo/jjml/llama/LlamaCppModelParams")));
//
//	// FIELDS
//	LlamaCppModelParams$gpuLayerCount = env->GetFieldID(LlamaCppModelParams,
//			"gpuLayersCount", "I");
//	LlamaCppModelParams$vocabOnly = env->GetFieldID(LlamaCppModelParams,
//			"vocabOnly", "Z");
//	LlamaCppModelParams$useMlock = env->GetFieldID(LlamaCppModelParams,
//			"useMlock", "Z");
//	// CONSTRUCTORS
//	LlamaCppModelParams$LlamaCppModelParams = env->GetMethodID(
//			LlamaCppModelParams, "<init>", "()V");

	/*
	 * LlamaCppContextParams
	 */
//	LlamaCppContextParams = static_cast<jclass>(env->NewWeakGlobalRef(
//			env->FindClass("org/argeo/jjml/llama/LlamaCppContextParams")));
//	// FIELDS
//	// integers
//	LlamaCppContextParams$contextSize = env->GetFieldID(LlamaCppContextParams,
//			"contextSize", "I");
//	LlamaCppContextParams$maxBatchSize = env->GetFieldID(LlamaCppContextParams,
//			"maxBatchSize", "I");
//	LlamaCppContextParams$physicalMaxBatchSize = env->GetFieldID(
//			LlamaCppContextParams, "physicalMaxBatchSize", "I");
//	LlamaCppContextParams$maxSequencesCount = env->GetFieldID(
//			LlamaCppContextParams, "maxSequencesCount", "I");
//	LlamaCppContextParams$generationThreadCount = env->GetFieldID(
//			LlamaCppContextParams, "generationThreadCount", "I");
//	LlamaCppContextParams$batchThreadCount = env->GetFieldID(
//			LlamaCppContextParams, "batchThreadCount", "I");
//	// enums
//	LlamaCppContextParams$poolingTypeCode = env->GetFieldID(
//			LlamaCppContextParams, "poolingTypeCode", "I");
//	// booleans
//	LlamaCppContextParams$embeddings = env->GetFieldID(LlamaCppContextParams,
//			"embeddings", "Z");
//	// CONSTRUCTORS
//	LlamaCppContextParams$LlamaCppContextParams = env->GetMethodID(
//			LlamaCppContextParams, "<init>", "()V");

	/*
	 * NativeReference
	 */
//	jclass NativeReference = env->FindClass(
//			"org/argeo/jjml/llama/NativeReference");
//	// METHODS
//	NativeReference$getPointer = env->GetMethodID(NativeReference, "getPointer",
//			"()J");
}

/*
 * JNI
 */
/** Called when the library is loaded, before any other function. */
JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
	//std::cerr << "JNI_OnLoad";

	// load a new JNIEnv
	JNIEnv *env;
	vm->AttachCurrentThreadAsDaemon((void**) &env, nullptr);

	// cache Java references
	org_argeo_jjml_llama_(env);

	vm->DetachCurrentThread();
	return JNI_VERSION_10;
}
