#include <cassert>
#include <iostream>

#include <ggml.h>
#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppBackend.h" // IWYU pragma: keep

/*
 * Standard Java
 */
// METHODS
jmethodID Integer$valueOf;
jmethodID DoublePredicate$test;
jmethodID CompletionHandler$completed;
jmethodID CompletionHandler$failed;

/*
 * org.argeo.jjml.llama package
 */
jmethodID ModelParams$init;
jmethodID ContextParams$init;

/*
 * LOCAL
 */
static bool backend_initialized = false;

/** Initialization of common variables.*/
static void org_argeo_jjml_llama_(JNIEnv *env) {
	/*
	 * Standard Java
	 */
	jclass Integer = argeo::jni::find_jclass(env, "java/lang/Integer");
	Integer$valueOf = argeo::jni::jmethod_id_static(env, Integer, //
			"valueOf", "(I)Ljava/lang/Integer;");

	// METHODS
	jclass DoublePredicate = argeo::jni::find_jclass(env,
			"java/util/function/DoublePredicate");
	DoublePredicate$test = argeo::jni::jmethod_id(env, DoublePredicate, //
			"test", "(D)Z");

//	jclass IntBuffer = env->FindClass("java/nio/Buffer");
//	IntBuffer$limit = env->GetMethodID(IntBuffer, "limit", "()I");
//	IntBuffer$limitI = env->GetMethodID(IntBuffer, "limit",
//			"(I)Ljava/nio/Buffer;");
//	IntBuffer$position = env->GetMethodID(IntBuffer, "position", "()I");
//	IntBuffer$positionI = env->GetMethodID(IntBuffer, "position",
//			"(I)Ljava/nio/Buffer;");

	jclass CompletionHandler = argeo::jni::find_jclass(env,
			"java/nio/channels/CompletionHandler");
	CompletionHandler$completed = argeo::jni::jmethod_id(env, CompletionHandler,
			"completed", "(Ljava/lang/Object;Ljava/lang/Object;)V");
	CompletionHandler$failed = argeo::jni::jmethod_id(env, CompletionHandler,
			"failed", "(Ljava/lang/Throwable;Ljava/lang/Object;)V");

	/*
	 * org.argeo.jjml.llama package
	 */
	// We define the constructors here so that they fail right away when signatures change
	jclass ModelParams = argeo::jni::find_jclass(env, JCLASS_MODEL_PARAMS);
	ModelParams$init = argeo::jni::jmethod_id(env, ModelParams, //
			"<init>", "(IZZZ)V");
	jclass ContextParams = argeo::jni::find_jclass(env, JCLASS_CONTEXT_PARAMS);
	ContextParams$init = argeo::jni::jmethod_id(env, ContextParams, //
			"<init>", "(IIIIIIIIIFFFFFFIFIIZZZZ)V");
	// Tip: in order to find a constructor signature, use:
	// javap -s '../org.argeo.jjml/bin/org/argeo/jjml/llama/params/ContextParams.class'
}

/*
 * LOCAL UTILITIES
 */
static void jjml_llama_init_backend() {
	if (!backend_initialized) {
		llama_backend_init();
		backend_initialized = true;
	}
}

static void jjml_llama_free_backend() {
	if (backend_initialized) {
		llama_backend_free();
		backend_initialized = false;
	}
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

	// initialize llama.cpp backend
	jjml_llama_init_backend();

	return JNI_VERSION_10;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
	// free llama.cpp backend
	jjml_llama_free_backend();
}
/*
 * BACKEND
 */
//JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doInit(
//		JNIEnv *env, jclass) {
//	jjml_llama_init_backend();
//}
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doNumaInit(
		JNIEnv*, jclass, jint numaStrategy) {
	switch (numaStrategy) {
	case GGML_NUMA_STRATEGY_DISABLED:
		llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
		break;
	case GGML_NUMA_STRATEGY_DISTRIBUTE:
		llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
		break;
	case GGML_NUMA_STRATEGY_ISOLATE:
		llama_numa_init(GGML_NUMA_STRATEGY_ISOLATE);
		break;
	case GGML_NUMA_STRATEGY_NUMACTL:
		llama_numa_init(GGML_NUMA_STRATEGY_NUMACTL);
		break;
	case GGML_NUMA_STRATEGY_MIRROR:
		llama_numa_init(GGML_NUMA_STRATEGY_MIRROR);
		break;
	default:
		assert(!"Invalid NUMA strategy enum value");
		break;
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doDestroy(
		JNIEnv*, jclass) {
	jjml_llama_free_backend();
}

/*
 * COMMON NATIVE
 */
JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_supportsMmap(
		JNIEnv*, jclass) {
	return llama_supports_mmap();
}

JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_supportsMlock(
		JNIEnv*, jclass) {
	return llama_supports_mlock();
}

JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_supportsGpuOffload(
		JNIEnv*, jclass) {
	return llama_supports_gpu_offload();
}

