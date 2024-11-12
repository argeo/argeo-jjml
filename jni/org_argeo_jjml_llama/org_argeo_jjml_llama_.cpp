#include <cassert>

#include <ggml.h>
#include <llama.h>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppBackend.h" // IWYU pragma: keep

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
 * LOCAL
 */
static bool backend_initialized = false;

/** Initialization of common variables.*/
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

	jclass IntBuffer = env->FindClass("java/nio/Buffer");
	IntBuffer$limit = env->GetMethodID(IntBuffer, "limit", "()I");
	IntBuffer$limitI = env->GetMethodID(IntBuffer, "limit",
			"(I)Ljava/nio/Buffer;");
	IntBuffer$position = env->GetMethodID(IntBuffer, "position", "()I");
	IntBuffer$positionI = env->GetMethodID(IntBuffer, "position",
			"(I)Ljava/nio/Buffer;");

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

