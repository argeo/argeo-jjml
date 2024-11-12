#ifndef argeo_jni_h
#define argeo_jni_h

#include <cstdarg>
#include <string>
#include <stdexcept>

#include <jni.h>

/*
 * PRE-PROCESSING
 */
//#define ARGEO_PERF 1
#ifdef ARGEO_PERF
#include <chrono>
#include <iomanip>
#include <iostream>

#define PERF_DURATION(_begin_) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _begin_).count()
#define PERF_NS(_begin_, msg) "## ARGEO_PERF ## " << std::setfill(' ') << std::setw(16) << PERF_DURATION(_begin_) << " ns " << msg << std::endl
#define PERF_BEGIN() auto _begin_ = std::chrono::high_resolution_clock::now()
#define PERF_END(msg) std::cout << PERF_NS(_begin_, msg)
#else
#define PERF_DURATION(_begin_)
#define PERF_NS(_begin_, msg)
#define PERF_BEGIN()
#define PERF_END(msg)
#endif

/*
 * NAMESPACE INDEPENDENT
 */

namespace argeo::jni {
/*
 * JAVA MACROS
 */
// exceptions
#define RuntimeException(env) env->FindClass("java/lang/RuntimeException")
#define IllegalArgumentException(env) env->FindClass("java/lang/IllegalArgumentException")
#define IndexOutOfBoundsException(env) env->FindClass("java/lang/IndexOutOfBoundsException")
#define IllegalStateException(env) env->FindClass("java/lang/IllegalStateException")

#define LongSupplier$getAsLong(env) env->GetMethodID( \
		env->FindClass("java/util/function/LongSupplier"), "getAsLong", "()J")

/*
 * EXCEPTION HANDLING
 */
/** Catches a standard C++ exception and throws an unchecked Java exception.
 * Does nothing if a Java execution is already pending
 * (that is, the already thrown java exception has priority.
 * A best effort is made to use an appropriate standard Java exception type.
 * Returns a <code>nullptr</code> as a convenience,
 * so that a call to it can be directly returned as the result of a (failed) non-void C++ function.
 * */
inline std::nullptr_t throw_to_java(JNIEnv *env, const std::exception &ex) {
	if (env->ExceptionCheck())
		return nullptr;
	if (typeid(std::invalid_argument) == typeid(ex)) {
		env->ThrowNew(IllegalArgumentException(env), ex.what());
	} else if (typeid(std::range_error) == typeid(ex)) {
		env->ThrowNew(IndexOutOfBoundsException(env), ex.what());
	} else if (typeid(std::runtime_error) == typeid(ex)) {
		env->ThrowNew(IllegalStateException(env), ex.what());
	} else {
		env->ThrowNew(RuntimeException(env), ex.what());
	}
	return nullptr;
}
/*
 * SAFE FIELDS AN METHODS ACCESSORS
 */
// ! We assume modified UTF-8 will work like C-strings
/** The name of this Java class. */
inline std::string jclass_name(JNIEnv *env, jclass clazz) {
	// we first need to get the class descriptor as an object ...
	jclass clsObj = env->GetObjectClass(clazz);
	// ... in order to find the proper method ...
	jmethodID Class$getName = env->GetMethodID(clsObj, "getName",
			"()Ljava/lang/String;");
	// ... to apply on the class passed as argument
	jstring name = (jstring) env->CallObjectMethod(clazz, Class$getName);
	jsize length = env->GetStringLength(name);
	std::string str;
	str.resize(length);
	env->GetStringUTFRegion(name, 0, length, str.data());
	return str;
}

/** This Java field. */
inline jfieldID jfield_id(JNIEnv *env, jclass clazz, std::string name,
		std::string sig) {
	jfieldID res = env->GetFieldID(clazz, name.c_str(), sig.c_str());
	if (res == NULL)
		throw std::invalid_argument(
				"Invalid field '" + name + "' with signature " + sig
						+ " of class " + jclass_name(env, clazz));
	return res;
}

/** This Java method. */
inline jmethodID jmethod_id(JNIEnv *env, jclass clazz, std::string name,
		std::string sig) {
	jmethodID res = env->GetMethodID(clazz, name.c_str(), sig.c_str());
	if (res == NULL)
		throw std::invalid_argument(
				"Invalid method '" + name + "' with signature " + sig
						+ " of class " + jclass_name(env, clazz));
	return res;
}

/** This Java static method. */
inline jmethodID jmethod_id_static(JNIEnv *env, jclass clazz, std::string name,
		std::string sig) {
	jmethodID res = env->GetStaticMethodID(clazz, name.c_str(), sig.c_str());
	if (res == NULL)
		throw std::invalid_argument(
				"Invalid static method '" + name + "' with signature " + sig
						+ " of class " + jclass_name(env, clazz));
	return res;
}

/** This Java class. */
inline jclass find_jclass(JNIEnv *env, std::string name) {
	jclass res = env->FindClass(name.c_str());
	if (res == NULL)
		throw std::invalid_argument("Invalid class " + name);
	return res;
}

/*
 * CALLBACKS
 */
/**
 * A structure holding the reference required for executing callback in the Java code.
 */
typedef struct java_callback {
	jobject callback = nullptr;
	jmethodID method = nullptr;
	JavaVM *jvm = nullptr;
} java_callback;

/**
 * Return the JNI environment for the current thread or attach it if it was detached.
 * @return true if the current thread was detached before this call.
 */
inline bool load_thread_jnienv(JavaVM *jvm, void **penv) {
	jint status = jvm->GetEnv(penv, JNI_VERSION_10);
	bool wasDetached = true;
	if (JNI_EVERSION == status) {
		// JNI version unsupported
		// TODO throw exception?
		wasDetached = true;
	} else if (JNI_EDETACHED == status) {
		wasDetached = true;
		jvm->AttachCurrentThreadAsDaemon(penv, nullptr);
	} else {
		//std::assert(threadEnv != nullptr,"");
		wasDetached = false;
	}
	return wasDetached;
}

/** Calls the callback's method on the provided Java object.*/
inline jboolean exec_boolean_callback(java_callback *cb, ...) {
	JNIEnv *threadEnv;
	bool wasDetached = load_thread_jnienv(cb->jvm, (void**) &threadEnv);
	jobject obj = threadEnv->NewLocalRef(cb->callback);

	// process variable arguments
	va_list args;
	jboolean result;
	va_start(args, cb);
	result = threadEnv->CallBooleanMethodV(obj, cb->method, args);
	va_end(args);

	if (wasDetached)
		cb->jvm->DetachCurrentThread();
	return result;
}

/*
 * POINTERS
 */

/** Cast a jlong to a pointer. */
template<typename T>
inline T as_pointer(jlong pointer) {
	static_assert(std::is_pointer<T>::value);
	// Check the (unlikely) case where casting pointers as jlong would fail,
	// since it is relied upon in order to map Java and native structures
	// TODO provide alternative mechanism, such as a registry?
	static_assert(sizeof(T) <= sizeof(jlong));
	return reinterpret_cast<T>(pointer);
}

/** Cast a java.util.function.LongSupplier to a native pointer.
 *
 * @param env the JNI environment
 * @param reference the Java object implementing java.util.function.LongSupplier
 * @return the native pointer
 */
template<typename T>
inline T as_pointer(JNIEnv *env, jobject reference) {
	jlong pointer = env->CallLongMethod(reference, LongSupplier$getAsLong(env));
	return as_pointer<T>(pointer);
}

}  // namespace argeo::jni
#endif
