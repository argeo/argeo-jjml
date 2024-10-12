#include <jni.h>

#ifndef _argeo_jni_utils_h
#define _argeo_jni_utils_h

/*
 * Canonical JNI utilities to be inlined.
 */

///**
// * Throws a java.lang.IllegalStateException.
// *
// * @param env the JNI environment
// * @param message the error message
// */
//inline void ThrowIllegalStateException(JNIEnv *env, char const *message) {
//	jclass IllegalStateException = env->FindClass(
//			"java/lang/IllegalStateException");
//	env->ThrowNew(IllegalStateException, message);
//}

/**
 * Throws a java.lang.IllegalArgumentException.
 *
 * @param env the JNI environment
 * @param message the error message
 */
inline void ThrowIllegalArgumentException(JNIEnv *env, char const *message) {
	jclass IllegalArgumentException = env->FindClass(
			"java/lang/IllegalArgumentException");
	env->ThrowNew(IllegalArgumentException, message);
}

/**
 * Get a native pointer from a Java object implementing a getPointer() method returning a (Java) long.
 *
 * @param env the JNI environment
 * @param obj the Java object
 * @return the native pointer
 */
inline void* getPointer(JNIEnv *env, jobject obj) {
	jclass clss = env->GetObjectClass(obj);
	jmethodID method = env->GetMethodID(clss, "getPointer", "()J");
	if (nullptr == method) {
		ThrowIllegalArgumentException(env, "No getPointer() method available");
		return nullptr;
	}
	jlong pointer = env->CallLongMethod(obj, method);
	// return static_cast<void*>(pointer); // cannot work since jlong is a 'long int'
	return reinterpret_cast<void*>(pointer);
}


#endif
