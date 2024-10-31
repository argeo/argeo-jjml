/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_argeo_jjml_llama_LlamaCppSamplers */

#ifndef _Included_org_argeo_jjml_llama_LlamaCppSamplers
#define _Included_org_argeo_jjml_llama_LlamaCppSamplers
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_argeo_jjml_llama_LlamaCppSamplers
 * Method:    doInitGreedy
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitGreedy
  (JNIEnv *, jclass);

/*
 * Class:     org_argeo_jjml_llama_LlamaCppSamplers
 * Method:    doInitPenalties
 * Signature: (Lorg/argeo/jjml/llama/LlamaCppModel;IFFFZZ)J
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitPenalties
  (JNIEnv *, jclass, jobject, jint, jfloat, jfloat, jfloat, jboolean, jboolean);

/*
 * Class:     org_argeo_jjml_llama_LlamaCppSamplers
 * Method:    doInitTopK
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTopK
  (JNIEnv *, jclass, jint);

/*
 * Class:     org_argeo_jjml_llama_LlamaCppSamplers
 * Method:    doInitTopP
 * Signature: (FF)J
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTopP
  (JNIEnv *, jclass, jfloat, jfloat);

/*
 * Class:     org_argeo_jjml_llama_LlamaCppSamplers
 * Method:    doInitTemp
 * Signature: (F)J
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTemp
  (JNIEnv *, jclass, jfloat);

/*
 * Class:     org_argeo_jjml_llama_LlamaCppSamplers
 * Method:    doInitDist
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitDist__
  (JNIEnv *, jclass);

/*
 * Class:     org_argeo_jjml_llama_LlamaCppSamplers
 * Method:    doInitDist
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitDist__I
  (JNIEnv *, jclass, jint);

#ifdef __cplusplus
}
#endif
#endif
