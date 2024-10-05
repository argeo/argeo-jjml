/*
 * package org.argeo.jjml.llama
 */

// We favor here clarity and consistency over optimization, since findClass() will be called repeatedly.
// Such patterns should therefore be used for non-critical code, such as setting parameters.
/*
 * Standard Java
 */
// METHOD
#define DoublePredicate$test(env) env->GetMethodID(env->FindClass("java/util/function/DoublePredicate"),"test","(D)Z")

/*
 * LlamaCppModel
 */
#define LlamaCppModelParams(env) env->FindClass("org/argeo/jjml/llama/LlamaCppModelParams")
// FIELDS
#define LlamaCppModelParams$gpuLayerCount(env) env->GetFieldID(LlamaCppModelParams(env), "gpuLayersCount", "I")
#define LlamaCppModelParams$vocabOnly(env) env->GetFieldID(LlamaCppModelParams(env), "vocabOnly", "Z")
#define LlamaCppModelParams$useMlock(env) env->GetFieldID(LlamaCppModelParams(env), "useMlock", "Z")
// METHODS
#define LlamaCppModelParams$LlamaCppModelParams(env) env->GetMethodID(LlamaCppModelParams(env), "<init>","()V")

