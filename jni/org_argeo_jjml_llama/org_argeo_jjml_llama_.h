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
 * LlamaCppModelParams
 */
#define LlamaCppModelParams(env) env->FindClass("org/argeo/jjml/llama/LlamaCppModelParams")
// FIELDS
#define LlamaCppModelParams$gpuLayerCount(env) env->GetFieldID(LlamaCppModelParams(env), "gpuLayersCount", "I")
#define LlamaCppModelParams$vocabOnly(env) env->GetFieldID(LlamaCppModelParams(env), "vocabOnly", "Z")
#define LlamaCppModelParams$useMlock(env) env->GetFieldID(LlamaCppModelParams(env), "useMlock", "Z")
// CONSTRUCTORS
#define LlamaCppModelParams$LlamaCppModelParams(env) env->GetMethodID(LlamaCppModelParams(env), "<init>","()V")

/*
 * LlamaCppContextParams
 */
#define LlamaCppContextParams(env) env->FindClass("org/argeo/jjml/llama/LlamaCppContextParams")
// FIELDS
// integers
#define LlamaCppContextParams$contextSize(env) env->GetFieldID(LlamaCppContextParams(env), "contextSize", "I")
#define LlamaCppContextParams$maxBatchSize(env) env->GetFieldID(LlamaCppContextParams(env), "maxBatchSize", "I")
#define LlamaCppContextParams$physicalMaxBatchSize(env) env->GetFieldID(LlamaCppContextParams(env), "physicalMaxBatchSize", "I")
#define LlamaCppContextParams$maxSequencesCount(env) env->GetFieldID(LlamaCppContextParams(env), "maxSequencesCount", "I")
#define LlamaCppContextParams$generationThreadCount(env) env->GetFieldID(LlamaCppContextParams(env), "generationThreadCount", "I")
#define LlamaCppContextParams$batchThreadCount(env) env->GetFieldID(LlamaCppContextParams(env), "batchThreadCount", "I")
// enums
#define LlamaCppContextParams$poolingTypeCode(env) env->GetFieldID(LlamaCppContextParams(env), "poolingTypeCode", "I")
// booleans
#define LlamaCppContextParams$embeddings(env) env->GetFieldID(LlamaCppContextParams(env), "embeddings", "Z")

// CONSTRUCTORS
#define LlamaCppContextParams$LlamaCppContextParams(env) env->GetMethodID(LlamaCppContextParams(env), "<init>","()V")
