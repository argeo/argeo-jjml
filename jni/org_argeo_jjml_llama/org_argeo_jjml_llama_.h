#include <jni.h>
/*
 * package org.argeo.jjml.llama
 */

// NOTE: Only standard Java or this package's classes should be cached,
// as the class loader may change in a dynamic environment (such as OSGi)

/*
 * Standard Java
 */
// METHODS
extern jmethodID DoublePredicate$test;

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
