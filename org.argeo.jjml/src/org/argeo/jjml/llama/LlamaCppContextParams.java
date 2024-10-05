package org.argeo.jjml.llama;

/**
 * Context parameters.
 * 
 * @see llama.h - struct llama_context_params
 */
public class LlamaCppContextParams {
	int contextSize;
	int maxBatchSize;
	int physicalMaxBatchSize;
	int maxSequencesCount;
	int generationThreadCount;
	int batchThreadCount;

	int poolingTypeCode;

	boolean embeddings;

	LlamaCppContextParams() {
	}

	/**
	 * The default context parameters.
	 * 
	 * @see llama.h - llama_context_default_params()
	 */
	public static LlamaCppContextParams defaultContextParams() {
		LlamaCppNative.ensureLibrariesLoaded();
		return LlamaCppNative.newContextParams();
	}

	/**
	 * When used for embeddings, physical batch size must be the same as logical
	 * batch size.
	 */
	protected void ensureEmbeddingsBatchSizes() {
		if (isEmbeddings())
			setPhysicalMaxBatchSize(getMaxBatchSize());
	}

	public int getContextSize() {
		return contextSize;
	}

	public void setContextSize(int contextSize) {
		this.contextSize = contextSize;
	}

	public int getMaxBatchSize() {
		return maxBatchSize;
	}

	public void setMaxBatchSize(int maxBatchSize) {
		this.maxBatchSize = maxBatchSize;
		ensureEmbeddingsBatchSizes();
	}

	public int getPhysicalMaxBatchSize() {
		return physicalMaxBatchSize;
	}

	public void setPhysicalMaxBatchSize(int physicalMaxBatchSize) {
		if (isEmbeddings() && getMaxBatchSize() != physicalMaxBatchSize)
			throw new IllegalStateException(
					"When embeddings are used, physical batch size cannot be set differently than logical batch size.");
		this.physicalMaxBatchSize = physicalMaxBatchSize;
	}

	public int getMaxSequencesCount() {
		return maxSequencesCount;
	}

	public void setMaxSequencesCount(int maxSequencesCount) {
		this.maxSequencesCount = maxSequencesCount;
	}

	public int getGenerationThreadCount() {
		return generationThreadCount;
	}

	public void setGenerationThreadCount(int generationThreadCount) {
		this.generationThreadCount = generationThreadCount;
	}

	public int getBatchThreadCount() {
		return batchThreadCount;
	}

	public void setBatchThreadCount(int batchThreadCount) {
		this.batchThreadCount = batchThreadCount;
	}

	public LlamaCppPoolingType getPoolingType() {
		return LlamaCppPoolingType.byCode(poolingTypeCode);
	}

	public void setPoolingType(LlamaCppPoolingType poolingType) {
		this.poolingTypeCode = poolingType.getCode();
	}

	public boolean isEmbeddings() {
		return embeddings;
	}

	public void setEmbeddings(boolean embeddings) {
		this.embeddings = embeddings;
		ensureEmbeddingsBatchSizes();
	}

}
