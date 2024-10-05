package org.argeo.jjml.llama;

/** Initialization parameters of a model. */
public class LlamaCppModelParams {
	int gpuLayersCount;

	boolean vocabOnly;

	boolean useMlock;

	LlamaCppModelParams() {
	}

	/**
	 * The default model parameters.
	 * 
	 * @see llama.h - llama_model_default_params()
	 */
	public static LlamaCppModelParams defaultModelParameters() {
		LlamaCppNative.ensureLibrariesLoaded();
		return LlamaCppNative.newModelParameters();
	}

	public int getGpuLayersCount() {
		return gpuLayersCount;
	}

	public void setGpuLayersCount(int gpuLayersCount) {
		this.gpuLayersCount = gpuLayersCount;
	}

	public boolean isVocabOnly() {
		return vocabOnly;
	}

	public void setVocabOnly(boolean vocabOnly) {
		this.vocabOnly = vocabOnly;
	}

	public boolean isUseMlock() {
		return useMlock;
	}

	public void setUseMlock(boolean useMlock) {
		this.useMlock = useMlock;
	}

}
